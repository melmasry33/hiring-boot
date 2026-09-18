"""
Telegram front end.

Every non-command message goes straight into the agent's persistent thread —
there is no keyword routing, no "is this a URL?" branch. That is the point: the
bot is a chat window onto one continuous agent conversation, so "find me AI jobs
in Cairo", a pasted JD, "make that email shorter" and "what did I apply to this
week?" are all handled by the same loop with the same context.

Commands exist only for things that are genuinely out-of-band: resetting the
thread, forcing a send, reading the tracker, and checking config.
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import graph_agent as agent
import email_sender
import store
from config import (
    LLM_MODEL,
    SMTP_USER,
    TELEGRAM_ALLOWED_USER_IDS,
    TELEGRAM_BOT_TOKEN,
    logger,
    startup_report,
)
from health import start_health_server

TELEGRAM_MAX = 3900  # under the 4096 hard limit, leaving room for entities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_allowed(user_id: int) -> bool:
    if not TELEGRAM_ALLOWED_USER_IDS:
        return True
    return user_id in TELEGRAM_ALLOWED_USER_IDS


async def deny(update: Update) -> None:
    uid = update.effective_user.id if update.effective_user else "?"
    logger.warning(f"Blocked message from unauthorised user {uid}")
    if update.message:
        await update.message.reply_text(
            f"This is a private assistant. Your Telegram ID is {uid} — "
            "add it to TELEGRAM_ALLOWED_USER_ID if it's yours."
        )


def chunks(text: str, size: int = TELEGRAM_MAX):
    """Split long replies on paragraph boundaries where possible."""
    text = text or ""
    while len(text) > size:
        cut = text.rfind("\n\n", 0, size)
        if cut < size // 2:
            cut = text.rfind("\n", 0, size)
        if cut < size // 2:
            cut = size
        yield text[:cut]
        text = text[cut:].lstrip()
    if text:
        yield text


async def reply_long(update: Update, text: str) -> None:
    target = update.message or (update.callback_query.message if update.callback_query else None)
    if not target:
        return
    for part in chunks(text):
        await target.reply_text(part, disable_web_page_preview=True)


async def keep_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int, stop: asyncio.Event) -> None:
    """Hold the 'typing…' indicator while the agent works (it expires every ~5s)."""
    while not stop.is_set():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except TelegramError:
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.5)
        except asyncio.TimeoutError:
            continue


def approval_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Approve & send", callback_data="approve_send"),
                InlineKeyboardButton("✋ Don't send", callback_data="reject_send"),
            ]
        ]
    )


# ---------------------------------------------------------------------------
# Rendering an agent turn
# ---------------------------------------------------------------------------


async def render_turn(update: Update, user_id: int, turn: "agent.AgentTurn") -> None:
    target = update.message or (update.callback_query.message if update.callback_query else None)

    for att in turn.attachments:
        path = att.get("path")
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    await target.reply_document(
                        document=fh,
                        filename=os.path.basename(path),
                        caption=att.get("caption", "")[:1000],
                    )
            except Exception as e:
                logger.error(f"Could not send attachment {path}: {e}")

    if turn.status == "ERROR":
        await reply_long(update, f"⚠️ {turn.error}")
        return

    if turn.text:
        await reply_long(update, turn.text)

    if turn.status == "NEEDS_APPROVAL" and turn.approval:
        ap = turn.approval
        store.save_draft(user_id, {**(store.load_draft(user_id) or {}), "pending_approval": ap})

        if not ap.get("to") or "@" not in ap.get("to", ""):
            await target.reply_text(
                "I don't have a recipient address for this one. Send it as:\n"
                "/send someone@company.com"
            )
            return

        attach_note = (
            f"\n📎 Attachment: {os.path.basename(ap['pdf_path'])}"
            if ap.get("pdf_path") and os.path.exists(ap["pdf_path"])
            else "\n📎 No CV attached (none generated yet)."
        )
        preview = (
            f"✉️ Ready to send\n\n"
            f"To: {ap['to']}\n"
            f"Subject: {ap['subject']}\n"
            f"{attach_note}\n\n"
            f"{ap['body']}"
        )
        parts = list(chunks(preview))
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            await target.reply_text(
                part,
                reply_markup=approval_keyboard() if is_last else None,
                disable_web_page_preview=True,
            )


async def run_agent(update: Update, context: ContextTypes.DEFAULT_TYPE, **kwargs) -> None:
    """Run one agent turn with a typing indicator and hard error containment."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    stop = asyncio.Event()
    typing = asyncio.create_task(keep_typing(context, chat_id, stop))
    try:
        turn = await agent.run_turn(user_id, **kwargs)
    except Exception as e:
        logger.error(f"Agent crashed for user {user_id}: {e}", exc_info=True)
        turn = agent.AgentTurn(status="ERROR", error=f"Something broke on my side: {e}")
    finally:
        stop.set()
        try:
            await typing
        except Exception:
            pass
    await render_turn(update, user_id, turn)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user.id):
        return await deny(update)

    name = update.effective_user.first_name or "there"
    await update.message.reply_text(
        f"Hi {name} — I'm your career agent, and I actually do the work.\n\n"
        "Talk to me normally. For example:\n"
        "• “Find AI Engineer jobs in Cairo posted this week”\n"
        "• paste a job link or the description text\n"
        "• “Is this a good fit for me?”\n"
        "• “Make the email shorter and less formal”\n"
        "• “What did I apply to this week?”\n"
        "• “Add my new project X to my profile”\n\n"
        "I'll read your full profile, judge the fit honestly (including the gaps), "
        "build a tailored CV PDF and draft the outreach email. Nothing gets sent "
        "until you tap Approve.\n\n"
        "Commands: /profile /apps /stats /send /reset /whoami /diag"
    )


async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user.id):
        return await deny(update)
    await update.message.reply_text("👤 Your profile on file:\n\n" + store.profile_summary_text())


async def cmd_apps(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user.id):
        return await deny(update)
    apps = store.load_applications()[-15:]
    if not apps:
        await update.message.reply_text("No applications tracked yet.")
        return
    lines = []
    for a in reversed(apps):
        date = time.strftime("%d %b", time.localtime(a.get("created_at", 0)))
        score = f" · {a['match_score']}%" if a.get("match_score") else ""
        lines.append(f"• {date} — {a.get('role','?')} @ {a.get('company','?')} [{a.get('status','?')}]{score}")
    await reply_long(update, "📋 Recent applications\n\n" + "\n".join(lines))


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user.id):
        return await deny(update)
    s = store.application_stats()
    by = "\n".join(f"  {k}: {v}" for k, v in sorted(s["by_status"].items())) or "  (none)"
    await update.message.reply_text(
        f"📊 Applications\n\nTotal: {s['total']}\nLast 7 days: {s['last_7_days']}\n\nBy status:\n{by}"
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user.id):
        return await deny(update)
    uid = update.effective_user.id
    store.clear_conversation(uid)
    await update.message.reply_text(
        "🧹 Conversation cleared. Your profile and application history are untouched."
    )


async def cmd_whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    await update.message.reply_text(
        f"Telegram ID: {u.id}\nUsername: @{u.username or '—'}\n"
        f"Authorised: {'yes' if is_allowed(u.id) else 'no'}"
    )


async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Config self-check — the first thing to run after deploying."""
    if not is_allowed(update.effective_user.id):
        return await deny(update)
    profile = store.load_profile()
    problems = startup_report()
    lines = [
        "🔧 Diagnostics",
        f"Model: {LLM_MODEL}",
        f"LLM key: {'set' if agent.get_client() else 'MISSING'}",
        f"SMTP: {'set (' + SMTP_USER + ')' if SMTP_USER else 'MISSING'}",
        f"Profile: {'loaded — ' + (profile.get('name') or '?') if profile else 'MISSING'}",
        f"Access control: {'restricted to ' + str(TELEGRAM_ALLOWED_USER_IDS) if TELEGRAM_ALLOWED_USER_IDS else 'OPEN TO EVERYONE'}",
        f"Applications logged: {store.application_stats()['total']}",
    ]
    if problems:
        lines.append("\nIssues:\n" + "\n".join(f"• {p}" for p in problems))
    await reply_long(update, "\n".join(lines))


async def cmd_send(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Force-send the staged draft to an explicit address."""
    if not is_allowed(update.effective_user.id):
        return await deny(update)
    uid = update.effective_user.id
    draft = store.load_draft(uid)
    if not draft:
        await update.message.reply_text("No draft staged yet. Send me a job first.")
        return
    args = context.args or []
    if not args or "@" not in args[0]:
        await update.message.reply_text("Usage: /send recruiter@company.com")
        return

    to = args[0].strip()
    pending = draft.get("pending_approval") or {}
    await update.message.reply_text(f"📤 Sending to {to} …")
    result = await email_sender.send_application_email(
        recipient_email=to,
        subject=pending.get("subject") or draft.get("email_subject", ""),
        body=pending.get("body") or draft.get("email_body", ""),
        cv_file_path=draft.get("pdf_path"),
        approved=True,
    )
    await _after_send(update, context, uid, draft, to, result)


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user.id):
        return await deny(update)
    store.clear_draft(update.effective_user.id)
    await update.message.reply_text("Draft discarded. The conversation is still here.")


# ---------------------------------------------------------------------------
# Message + callback handlers
# ---------------------------------------------------------------------------


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user.id):
        return await deny(update)
    text = (update.message.text or "").strip()
    if not text:
        return
    await run_agent(update, context, user_message=text)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = update.effective_user.id
    if not is_allowed(uid):
        await query.edit_message_text("Not authorised.")
        return

    draft = store.load_draft(uid) or {}
    pending: Optional[Dict[str, Any]] = draft.get("pending_approval")
    if not pending:
        await query.edit_message_text("That draft has expired. Send me the job again.")
        return

    if query.data == "reject_send":
        store.save_draft(uid, {k: v for k, v in draft.items() if k != "pending_approval"})
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("Held back — nothing was sent.")
        await run_agent(
            update,
            context,
            resume_tool_result={
                "tool_call_id": pending["tool_call_id"],
                "content": {
                    "sent": False,
                    "reason": "The user declined to send this email. Ask what they want changed.",
                },
            },
        )
        return

    if query.data == "approve_send":
        await query.edit_message_reply_markup(reply_markup=None)
        status = await query.message.reply_text("📤 Sending …")
        result = await email_sender.send_application_email(
            recipient_email=pending.get("to", ""),
            subject=pending.get("subject", ""),
            body=pending.get("body", ""),
            cv_file_path=pending.get("pdf_path"),
            approved=True,
        )
        try:
            await status.delete()
        except TelegramError:
            pass
        await _after_send(update, context, uid, draft, pending.get("to", ""), result, pending)


async def _after_send(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    uid: int,
    draft: Dict[str, Any],
    to: str,
    result: Dict[str, Any],
    pending: Optional[Dict[str, Any]] = None,
) -> None:
    """Report the send, update the tracker, and feed the outcome back to the agent."""
    target = update.message or update.callback_query.message

    if result.get("success"):
        store.log_application(
            {
                "company": draft.get("company", ""),
                "role": draft.get("role", ""),
                "job_url": draft.get("job_url", ""),
                "status": "sent",
                "notes": f"Sent to {to}",
            }
        )
        store.save_draft(uid, {k: v for k, v in draft.items() if k != "pending_approval"})
        await target.reply_text(
            f"✅ Sent to {to}\nSubject: {draft.get('email_subject') or (pending or {}).get('subject','')}"
        )
    else:
        await target.reply_text(f"❌ Not sent — {result.get('message')}")

    if pending:
        await run_agent(
            update,
            context,
            resume_tool_result={
                "tool_call_id": pending["tool_call_id"],
                "content": {
                    "sent": bool(result.get("success")),
                    "recipient": to,
                    "status": result.get("status"),
                    "message": result.get("message"),
                },
            },
        )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled Telegram error", exc_info=context.error)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def _post_init(application) -> None:
    from telegram import BotCommand

    await application.bot.set_my_commands(
        [
            BotCommand("start", "What this bot does"),
            BotCommand("profile", "Show the profile on file"),
            BotCommand("apps", "Recent applications"),
            BotCommand("stats", "Application stats"),
            BotCommand("send", "Send the staged draft to an address"),
            BotCommand("cancel", "Discard the staged draft"),
            BotCommand("reset", "Clear the conversation"),
            BotCommand("diag", "Configuration self-check"),
            BotCommand("whoami", "Show your Telegram ID"),
        ]
    )
    me = await application.bot.get_me()
    logger.info(f"Connected to Telegram as @{me.username} (id {me.id})")


def main() -> None:
    store.bootstrap()

    for problem in startup_report():
        (logger.error if problem.startswith("FATAL") else logger.warning)(problem)

    if not TELEGRAM_BOT_TOKEN:
        logger.error("Cannot start without TELEGRAM_BOT_TOKEN. Exiting.")
        sys.exit(1)

    start_health_server()

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(_post_init)
        .concurrent_updates(True)
        .build()
    )

    application.add_handler(CommandHandler(["start", "help"], cmd_start))
    application.add_handler(CommandHandler("profile", cmd_profile))
    application.add_handler(CommandHandler(["apps", "applications"], cmd_apps))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("reset", cmd_reset))
    application.add_handler(CommandHandler("cancel", cmd_cancel))
    application.add_handler(CommandHandler("send", cmd_send))
    application.add_handler(CommandHandler(["whoami", "id"], cmd_whoami))
    application.add_handler(CommandHandler(["diag", "health"], cmd_diag))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    application.add_error_handler(on_error)

    logger.info(f"Career agent starting — model {LLM_MODEL}")
    # drop_pending_updates avoids replaying a backlog of messages after a redeploy.
    application.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
