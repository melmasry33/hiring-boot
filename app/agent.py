"""
The agent.

What makes this a real agent rather than a prompt template:

- It is CONVERSATIONAL. There is one persistent thread per Telegram user,
  stored on disk. Any message — "hi", "what did I apply to this week?", a
  pasted JD, "make the email shorter" — goes into the same thread, and the
  model decides what to do with it. The old version treated literally every
  message as a new job posting, which is why it could not hold a conversation.

- It CHOOSES ITS ACTIONS. It can read the full profile, search LinkedIn, read a
  posting, score the match deterministically, edit the stored profile, build a
  tailored CV, look up its own application history, and request permission to
  send an email. Nine tools, called in whatever order the situation needs.

- It ASKS instead of guessing. There is no special "ask_user" tool because
  none is needed: when the model replies with plain text, that text goes to the
  user and the thread simply waits. The next message resumes the same reasoning
  thread, with full history.

- It CANNOT send anything on its own. `send_email` is an interrupting tool: it
  returns control to bot.py with a pending-approval payload, and the send only
  happens after the human taps Approve. The tool result is then fed back so the
  agent knows what happened and can carry on talking about it.

- It is HONEST. `build_application` forces a separate `gap_notes` field, so the
  model has to state what the profile does not cover instead of quietly padding.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

import cv_generator
import jobs as job_source
import matching
import store
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MAX_HISTORY_MESSAGES,
    MAX_TOOL_STEPS,
    logger,
)

# Tools that stop the loop and hand control back to the human.
INTERRUPTING_TOOLS = {"send_email"}

# Profile fields the agent is allowed to rewrite through update_profile.
EDITABLE_PROFILE_FIELDS = {
    "name", "headline", "location", "phone", "email", "linkedin", "github",
    "summary", "target_roles", "experience", "projects", "skills",
    "skills_categories", "education", "certifications", "languages",
}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_profile",
            "description": (
                "Read the candidate's COMPLETE stored profile: every experience entry with "
                "full bullets, every project with bullets and tech stack, all skill "
                "categories, education and certifications. Call this before writing any CV "
                "or email content, and whenever the user asks what you know about them."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_profile",
            "description": (
                "Permanently update one field of the stored profile — use when the user "
                "tells you something new about themselves (a new project, a new job, a "
                "corrected phone number, an extra skill). Read the profile first so you "
                "write back the full corrected value, not a fragment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": (
                            "One of: name, headline, location, phone, email, linkedin, "
                            "github, summary, target_roles, experience, projects, skills, "
                            "skills_categories, education, certifications, languages."
                        ),
                    },
                    "value_json": {
                        "type": "string",
                        "description": (
                            "The complete new value, JSON-encoded. A string field takes "
                            '"\\"text\\"", a list field takes a full JSON array (the whole '
                            "list, including items that are not changing)."
                        ),
                    },
                },
                "required": ["field", "value_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": (
                "Search LinkedIn's public job feed for openings. Use when the user asks you "
                "to find jobs rather than handing you one. Returns titles, companies and "
                "links but no descriptions — call read_job on anything promising."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "e.g. 'AI Engineer' or 'Data Analyst Power BI'"},
                    "location": {"type": "string", "description": "e.g. 'Egypt', 'Cairo', 'Remote'. Default Egypt."},
                    "limit": {"type": "integer", "description": "How many results, 1-25. Default 10."},
                    "remote_only": {"type": "boolean", "description": "Restrict to remote roles."},
                    "posted_within_days": {"type": "integer", "description": "Only jobs posted in the last N days."},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_job",
            "description": (
                "Read the full text of a job posting from its URL (LinkedIn or any other "
                "job board). Always call this when the user sends a link instead of "
                "guessing from the URL. Also reports any contact emails found in the post."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch any other web page as text — a company's about/careers page, a "
                "GitHub repo to verify what a project actually contains before claiming it "
                "on the CV, a recruiter's post. Not for job postings (use read_job)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "reason": {"type": "string", "description": "Why you need it (for the logs)."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_match",
            "description": (
                "Run a deterministic keyword/seniority match of a job description against "
                "the stored profile. Returns a 0-100 score, matched skills, missing skills "
                "and a seniority warning. Use it as an objective sanity check before "
                "telling the user a job is a good fit — it is not an opinion, it is math."
            ),
            "parameters": {
                "type": "object",
                "properties": {"job_description": {"type": "string"}},
                "required": ["job_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_application",
            "description": (
                "Produce the tailored application package: generates a real ATS-friendly "
                "PDF CV (which is sent to the user automatically) and stores the email "
                "draft. Call this only once you have read the profile and the actual job "
                "text, and have resolved anything ambiguous by asking the user. Select and "
                "reorder content for THIS job — do not dump everything."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": (
                            "The exact job title you are applying for (e.g. 'Senior Data "
                            "Engineer'). This is NOT the company name."
                        ),
                    },
                    "company": {
                        "type": "string",
                        "description": "The hiring company's name only.",
                    },
                    "job_url": {"type": "string"},
                    "recruiter_email": {"type": "string", "description": "Leave empty if genuinely unknown."},
                    "fit_summary": {"type": "string", "description": "1-2 honest sentences on fit."},
                    "gap_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Real requirements in the posting that the profile does NOT cover. "
                            "Empty only if there genuinely are none. Never hide a gap to make "
                            "the match look better."
                        ),
                    },
                    "headline": {
                        "type": "string",
                        "description": (
                            "Max 60 characters. A short professional headline for the top "
                            "of the CV, describing the candidate's role/specialty (e.g. "
                            "'Data Engineer - Python, SQL, AI Systems'). Must NOT be the "
                            "company name or the job posting's company."
                        ),
                    },
                    "summary": {"type": "string", "description": "3-4 sentences, built only from real profile facts."},
                    "selected_experience": {
                        "type": "array",
                        "description": (
                            "Experience entries from the real profile in display order. You may "
                            "reword bullets for emphasis but must not invent achievements or numbers."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "description": "The candidate's job title at this past employer.",
                                },
                                "company": {
                                    "type": "string",
                                    "description": "The name of that past employer.",
                                },
                                "period": {"type": "string"},
                                "bullets": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                    "selected_projects": {
                        "type": "array",
                        "description": "Only the projects actually relevant to this job, in relevance order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "tech": {"type": "string"},
                                "links": {"type": "string"},
                                "bullets": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                    "skills_categories": {
                        "type": "object",
                        "description": "Profile skill categories, reordered so the job-relevant ones lead.",
                    },
                    "selected_certifications": {"type": "array", "items": {"type": "string"}},
                    "email_subject": {"type": "string", "description": "Max 80 characters."},
                    "email_body": {
                        "type": "string",
                        "description": "Application email referencing the SAME projects as the CV above.",
                    },
                },
                "required": [
                    "role", "company", "fit_summary", "gap_notes", "headline", "summary",
                    "selected_experience", "selected_projects", "skills_categories",
                    "email_subject", "email_body",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Request to send the current application email with the tailored CV "
                "attached. This does NOT send immediately — it shows the user an approval "
                "button and pauses. Never claim an email was sent until this tool returns "
                "a result saying so."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient address."},
                    "subject": {"type": "string", "description": "Leave empty to use the current draft's subject."},
                    "body": {"type": "string", "description": "Leave empty to use the current draft's body."},
                },
                "required": ["to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_application",
            "description": (
                "Record or update an application in the tracker: company, role, status "
                "(drafted / sent / interview / rejected / offer), notes. Keep it current so "
                "the user can ask what they applied to and when."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "role": {"type": "string"},
                    "job_url": {"type": "string"},
                    "status": {"type": "string"},
                    "notes": {"type": "string"},
                    "match_score": {"type": "integer"},
                },
                "required": ["company", "role", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_applications",
            "description": (
                "Read the application history and summary stats. Use for questions like "
                "'what have I applied to', 'how many this week', 'did I already apply here'. "
                "Always check this before drafting, to avoid applying to the same job twice."
            ),
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Most recent N. Default 20."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Store a durable note about the user's preferences or situation — salary "
                "floor, willingness to relocate, companies to avoid, tone they like in "
                "emails. These are injected into every future conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
]


BASE_SYSTEM_PROMPT = """You are a career agent working for one specific person: a job seeker who needs real interviews, fast. You are not a chatbot that answers questions about job hunting — you do the work: find postings, judge fit honestly, tailor the CV, draft the outreach, and keep the tracker current.

How you behave:

1. Be conversational and direct. Reply in plain text when you are talking; call a tool when you are acting. If something is ambiguous, just ask — a short question beats a confident guess, and the user will reply in the same thread.
2. NEVER invent experience, projects, skills, employers, metrics or certifications. Every line on the CV must trace to get_profile output or something the user told you in this conversation. If a posting wants something they do not have, say so in gap_notes.
3. Always call get_profile before writing CV or email content. Do not work from memory of an earlier turn's summary.
4. When given a link, call read_job. When given pasted text, use it directly. If LinkedIn refuses the server (it throttles cloud IPs), say so plainly and ask the user to paste the description — do not pretend you read it.
5. Run score_match before you tell someone a job is a good fit. If the score is low or it is clearly a senior role, tell them that honestly instead of flattering the application.
6. Select and reorder content for the specific job. Do not attach every project. Lead with what this employer is buying.
7. Keep the email consistent with the CV — same projects, same claims.
8. You cannot send anything on your own. send_email only asks for permission; the human approves. Never say an email was sent unless a tool result told you it was.
9. Call track_application after every draft and every send, so the history stays useful.
10. Be efficient with the user's time. Short messages, no filler, no repeating what they just said back to them. Use Telegram-friendly formatting: short paragraphs, occasional bullets, no markdown tables.
11. Write in the language the user writes to you in. If they write Arabic, answer in Arabic — but keep CV and application-email content in English unless they ask otherwise, since that is what most employers expect.
"""


@dataclass
class AgentTurn:
    """Everything bot.py needs to render one turn."""

    status: str  # "MESSAGE" | "NEEDS_APPROVAL" | "ERROR"
    text: str = ""
    attachments: List[Dict[str, str]] = field(default_factory=list)
    approval: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tool_trace: List[str] = field(default_factory=list)


_client: Optional[AsyncOpenAI] = None


def get_client() -> Optional[AsyncOpenAI]:
    global _client
    if not LLM_API_KEY:
        return None
    if _client is None:
        _client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)
    return _client


def _system_prompt() -> str:
    """Base prompt plus live context, so the model starts each turn oriented."""
    parts = [BASE_SYSTEM_PROMPT]

    profile = store.load_profile()
    if profile:
        parts.append(
            "Candidate snapshot (call get_profile for the full record):\n"
            + store.profile_summary_text(profile)
        )
    else:
        parts.append(
            "There is NO profile on file yet. Your first job is to collect one from the "
            "user conversationally and save it with update_profile."
        )

    notes = store.load_notes()
    if notes:
        recent = "\n".join(f"- {n.get('text', '')}" for n in notes[-12:])
        parts.append(f"Standing notes from the user:\n{recent}")

    stats = store.application_stats()
    parts.append(
        f"Tracker: {stats['total']} applications logged, {stats['last_7_days']} in the last 7 days."
    )
    parts.append(f"Today's date: {time.strftime('%Y-%m-%d')}.")
    return "\n\n".join(parts)


def _trim(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep history bounded without leaving an orphaned tool result at the front
    (an API error) — drop from the front until the oldest message is a user turn.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    trimmed = messages[-MAX_HISTORY_MESSAGES:]
    while trimmed and trimmed[0].get("role") != "user":
        trimmed.pop(0)
    return trimmed


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def _execute_tool(
    name: str, args: Dict[str, Any], user_id: int, turn: AgentTurn
) -> Any:
    """Run one tool. Always returns a JSON-serialisable result; never raises."""
    try:
        if name == "get_profile":
            profile = store.load_profile()
            return profile or {"error": "No profile stored yet."}

        if name == "update_profile":
            field_name = (args.get("field") or "").strip()
            if field_name not in EDITABLE_PROFILE_FIELDS:
                return {
                    "ok": False,
                    "error": f"'{field_name}' is not an editable field. Allowed: {sorted(EDITABLE_PROFILE_FIELDS)}",
                }
            raw = args.get("value_json", "")
            try:
                value = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                value = raw  # model sent a bare string; accept it
            store.update_profile_field(field_name, value)
            return {"ok": True, "field": field_name, "message": "Profile updated and saved."}

        if name == "search_jobs":
            return await job_source.search_linkedin_jobs(
                keywords=args.get("keywords", ""),
                location=args.get("location") or "Egypt",
                limit=min(int(args.get("limit") or 10), 25),
                remote_only=bool(args.get("remote_only")),
                posted_within_days=int(args.get("posted_within_days") or 0),
            )

        if name == "read_job":
            return await job_source.get_linkedin_job(args.get("url", ""))

        if name == "fetch_url":
            return await job_source.fetch_page(args.get("url", ""))

        if name == "score_match":
            return matching.score_job(args.get("job_description", ""))

        if name == "build_application":
            return await _build_application(args, user_id, turn)

        if name == "track_application":
            entry = store.log_application(
                {
                    "company": args.get("company", ""),
                    "role": args.get("role", ""),
                    "job_url": args.get("job_url", ""),
                    "status": args.get("status", "drafted"),
                    "notes": args.get("notes", ""),
                    "match_score": args.get("match_score"),
                }
            )
            return {"ok": True, "logged": entry}

        if name == "list_applications":
            limit = int(args.get("limit") or 20)
            apps = store.load_applications()[-limit:]
            return {
                "stats": store.application_stats(),
                "applications": [
                    {
                        "company": a.get("company"),
                        "role": a.get("role"),
                        "status": a.get("status"),
                        "job_url": a.get("job_url"),
                        "match_score": a.get("match_score"),
                        "date": time.strftime("%Y-%m-%d", time.localtime(a.get("created_at", 0))),
                    }
                    for a in apps
                ],
            }

        if name == "remember":
            note = store.add_note(args.get("note", ""))
            return {"ok": True, "saved": note["text"]}

        return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        logger.error(f"Tool {name} raised: {e}", exc_info=True)
        return {"error": f"Tool '{name}' failed: {e}"}


async def _build_application(args: Dict[str, Any], user_id: int, turn: AgentTurn) -> Dict[str, Any]:
    """Render the tailored package into a PDF and stage it as the pending draft."""
    profile = store.load_profile()
    cv_data = {
        "name": profile.get("name", ""),
        "headline": args.get("headline") or profile.get("headline", ""),
        "summary": args.get("summary") or profile.get("summary", ""),
        "location": profile.get("location", ""),
        "phone": profile.get("phone", ""),
        "email": profile.get("email", ""),
        "linkedin": profile.get("linkedin", ""),
        "github": profile.get("github", ""),
        "experience": args.get("selected_experience") or profile.get("experience", []),
        "projects": args.get("selected_projects") or profile.get("projects", []),
        "education": profile.get("education", []),
        "certifications": args.get("selected_certifications") or profile.get("certifications", []),
        "skills_categories": args.get("skills_categories") or profile.get("skills_categories", {}),
    }

    company = args.get("company", "Company")
    role = args.get("role", "Role")
    safe = "".join(c if c.isalnum() else "_" for c in f"{profile.get('name','CV')}_{company}")[:60]
    pdf_path = cv_generator.generate_pdf_cv(cv_data, filename=f"{safe}.pdf")

    subject = (args.get("email_subject") or "")[:80]
    draft = {
        "role": role,
        "company": company,
        "job_url": args.get("job_url", ""),
        "recruiter_email": (args.get("recruiter_email") or "").strip(),
        "email_subject": subject,
        "email_body": args.get("email_body", ""),
        "pdf_path": pdf_path,
        "fit_summary": args.get("fit_summary", ""),
        "gap_notes": args.get("gap_notes") or [],
        "created_at": time.time(),
    }
    store.save_draft(user_id, draft)

    turn.attachments.append(
        {"path": pdf_path, "caption": f"Tailored CV — {role} at {company}"}
    )

    return {
        "ok": True,
        "cv_generated": True,
        "cv_file": pdf_path,
        "message": (
            "The tailored CV PDF has been sent to the user in this chat and the email "
            "draft is staged. Now summarise the fit and the gaps in your reply, show the "
            "email subject and body, and tell them to approve sending or to give you a "
            "recruiter address. Do not re-paste the CV contents."
        ),
        "draft_subject": subject,
        "draft_recruiter_email": draft["recruiter_email"] or None,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_turn(
    user_id: int,
    user_message: Optional[str] = None,
    resume_tool_result: Optional[Dict[str, Any]] = None,
) -> AgentTurn:
    """
    Advance the user's conversation by one turn.

    - Normal message: pass user_message.
    - Resuming after an approval decision: pass resume_tool_result, which is
      {"tool_call_id": ..., "content": {...}} — the outcome of the send the
      human just approved or rejected, fed back so the agent knows what happened.
    """
    turn = AgentTurn(status="MESSAGE")

    client = get_client()
    if not client:
        return AgentTurn(
            status="ERROR",
            error="LLM_API_KEY is not configured — set it in Railway → Variables.",
        )

    messages = store.load_conversation(user_id)

    # Refresh the system prompt every turn so profile/tracker changes are seen.
    system_msg = {"role": "system", "content": _system_prompt()}
    messages = [m for m in messages if m.get("role") != "system"]
    messages = _trim(messages)

    if resume_tool_result:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": resume_tool_result["tool_call_id"],
                "content": json.dumps(resume_tool_result["content"])[:4000],
            }
        )
    elif user_message:
        messages.append({"role": "user", "content": user_message})

    for step in range(MAX_TOOL_STEPS):
        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[system_msg] + messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        except Exception as e:
            logger.error(f"LLM call failed (step {step}): {e}")
            store.save_conversation(user_id, messages)
            return AgentTurn(
                status="ERROR",
                error=f"The AI provider rejected the request: {e}",
                attachments=turn.attachments,
            )

        msg = response.choices[0].message
        tool_calls = list(msg.tool_calls or [])

        # No tool calls: the model is talking. That IS the reply to the human,
        # and the thread simply waits for their next message.
        if not tool_calls:
            text = (msg.content or "").strip()
            messages.append({"role": "assistant", "content": text})
            store.save_conversation(user_id, _trim(messages))
            turn.status = "MESSAGE"
            turn.text = text or "…"
            return turn

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [_serialise_tool_call(tc) for tc in tool_calls],
            }
        )

        pending_approval: Optional[Dict[str, Any]] = None

        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            turn.tool_trace.append(fn_name)
            logger.info(f"[user {user_id}] tool: {fn_name}")

            if fn_name in INTERRUPTING_TOOLS:
                # Do not answer this tool call yet — the human decides first.
                pending_approval = {
                    "tool_call_id": tc.id,
                    "tool": fn_name,
                    "args": fn_args,
                }
                continue

            result = await _execute_tool(fn_name, fn_args, user_id, turn)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str)[:12000],
                }
            )

        if pending_approval:
            draft = store.load_draft(user_id) or {}
            args = pending_approval["args"]
            approval = {
                "tool_call_id": pending_approval["tool_call_id"],
                "to": (args.get("to") or draft.get("recruiter_email") or "").strip(),
                "subject": (args.get("subject") or draft.get("email_subject") or "").strip(),
                "body": (args.get("body") or draft.get("email_body") or "").strip(),
                "pdf_path": draft.get("pdf_path"),
                "company": draft.get("company", ""),
                "role": draft.get("role", ""),
                "job_url": draft.get("job_url", ""),
            }
            store.save_conversation(user_id, messages)
            turn.status = "NEEDS_APPROVAL"
            turn.approval = approval
            turn.text = (msg.content or "").strip()
            return turn

    # Ran out of steps — save what we have and say so rather than hanging.
    store.save_conversation(user_id, _trim(messages))
    return AgentTurn(
        status="ERROR",
        error=f"I got stuck after {MAX_TOOL_STEPS} steps without finishing. Try /reset and rephrase.",
        attachments=turn.attachments,
    )


def _serialise_tool_call(tc: Any) -> Dict[str, Any]:
    """Normalise a tool call into the plain dict shape the API expects back."""
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
    }
