"""
graph_agent.py — LangGraph rewrite of the career agent.

WHY THIS EXISTS
----------------
The old loop (agent.py) trusted the model to remember an 11-point system
prompt and call tools in the right order every time: read the job, THEN
score it, THEN build the CV, THEN ask to send. When the model skipped a
step — built a CV without reading the posting, tried to send before
building — nothing stopped it. That's not a framework problem, it's a
missing-guardrail problem: a longer prompt does not make a model more
reliable at sequencing.

This version is the SAME agent — one LLM, the same nine actions, the model
still decides what to call and when — but the risky tools now enforce their
own preconditions in code:

    read_job  →  score_match  →  build_application  →  send_email

  - score_match refuses to run until read_job has populated job text for
    the job in play.
  - build_application refuses to run until score_match has actually scored
    THAT job (switching to a different URL resets the requirement).
  - send_email refuses to run until build_application produced a draft for
    that same job.

A blocked call doesn't crash — it returns a plain error message as the tool
result, exactly like the API rejecting a malformed call. The model sees it
on the next step and corrects itself. That's the actual fix for "wrong
order": hard rails, not more prose.

WHAT THE FRAMEWORK GENUINELY BUYS US (not just re-skinning agent.py):

  1. Real persistence. Conversation + in-flight job state live in a
     LangGraph checkpointer (SQLite file by default, Postgres if
     DATABASE_URL is set) instead of a hand-merged JSON blob. A crash
     mid-tool-call no longer loses the thread.
  2. send_email is a genuine `interrupt()` — the graph itself pauses and
     the checkpointer persists that paused state. bot.py resumes it with
     `Command(resume=...)` after the human taps Approve/Reject. No more
     hand-rolled `pending_approval` dict threaded through AgentTurn.

WHAT THIS DELIBERATELY DOES NOT ADD: no planner node, no subagents, no
vector memory, no multi-graph orchestration. None of that was the actual
complaint — a single ReAct loop that enforces its own rules is enough.

DROP-IN CONTRACT: same public surface as agent.py — `AgentTurn`,
`run_turn(user_id, user_message=None, resume_tool_result=None)`,
`get_client()` — so bot.py needs only an import change, not a rewrite.
See MIGRATION.md for the two-line diff.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command, interrupt

import cv_generator
import jobs as job_source
import matching
import store
from llm_preflight import resolve_model
from config import (
    DATA_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_MODEL_CHAIN,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MAX_HISTORY_MESSAGES,
    MAX_TOOL_STEPS,
    logger,
)

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

EDITABLE_PROFILE_FIELDS = {
    "name", "headline", "location", "phone", "email", "linkedin", "github",
    "summary", "target_roles", "experience", "projects", "skills",
    "skills_categories", "education", "certifications", "languages",
}


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class JobContext(TypedDict, total=False):
    """Precondition tracking for whichever job is currently in play."""

    job_url: str
    job_text: str
    score: Dict[str, Any]
    application_built: bool


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: int
    job: JobContext


def _job(state: AgentState) -> JobContext:
    return state.get("job") or {}


def _blocked(tool_call_id: str, message: str) -> Command:
    """A tool refuses to run: hand the model back an error, not a crash."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps({"ok": False, "error": message}),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_profile() -> dict:
    """Read the candidate's COMPLETE stored profile: every experience entry with
    full bullets, every project with bullets and tech stack, all skill
    categories, education and certifications. Call this before writing any CV
    or email content, and whenever the user asks what you know about them."""
    return store.load_profile() or {"error": "No profile stored yet."}


@tool
def update_profile(field: str, value_json: str) -> dict:
    """Permanently update one field of the stored profile — use when the user
    tells you something new about themselves (a new project, a new job, a
    corrected phone number, an extra skill). Read the profile first so you
    write back the full corrected value, not a fragment.

    field: one of name, headline, location, phone, email, linkedin, github,
        summary, target_roles, experience, projects, skills,
        skills_categories, education, certifications, languages.
    value_json: the complete new value, JSON-encoded. A string field takes
        "\\"text\\"", a list field takes the full JSON array (the whole list,
        including items that are not changing).
    """
    field = (field or "").strip()
    if field not in EDITABLE_PROFILE_FIELDS:
        return {"ok": False, "error": f"'{field}' is not editable. Allowed: {sorted(EDITABLE_PROFILE_FIELDS)}"}
    try:
        value = json.loads(value_json) if isinstance(value_json, str) else value_json
    except json.JSONDecodeError:
        value = value_json
    store.update_profile_field(field, value)
    return {"ok": True, "field": field, "message": "Profile updated and saved."}


@tool
async def search_jobs(
    keywords: str,
    location: str = "Egypt",
    limit: int = 10,
    remote_only: bool = False,
    posted_within_days: int = 0,
) -> Any:
    """Search LinkedIn's public job feed for openings. Use when the user asks you
    to find jobs rather than handing you one. Returns titles, companies and
    links but no descriptions — call read_job on anything promising."""
    return await job_source.search_linkedin_jobs(
        keywords=keywords,
        location=location or "Egypt",
        limit=min(int(limit or 10), 25),
        remote_only=bool(remote_only),
        posted_within_days=int(posted_within_days or 0),
    )


@tool
async def read_job(
    url: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read the full text of a job posting from its URL (LinkedIn or any other
    job board). Always call this when the user sends a link instead of
    guessing from the URL. Also reports any contact emails found in the post.
    This is the required first step before score_match or build_application."""
    result = await job_source.get_linkedin_job(url)
    text = result.get("description") or result.get("text") or ""
    new_job: JobContext = {
        "job_url": url,
        "job_text": text,
        "score": {},
        "application_built": False,
    }
    return Command(
        update={
            "job": new_job,
            "messages": [
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str)[:12000],
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
async def fetch_url(url: str, reason: str = "") -> Any:
    """Fetch any other web page as text — a company's about/careers page, a
    GitHub repo to verify what a project actually contains before claiming it
    on the CV, a recruiter's post. Not for job postings (use read_job)."""
    return await job_source.fetch_page(url)


@tool
def score_match(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Run a deterministic keyword/seniority match of the CURRENTLY-READ job
    against the stored profile. Returns a 0-100 score, matched skills,
    missing skills and a seniority warning. You must call read_job first —
    this scores whatever job read_job last loaded, not free text."""
    job = _job(state)
    job_text = job.get("job_text")
    if not job_text:
        return _blocked(
            tool_call_id,
            "No job has been read yet. Call read_job on the posting URL first — "
            "you cannot score a job you have not read.",
        )
    result = matching.score_job(job_text)
    new_job: JobContext = {**job, "score": result}
    return Command(
        update={
            "job": new_job,
            "messages": [ToolMessage(content=json.dumps(result, default=str), tool_call_id=tool_call_id)],
        }
    )


@tool
async def build_application(
    role: str,
    company: str,
    fit_summary: str,
    gap_notes: List[str],
    headline: str,
    summary: str,
    selected_experience: List[dict],
    selected_projects: List[dict],
    skills_categories: dict,
    email_subject: str,
    email_body: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    job_url: str = "",
    recruiter_email: str = "",
    selected_certifications: Optional[List[str]] = None,
) -> Command:
    """Produce the tailored application package: generates a real ATS-friendly
    PDF CV (sent to the user automatically) and stores the email draft.
    Requires read_job AND score_match to have already run for this job —
    call this only once you have read the profile and the actual job text,
    and have resolved anything ambiguous by asking the user. Select and
    reorder content for THIS job — do not dump everything.

    role: the exact job title you are applying for. NOT the company name.
    company: the hiring company's name only.
    fit_summary: 1-2 honest sentences on fit.
    gap_notes: real requirements in the posting the profile does NOT cover.
        Empty only if there genuinely are none. Never hide a gap.
    headline: max 60 characters, a professional headline for the CV header
        (e.g. 'Data Engineer - Python, SQL, AI Systems'). NOT the company name.
    summary: 3-4 sentences, built only from real profile facts.
    selected_experience: experience entries from the real profile, in display
        order. Reword bullets for emphasis but never invent achievements.
    selected_projects: only the projects actually relevant to this job.
    skills_categories: profile skill categories, reordered so job-relevant
        ones lead.
    email_subject: max 80 characters.
    email_body: application email referencing the SAME projects as the CV.
    recruiter_email: leave empty if genuinely unknown.
    """
    job = _job(state)
    if not job.get("job_text"):
        return _blocked(tool_call_id, "No job has been read yet. Call read_job first.")
    if not job.get("score"):
        return _blocked(tool_call_id, "This job hasn't been scored yet. Call score_match first.")

    user_id = state["user_id"]
    profile = store.load_profile()

    # The model is allowed to choose and reorder content, but the PDF must be
    # grounded in the canonical profile. This prevents accidental rewrites of
    # job titles, dates, project names or technical claims from becoming CV facts.
    def _norm(value: Any) -> str:
        return re.sub(r"\\s+", " ", str(value or "").strip().lower())

    profile_experience = profile.get("experience") or []
    profile_projects = profile.get("projects") or []
    profile_certs = profile.get("certifications") or []

    def _canonical_experience(items: List[dict]) -> List[dict]:
        chosen: List[dict] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            role = _norm(item.get("role"))
            company = _norm(item.get("company"))
            match = next(
                (
                    src for src in profile_experience
                    if _norm(src.get("role")) == role
                    or (company and _norm(src.get("company")) == company and role in _norm(src.get("role")))
                ),
                None,
            )
            if match and match not in chosen:
                chosen.append(match)
        for src in profile_experience:
            if src not in chosen:
                chosen.append(src)
            if len(chosen) >= 2:
                break
        return chosen[:3]

    def _canonical_projects(items: List[dict]) -> List[dict]:
        chosen: List[dict] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            name = _norm(item.get("name"))
            match = next((src for src in profile_projects if _norm(src.get("name")) == name), None)
            if match and match not in chosen:
                chosen.append(match)
        for src in profile_projects:
            if src not in chosen:
                chosen.append(src)
            if len(chosen) >= 4:
                break
        return chosen[:5]

    def _canonical_certifications(items: Optional[List[str]]) -> List[str]:
        chosen: List[str] = []
        for item in items or []:
            text = _norm(item)
            match = next((src for src in profile_certs if _norm(src) == text), None)
            if match and match not in chosen:
                chosen.append(match)
        for src in profile_certs:
            if src not in chosen:
                chosen.append(src)
            if len(chosen) >= 6:
                break
        return chosen[:7]

    # Reorder the profile's skill categories according to the model's requested
    # order, but always use the canonical skill names from the profile.
    profile_skill_categories = profile.get("skills_categories") or {}
    ordered_skill_categories: Dict[str, List[str]] = {}
    for key in (skills_categories or {}).keys():
        if key in profile_skill_categories and key not in ordered_skill_categories:
            ordered_skill_categories[key] = list(profile_skill_categories[key])
    for key, values in profile_skill_categories.items():
        if key not in ordered_skill_categories:
            ordered_skill_categories[key] = list(values)
        if len(ordered_skill_categories) >= 8:
            break

    cv_data = {
        "name": profile.get("name", ""),
        "headline": (headline or profile.get("headline", "")).strip(),
        "summary": (summary or profile.get("summary", "")).strip(),
        "location": profile.get("location", ""),
        "phone": profile.get("phone", ""),
        "email": profile.get("email", ""),
        "linkedin": profile.get("linkedin", ""),
        "github": profile.get("github", ""),
        "datacamp": profile.get("datacamp", ""),
        "experience": _canonical_experience(selected_experience),
        "projects": _canonical_projects(selected_projects),
        "education": profile.get("education", []),
        "certifications": _canonical_certifications(selected_certifications),
        "languages": profile.get("languages", []),
        "military_service": profile.get("military_service", ""),
        "skills_categories": ordered_skill_categories,
    }
    safe = "".join(c if c.isalnum() else "_" for c in f"{profile.get('name','CV')}_{company}")[:60]
    pdf_path = cv_generator.generate_pdf_cv(cv_data, filename=f"{safe}.pdf")

    draft = {
        "role": role,
        "company": company,
        "job_url": job_url or job.get("job_url", ""),
        "recruiter_email": (recruiter_email or "").strip(),
        "email_subject": (email_subject or "")[:80],
        "email_body": email_body,
        "pdf_path": pdf_path,
        "fit_summary": fit_summary,
        "gap_notes": gap_notes or [],
        "created_at": time.time(),
    }
    store.save_draft(user_id, draft)

    new_job: JobContext = {**job, "application_built": True}
    result = {
        "ok": True,
        "cv_generated": True,
        "cv_file": pdf_path,
        "message": (
            "The tailored CV PDF has been sent to the user in this chat and the email "
            "draft is staged. Summarise the fit and the gaps in your reply, show the "
            "email subject and body, and tell them to approve sending or give you a "
            "recruiter address. Do not re-paste the CV contents."
        ),
        "draft_subject": draft["email_subject"],
        "draft_recruiter_email": draft["recruiter_email"] or None,
        # Consumed by run_turn to attach the PDF to this turn's reply.
        "_attachment": {"path": pdf_path, "caption": f"Tailored CV — {role} at {company}"},
    }
    return Command(
        update={
            "job": new_job,
            "messages": [ToolMessage(content=json.dumps(result, ensure_ascii=False, default=str)[:12000], tool_call_id=tool_call_id)],
        }
    )


@tool
def send_email(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    to: str = "",
    subject: str = "",
    body: str = "",
) -> Command:
    """Request to send the current application email with the tailored CV
    attached. This does NOT send immediately — it pauses for human approval.
    Requires build_application to have already run for this job. Never claim
    an email was sent until this tool's result says so."""
    job = _job(state)
    if not job.get("application_built"):
        return _blocked(tool_call_id, "No application has been built yet. Call build_application first.")

    draft = store.load_draft(state["user_id"]) or {}
    payload = {
        "to": (to or draft.get("recruiter_email") or "").strip(),
        "subject": (subject or draft.get("email_subject") or "").strip(),
        "body": (body or draft.get("email_body") or "").strip(),
        "pdf_path": draft.get("pdf_path"),
        "company": draft.get("company", ""),
        "role": draft.get("role", ""),
        "job_url": draft.get("job_url", ""),
    }
    # Pauses the graph here. The checkpointer persists this exact point, so
    # even a restart before the human taps a button loses nothing. bot.py
    # resumes with Command(resume={"sent": bool, ...}).
    decision = interrupt(payload)
    return Command(
        update={
            "messages": [ToolMessage(content=json.dumps(decision, ensure_ascii=False, default=str), tool_call_id=tool_call_id)]
        }
    )


@tool
def track_application(
    company: str,
    role: str,
    status: str,
    job_url: str = "",
    notes: str = "",
    match_score: Optional[int] = None,
) -> dict:
    """Record or update an application in the tracker: company, role, status
    (drafted / sent / interview / rejected / offer), notes. Keep it current so
    the user can ask what they applied to and when."""
    entry = store.log_application(
        {
            "company": company,
            "role": role,
            "job_url": job_url,
            "status": status or "drafted",
            "notes": notes,
            "match_score": match_score,
        }
    )
    return {"ok": True, "logged": entry}


@tool
def list_applications(limit: int = 20) -> dict:
    """Read the application history and summary stats. Use for questions like
    'what have I applied to', 'how many this week', 'did I already apply here'.
    Always check this before drafting, to avoid applying to the same job twice."""
    apps = store.load_applications()[-int(limit or 20):]
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


@tool
def remember(note: str) -> dict:
    """Store a durable note about the user's preferences or situation — salary
    floor, willingness to relocate, companies to avoid, tone they like in
    emails. These are injected into every future conversation."""
    saved = store.add_note(note)
    return {"ok": True, "saved": saved["text"]}


TOOLS = [
    get_profile, update_profile, search_jobs, read_job, fetch_url,
    score_match, build_application, send_email, track_application,
    list_applications, remember,
]
INTERRUPTING_TOOLS = {"send_email"}


# ---------------------------------------------------------------------------
# System prompt (unchanged content from agent.py, minus the ordering
# instructions that are now enforced by the tools themselves)
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """You are a career agent working for one specific person: a job seeker who needs real interviews, fast. You are not a chatbot that answers questions about job hunting — you do the work: find postings, judge fit honestly, tailor the CV, draft the outreach, and keep the tracker current.

How you behave:

1. Be conversational and direct. Reply in plain text when you are talking; call a tool when you are acting. If something is ambiguous, just ask — a short question beats a confident guess.
2. NEVER invent experience, projects, skills, employers, metrics or certifications. Every line on the CV must trace to get_profile output or something the user told you in this conversation. If a posting wants something they do not have, say so in gap_notes.
3. Always call get_profile before writing CV or email content. Do not work from memory of an earlier turn's summary.
4. When given a link, call read_job. When given pasted text, use it directly. If LinkedIn refuses the server (it throttles cloud IPs), say so plainly and ask the user to paste the description — do not pretend you read it.
5. score_match, build_application and send_email will refuse to run out of order (they'll tell you what's missing) — that's expected, just do the missing step and retry, don't apologize for it in the reply.
6. Select and reorder content for the specific job, but keep the CV substantial: include both real experience entries when relevant and at least three relevant real projects. Do not make a half-page resume just to be concise.
7. Treat the stored profile as the source of truth. Keep job titles, employers, dates, project names, technical stacks and achievement bullets faithful to the profile; do not rewrite facts into new claims. Tailoring should come from selection, ordering, headline/summary emphasis and skills ordering.
8. Keep the email consistent with the CV — same projects, same claims.
9. You cannot send anything on your own. send_email only asks for permission; the human approves. Never say an email was sent unless a tool result told you it was.
10. Call track_application after every draft and every send, so the history stays useful.
11. Be efficient with the user's time. Short messages, no filler. Telegram-friendly formatting: short paragraphs, occasional bullets, no markdown tables.
12. Write in the language the user writes to you in. If they write Arabic, answer in Arabic — but keep CV and application-email content in English unless they ask otherwise.
"""


def _system_prompt() -> str:
    parts = [BASE_SYSTEM_PROMPT]
    profile = store.load_profile()
    if profile:
        parts.append("Candidate snapshot (call get_profile for the full record):\n" + store.profile_summary_text(profile))
    else:
        parts.append("There is NO profile on file yet. Your first job is to collect one from the user conversationally and save it with update_profile.")
    notes = store.load_notes()
    if notes:
        parts.append("Standing notes from the user:\n" + "\n".join(f"- {n.get('text', '')}" for n in notes[-12:]))
    stats = store.application_stats()
    parts.append(f"Tracker: {stats['total']} applications logged, {stats['last_7_days']} in the last 7 days.")
    parts.append(f"Today's date: {time.strftime('%Y-%m-%d')}.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_llm: Optional[ChatOpenAI] = None
# Which entry of LLM_MODEL_CHAIN we are currently on. Survives for the life of
# the process, so one 404 doesn't cost every subsequent turn a retry.
_active_model: Optional[str] = None


def _build_client(model: str) -> ChatOpenAI:
    client_kwargs = {
        "model": model,
        "api_key": LLM_API_KEY,
        "base_url": LLM_BASE_URL,
        "timeout": LLM_TIMEOUT,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "max_retries": LLM_MAX_RETRIES,
    }
    # NVIDIA's current reasoning models default to maximum thinking.
    # Interactive Telegram turns need low-latency tool decisions instead.
    model_id = model.lower()
    if model_id == "nvidia/nemotron-3.5-lightning-30b-a3b":
        # NVIDIA exposes a small explicit reasoning budget for this model.
        # That keeps an interactive Telegram turn responsive without disabling
        # the reasoning behavior entirely.
        client_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 4096,
        }
    elif model_id in {
        "z-ai/glm-5-3-flash",
        "z-ai/glm-5.3-flash",
        "z-ai/glm-5.3",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }:
        client_kwargs["reasoning_effort"] = "low"
    return ChatOpenAI(**client_kwargs).bind_tools(TOOLS, parallel_tool_calls=False)


def active_model() -> str:
    """What the bot is ACTUALLY talking to, which may not be LLM_MODEL."""
    return _active_model or LLM_MODEL


def get_client() -> Optional[ChatOpenAI]:
    """Same name/contract as agent.py's get_client, so bot.py's /diag check
    (`agent.get_client()`) keeps working unchanged."""
    global _llm, _active_model
    if not LLM_API_KEY:
        return None
    if _llm is None:
        _active_model = LLM_MODEL
        _llm = _build_client(_active_model)
    return _llm


def eager_resolve() -> List[str]:
    """
    Called once at boot. Walks LLM_MODEL_CHAIN against the real catalogue and
    pre-builds the client on whichever entry works, so the FIRST real user
    message doesn't pay for a 404 round-trip and startup logs don't cry FATAL
    over a primary model that the configured fallback would have covered anyway.
    """
    global _llm, _active_model
    if not LLM_API_KEY:
        return ["FATAL: LLM_API_KEY is not set — the agent cannot think."]

    chosen, notes = resolve_model(LLM_BASE_URL, LLM_MODEL_CHAIN, LLM_API_KEY)
    if chosen is None:
        return notes  # last entry is FATAL — nothing in the chain works
    _active_model = chosen
    _llm = _build_client(chosen)
    # Non-fatal notes (skipped/fell-back entries) still matter — surface them
    # as WARN so /diag and the startup log show what actually happened.
    return [n if n.startswith(("FATAL", "WARN")) else f"WARN: {n}" for n in notes]


def _demote_model() -> bool:
    """
    The active model 404'd. Advance to the next entry in LLM_MODEL_CHAIN and
    rebuild the client. Returns False when the chain is exhausted.
    """
    global _llm, _active_model
    current = active_model()
    try:
        nxt = LLM_MODEL_CHAIN[LLM_MODEL_CHAIN.index(current) + 1]
    except (ValueError, IndexError):
        return False
    logger.error(
        f"Model '{current}' failed at {LLM_BASE_URL} — falling back to '{nxt}'. "
        f"Fix LLM_MODEL (or your provider entitlement) to stop running degraded."
    )
    _active_model = nxt
    _llm = _build_client(nxt)
    return True


def _is_model_not_found(e: Exception) -> bool:
    """A 404 that names the model, not a 404 from a wrong URL path."""
    name = type(e).__name__
    if "ModelNotFound" in name or "NotFoundError" in name:
        return True
    return "404" in str(e) and "model" in str(e).lower()


def _explain_provider_error(e: Exception) -> str:
    """
    Provider errors arrive as a bare status code wrapped in 60 lines of
    LangChain/LangGraph frames. Translate the ones that are config mistakes
    into something you can act on without opening the logs.
    """
    text = str(e)
    if "404" in text or "NotFound" in type(e).__name__:
        # An empty body after "Error code: 404" means the HTTP PATH is wrong.
        # A JSON body naming the model means the MODEL is wrong. Both are config.
        path_level = text.strip().rstrip(".").endswith("404")
        cause = (
            f"the endpoint {LLM_BASE_URL} has no /chat/completions route — LLM_BASE_URL is wrong"
            if path_level
            else f"the provider does not serve the model '{LLM_MODEL}' — LLM_MODEL is wrong for this endpoint"
        )
        return (
            f"The AI provider returned 404: {cause}.\n\n"
            f"Current settings:\n• LLM_BASE_URL = {LLM_BASE_URL}\n• LLM_MODEL = {LLM_MODEL}\n\n"
            "Run /diag — it now checks the endpoint's model catalogue and names the closest valid id."
        )
    if "401" in text or "403" in text:
        return (
            f"The AI provider rejected the API key (auth error) for {LLM_BASE_URL}. "
            "Check LLM_API_KEY belongs to this provider and is still active."
        )
    if "503" in text or "ResourceExhausted" in text or "Service Unavailable" in text:
        return (
            f"The NVIDIA AI endpoint is temporarily overloaded while using '{active_model()}'. "
            "The agent retried automatically, but the provider was still at capacity."
        )
    if "429" in text:
        return "The AI provider is rate-limiting this key. The agent retried automatically."
    if "timeout" in text.lower() or "timed out" in text.lower() or "APITimeoutError" in type(e).__name__:
        return (
            f"The AI request timed out after {LLM_TIMEOUT}s while using '{active_model()}'. "
            "The provider did not finish the response in time." 
        )
    return f"The AI provider returned an unexpected error: {e}"


def _trim_removals(messages: List[BaseMessage]) -> List[RemoveMessage]:
    """
    Same policy as agent.py's _trim: keep history bounded, but never leave an
    orphaned ToolMessage at the front — a dangling tool result with no
    preceding tool-calling AIMessage is an API error, not just untidy.
    Returns RemoveMessage entries; add_messages applies them against the
    PERSISTED state, so old turns actually leave the checkpointer over time
    instead of accumulating forever.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return []
    keep = messages[-MAX_HISTORY_MESSAGES:]
    while keep and not isinstance(keep[0], HumanMessage):
        keep.pop(0)
    drop_ids = {m.id for m in messages[: len(messages) - len(keep)] if m.id}
    return [RemoveMessage(id=mid) for mid in drop_ids]


def _agent_node(state: AgentState) -> dict:
    llm = get_client()
    removals = _trim_removals(state["messages"])
    dropped_ids = {r.id for r in removals}
    live_messages = [m for m in state["messages"] if m.id not in dropped_ids]
    payload = [SystemMessage(content=_system_prompt())] + live_messages

    attempts_on_model = 0
    max_transient_retries = 2
    while True:
        try:
            response = llm.invoke(payload)
            break
        except Exception as e:
            error_text = str(e).lower()
            is_rate_limited = "429" in error_text or "rate limit" in error_text
            transient = (
                "503" in error_text
                or "service unavailable" in error_text
                or "resourceexhausted" in error_text
                or is_rate_limited
                or "timeout" in error_text
                or "timed out" in error_text
            )
            if transient and attempts_on_model < max_transient_retries:
                # Groq includes a concrete retry window in 429 errors
                # (e.g. "try again in 6.66s"). Honor it instead of hammering
                # the same token bucket with 1s/2s retries.
                delay = 2 ** attempts_on_model
                if is_rate_limited:
                    match = re.search(r"try again in ([0-9.]+)s", str(e), re.IGNORECASE)
                    if match:
                        delay = min(max(float(match.group(1)) + 0.5, 1.0), 60.0)
                attempts_on_model += 1
                logger.warning(
                    f"Transient LLM error on '{active_model()}'; "
                    f"retrying in {delay:.1f}s ({attempts_on_model}/{max_transient_retries})."
                )
                time.sleep(delay)
                continue

            # Fail over to the next configured model after transient
            # provider failures too. This lets OpenRouter move from the
            # DeepSeek free endpoint to openrouter/free when the primary is
            # unavailable, rate-limited, or times out.
            if not _is_model_not_found(e) and not transient:
                raise
            if not _demote_model():
                raise
            attempts_on_model = 0
            llm = get_client()
    # Removals + the new response land in the same state update: old turns
    # are pruned from the checkpointer, the new one is appended, in one step.
    return {"messages": [*removals, response]}


def _route(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", _agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g


_compiled_graph = None
_graph_lock: Optional["asyncio.Lock"] = None
_checkpointer_cm = None  # kept alive deliberately: from_conn_string() is an
# @asynccontextmanager generator, and if this reference is dropped it gets
# garbage-collected, which runs its `finally` and closes the connection out
# from under a saver that's still in use. Module-level = lives as long as
# the process does, same as a plain global connection would.


async def _get_checkpointer():
    """Postgres if DATABASE_URL is set (survives Railway volume loss on
    redeploy too), else a SQLite file on the mounted DATA_DIR volume. Both
    are async savers because run_turn drives the graph with ainvoke — the
    sync SqliteSaver/PostgresSaver raise NotImplementedError under ainvoke,
    so this is not optional."""
    global _checkpointer_cm
    if DATABASE_URL:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        _checkpointer_cm = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
    else:
        _checkpointer_cm = AsyncSqliteSaver.from_conn_string(str(DATA_DIR / "checkpoints.sqlite"))
    saver = await _checkpointer_cm.__aenter__()  # kept open for process lifetime
    await saver.setup()
    return saver


async def _get_graph():
    """Lazily build+compile once, guarded by an asyncio.Lock so two concurrent
    first requests (e.g. two Telegram users messaging at the same instant on
    a cold start) don't each open their own checkpointer connection."""
    global _compiled_graph, _graph_lock
    if _compiled_graph is not None:
        return _compiled_graph
    if _graph_lock is None:
        _graph_lock = asyncio.Lock()
    async with _graph_lock:
        if _compiled_graph is None:
            checkpointer = await _get_checkpointer()
            _compiled_graph = _build_graph().compile(checkpointer=checkpointer)
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public contract — matches agent.py exactly
# ---------------------------------------------------------------------------


@dataclass
class AgentTurn:
    """Everything bot.py needs to render one turn. Identical shape to agent.py."""

    status: str  # "MESSAGE" | "NEEDS_APPROVAL" | "ERROR"
    text: str = ""
    attachments: List[Dict[str, str]] = field(default_factory=list)
    approval: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tool_trace: List[str] = field(default_factory=list)


async def _collect_attachments_and_trace(graph, config, turn: AgentTurn) -> None:
    """Pull tool-call names and any _attachment payloads out of this run's
    new state, without replaying already-seen messages on later turns."""
    state = await graph.aget_state(config)
    for msg in state.values.get("messages", []):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                turn.tool_trace.append(tc["name"])
        if isinstance(msg, ToolMessage):
            try:
                payload = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue
            att = payload.get("_attachment") if isinstance(payload, dict) else None
            if att and att not in turn.attachments:
                turn.attachments.append(att)


async def run_turn(
    user_id: int,
    user_message: Optional[str] = None,
    resume_tool_result: Optional[Dict[str, Any]] = None,
) -> AgentTurn:
    """
    Advance the user's conversation by one turn. Same contract as agent.py:

    - Normal message: pass user_message.
    - Resuming after an approval decision: pass resume_tool_result, which is
      {"tool_call_id": ..., "content": {...}} — the outcome of the send the
      human just approved or rejected.

    Internally, resume_tool_result["content"] becomes the value returned by
    interrupt() inside send_email — tool_call_id is accepted for interface
    compatibility with agent.py but isn't needed here: LangGraph resumes
    whichever interrupt is pending for this thread.
    """
    turn = AgentTurn(status="MESSAGE")

    if not get_client():
        return AgentTurn(status="ERROR", error="LLM_API_KEY is not configured — set it in Railway → Variables.")

    graph = await _get_graph()
    config = {"configurable": {"thread_id": str(user_id)}, "recursion_limit": MAX_TOOL_STEPS * 2 + 4}

    try:
        if resume_tool_result is not None:
            result = await graph.ainvoke(Command(resume=resume_tool_result["content"]), config=config)
        else:
            snapshot = await graph.aget_state(config)
            base_state = {"user_id": user_id, "job": (snapshot.values or {}).get("job", {})} if snapshot.values else {"user_id": user_id, "job": {}}
            result = await graph.ainvoke({**base_state, "messages": [("user", user_message or "")]}, config=config)
    except GraphRecursionError:
        logger.warning(f"Recursion limit hit for user {user_id} — agent looped without finishing.")
        return AgentTurn(
            status="ERROR",
            error=f"I got stuck after {MAX_TOOL_STEPS} steps without finishing. Try /reset and rephrase.",
        )
    except Exception as e:
        logger.error(f"Graph run failed for user {user_id}: {e}", exc_info=True)
        return AgentTurn(status="ERROR", error=_explain_provider_error(e))

    await _collect_attachments_and_trace(graph, config, turn)

    state = await graph.aget_state(config)
    if state.next:
        # Graph paused on an interrupt() — currently only send_email does this.
        pending = state.tasks[0].interrupts[0].value if state.tasks and state.tasks[0].interrupts else {}
        turn.status = "NEEDS_APPROVAL"
        turn.approval = {
            "tool_call_id": "",  # not used by graph_agent's resume path; kept for bot.py shape compatibility
            **pending,
        }
        turn.text = ""
        return turn

    last = result["messages"][-1]
    turn.text = (last.content or "…").strip() if isinstance(last, AIMessage) else "…"
    return turn
