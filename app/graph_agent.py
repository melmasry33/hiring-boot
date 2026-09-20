"""Production LangGraph surface for the staged canonical CV pipeline."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command, interrupt

import jobs as job_source
import pipeline
import store
from candidate import canonical_payload, list_variant_catalog, load_identity
from config import (DATA_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_MAX_RETRIES, LLM_MAX_TOKENS,
                    LLM_MODEL, LLM_MODEL_CHAIN, LLM_TEMPERATURE, LLM_TIMEOUT,
                    MAX_HISTORY_MESSAGES, MAX_TOOL_STEPS, logger)
from validator import PipelineError

EDITABLE_PROFILE_FIELDS = {"name", "location", "phone", "email", "linkedin", "github", "datacamp"}


class JobContext(TypedDict, total=False):
    job_url: str
    job_text: str
    job_text_chars: int
    job_text_truncated: bool
    role_hint: str
    company_hint: str
    stage: str
    analysis: Dict[str, Any]
    selected_variant: str
    variant_reasons: List[str]
    hr_screen: Dict[str, Any]
    tailoring_brief: Dict[str, Any]
    selection: Dict[str, Any]
    tailoring: Dict[str, Any]
    email: Dict[str, Any]
    cv_data: Dict[str, Any]
    pdf_path: str
    pdf_validation: Dict[str, Any]
    email_validation: Dict[str, Any]
    application_built: bool


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: int
    job: JobContext


def _job(state: AgentState) -> JobContext:
    return state.get("job") or {}


def _result(call_id: str, payload: Dict[str, Any], job: Optional[JobContext] = None) -> Command:
    update: Dict[str, Any] = {"messages": [ToolMessage(content=json.dumps(payload, ensure_ascii=False, default=str), tool_call_id=call_id)]}
    if job is not None:
        update["job"] = job
    return Command(update=update)


def _blocked(call_id: str, message: str) -> Command:
    return _result(call_id, {"ok": False, "error": message})


def _failure(call_id: str, exc: PipelineError, job: JobContext) -> Command:
    return _result(call_id, exc.as_dict(), job)


@tool
def get_profile() -> dict:
    """Return contact metadata and available canonical variants, never a career-profile blob."""
    return {"ok": True, "identity": load_identity(), "variants": list_variant_catalog()}


@tool
def get_candidate_payload(
    state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Return selected CV facts, with canonical IDs, only after variant analysis."""
    job = _job(state)
    if pipeline.require_stage(job, pipeline.Stage.VARIANT_SELECTED):
        return _blocked(tool_call_id, "Analyze the job first; no canonical variant is selected.")
    return _result(tool_call_id, {"ok": True, "candidate": canonical_payload(job["selected_variant"])})


@tool
def update_profile(field: str, value_json: str) -> dict:
    """Update contact metadata only; career facts are authored in cv_variants.json."""
    if field not in EDITABLE_PROFILE_FIELDS:
        return {"ok": False, "error": f"Only identity fields are editable: {sorted(EDITABLE_PROFILE_FIELDS)}"}
    try:
        value = json.loads(value_json)
    except json.JSONDecodeError:
        value = value_json
    store.update_profile_field(field, value)
    return {"ok": True, "field": field}


@tool
async def search_jobs(keywords: str, location: str = "Egypt", limit: int = 10, remote_only: bool = False, posted_within_days: int = 0) -> Any:
    """Search job listings; read a chosen job before screening it."""
    return await job_source.search_linkedin_jobs(keywords, location or "Egypt", min(int(limit or 10), 25), bool(remote_only), int(posted_within_days or 0))


@tool
async def read_job(url: str, state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Fetch full job text and create canonical JOB_INGESTED pipeline state."""
    result = await job_source.get_linkedin_job(url)
    text = result.get("description") or result.get("text") or ""
    if len(text.strip()) < 40:
        return _blocked(tool_call_id, "Could not obtain enough job text. Paste the full description.")
    job = pipeline.ingest_job(text, url, result.get("title") or "", result.get("company") or "")
    return _result(tool_call_id, {"ok": True, "stage": job["stage"], "job_text_chars": job["job_text_chars"], "job_text_truncated": False, "contacts": result.get("emails") or []}, job)


@tool
def ingest_job_text(
    job_text: str, state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId],
    job_url: str = "", role_hint: str = "", company_hint: str = "",
) -> Command:
    """Store a complete pasted JD. Source text is never silently truncated."""
    if len((job_text or "").strip()) < 40:
        return _blocked(tool_call_id, "Job text is too short to screen; paste responsibilities and requirements.")
    job = pipeline.ingest_job(job_text, job_url, role_hint, company_hint)
    return _result(tool_call_id, {"ok": True, "stage": job["stage"], "job_text_chars": job["job_text_chars"], "job_text_truncated": False}, job)


@tool
async def fetch_url(url: str, reason: str = "") -> Any:
    """Fetch a non-job supporting URL as text."""
    return await job_source.fetch_page(url)


@tool
def analyze_job(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], requested_variant: str = "auto") -> Command:
    """Analyze the full JD and deterministically select its canonical CV variant."""
    job = _job(state)
    try:
        pipeline.analyze_and_select_variant(job, requested_variant)
    except PipelineError as exc:
        return _failure(tool_call_id, exc, job)
    return _result(tool_call_id, {"ok": True, "stage": job["stage"], "selected_variant": job["selected_variant"], "analysis": job["analysis"], "variant_reasons": job["variant_reasons"]}, job)


@tool
def score_match(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Compatibility name for analyze_job; production matching is matcher.py via pipeline."""
    return analyze_job.func(state, tool_call_id, "auto")


@tool
def hr_screen(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Screen the selected variant and produce a relevance-ranked tailoring brief."""
    job = _job(state)
    try:
        pipeline.run_hr_screen(job)
    except PipelineError as exc:
        return _failure(tool_call_id, exc, job)
    return _result(tool_call_id, {"ok": True, "stage": job["stage"], "review": job["hr_screen"], "tailoring_brief": job["tailoring_brief"]}, job)


@tool
def select_evidence(
    selected_project_ids: List[str], selected_experience_ids: List[str], selected_skill_categories: List[str],
    state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId],
    selected_certification_ids: Optional[List[str]] = None,
) -> Command:
    """Rank canonical evidence by stable ID. Selection never removes CV content."""
    job = _job(state)
    selection = {"selected_project_ids": selected_project_ids or [], "selected_experience_ids": selected_experience_ids or [], "selected_skill_categories": selected_skill_categories or [], "selected_certification_ids": selected_certification_ids or []}
    try:
        pipeline.apply_selection(job, selection)
    except PipelineError as exc:
        return _failure(tool_call_id, exc, job)
    return _result(tool_call_id, {"ok": True, "stage": job["stage"], "selection": job["selection"]}, job)


@tool
def generate_tailoring(
    headline: str, summary: str, rewritten_bullets_by_experience_id: Dict[str, List[str]],
    rewritten_project_descriptions_by_project_id: Dict[str, List[str]],
    state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Validate headline/summary and ID-keyed rewrites. Existing bullet counts must be preserved."""
    job = _job(state)
    tailoring = {"headline": headline, "summary": summary, "rewritten_bullets_by_experience_id": rewritten_bullets_by_experience_id or {}, "rewritten_project_descriptions_by_project_id": rewritten_project_descriptions_by_project_id or {}}
    try:
        pipeline.apply_tailoring(job, tailoring)
    except PipelineError as exc:
        return _failure(tool_call_id, exc, job)
    return _result(tool_call_id, {"ok": True, "stage": job["stage"], "tailoring": job["tailoring"]}, job)


@tool
def build_pdf(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Assemble all canonical content, render the PDF, and validate extracted text."""
    job = _job(state)
    try:
        pipeline.build_pdf(job)
    except PipelineError as exc:
        return _failure(tool_call_id, exc, job)
    path = job.get("pdf_path") or ""
    return _result(tool_call_id, {"ok": True, "stage": job["stage"], "cv_file": path, "pdf_validation": job.get("pdf_validation"), "_attachment": {"path": path, "caption": f"Tailored CV [{job.get('selected_variant')}]"}}, job)


def _save_pipeline_draft(user_id: int, job: JobContext) -> None:
    analysis, email = job.get("analysis") or {}, job.get("email") or {}
    store.save_draft(user_id, {"role": job.get("role_hint") or analysis.get("role") or "", "company": job.get("company_hint") or analysis.get("company") or "", "job_url": job.get("job_url") or "", "recruiter_email": email.get("recruiter_email") or "", "email_subject": email.get("subject") or "", "email_body": email.get("body") or "", "pdf_path": job.get("pdf_path") or "", "fit_summary": email.get("fit_summary") or "", "gap_notes": email.get("gap_notes") or [], "stage": job.get("stage"), "selected_variant": job.get("selected_variant"), "created_at": time.time()})


@tool
def generate_email(
    subject: str, body: str, fit_summary: str, gap_notes: List[str],
    state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], recruiter_email: str = "",
) -> Command:
    """Validate raw plain-text email against candidate facts only; invalid markdown is rejected, never repaired."""
    job = _job(state)
    try:
        pipeline.apply_email(job, {"subject": subject, "body": body, "fit_summary": fit_summary, "gap_notes": gap_notes or [], "recruiter_email": recruiter_email})
        pipeline.mark_ready_if_complete(job)
    except PipelineError as exc:
        return _failure(tool_call_id, exc, job)
    _save_pipeline_draft(state["user_id"], job)
    return _result(tool_call_id, {"ok": True, "stage": job["stage"], "email": job["email"]}, job)


@tool
def critique_cv_package(state: Annotated[AgentState, InjectedState]) -> dict:
    """Report validation state for the pipeline-produced draft."""
    job = _job(state)
    return {"ok": bool(job.get("application_built")), "stage": job.get("stage"), "pdf_validation": job.get("pdf_validation"), "email_validation": job.get("email_validation"), "gaps": (job.get("email") or {}).get("gap_notes") or []}


@tool
def send_email(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], to: str = "") -> Command:
    """Pause for human approval; requires READY_TO_SEND and never sends itself."""
    job = _job(state)
    if pipeline.current_stage(job) != pipeline.Stage.READY_TO_SEND:
        return _blocked(tool_call_id, "Application is not READY_TO_SEND: validate tailoring, PDF, and email first.")
    draft = store.load_draft(state["user_id"]) or {}
    payload = {"to": (to or draft.get("recruiter_email") or "").strip(), "subject": draft.get("email_subject") or "", "body": draft.get("email_body") or "", "pdf_path": draft.get("pdf_path") or "", "company": draft.get("company") or "", "role": draft.get("role") or "", "job_url": draft.get("job_url") or ""}
    return Command(update={"messages": [ToolMessage(content=json.dumps(interrupt(payload), ensure_ascii=False, default=str), tool_call_id=tool_call_id)]})


@tool
def track_application(company: str, role: str, status: str, job_url: str = "", notes: str = "", match_score: Optional[int] = None) -> dict:
    """Record an application status in the durable tracker."""
    return {"ok": True, "logged": store.log_application({"company": company, "role": role, "job_url": job_url, "status": status or "drafted", "notes": notes, "match_score": match_score})}


@tool
def list_applications(limit: int = 20) -> dict:
    """List recently tracked applications and aggregate status counts."""
    return {"stats": store.application_stats(), "applications": store.load_applications()[-int(limit or 20):]}


@tool
def remember(note: str) -> dict:
    """Save a durable user preference note."""
    return {"ok": True, "saved": store.add_note(note)["text"]}


TOOLS = [get_profile, get_candidate_payload, update_profile, search_jobs, read_job, ingest_job_text, fetch_url, analyze_job, score_match, hr_screen, select_evidence, generate_tailoring, build_pdf, generate_email, critique_cv_package, send_email, track_application, list_applications, remember]
BASE_SYSTEM_PROMPT = """You are a truthful CV writer and HR screener for one candidate. Career facts exist only in the selected canonical CV variant; profile data is contact metadata. Required sequence: ingest/read job → analyze_job → hr_screen → get_candidate_payload → select_evidence → generate_tailoring → build_pdf → generate_email → send_email. Selections rank evidence only; every canonical experience and project stays in the PDF. Never invent facts. Email is plain text, 120–180 words. Nothing sends without human approval."""


def _system_prompt() -> str:
    return "\n\n".join([BASE_SYSTEM_PROMPT, "Identity:\n" + store.profile_summary_text(), f"Today: {time.strftime('%Y-%m-%d')}"])


_llm: Optional[ChatOpenAI] = None
_active_model: Optional[str] = None


def _build_client(model: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS, max_retries=LLM_MAX_RETRIES).bind_tools(TOOLS, parallel_tool_calls=False)


def active_model() -> str:
    return _active_model or LLM_MODEL


def get_client() -> Optional[ChatOpenAI]:
    global _llm, _active_model
    if _llm is not None:
        return _llm
    if not LLM_API_KEY:
        return None
    _active_model = (list(LLM_MODEL_CHAIN) or [LLM_MODEL])[0]
    _llm = _build_client(_active_model)
    return _llm


def eager_resolve() -> List[str]:
    return [active_model()] if get_client() else []


def _agent_node(state: AgentState) -> dict:
    client = get_client()
    if client is None:
        raise RuntimeError("LLM_API_KEY is not configured")
    messages = list(state.get("messages") or [])[-MAX_HISTORY_MESSAGES:]
    return {"messages": [client.invoke([SystemMessage(content=_system_prompt()), *messages])]}


def _route(state: AgentState) -> str:
    last = (state.get("messages") or [None])[-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else END


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", _agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g


_checkpointer = None
_checkpointer_context = None
_graph = None


async def _get_checkpointer():
    global _checkpointer, _checkpointer_context
    if _checkpointer is None:
        # Current langgraph exposes from_conn_string as an async context
        # manager. Keep that context open for the process lifetime so its
        # aiosqlite connection remains valid for the compiled graph.
        _checkpointer_context = AsyncSqliteSaver.from_conn_string(str(DATA_DIR / "graph_checkpoints.sqlite"))
        _checkpointer = await _checkpointer_context.__aenter__()
    return _checkpointer


async def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph().compile(checkpointer=await _get_checkpointer())
    return _graph


@dataclass
class AgentTurn:
    status: str
    text: str = ""
    attachments: List[Dict[str, str]] = field(default_factory=list)
    approval: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tool_trace: List[str] = field(default_factory=list)


async def _collect_new_messages(graph, config, before_ids: set, turn: AgentTurn) -> None:
    state = await graph.aget_state(config)
    for msg in state.values.get("messages", []):
        if getattr(msg, "id", None) in before_ids:
            continue
        if isinstance(msg, AIMessage):
            turn.tool_trace.extend(tc["name"] for tc in (msg.tool_calls or []))
        elif isinstance(msg, ToolMessage):
            try:
                payload = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue
            attachment = payload.get("_attachment") if isinstance(payload, dict) else None
            if attachment and attachment not in turn.attachments:
                turn.attachments.append(attachment)


async def run_turn(user_id: int, user_message: Optional[str] = None, resume_tool_result: Optional[Dict[str, Any]] = None) -> AgentTurn:
    if not get_client():
        return AgentTurn(status="ERROR", error="LLM_API_KEY is not configured — set it in Railway → Variables.")
    graph = await _get_graph()
    config = {"configurable": {"thread_id": str(user_id)}, "recursion_limit": MAX_TOOL_STEPS * 2 + 4}
    before = await graph.aget_state(config)
    before_ids = {getattr(m, "id", None) for m in (before.values or {}).get("messages", []) if getattr(m, "id", None)}
    try:
        if resume_tool_result is not None:
            result = await graph.ainvoke(Command(resume=resume_tool_result["content"]), config=config)
        else:
            existing = (before.values or {}).get("job", {}) if before.values else {}
            result = await graph.ainvoke({"user_id": user_id, "job": existing, "messages": [HumanMessage(content=user_message or "")]}, config=config)
    except GraphRecursionError:
        return AgentTurn(status="ERROR", error=f"I got stuck after {MAX_TOOL_STEPS} steps. Try /reset and rephrase.")
    except Exception as exc:
        logger.error("Graph run failed for user %s: %s", user_id, exc, exc_info=True)
        return AgentTurn(status="ERROR", error=str(exc))
    turn = AgentTurn(status="MESSAGE")
    await _collect_new_messages(graph, config, before_ids, turn)
    state = await graph.aget_state(config)
    if state.next:
        pending = state.tasks[0].interrupts[0].value if state.tasks and state.tasks[0].interrupts else {}
        turn.status, turn.approval = "NEEDS_APPROVAL", {"tool_call_id": "", **pending}
        return turn
    last = (result.get("messages") or [None])[-1]
    turn.text = (last.content or "…").strip() if isinstance(last, AIMessage) else "…"
    return turn
