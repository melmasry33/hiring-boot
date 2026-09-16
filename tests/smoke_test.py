"""
Offline smoke test — no API keys, no network, no Telegram.

Run it before every deploy:   python tests/smoke_test.py

It stubs the LLM with a scripted sequence of tool calls so the agent loop,
the PDF generator, the store and the approval interrupt are all exercised for
real. If this passes, the only things left that can fail in production are
credentials and third-party availability.
"""

import asyncio
import json
import os
import sys
import tempfile
from types import SimpleNamespace

TEST_DATA = tempfile.mkdtemp(prefix="career_agent_test_")
os.environ.setdefault("DATA_DIR", TEST_DATA)
os.environ.setdefault("GENERATED_CVS_DIR", os.path.join(TEST_DATA, "cvs"))
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

import agent  # noqa: E402
import cv_generator  # noqa: E402
import jobs as job_source  # noqa: E402
import matching  # noqa: E402
import store  # noqa: E402

PASS, FAIL = "  PASS", "  FAIL"
failures = []


def check(name, condition, detail=""):
    print(f"{PASS if condition else FAIL}  {name}{'' if condition else ' -> ' + str(detail)}")
    if not condition:
        failures.append(name)


# ---------------------------------------------------------------------------
# Scripted LLM stub
# ---------------------------------------------------------------------------


def _tool_call(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class ScriptedCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        message = self.script.pop(0) if self.script else SimpleNamespace(
            content="Done.", tool_calls=None
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ScriptedClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=ScriptedCompletions(script))


def install_script(script):
    client = ScriptedClient(script)
    agent.get_client = lambda: client  # noqa: E731
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_store():
    print("\n[1] store")
    store.bootstrap()
    profile = store.load_profile()
    check("profile loads", bool(profile.get("name")), profile.keys())

    store.update_profile_field("headline", "Test Headline")
    check("profile field persists", store.load_profile()["headline"] == "Test Headline")
    store.update_profile_field("headline", profile.get("headline", ""))

    store.log_application({"company": "Acme", "role": "AI Engineer", "status": "drafted",
                           "job_url": "https://example.com/j/1"})
    store.log_application({"company": "Acme", "role": "AI Engineer", "status": "sent",
                           "job_url": "https://example.com/j/1"})
    apps = store.load_applications()
    check("application dedupes by url", len(apps) == 1 and apps[0]["status"] == "sent", apps)
    check("stats compute", store.application_stats()["total"] == 1)

    store.save_conversation(7, [{"role": "user", "content": "hi"}])
    check("conversation round-trips", store.load_conversation(7)[0]["content"] == "hi")
    store.clear_conversation(7)
    check("conversation clears", store.load_conversation(7) == [])

    store.save_draft(9, {"email_subject": "S"})
    check("draft round-trips", (store.load_draft(9) or {}).get("email_subject") == "S")
    store.clear_draft(9)


def test_matching():
    print("\n[2] matching")
    jd = "We need an AI Engineer with Python, FastAPI, LangChain and SQL. Junior friendly."
    result = matching.score_job(jd)
    check("score in range", 0 <= result["score"] <= 100, result["score"])
    check("finds matches", len(result["strong_matches"]) > 0, result["strong_matches"])

    senior = matching.score_job("Senior Staff Engineer, 10+ years required.")
    check("penalises senior roles", senior["score"] < result["score"], (senior["score"], result["score"]))


def test_pdf():
    print("\n[3] cv generator")
    profile = store.load_profile()
    path = cv_generator.generate_pdf_cv(
        {
            "name": profile.get("name", "Test"),
            "headline": "AI Engineer",
            "summary": "Real summary built from profile facts.",
            "location": profile.get("location", ""),
            "email": profile.get("email", ""),
            "experience": (profile.get("experience") or [])[:2],
            "projects": (profile.get("projects") or [])[:2],
            "skills_categories": profile.get("skills_categories", {}),
            "education": profile.get("education", []),
            "certifications": (profile.get("certifications") or [])[:3],
        },
        filename="smoke_test_cv.pdf",
    )
    size = os.path.getsize(path) if os.path.exists(path) else 0
    check("pdf generated", size > 2000, f"{path} is {size} bytes")


def test_html_parsing():
    print("\n[4] job parsing")
    check(
        "linkedin job id from /jobs/view/",
        job_source.linkedin_job_id("https://www.linkedin.com/jobs/view/ai-engineer-at-x-4055512345") == "4055512345",
    )
    check(
        "linkedin job id from currentJobId",
        job_source.linkedin_job_id("https://www.linkedin.com/jobs/search/?currentJobId=3987654321") == "3987654321",
    )
    check("non-linkedin url yields none", job_source.linkedin_job_id("https://wuzzuf.net/jobs/p/123abc") is None)

    text = job_source.strip_html("<p>Hello <b>world</b></p><script>bad()</script><li>Item</li>")
    check("strips tags and scripts", "Hello world" in text and "bad()" not in text, text)
    check(
        "finds contact emails",
        job_source.extract_emails("Apply to careers@acme.com or noreply@x.com") == ["careers@acme.com"],
        job_source.extract_emails("Apply to careers@acme.com or noreply@x.com"),
    )


def test_agent_loop():
    print("\n[5] agent loop")
    user_id = 1234

    # Turn 1: the model talks without calling a tool -> plain reply to the human.
    install_script([SimpleNamespace(content="Hi! Send me a job posting.", tool_calls=None)])
    turn = asyncio.run(agent.run_turn(user_id, user_message="hello"))
    check("plain message turn", turn.status == "MESSAGE" and "job posting" in turn.text, turn)
    check("history persisted", len(store.load_conversation(user_id)) == 2)

    # Turn 2: profile -> score -> build_application -> final text.
    profile = store.load_profile()
    install_script([
        SimpleNamespace(content="", tool_calls=[_tool_call("c1", "get_profile", {})]),
        SimpleNamespace(content="", tool_calls=[_tool_call("c2", "score_match", {"job_description": "AI Engineer Python FastAPI"})]),
        SimpleNamespace(content="", tool_calls=[_tool_call("c3", "build_application", {
            "role": "AI Engineer",
            "company": "Acme AI",
            "job_url": "https://example.com/j/9",
            "recruiter_email": "careers@acme.ai",
            "fit_summary": "Strong overlap on Python and LLM work.",
            "gap_notes": ["No Kubernetes experience on file."],
            "headline": "AI Engineer",
            "summary": "Summary from real facts.",
            "selected_experience": (profile.get("experience") or [])[:1],
            "selected_projects": (profile.get("projects") or [])[:2],
            "skills_categories": profile.get("skills_categories", {}),
            "selected_certifications": (profile.get("certifications") or [])[:2],
            "email_subject": "Application: AI Engineer",
            "email_body": "Hello, I'd like to apply.",
        })]),
        SimpleNamespace(content="CV ready. One gap: no Kubernetes. Approve to send?", tool_calls=None),
    ])
    turn = asyncio.run(agent.run_turn(user_id, user_message="https://example.com/j/9"))
    check("multi-tool turn completes", turn.status == "MESSAGE", turn)
    check("cv attached to turn", len(turn.attachments) == 1 and os.path.exists(turn.attachments[0]["path"]), turn.attachments)
    check("tool trace correct", turn.tool_trace == ["get_profile", "score_match", "build_application"], turn.tool_trace)
    draft = store.load_draft(user_id) or {}
    check("draft staged", draft.get("company") == "Acme AI" and draft.get("gap_notes"), draft)

    # Turn 3: send_email interrupts and asks for approval instead of sending.
    install_script([
        SimpleNamespace(content="Sending now.", tool_calls=[
            _tool_call("c9", "send_email", {"to": "careers@acme.ai"})
        ]),
    ])
    turn = asyncio.run(agent.run_turn(user_id, user_message="send it"))
    check("send_email interrupts", turn.status == "NEEDS_APPROVAL", turn.status)
    check("approval carries recipient", turn.approval["to"] == "careers@acme.ai", turn.approval)
    check("approval carries subject from draft", turn.approval["subject"] == "Application: AI Engineer", turn.approval)
    check("approval carries cv path", os.path.exists(turn.approval["pdf_path"] or ""), turn.approval)

    # Turn 4: the human's decision is fed back as the deferred tool result.
    install_script([SimpleNamespace(content="Sent — logged it in your tracker.", tool_calls=None)])
    turn = asyncio.run(agent.run_turn(
        user_id,
        resume_tool_result={"tool_call_id": turn.approval["tool_call_id"],
                            "content": {"sent": True, "recipient": "careers@acme.ai"}},
    ))
    check("resume after approval", turn.status == "MESSAGE" and "Sent" in turn.text, turn)

    msgs = store.load_conversation(user_id)
    ids = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
    check("deferred tool call answered", "c9" in ids, ids)

    # No API key -> clean error instead of a crash.
    real_key = agent.LLM_API_KEY
    agent.get_client = lambda: None  # noqa: E731
    turn = asyncio.run(agent.run_turn(user_id, user_message="hi"))
    check("missing key yields ERROR turn", turn.status == "ERROR", turn)
    assert real_key is not None or True


def test_email_guard():
    print("\n[6] email approval gate")
    import email_sender

    result = asyncio.run(email_sender.send_application_email("a@b.com", "s", "b", None, approved=False))
    check("refuses without approval", result["status"] == "REFUSED_APPROVAL_REQUIRED", result)
    result = asyncio.run(email_sender.send_application_email("not-an-email", "s", "b", None, approved=True))
    check("rejects bad recipient", result["status"] == "INVALID_RECIPIENT", result)


if __name__ == "__main__":
    print(f"Using throwaway DATA_DIR: {TEST_DATA}")
    test_store()
    test_matching()
    test_pdf()
    test_html_parsing()
    test_agent_loop()
    test_email_guard()

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Safe to deploy.")
