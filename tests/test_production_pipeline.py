import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

import graph_agent
import pipeline
from candidate import canonical_payload, choose_variant, get_variant, load_identity, select_variant
from matcher import analyze_job, analyze_seniority, candidate_skill_set
from validator import validate_email, validate_pdf, validate_selection, validate_tailoring
import cv_generator
from langchain_core.messages import AIMessage, ToolMessage
from types import SimpleNamespace


def test_candidate_source_of_truth_and_deterministic_ids():
    first, second = canonical_payload("ai"), canonical_payload("ai")
    assert first["projects"] == second["projects"]
    assert "experience" not in graph_agent.get_profile.invoke({})["identity"]


def test_variant_selection_and_ambiguous_signal():
    assert select_variant("Power BI Developer", "", "auto")[0] == "bi"
    _, reasons = select_variant("", "Tableau dashboard and Python", "auto")
    assert reasons
    choice = choose_variant("Data Scientist / BI Developer", "", "auto")
    assert choice["ambiguous"] is True


def test_long_jd_is_preserved():
    text = "Python required. " * 3000
    job = pipeline.ingest_job(text)
    assert job["job_text"] == text.strip()
    assert job["job_text_chars"] == len(text.strip())
    assert job["job_text_truncated"] is False


def test_exact_related_and_non_equivalent_technologies():
    variant = get_variant("ai")
    skills = candidate_skill_set(variant)
    assert skills["langgraph"] in {"exact", "related", "not_equivalent"}
    # Definitions prohibit treating distinct stack items as direct equivalents.
    for required, candidate in (("xgboost", "lightgbm"), ("react", "next.js"), ("langchain", "langgraph")):
        assert candidate_skill_set({"skills_categories": {"x": [candidate]}})[required] != "exact"


def test_seniority_requires_candidate_requirement_context():
    assert analyze_seniority("Our company has 5 years of history.")["years_requirement"] is None
    assert analyze_seniority("At least 2 years of Python experience required.")["years_requirement"] == 2
    assert analyze_seniority("Data Engineer")["level"] == "unspecified"
    assert analyze_seniority("Junior Data Engineer")["level"] == "junior"
    assert analyze_seniority("Senior Data Engineer")["level"] == "senior"
    assert analyze_seniority("5 years managing a team.")["years_requirement"] is None


def test_selection_rejects_duplicates_and_preserves_all_projects():
    variant = get_variant("ai")
    project_id = variant["projects"][0]["id"]
    result = validate_selection({"selected_project_ids": [project_id, project_id]}, variant)
    assert result["code"] == "DUPLICATE_SELECTION_ID"
    job = pipeline.ingest_job("AI Engineer. Python required.", role_hint="AI Engineer")
    pipeline.analyze_and_select_variant(job)
    pipeline.run_hr_screen(job)
    pipeline.apply_selection(job, {"selected_project_ids": [project_id], "selected_experience_ids": [], "selected_skill_categories": [], "selected_certification_ids": []})
    pipeline.apply_tailoring(job, {"headline": variant["headline"], "summary": variant["summary"], "rewritten_bullets_by_experience_id": {}, "rewritten_project_descriptions_by_project_id": {}})
    cv = pipeline.assemble_cv(job)
    assert len(cv["projects"]) == len(variant["projects"])
    assert len(cv["experience"]) == len(variant["experience"])


def test_assemble_cv_preserves_every_variant_baseline_section():
    for key in ("ai", "bi", "data_analyst", "data_scientist"):
        variant = get_variant(key)
        job = pipeline.ingest_job("General technology role with Python and SQL requirements.", role_hint="Technology role")
        pipeline.analyze_job(job)
        pipeline.select_job_variant(job, requested_variant=key)
        pipeline.run_hr_screen(job)
        pipeline.apply_selection(job, {"selected_project_ids": [], "selected_experience_ids": [], "selected_skill_categories": [], "selected_certification_ids": []})
        pipeline.apply_tailoring(job, {"headline": variant["headline"], "summary": variant["summary"], "rewritten_bullets_by_experience_id": {}, "rewritten_project_descriptions_by_project_id": {}})
        cv = pipeline.assemble_cv(job)
        assert len(cv["projects"]) == len(variant["projects"])
        assert len(cv["experience"]) == len(variant["experience"])
        assert cv["certifications"] == variant["certifications"]
        assert cv["education"] == variant["education"]
        assert cv["training"] == variant["training"]
        assert cv["languages"] == variant["languages"]
        assert cv["skills_categories"] == variant["skills_categories"]


def test_tailoring_and_email_reject_unsupported_claims():
    variant = get_variant("ai")
    exp = variant["experience"][0]
    tailoring = {"headline": variant["headline"], "summary": "Worked at Company X and led a team of 12.", "rewritten_bullets_by_experience_id": {exp["id"]: list(exp["bullets"])}, "rewritten_project_descriptions_by_project_id": {}}
    assert not validate_tailoring(tailoring, variant)["ok"]
    email = {"subject": "Application", "body": ("I have Kubernetes experience. " * 45)}
    assert not validate_email(email, variant, load_identity(), "Kubernetes required")["ok"]
    markdown = {"subject": "Application", "body": ("**Python experience** " * 50)}
    assert not validate_email(markdown, variant, load_identity())["ok"]
    assert not validate_email({"subject": "Application", "body": ("- Python experience\n" * 100)}, variant, load_identity())["ok"]
    assert not validate_email({"subject": "Application", "body": ("I am excited about Python. " * 45)}, variant, load_identity())["ok"]


def test_pipeline_stage_preconditions_and_graph_production_tools():
    job = pipeline.ingest_job("AI Engineer. Python required.")
    assert job["stage"] == pipeline.Stage.JOB_INGESTED.name
    pipeline.analyze_job(job)
    assert job["stage"] == pipeline.Stage.JOB_ANALYZED.name
    pipeline.select_job_variant(job)
    assert job["stage"] == pipeline.Stage.VARIANT_SELECTED.name
    try:
        pipeline.build_pdf(job)
    except Exception as exc:
        assert getattr(exc, "code", "") == "MISSING_REQUIRED_STAGE"
    else:
        assert False, "build_pdf must require validated tailoring"
    names = {tool.name for tool in graph_agent.TOOLS}
    assert "build_application" not in names
    assert {"analyze_job", "select_evidence", "generate_tailoring", "build_pdf", "generate_email"} <= names


def test_graph_tools_advance_the_single_pipeline_state():
    """Exercise production tool implementations without a live LLM."""
    intake = graph_agent.ingest_job_text.func(
        "AI Engineer role. Python and FastAPI are required for production services.",
        {},
        "intake",
        role_hint="AI Engineer",
    )
    job = intake.update["job"]
    analyzed = graph_agent.analyze_job.func({"job": job}, "analyze")
    job = analyzed.update["job"]
    assert job["stage"] == pipeline.Stage.JOB_ANALYZED.name
    chosen = graph_agent.select_variant.func({"job": job}, "variant", "auto")
    job = chosen.update["job"]
    assert job["stage"] == pipeline.Stage.VARIANT_SELECTED.name
    screened = graph_agent.hr_screen.func({"job": job}, "screen")
    job = screened.update["job"]
    variant = get_variant(job["selected_variant"])
    selected = graph_agent.select_evidence.func(
        [variant["projects"][0]["id"]],
        [variant["experience"][0]["id"]],
        [],
        {"job": job},
        "selection",
    )
    assert selected.update["job"]["stage"] == pipeline.Stage.SELECTION_VALIDATED.name
    blocked_send = graph_agent.send_email.func({"job": selected.update["job"], "user_id": 7}, "send")
    assert "READY_TO_SEND" in blocked_send.update["messages"][0].content


def test_graph_compiles_with_persistent_sqlite_checkpointer():
    assert asyncio.run(graph_agent._get_graph()) is not None


def test_evidence_ranking_prefers_matching_project():
    variant = {
        "key": "test",
        "roles": ["Data Engineer"],
        "experience": [],
        "projects": [
            {"id": "a", "name": "Python ETL", "tech": "Python SQL", "bullets": ["Built Python ETL pipelines."]},
            {"id": "b", "name": "Portfolio Site", "tech": "HTML CSS", "bullets": ["Static page."]},
        ],
        "skills_categories": {},
    }
    analysis = analyze_job("Data Engineer. Python and SQL required.", variant, role_hint="Data Engineer")
    ranking = {item["id"]: item["relevance_score"] for item in analysis["evidence"] if item["type"] == "project"}
    assert ranking["a"] > ranking["b"]


def test_pdf_validation_reports_real_page_count_and_all_sections():
    cv = {
        "name": "Candidate With An Intentionally Long Professional Name",
        "headline": "Data Engineer",
        "summary": "A concise verified summary that is long enough for extraction.",
        "location": "An intentionally long location field that should wrap safely in the contact row",
        "email": "long.address.for.a.candidate@example.com",
        "linkedin": "https://linkedin.example.com/in/a-very-long-profile-address",
        "github": "https://github.example.com/an-intentionally-long-account-name",
        "experience": [{"company": "A Very Long Company Name Incorporated", "role": "Senior Data Engineering Specialist", "period": "2020 - Present", "bullets": ["Built reliable Python systems."]}],
        "projects": [{"name": "A Long Project Name That Must Wrap Rather Than Clip", "tech": "Python, SQL", "bullets": ["Created an extractable project description."]}],
        "skills_categories": {"A Very Long Skills Category Name": ["Python", "SQL"]},
        "education": ["BSc Computer Science"],
        "certifications": ["Python Certificate"],
        "training": ["Data Engineering Simulation"],
        "languages": ["English (Professional)"],
        "military_service": "Completed",
    }
    cv_generator.GENERATED_CVS_DIR = Path(tempfile.mkdtemp(prefix="cv_long_content_"))
    path = cv_generator.generate_pdf_cv(cv, filename="long_content_regression.pdf")
    result = validate_pdf(path, cv)
    assert result["ok"], result
    assert result["pages"] >= 1
    missing = dict(cv)
    missing["training"] = ["Absent Training Record"]
    assert not validate_pdf(path, missing)["ok"]


def test_attachment_collector_excludes_messages_before_turn():
    old = AIMessage(id="old", content="", tool_calls=[{"name": "build_pdf", "args": {}, "id": "old-call", "type": "tool_call"}])
    new = ToolMessage(id="new", tool_call_id="new-call", content='{"_attachment":{"path":"new.pdf"}}')

    class FakeGraph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={"messages": [old, new]})

    turn = graph_agent.AgentTurn(status="MESSAGE")
    asyncio.run(graph_agent._collect_new_messages(FakeGraph(), {}, {"old"}, turn))
    assert turn.tool_trace == []
    assert turn.attachments == [{"path": "new.pdf"}]
