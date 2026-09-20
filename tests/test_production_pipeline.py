import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

import graph_agent
import pipeline
from candidate import canonical_payload, get_variant, load_identity, select_variant
from matcher import analyze_job, analyze_seniority, candidate_skill_set
from validator import validate_email, validate_selection, validate_tailoring


def test_candidate_source_of_truth_and_deterministic_ids():
    first, second = canonical_payload("ai"), canonical_payload("ai")
    assert first["projects"] == second["projects"]
    assert "experience" not in graph_agent.get_profile.invoke({})["identity"]


def test_variant_selection_and_ambiguous_signal():
    assert select_variant("Power BI Developer", "", "auto")[0] == "bi"
    _, reasons = select_variant("", "Tableau dashboard and Python", "auto")
    assert reasons


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


def test_tailoring_and_email_reject_unsupported_claims():
    variant = get_variant("ai")
    exp = variant["experience"][0]
    tailoring = {"headline": variant["headline"], "summary": "Worked at Company X and led a team of 12.", "rewritten_bullets_by_experience_id": {exp["id"]: list(exp["bullets"])}, "rewritten_project_descriptions_by_project_id": {}}
    assert not validate_tailoring(tailoring, variant)["ok"]
    email = {"subject": "Application", "body": ("I have Kubernetes experience. " * 45)}
    assert not validate_email(email, variant, load_identity(), "Kubernetes required")["ok"]
    markdown = {"subject": "Application", "body": ("**Python experience** " * 50)}
    assert not validate_email(markdown, variant, load_identity())["ok"]


def test_pipeline_stage_preconditions_and_graph_production_tools():
    job = pipeline.ingest_job("AI Engineer. Python required.")
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
    analyzed = graph_agent.analyze_job.func({"job": job}, "analyze", "auto")
    job = analyzed.update["job"]
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
