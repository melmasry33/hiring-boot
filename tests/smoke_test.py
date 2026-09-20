"""Offline production smoke test. Run: python tests/smoke_test.py."""
import os
import sys
import tempfile

TEST_DATA = tempfile.mkdtemp(prefix="hiring_pipeline_smoke_")
os.environ.setdefault("DATA_DIR", TEST_DATA)
os.environ.setdefault("GENERATED_CVS_DIR", os.path.join(TEST_DATA, "cvs"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

import graph_agent  # noqa: E402
import pipeline  # noqa: E402
import store  # noqa: E402
from candidate import get_variant, load_identity  # noqa: E402
from validator import validate_email, validate_selection  # noqa: E402


def main():
    store.bootstrap()
    assert "build_application" not in [tool.name for tool in graph_agent.TOOLS]
    required = {"ingest_job_text", "analyze_job", "hr_screen", "select_evidence", "generate_tailoring", "build_pdf", "generate_email", "send_email"}
    assert required <= {tool.name for tool in graph_agent.TOOLS}

    job = pipeline.ingest_job(
        "AI Engineer\nRequirements\nPython and FastAPI required.\nNice to have: Kubernetes.",
        role_hint="AI Engineer",
    )
    pipeline.analyze_and_select_variant(job)
    pipeline.run_hr_screen(job)
    variant = get_variant(job["selected_variant"])
    selection = {
        "selected_project_ids": [variant["projects"][0]["id"]],
        "selected_experience_ids": [variant["experience"][0]["id"]],
        "selected_skill_categories": list(variant["skills_categories"])[:1],
        "selected_certification_ids": [],
    }
    pipeline.apply_selection(job, selection)
    pipeline.apply_tailoring(job, {
        "headline": variant["headline"],
        "summary": variant["summary"],
        "rewritten_bullets_by_experience_id": {},
        "rewritten_project_descriptions_by_project_id": {},
    })
    pipeline.build_pdf(job)
    body = ("Dear Hiring Team,\n\nI am applying for the AI Engineer role. My canonical CV documents "
            "Python and FastAPI work alongside AI product delivery. I would welcome the opportunity to "
            "discuss how that experience can support your team.\n\nKind regards,\nCandidate")
    # Pad the deterministic fixture without making a new factual claim.
    body = body.replace("I would welcome", "I would welcome " + "the opportunity " * 70)
    pipeline.apply_email(job, {"subject": "Application: AI Engineer", "body": body, "fit_summary": "Python and FastAPI evidence.", "gap_notes": ["Kubernetes is not documented." ]}, load_identity())
    pipeline.mark_ready_if_complete(job)
    assert job["stage"] == pipeline.Stage.READY_TO_SEND.name
    assert os.path.exists(job["pdf_path"])
    assert len(job["cv_data"]["projects"]) == len(variant["projects"])
    assert not validate_selection({**selection, "selected_project_ids": selection["selected_project_ids"] * 2}, variant)["ok"]
    assert not validate_email({"subject": "s", "body": "**Markdown** " * 100}, variant, load_identity(), "Kubernetes required")["ok"]
    print("Production pipeline smoke test passed.")


if __name__ == "__main__":
    main()
