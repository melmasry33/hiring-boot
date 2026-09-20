"""Application pipeline: explicit stages, canonicalization, no silent fallbacks."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from candidate import (
    canonical_payload,
    certifications_by_id,
    experience_by_id,
    get_variant,
    load_identity,
    projects_by_id,
    select_variant,
)
from config import logger
from matcher import analyze_job as analyze_candidate_match, analyze_seniority, build_tailoring_brief, extract_jd_skills, format_hr_review_for_chat
from validator import (
    PipelineError,
    validate_email,
    validate_pdf,
    validate_selection,
    validate_tailoring,
)


class Stage(IntEnum):
    EMPTY = 0
    JOB_INGESTED = 1
    JOB_ANALYZED = 2
    VARIANT_SELECTED = 3
    HR_SCREENED = 4
    SELECTION_GENERATED = 5
    SELECTION_VALIDATED = 6
    TAILORING_GENERATED = 7
    TAILORING_VALIDATED = 8
    PDF_GENERATED = 9
    PDF_VALIDATED = 10
    EMAIL_GENERATED = 11
    EMAIL_VALIDATED = 12
    READY_TO_SEND = 13


STAGE_ORDER = [s.name for s in Stage]


def current_stage(job: Dict[str, Any]) -> Stage:
    name = job.get("stage") or "EMPTY"
    try:
        return Stage[name]
    except KeyError:
        return Stage.EMPTY


def require_stage(job: Dict[str, Any], minimum: Stage) -> Optional[str]:
    stage = current_stage(job)
    if stage < minimum:
        return (
            f"MISSING_REQUIRED_STAGE: need {minimum.name}, currently {stage.name}. "
            "Complete the earlier pipeline step first."
        )
    return None


def _log_stage(job: Dict[str, Any], extra: str = "") -> None:
    logger.info(
        "pipeline stage=%s variant=%s score=%s job_url=%s %s",
        job.get("stage"),
        job.get("selected_variant"),
        (job.get("analysis") or {}).get("score"),
        job.get("job_url") or "",
        extra,
    )


def ingest_job(
    job_text: str,
    job_url: str = "",
    role_hint: str = "",
    company_hint: str = "",
) -> Dict[str, Any]:
    text = (job_text or "").strip()
    job: Dict[str, Any] = {
        "job_url": (job_url or "").strip(),
        "job_text": text,
        "job_text_chars": len(text),
        "job_text_truncated": False,
        "role_hint": (role_hint or "").strip(),
        "company_hint": (company_hint or "").strip(),
        "stage": Stage.JOB_INGESTED.name,
        "analysis": {},
        "selected_variant": "",
        "variant_reasons": [],
        "hr_screen": {},
        "tailoring_brief": {},
        "selection": {},
        "tailoring": {},
        "email": {},
        "cv_data": {},
        "pdf_path": "",
        "pdf_validation": {},
        "application_built": False,
    }
    _log_stage(job, extra=f"chars={len(text)}")
    return job


def analyze_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Extract job-only facts before candidate variant selection."""
    err = require_stage(job, Stage.JOB_INGESTED)
    if err:
        raise PipelineError("MISSING_REQUIRED_STAGE", err)
    text = job.get("job_text") or ""
    job["analysis"] = {
        "job_skills": extract_jd_skills(text),
        "seniority": analyze_seniority(text),
        "role": (job.get("role_hint") or "").strip(),
        "company": (job.get("company_hint") or "").strip(),
    }
    job["stage"] = Stage.JOB_ANALYZED.name
    _log_stage(job)
    return job


def select_job_variant(job: Dict[str, Any], requested_variant: str = "auto") -> Dict[str, Any]:
    """Select a canonical variant after job-only analysis, then add candidate match analysis."""
    err = require_stage(job, Stage.JOB_ANALYZED)
    if err:
        raise PipelineError("MISSING_REQUIRED_STAGE", err)
    text = job.get("job_text") or ""
    role = job.get("role_hint") or ""
    key, reasons = select_variant(role, text, requested=requested_variant)
    variant = get_variant(key)
    job_only_analysis = dict(job.get("analysis") or {})
    analysis = analyze_candidate_match(text, variant, role_hint=role, company_hint=job.get("company_hint") or "")
    analysis["job_analysis"] = job_only_analysis
    analysis["selected_variant"] = key
    analysis["variant_reasons"] = reasons
    job["selected_variant"] = key
    job["variant_reasons"] = reasons
    job["analysis"] = analysis
    job["stage"] = Stage.VARIANT_SELECTED.name
    _log_stage(job, extra=f"reasons={reasons}")
    return job


def analyze_and_select_variant(job: Dict[str, Any], requested_variant: str = "auto") -> Dict[str, Any]:
    """Compatibility convenience wrapper preserving every explicit transition."""
    analyze_job(job)
    return select_job_variant(job, requested_variant)


def run_hr_screen(job: Dict[str, Any]) -> Dict[str, Any]:
    err = require_stage(job, Stage.VARIANT_SELECTED)
    if err:
        raise PipelineError("MISSING_REQUIRED_STAGE", err)
    variant = get_variant(job["selected_variant"])
    analysis = job.get("analysis") or {}
    if not analysis:
        analysis = analyze_candidate_match(
            job.get("job_text") or "",
            variant,
            role_hint=job.get("role_hint") or "",
            company_hint=job.get("company_hint") or "",
        )
        job["analysis"] = analysis
    brief = build_tailoring_brief(analysis, variant)
    review = dict(analysis)
    review["chat_summary"] = format_hr_review_for_chat(analysis)
    review["canonical_candidate"] = canonical_payload(job["selected_variant"])
    review["tailoring_brief"] = brief
    job["hr_screen"] = review
    job["tailoring_brief"] = brief
    job["stage"] = Stage.HR_SCREENED.name
    _log_stage(job, extra=f"verdict={analysis.get('verdict')}")
    return job


def apply_selection(job: Dict[str, Any], selection: Dict[str, Any]) -> Dict[str, Any]:
    err = require_stage(job, Stage.HR_SCREENED)
    if err:
        raise PipelineError("MISSING_REQUIRED_STAGE", err)
    variant = get_variant(job["selected_variant"])
    job["selection"] = {
        "selected_project_ids": list(selection.get("selected_project_ids") or []),
        "selected_experience_ids": list(selection.get("selected_experience_ids") or []),
        "selected_skill_categories": list(selection.get("selected_skill_categories") or []),
        "selected_certification_ids": list(selection.get("selected_certification_ids") or []),
    }
    job["stage"] = Stage.SELECTION_GENERATED.name
    result = validate_selection(job["selection"], variant)
    if not result["ok"]:
        job["selection_validation"] = result
        raise PipelineError(result.get("code") or "INVALID_SELECTION", result.get("error") or "Selection failed")
    job["selection"] = result["selection"]
    job["selection_validation"] = result
    job["stage"] = Stage.SELECTION_VALIDATED.name
    _log_stage(job, extra=f"projects={len(job['selection']['selected_project_ids'])}")
    return job


def apply_tailoring(job: Dict[str, Any], tailoring: Dict[str, Any]) -> Dict[str, Any]:
    err = require_stage(job, Stage.SELECTION_VALIDATED)
    if err:
        raise PipelineError("MISSING_REQUIRED_STAGE", err)
    variant = get_variant(job["selected_variant"])
    job["tailoring"] = dict(tailoring or {})
    job["stage"] = Stage.TAILORING_GENERATED.name
    result = validate_tailoring(job["tailoring"], variant, job.get("job_text") or "")
    if not result["ok"]:
        job["tailoring_validation"] = result
        raise PipelineError(result.get("code") or "UNSUPPORTED_CLAIM", result.get("error") or "Tailoring failed")
    job["tailoring"] = result["tailoring"]
    job["tailoring_validation"] = result
    job["stage"] = Stage.TAILORING_VALIDATED.name
    _log_stage(job)
    return job


def apply_email(job: Dict[str, Any], email: Dict[str, Any], identity: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    err = require_stage(job, Stage.PDF_VALIDATED)
    if err:
        raise PipelineError("MISSING_REQUIRED_STAGE", err)
    identity = identity or load_identity()
    variant = get_variant(job["selected_variant"])
    job["email"] = {
        "subject": (email.get("subject") or "").strip(),
        "body": email.get("body") or "",
        "fit_summary": email.get("fit_summary") or "",
        "gap_notes": list(email.get("gap_notes") or []),
        "recruiter_email": (email.get("recruiter_email") or "").strip(),
    }
    job["stage"] = Stage.EMAIL_GENERATED.name
    result = validate_email(job["email"], variant, identity, job.get("job_text") or "")
    if not result["ok"]:
        job["email_validation"] = result
        raise PipelineError(result.get("code") or "EMAIL_VALIDATION_FAILED", result.get("error") or "Email failed")
    job["email"] = result["email"]
    job["email_validation"] = result
    job["stage"] = Stage.EMAIL_VALIDATED.name
    _log_stage(job)
    return job


def _reorder(items: List[Dict[str, Any]], selected_ids: List[str]) -> List[Dict[str, Any]]:
    by_id = {i["id"]: i for i in items}
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for ident in selected_ids:
        if ident in by_id and ident not in seen:
            ordered.append(dict(by_id[ident]))
            seen.add(ident)
    for item in items:
        if item["id"] not in seen:
            ordered.append(dict(item))
            seen.add(item["id"])
    return ordered


def assemble_cv(job: Dict[str, Any], identity: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Preserve every baseline section; selected IDs only change order/emphasis."""
    identity = identity or load_identity()
    variant = get_variant(job["selected_variant"])
    selection = job.get("selection") or {}
    tailoring = job.get("tailoring") or {}

    exp_map = experience_by_id(variant)
    proj_map = projects_by_id(variant)
    experience = _reorder(list(variant.get("experience") or []), selection.get("selected_experience_ids") or [])
    rewritten_exp = tailoring.get("rewritten_bullets_by_experience_id") or {}
    for item in experience:
        proposed = rewritten_exp.get(item["id"])
        if proposed:
            original = list(item.get("bullets") or [])
            merged = []
            for i, orig in enumerate(original):
                merged.append(proposed[i] if i < len(proposed) else orig)
            item["bullets"] = merged

    projects = _reorder(list(variant.get("projects") or []), selection.get("selected_project_ids") or [])
    rewritten_proj = tailoring.get("rewritten_project_descriptions_by_project_id") or {}
    for item in projects:
        proposed = rewritten_proj.get(item["id"])
        if proposed:
            original = list(item.get("bullets") or [])
            merged = []
            for i, orig in enumerate(original):
                merged.append(proposed[i] if i < len(proposed) else orig)
            item["bullets"] = merged

    cats = dict(variant.get("skills_categories") or {})
    ordered_cats: Dict[str, List[str]] = {}
    for key in selection.get("selected_skill_categories") or []:
        if key in cats and key not in ordered_cats:
            ordered_cats[key] = list(cats[key])
    for key, values in cats.items():
        if key not in ordered_cats:
            ordered_cats[key] = list(values)

    cert_records = _reorder(
        list(variant.get("certification_records") or []),
        selection.get("selected_certification_ids") or [],
    )

    headline = (tailoring.get("headline") or variant.get("headline") or "").strip()
    summary = (tailoring.get("summary") or variant.get("summary") or "").strip()
    military = variant.get("additional_information") or ""

    cv_data = {
        "name": identity.get("name") or "",
        "headline": headline,
        "summary": summary,
        "location": identity.get("location") or "",
        "phone": identity.get("phone") or "",
        "email": identity.get("email") or "",
        "linkedin": identity.get("linkedin") or "",
        "github": identity.get("github") or "",
        "datacamp": identity.get("datacamp") or "",
        "experience": experience,
        "projects": projects,
        "education": list(variant.get("education") or []),
        "certifications": [c["text"] for c in cert_records],
        "training": list(variant.get("training") or []),
        "languages": list(variant.get("languages") or []),
        "military_service": military,
        "skills_categories": ordered_cats,
        "variant_key": job["selected_variant"],
    }
    job["cv_data"] = cv_data
    return cv_data


def build_pdf(job: Dict[str, Any], identity: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    err = require_stage(job, Stage.TAILORING_VALIDATED)
    if err:
        raise PipelineError("MISSING_REQUIRED_STAGE", err)
    identity = identity or load_identity()
    cv_data = assemble_cv(job, identity)
    import cv_generator

    role = (job.get("role_hint") or (job.get("email") or {}).get("role") or "Role")
    name = identity.get("name") or "CV"
    safe = "".join(c if c.isalnum() else "_" for c in f"{name}_{role}")[:60]
    pdf_path = cv_generator.generate_pdf_cv(cv_data, filename=f"{safe}.pdf")
    job["pdf_path"] = pdf_path
    job["stage"] = Stage.PDF_GENERATED.name
    result = validate_pdf(pdf_path, cv_data)
    job["pdf_validation"] = result
    if not result["ok"]:
        raise PipelineError(result.get("code") or "PDF_CONTENT_MISMATCH", result.get("error") or "PDF failed")
    job["stage"] = Stage.PDF_VALIDATED.name
    job["application_built"] = True
    _log_stage(job, extra=f"pdf={pdf_path}")
    return job


def mark_ready_if_complete(job: Dict[str, Any]) -> Dict[str, Any]:
    if job.get("pdf_validation", {}).get("ok") and job.get("email_validation", {}).get("ok"):
        job["stage"] = Stage.READY_TO_SEND.name
        job["application_built"] = True
    return job
