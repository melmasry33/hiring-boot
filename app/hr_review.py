"""HR screening facade over the shared matcher analysis."""

from __future__ import annotations

from typing import Any, Dict, Optional

from candidate import get_variant, select_variant
from matcher import analyze_job, format_hr_review_for_chat as _format


def hr_screen(job_text: str, profile: Optional[Dict[str, Any]] = None, variant_key: str = "") -> Dict[str, Any]:
    """Recruiter screen against the selected canonical variant, not profile.json."""
    if not (job_text or "").strip():
        return {"ok": False, "error": "Empty job text — ingest or read a posting first."}
    key = variant_key
    reasons = ["explicit variant"] if key else []
    if not key:
        key, reasons = select_variant("", job_text, requested="auto")
    variant = get_variant(key)
    analysis = analyze_job(job_text, variant)
    analysis["ok"] = True
    analysis["selected_variant"] = key
    analysis["variant_reasons"] = reasons
    analysis["must_have_skills"] = analysis.get("required_skills") or []
    analysis["nice_to_have_skills"] = analysis.get("preferred_skills") or []
    analysis["covered_must_haves"] = (analysis.get("candidate_skill_matches") or {}).get("exact_required") or []
    analysis["missing_must_haves"] = analysis.get("missing_skills") or []
    analysis["ats_keywords_to_echo"] = analysis["covered_must_haves"][:12]
    analysis["dealbreakers"] = [
        {"kind": c.get("kind"), "detail": f"[{c.get('strength')}] {c.get('span')}"}
        for c in analysis.get("constraints") or []
        if c.get("strength") == "required"
    ]
    analysis["rewrite_priorities"] = [
        "Lead with exact matched must-haves; never claim related skills as direct experience.",
        "Reorder canonical evidence; do not drop baseline sections.",
        "Rewrite existing bullets only; never invent metrics or tools.",
    ]
    return analysis


def format_hr_review_for_chat(review: Dict[str, Any]) -> str:
    return _format(review)
