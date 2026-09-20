"""Compatibility facade.

Production scoring lives in matcher.analyze_job. This module keeps the
historical score_job() shape used by older callers and smoke tests.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from candidate import get_variant, select_variant
from matcher import analyze_job, extract_jd_skills


def score_job(job_description: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score a job 0–100 against the selected canonical CV variant."""
    key, reasons = select_variant("", job_description or "", requested="auto")
    variant = get_variant(key)
    analysis = analyze_job(job_description or "", variant)
    matches = analysis.get("candidate_skill_matches") or {}
    roles = analysis.get("role_alignment") or {}
    seniority = analysis.get("seniority") or {}
    return {
        "score": analysis.get("score", 0),
        "strong_matches": matches.get("exact_required") or [],
        "missing_skills": analysis.get("missing_skills") or [],
        "seniority_level": seniority.get("level"),
        "role_matched": roles.get("matched_roles") or [],
        "role_alignment": roles.get("relation"),
        "selected_variant": key,
        "variant_reasons": reasons,
        "reason": analysis.get("recruiter_summary") or "",
        "analysis": analysis,
    }


def extract_skills(job_text: str) -> Dict[str, Any]:
    return extract_jd_skills(job_text)
