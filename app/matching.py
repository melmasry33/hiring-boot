"""
Deterministic job-fit scoring against the stored candidate profile.

Used as a hard gate before CV generation. Complements hr_review.hr_screen
with a simpler 0–100 keyword/seniority score the agent can quote quickly.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from store import load_profile

try:
    from hr_review import extract_jd_skills, _profile_skill_set, _seniority_signal, _role_alignment
except ImportError:  # pragma: no cover
    extract_jd_skills = None  # type: ignore
    _profile_skill_set = None  # type: ignore
    _seniority_signal = None  # type: ignore
    _role_alignment = None  # type: ignore


def _flat_skills(profile: Dict[str, Any]) -> List[str]:
    skills: List[str] = []
    for s in profile.get("skills") or []:
        if s:
            skills.append(str(s))
    cats = profile.get("skills_categories") or {}
    if isinstance(cats, dict):
        for values in cats.values():
            for v in values or []:
                if v and str(v) not in skills:
                    skills.append(str(v))
    return skills


def score_job(job_description: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Score a job description 0–100 against the local profile.

    Weights (approx):
      - Must-have / skill coverage: ~55
      - Seniority suitability: ~20
      - Target-role alignment: ~15
      - Nice-to-have / domain: ~10
    """
    if profile is None:
        profile = load_profile()

    desc = job_description or ""
    desc_lower = desc.lower()

    # Prefer lexicon-based JD skill extraction when available.
    if extract_jd_skills and _profile_skill_set:
        jd = extract_jd_skills(desc)
        have: Set[str] = _profile_skill_set(profile)
        must = jd["must_have"] or jd["all"]
        nice = jd["nice_to_have"]
        strong_matches = [s for s in must if s in have] + [s for s in nice if s in have and s not in must]
        # "missing" = JD skills the profile lacks (what HR cares about)
        missing_skills = [s for s in must if s not in have]
        skill_score = (len([s for s in must if s in have]) / max(len(must), 1)) * 55.0
        if len(strong_matches) >= 5:
            skill_score = min(55.0, skill_score + 8.0)
        nice_bonus = min(10.0, len([s for s in nice if s in have]) * 2.5)
    else:
        skills = _flat_skills(profile)
        strong_matches = []
        missing_skills = []
        for skill in skills:
            escaped = re.escape(skill.lower())
            pattern = rf"(?:\b|_){escaped}(?:\b|_)"
            if re.search(pattern, desc_lower):
                strong_matches.append(skill)
            else:
                missing_skills.append(skill)
        skill_score = (len(strong_matches) / max(len(skills), 1)) * 55.0
        if len(strong_matches) >= 5:
            skill_score = min(55.0, skill_score + 8.0)
        nice_bonus = 0.0

    if _role_alignment:
        role_info = _role_alignment(desc, profile)
        role_matched = role_info["matched_roles"]
    else:
        role_matched = []
        for role in profile.get("target_roles") or []:
            escaped = re.escape(str(role).lower())
            if re.search(rf"\b{escaped}\b", desc_lower):
                role_matched.append(role)

    if _seniority_signal:
        seniority = _seniority_signal(desc)
        level = seniority.get("level", "unknown")
        is_junior_friendly = seniority["flags"]["junior_friendly"]
        is_senior_heavy = seniority["flags"]["senior_heavy"]
    else:
        level = "unknown"
        is_junior_friendly = False
        is_senior_heavy = False

    # Seniority: reward junior/mid postings; penalize senior/lead bars.
    seniority_score = 12.0
    if level == "junior" or is_junior_friendly:
        seniority_score = 20.0
    elif level == "mid":
        seniority_score = 16.0
    elif level == "senior" or is_senior_heavy:
        seniority_score = 4.0

    role_score = 15.0 if role_matched else 5.0

    final_score = int(round(max(0.0, min(100.0, skill_score + seniority_score + role_score + nice_bonus))))

    reasons: List[str] = []
    if strong_matches:
        reasons.append(f"Matched {len(strong_matches)} skills: {', '.join(list(strong_matches)[:6])}")
    if missing_skills:
        reasons.append(f"Missing must-haves: {', '.join(missing_skills[:5])}")
    if role_matched:
        reasons.append(f"Matches target role ({', '.join(role_matched)})")
    if is_junior_friendly or level == "junior":
        reasons.append("Junior/entry-level friendly experience requirements")
    elif level == "mid":
        reasons.append("Mid-level experience band")
    if is_senior_heavy or level == "senior":
        reasons.append("Warning: senior/lead/5+ years requirements")
    if not reasons:
        reasons.append("General match based on keywords")

    return {
        "score": final_score,
        "strong_matches": list(strong_matches)[:12],
        "missing_skills": list(missing_skills)[:12],
        "seniority_level": level,
        "role_matched": role_matched,
        "reason": " | ".join(reasons),
    }
