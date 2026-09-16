import re
from typing import Any, Dict, List, Optional
from store import load_profile


def score_job(job_description: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Deterministic scoring of a job description against the local profile.
    
    Evaluates:
    - Skill keyword matches (from profile['skills'])
    - Target roles alignment
    - Experience level (rewards entry/junior/0-2 yrs, penalizes senior/lead/5+ yrs)
    - Education/Projects alignment
    
    Returns structured JSON with score (0-100), strong_matches, missing_skills, and reason.
    """
    if profile is None:
        profile = load_profile()

    skills: List[str] = profile.get("skills", [])
    target_roles: List[str] = profile.get("target_roles", [])
    projects: List[str] = profile.get("projects", [])

    desc_lower = job_description.lower()

    # 1. Match Skills
    strong_matches = []
    missing_skills = []

    for skill in skills:
        # Match word boundaries or exact tokens
        # Special cases like C++, C#, .NET, pgvector, etc.
        escaped_skill = re.escape(skill.lower())
        pattern = rf"(?:\b|_){escaped_skill}(?:\b|_)"
        if re.search(pattern, desc_lower):
            strong_matches.append(skill)
        else:
            missing_skills.append(skill)

    # 2. Check Target Roles
    role_matched = []
    for role in target_roles:
        escaped_role = re.escape(role.lower())
        if re.search(rf"\b{escaped_role}\b", desc_lower):
            role_matched.append(role)

    # 3. Check Seniority / Experience Level
    junior_keywords = [
        "junior", "entry level", "entry-level", "fresh graduate", "fresh grad",
        "0-1 year", "0-2 years", "1-2 years", "0 - 2 years", "1 - 3 years",
        "intern", "internship", "associate", "graduate"
    ]
    senior_keywords = [
        "senior", "sr.", "lead", "principal", "staff engineer", "head of",
        "director", "5+ years", "6+ years", "7+ years", "8+ years", "10+ years"
    ]

    is_junior_friendly = any(re.search(rf"\b{re.escape(kw)}\b", desc_lower) for kw in junior_keywords)
    is_senior_heavy = any(re.search(rf"\b{re.escape(kw)}\b", desc_lower) for kw in senior_keywords)

    # 4. Check Project / Domain mentions (e.g. RAG, Vector, Computer Vision, BI, LLM)
    domain_keywords = ["rag", "llm", "llms", "vector", "ai", "machine learning", "fastapi", "power bi", "bi", "sql"]
    domain_matches = [kw for kw in domain_keywords if re.search(rf"\b{re.escape(kw)}\b", desc_lower)]

    # Compute Score (0 - 100)
    # Skill weight: 50% max
    # Seniority suitability: 25% max
    # Role alignment: 15% max
    # Domain relevance: 10% max

    skill_score = (len(strong_matches) / max(len(skills), 1)) * 50
    
    # Cap skill score at 50, but give generous curve if 4+ core skills match
    if len(strong_matches) >= 5:
        skill_score = min(50.0, skill_score + 15.0)

    # Seniority adjustment
    seniority_score = 15.0  # neutral starting point
    if is_junior_friendly:
        seniority_score += 10.0  # max 25
    elif is_senior_heavy:
        seniority_score -= 15.0  # heavy penalty for senior roles

    # Role score
    role_score = 15.0 if role_matched else 5.0

    # Domain score
    domain_score = min(10.0, len(domain_matches) * 2.5)

    final_score = int(round(max(0.0, min(100.0, skill_score + seniority_score + role_score + domain_score))))

    # Construct Explanation / Reason
    reasons = []
    if strong_matches:
        reasons.append(f"Matched {len(strong_matches)} skills: {', '.join(strong_matches[:5])}")
    if role_matched:
        reasons.append(f"Matches target role ({', '.join(role_matched)})")
    if is_junior_friendly:
        reasons.append("Junior/entry-level friendly experience requirements")
    if is_senior_heavy:
        reasons.append("Warning: Job mentions senior/lead/5+ years requirements")
    if not reasons:
        reasons.append("General match based on keywords")

    return {
        "score": final_score,
        "strong_matches": strong_matches,
        "missing_skills": missing_skills,
        "reason": " | ".join(reasons)
    }
