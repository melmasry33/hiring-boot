"""
Professional HR screening lens.

Deterministic, explainable review of a job posting against the candidate
profile — the same questions a recruiter asks in the first 90 seconds of a
screen: must-haves, deal-breakers, ATS keyword coverage, seniority fit, and
whether to apply / stretch / skip.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from store import load_profile

# Canonical skill / keyword lexicon used for JD extraction and ATS coverage.
# Aliases map surface forms → a single token for matching.
SKILL_LEXICON: List[Tuple[str, List[str]]] = [
    ("python", ["python", "py"]),
    ("sql", ["sql", "t-sql", "tsql", "pl/sql"]),
    ("fastapi", ["fastapi"]),
    ("django", ["django"]),
    ("flask", ["flask"]),
    ("langchain", ["langchain", "langgraph"]),
    ("langsmith", ["langsmith"]),
    ("openai", ["openai", "gpt-4", "gpt-3.5", "chatgpt"]),
    ("llm", ["llm", "llms", "large language model", "large language models"]),
    ("rag", ["rag", "retrieval-augmented", "retrieval augmented"]),
    ("agents", ["agentic", "multi-agent", "ai agent", "autonomous agent"]),
    ("pytorch", ["pytorch", "torch"]),
    ("tensorflow", ["tensorflow", "keras"]),
    ("scikit-learn", ["scikit-learn", "sklearn", "scikit learn"]),
    ("xgboost", ["xgboost", "lightgbm", "catboost"]),
    ("pandas", ["pandas"]),
    ("numpy", ["numpy"]),
    ("power bi", ["power bi", "powerbi", "dax", "power query"]),
    ("tableau", ["tableau"]),
    ("excel", ["excel", "spreadsheet"]),
    ("postgresql", ["postgresql", "postgres", "pgvector"]),
    ("mysql", ["mysql"]),
    ("mongodb", ["mongodb", "mongo"]),
    ("redis", ["redis"]),
    ("docker", ["docker", "containerization"]),
    ("kubernetes", ["kubernetes", "k8s"]),
    ("aws", ["aws", "amazon web services", "s3", "ec2", "lambda", "sagemaker"]),
    ("azure", ["azure", "microsoft azure"]),
    ("gcp", ["gcp", "google cloud", "bigquery"]),
    ("spark", ["spark", "pyspark", "apache spark"]),
    ("airflow", ["airflow", "apache airflow"]),
    ("dbt", ["dbt"]),
    ("etl", ["etl", "elt", "data pipeline", "data pipelines"]),
    ("react", ["react", "react.js", "reactjs", "next.js", "nextjs"]),
    ("typescript", ["typescript", "ts"]),
    ("javascript", ["javascript", "js", "node.js", "nodejs"]),
    ("git", ["git", "github", "gitlab"]),
    ("rest api", ["rest", "rest api", "restful", "api design"]),
    ("machine learning", ["machine learning", "ml", "mlops"]),
    ("deep learning", ["deep learning", "neural network", "cnn", "rnn", "transformer"]),
    ("nlp", ["nlp", "natural language processing", "text classification"]),
    ("computer vision", ["computer vision", "cv", "opencv", "yolo"]),
    ("statistics", ["statistics", "statistical", "hypothesis testing", "a/b testing"]),
    ("data analysis", ["data analysis", "data analytics", "analytics"]),
    ("data science", ["data science", "data scientist"]),
    ("bi", ["business intelligence", "bi developer", "reporting"]),
    ("streamlit", ["streamlit", "gradio"]),
    ("huggingface", ["hugging face", "huggingface", "transformers"]),
    ("chromadb", ["chromadb", "chroma", "vector database", "vector db", "faiss"]),
    ("ci/cd", ["ci/cd", "cicd", "github actions", "jenkins"]),
    ("agile", ["agile", "scrum", "kanban"]),
]

MUST_HAVE_MARKERS = [
    r"must[- ]have",
    r"required",
    r"requirements?",
    r"you (?:will|should) have",
    r"minimum qualifications?",
    r"mandatory",
    r"essential",
]
NICE_TO_HAVE_MARKERS = [
    r"nice[- ]to[- ]have",
    r"preferred",
    r"bonus",
    r"plus",
    r"good to have",
    r"advantageous",
]
DEALBREAKER_PATTERNS = [
    (r"\b\d+\+?\s*years?\b", "years_of_experience"),
    (r"\bsenior\b|\blead\b|\bprincipal\b|\bstaff\b", "seniority_title"),
    (r"\bsecurity clearance\b|\bcitizenship\b|\bvisa\b", "eligibility"),
    (r"\bon[- ]site only\b|\bno remote\b|\bmust relocate\b", "location"),
    (r"\bphd\b|\bmaster'?s (?:degree )?required\b", "education"),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _profile_corpus(profile: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in ("headline", "summary"):
        chunks.append(str(profile.get(key) or ""))
    for skill in profile.get("skills") or []:
        chunks.append(str(skill))
    cats = profile.get("skills_categories") or {}
    if isinstance(cats, dict):
        for values in cats.values():
            for v in values or []:
                chunks.append(str(v))
    for exp in profile.get("experience") or []:
        if isinstance(exp, dict):
            chunks.append(str(exp.get("role") or ""))
            chunks.append(str(exp.get("company") or ""))
            for b in exp.get("bullets") or []:
                chunks.append(str(b))
    for proj in profile.get("projects") or []:
        if isinstance(proj, dict):
            chunks.append(str(proj.get("name") or ""))
            chunks.append(str(proj.get("tech") or ""))
            for b in proj.get("bullets") or []:
                chunks.append(str(b))
    for cert in profile.get("certifications") or []:
        chunks.append(str(cert))
    for role in profile.get("target_roles") or []:
        chunks.append(str(role))
    return _norm(" ".join(chunks))


def _profile_skill_set(profile: Dict[str, Any]) -> Set[str]:
    corpus = _profile_corpus(profile)
    found: Set[str] = set()
    for canonical, aliases in SKILL_LEXICON:
        for alias in aliases:
            if re.search(rf"(?:\b|_){re.escape(alias)}(?:\b|_)", corpus):
                found.add(canonical)
                break
    return found


def extract_jd_skills(job_text: str) -> Dict[str, List[str]]:
    text = _norm(job_text)
    found: List[str] = []
    for canonical, aliases in SKILL_LEXICON:
        for alias in aliases:
            if re.search(rf"(?:\b|_){re.escape(alias)}(?:\b|_)", text):
                found.append(canonical)
                break
    # Prefer skills that appear near must-have language.
    must: List[str] = []
    nice: List[str] = []
    alias_map = {c: a for c, a in SKILL_LEXICON}
    for skill in found:
        # Look for skill within ~120 chars of a must/preferred marker.
        window_hit_must = False
        window_hit_nice = False
        for alias in alias_map.get(skill, [skill]):
            for m in re.finditer(rf"(?:\b|_){re.escape(alias)}(?:\b|_)", text):
                start = max(0, m.start() - 120)
                end = min(len(text), m.end() + 120)
                window = text[start:end]
                if any(re.search(p, window) for p in MUST_HAVE_MARKERS):
                    window_hit_must = True
                if any(re.search(p, window) for p in NICE_TO_HAVE_MARKERS):
                    window_hit_nice = True
        if window_hit_must:
            must.append(skill)
        elif window_hit_nice:
            nice.append(skill)
        else:
            # Default: treat as required if listed generally in the JD.
            must.append(skill)
    # Dedupe while preserving order
    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {"must_have": uniq(must), "nice_to_have": uniq(nice), "all": uniq(found)}


def _seniority_signal(job_text: str) -> Dict[str, Any]:
    text = _norm(job_text)
    junior = any(
        re.search(rf"\b{re.escape(k)}\b", text)
        for k in (
            "junior", "entry level", "entry-level", "fresh graduate", "intern",
            "internship", "associate", "graduate", "0-1 year", "0-2 years",
            "1-2 years", "1-3 years",
        )
    )
    mid = any(
        re.search(rf"\b{re.escape(k)}\b", text)
        for k in ("mid-level", "mid level", "intermediate", "2-4 years", "3-5 years", "2+ years", "3+ years")
    )
    senior = any(
        re.search(rf"\b{re.escape(k)}\b", text)
        for k in (
            "senior", "sr.", "lead", "principal", "staff engineer", "head of",
            "director", "5+ years", "6+ years", "7+ years", "8+ years", "10+ years",
        )
    )
    years = re.findall(r"(\d+)\+?\s*years?", text)
    years_req = max((int(y) for y in years), default=0)
    level = "unknown"
    if senior or years_req >= 5:
        level = "senior"
    elif mid or (2 <= years_req <= 4):
        level = "mid"
    elif junior or years_req <= 1:
        level = "junior"
    return {"level": level, "years_mentioned": years_req, "flags": {
        "junior_friendly": junior,
        "mid_signal": mid,
        "senior_heavy": senior,
    }}


def _role_alignment(job_text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    text = _norm(job_text)
    targets = [str(r) for r in (profile.get("target_roles") or [])]
    matched = []
    for role in targets:
        r = _norm(role)
        if not r:
            continue
        # Partial token overlap (e.g. "AI Engineer" vs "AI/ML Engineer")
        tokens = [t for t in re.split(r"[\s/|-]+", r) if len(t) > 1]
        if tokens and all(re.search(rf"\b{re.escape(t)}\b", text) for t in tokens if t not in {"and", "or"}):
            matched.append(role)
        elif re.search(rf"\b{re.escape(r)}\b", text):
            matched.append(role)
    return {"matched_roles": matched, "target_roles": targets}


def _dealbreakers(job_text: str, profile: Dict[str, Any], seniority: Dict[str, Any]) -> List[Dict[str, str]]:
    text = _norm(job_text)
    findings: List[Dict[str, str]] = []
    for pattern, kind in DEALBREAKER_PATTERNS:
        if re.search(pattern, text):
            if kind == "seniority_title" and seniority.get("level") == "senior":
                findings.append({
                    "kind": kind,
                    "detail": "Posting signals senior/lead seniority — expect scrutiny on years and ownership depth.",
                })
            elif kind == "years_of_experience" and seniority.get("years_mentioned", 0) >= 5:
                findings.append({
                    "kind": kind,
                    "detail": f"Posting asks for ~{seniority['years_mentioned']}+ years — treat as a stretch unless evidence is strong.",
                })
            elif kind == "eligibility":
                findings.append({
                    "kind": kind,
                    "detail": "Citizenship, visa, or clearance language detected — confirm eligibility before applying.",
                })
            elif kind == "location":
                findings.append({
                    "kind": kind,
                    "detail": "On-site / relocation constraint detected — confirm location fit.",
                })
            elif kind == "education":
                findings.append({
                    "kind": kind,
                    "detail": "Advanced degree called out as required — note in gap_notes if not held.",
                })
    return findings


def hr_screen(job_text: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return a recruiter-grade screening packet for one job vs one profile.
    """
    if profile is None:
        profile = load_profile()
    if not profile:
        return {"ok": False, "error": "No profile on file — collect one before screening."}
    if not (job_text or "").strip():
        return {"ok": False, "error": "Empty job text — ingest or read a posting first."}

    jd_skills = extract_jd_skills(job_text)
    have = _profile_skill_set(profile)
    must = jd_skills["must_have"]
    nice = jd_skills["nice_to_have"]

    covered_must = [s for s in must if s in have]
    missing_must = [s for s in must if s not in have]
    covered_nice = [s for s in nice if s in have]
    missing_nice = [s for s in nice if s not in have]

    seniority = _seniority_signal(job_text)
    roles = _role_alignment(job_text, profile)
    dealbreakers = _dealbreakers(job_text, profile, seniority)

    must_cov = len(covered_must) / max(len(must), 1)
    nice_cov = len(covered_nice) / max(len(nice), 1) if nice else 0.5

    # Recruiter score: must-haves dominate; seniority and role tilt the rest.
    score = must_cov * 55 + nice_cov * 10
    if roles["matched_roles"]:
        score += 15
    else:
        score += 5
    level = seniority["level"]
    if level == "junior":
        score += 12
    elif level == "mid":
        score += 10
    elif level == "senior":
        score -= 12
    else:
        score += 6
    if dealbreakers:
        score -= min(15, 5 * len(dealbreakers))
    score = int(round(max(0.0, min(100.0, score))))

    if score >= 72 and len(missing_must) <= 1 and level != "senior":
        verdict = "apply"
        verdict_label = "Strong apply"
    elif score >= 55 and len(missing_must) <= 3:
        verdict = "stretch"
        verdict_label = "Stretch — apply with honest gaps"
    elif score >= 40:
        verdict = "weak"
        verdict_label = "Weak fit — only apply if strategic"
    else:
        verdict = "skip"
        verdict_label = "Skip — low probability of interview"

    strengths = covered_must[:6] or list(have)[:4]
    lead_with = []
    for exp in (profile.get("experience") or [])[:2]:
        if isinstance(exp, dict) and exp.get("role"):
            lead_with.append(f"{exp.get('role')} @ {exp.get('company', '')}".strip(" @"))
    for proj in (profile.get("projects") or [])[:3]:
        if isinstance(proj, dict) and proj.get("name"):
            lead_with.append(str(proj["name"]))

    ats_keywords = covered_must + [s for s in covered_nice if s not in covered_must]
    rewrite_priorities = []
    if missing_must:
        rewrite_priorities.append(
            "Mirror must-have keywords ONLY where true experience exists; never invent."
        )
    if level == "senior":
        rewrite_priorities.append(
            "Lead with ownership, production impact, and measurable outcomes — not coursework."
        )
    rewrite_priorities.append(
        "Put the 2–3 strongest matching projects first; rewrite bullets to echo JD language."
    )
    rewrite_priorities.append(
        "Open the summary with the target role + 1 proof point that maps to a must-have."
    )

    interview_risks = []
    for skill in missing_must[:5]:
        interview_risks.append(f"Expect grilling on missing must-have: {skill}")
    if level == "senior":
        interview_risks.append("Senior bar: be ready with depth, tradeoffs, and production war stories.")
    if not roles["matched_roles"]:
        interview_risks.append("Title mismatch — clarify target role family early in the email.")

    return {
        "ok": True,
        "score": score,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "must_have_skills": must,
        "nice_to_have_skills": nice,
        "covered_must_haves": covered_must,
        "missing_must_haves": missing_must,
        "covered_nice_to_haves": covered_nice,
        "missing_nice_to_haves": missing_nice,
        "ats_keywords_to_echo": ats_keywords[:12],
        "seniority": seniority,
        "role_alignment": roles,
        "dealbreakers": dealbreakers,
        "strengths_to_lead_with": strengths,
        "evidence_to_feature": lead_with[:5],
        "rewrite_priorities": rewrite_priorities,
        "interview_risks": interview_risks,
        "recruiter_summary": (
            f"{verdict_label} ({score}/100). "
            f"Must-have coverage {len(covered_must)}/{len(must) or 0}. "
            f"Seniority signal: {level}. "
            + (f"Gaps: {', '.join(missing_must[:5])}." if missing_must else "No critical skill gaps detected.")
        ),
    }


def format_hr_review_for_chat(review: Dict[str, Any]) -> str:
    """Compact Telegram-friendly rendering of an HR screen."""
    if not review.get("ok"):
        return review.get("error") or "HR review failed."
    lines = [
        f"HR screen — {review['verdict_label']} ({review['score']}/100)",
        "",
        review.get("recruiter_summary", ""),
        "",
        "Lead with:",
    ]
    for s in review.get("strengths_to_lead_with") or []:
        lines.append(f"  • {s}")
    if review.get("missing_must_haves"):
        lines.append("")
        lines.append("Must-have gaps:")
        for s in review["missing_must_haves"]:
            lines.append(f"  • {s}")
    if review.get("dealbreakers"):
        lines.append("")
        lines.append("Flags:")
        for d in review["dealbreakers"]:
            lines.append(f"  • {d.get('detail')}")
    if review.get("rewrite_priorities"):
        lines.append("")
        lines.append("CV rewrite priorities:")
        for p in review["rewrite_priorities"]:
            lines.append(f"  • {p}")
    return "\n".join(lines).strip()
