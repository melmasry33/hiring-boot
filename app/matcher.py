"""Single job-analysis / matching layer.

HR screening, numeric scoring, and tailoring all consume this result.
Exact skill matches are distinct from related or transferable evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from candidate import variant_corpus

MatchKind = str  # exact | related | transferable | not_equivalent
ConstraintStrength = str  # required | preferred | unclear


@dataclass
class SkillDef:
    canonical: str
    exact_aliases: Tuple[str, ...]
    related: Tuple[str, ...] = ()
    family: str = "general"


# Distinct technologies stay distinct. related[] is supporting evidence only.
SKILLS: List[SkillDef] = [
    SkillDef("python", ("python",), family="language"),
    SkillDef("sql", ("sql", "t-sql", "tsql", "pl/sql"), family="data"),
    SkillDef("fastapi", ("fastapi",), related=("flask", "django"), family="python_web"),
    SkillDef("django", ("django",), related=("flask", "fastapi"), family="python_web"),
    SkillDef("flask", ("flask",), related=("django", "fastapi"), family="python_web"),
    SkillDef("langchain", ("langchain",), related=("langgraph",), family="llm_orchestration"),
    SkillDef("langgraph", ("langgraph",), related=("langchain",), family="llm_orchestration"),
    SkillDef("langsmith", ("langsmith",), family="llm_orchestration"),
    SkillDef("openai", ("openai", "gpt-4", "gpt-3.5", "chatgpt"), family="llm_vendor"),
    SkillDef("llm", ("llm", "llms", "large language model", "large language models"), family="ai"),
    SkillDef("rag", ("rag", "retrieval-augmented", "retrieval augmented"), family="ai"),
    SkillDef("agents", ("agentic", "multi-agent", "ai agent", "autonomous agent"), family="ai"),
    SkillDef("pytorch", ("pytorch", "torch"), related=("tensorflow",), family="dl_framework"),
    SkillDef("tensorflow", ("tensorflow", "keras"), related=("pytorch",), family="dl_framework"),
    SkillDef("scikit-learn", ("scikit-learn", "sklearn", "scikit learn"), family="ml"),
    SkillDef("xgboost", ("xgboost",), related=("lightgbm", "catboost"), family="gbdt"),
    SkillDef("lightgbm", ("lightgbm",), related=("xgboost", "catboost"), family="gbdt"),
    SkillDef("catboost", ("catboost",), related=("xgboost", "lightgbm"), family="gbdt"),
    SkillDef("pandas", ("pandas",), family="data"),
    SkillDef("numpy", ("numpy",), family="data"),
    SkillDef("power bi", ("power bi", "powerbi"), related=("dax", "power query"), family="bi"),
    SkillDef("dax", ("dax",), related=("power bi",), family="bi"),
    SkillDef("power query", ("power query", "powerquery"), related=("power bi",), family="bi"),
    SkillDef("tableau", ("tableau",), related=("power bi",), family="bi"),
    SkillDef("excel", ("excel", "spreadsheet"), family="bi"),
    SkillDef("postgresql", ("postgresql", "postgres"), related=("pgvector",), family="db"),
    SkillDef("pgvector", ("pgvector",), related=("postgresql",), family="db"),
    SkillDef("mysql", ("mysql",), family="db"),
    SkillDef("mongodb", ("mongodb", "mongo"), family="db"),
    SkillDef("redis", ("redis",), family="infra"),
    SkillDef("docker", ("docker",), related=("kubernetes",), family="infra"),
    SkillDef("kubernetes", ("kubernetes", "k8s"), related=("docker",), family="infra"),
    SkillDef("aws", ("aws", "amazon web services"), related=("s3", "ec2", "lambda", "sagemaker"), family="aws"),
    SkillDef("s3", ("s3", "amazon s3"), related=("aws",), family="aws"),
    SkillDef("ec2", ("ec2", "amazon ec2"), related=("aws",), family="aws"),
    SkillDef("lambda", ("aws lambda", "amazon lambda"), related=("aws",), family="aws"),
    SkillDef("sagemaker", ("sagemaker", "amazon sagemaker"), related=("aws",), family="aws"),
    SkillDef("azure", ("azure", "microsoft azure"), family="cloud"),
    SkillDef("gcp", ("gcp", "google cloud"), related=("bigquery",), family="cloud"),
    SkillDef("bigquery", ("bigquery",), related=("gcp",), family="cloud"),
    SkillDef("spark", ("spark", "pyspark", "apache spark"), family="data_eng"),
    SkillDef("airflow", ("airflow", "apache airflow"), family="data_eng"),
    SkillDef("dbt", ("dbt",), family="data_eng"),
    SkillDef("etl", ("etl", "elt", "data pipeline", "data pipelines"), family="data_eng"),
    SkillDef("react", ("react", "react.js", "reactjs"), related=("next.js",), family="frontend"),
    SkillDef("next.js", ("next.js", "nextjs"), related=("react",), family="frontend"),
    SkillDef("typescript", ("typescript",), related=("javascript",), family="language"),
    SkillDef("javascript", ("javascript", "node.js", "nodejs"), related=("typescript",), family="language"),
    SkillDef("git", ("git",), related=("github", "gitlab"), family="tools"),
    SkillDef("github", ("github",), related=("git",), family="tools"),
    SkillDef("rest api", ("rest api", "restful", "api design"), family="backend"),
    SkillDef("machine learning", ("machine learning", "mlops"), family="ml"),
    SkillDef("deep learning", ("deep learning", "neural network"), family="ml"),
    SkillDef("nlp", ("nlp", "natural language processing"), family="ml"),
    SkillDef("computer vision", ("computer vision", "opencv", "yolo"), family="ml"),
    SkillDef("statistics", ("statistics", "hypothesis testing", "a/b testing"), family="analytics"),
    SkillDef("data analysis", ("data analysis", "data analytics"), family="analytics"),
    SkillDef("data science", ("data science",), family="analytics"),
    SkillDef("streamlit", ("streamlit",), related=("gradio",), family="app"),
    SkillDef("huggingface", ("hugging face", "huggingface"), family="llm_vendor"),
    SkillDef("chromadb", ("chromadb", "chroma"), related=("faiss", "vector database"), family="db"),
    SkillDef("ci/cd", ("ci/cd", "cicd", "github actions", "jenkins"), family="tools"),
    SkillDef("agile", ("agile", "scrum", "kanban"), family="process"),
]

SKILL_BY_CANONICAL = {s.canonical: s for s in SKILLS}

ROLE_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "ai_engineer": (
        "ai engineer",
        "llm engineer",
        "agentic ai engineer",
        "genai engineer",
        "generative ai engineer",
        "ai/ml engineer",
        "ai ml engineer",
    ),
    "ml_engineer": (
        "machine learning engineer",
        "ml engineer",
        "applied ml engineer",
        "applied machine learning engineer",
        "applied scientist",
    ),
    "data_scientist": (
        "data scientist",
        "junior data scientist",
        "applied scientist",
        "research scientist",
    ),
    "data_analyst": (
        "data analyst",
        "business analyst",
        "product analyst",
        "operations analyst",
        "supply chain analyst",
        "product analytics",
        "operations analytics",
    ),
    "bi": (
        "bi developer",
        "power bi developer",
        "business intelligence developer",
        "bi analyst",
        "reporting developer",
        "business intelligence",
    ),
    "data_engineer": (
        "data engineer",
        "analytics engineer",
        "etl developer",
    ),
}

CLOSE_FAMILIES = {
    frozenset({"ai_engineer", "ml_engineer"}),
    frozenset({"ml_engineer", "data_scientist"}),
    frozenset({"ai_engineer", "data_scientist"}),
    frozenset({"data_analyst", "bi"}),
}
ADJACENT_FAMILIES = {
    frozenset({"data_scientist", "data_analyst"}),
    frozenset({"data_scientist", "data_engineer"}),
    frozenset({"ml_engineer", "data_engineer"}),
    frozenset({"data_analyst", "data_engineer"}),
    frozenset({"bi", "data_engineer"}),
}

REQUIRED_MARKERS = (
    r"must[- ]have",
    r"\brequired\b",
    r"\bmandatory\b",
    r"\bessential\b",
    r"minimum qualifications?",
    r"you (?:will|should|must) have",
)
PREFERRED_MARKERS = (
    r"nice[- ]to[- ]have",
    r"\bpreferred\b",
    r"\bbonus\b",
    r"good to have",
    r"\badvantageous\b",
    r"plus if",
)
REQUIRED_HEADERS = (
    "requirements",
    "qualifications",
    "must have",
    "minimum qualifications",
    "what you'll need",
    "what you will need",
    "you have",
)
PREFERRED_HEADERS = (
    "nice to have",
    "preferred",
    "bonus",
    "plus",
    "good to have",
)
CONTEXT_HEADERS = (
    "responsibilities",
    "about the role",
    "about us",
    "what you'll do",
    "what you will do",
    "benefits",
    "we offer",
)

YEARS_CANDIDATE_CUES = (
    "experience",
    "exp",
    "required",
    "minimum",
    "at least",
    "you have",
    "candidate",
    "background in",
    "hands-on",
    "working with",
    "using",
    "in python",
    "in sql",
    "with python",
    "with sql",
    "software",
    "engineering",
    "development",
)
YEARS_IRRELEVANT_CUES = (
    "founded",
    "company has",
    "we've been",
    "we have been",
    "our team",
    "customers",
    "leadership",
    "managing a team",
    "market",
    "industry for",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _word_hit(alias: str, text: str) -> bool:
    return re.search(rf"(?:\b|_){re.escape(alias)}(?:\b|_)", text) is not None


def _skill_hits(text: str) -> List[str]:
    found: List[str] = []
    seen = set()
    for skill in SKILLS:
        if any(_word_hit(alias, text) for alias in skill.exact_aliases):
            if skill.canonical not in seen:
                seen.add(skill.canonical)
                found.append(skill.canonical)
    return found


def _split_sections(job_text: str) -> List[Tuple[str, str]]:
    """Return (section_kind, text) chunks using common JD headers."""
    lines = (job_text or "").splitlines()
    sections: List[Tuple[str, List[str]]] = [("body", [])]
    for line in lines:
        n = _norm(line).strip(" :")
        kind = None
        if any(n.startswith(h) or n == h for h in REQUIRED_HEADERS):
            kind = "required"
        elif any(n.startswith(h) or n == h for h in PREFERRED_HEADERS):
            kind = "preferred"
        elif any(n.startswith(h) or n == h for h in CONTEXT_HEADERS):
            kind = "context"
        if kind:
            sections.append((kind, [line]))
        else:
            sections[-1][1].append(line)
    return [(k, "\n".join(v)) for k, v in sections if "".join(v).strip()]


def extract_jd_skills(job_text: str) -> Dict[str, List[str]]:
    sections = _split_sections(job_text)
    required: List[str] = []
    preferred: List[str] = []
    contextual: List[str] = []
    unclear: List[str] = []

    def add(bucket: List[str], skills: List[str]) -> None:
        for s in skills:
            if s not in bucket:
                bucket.append(s)

    if len(sections) == 1:
        text = _norm(job_text)
        for skill in _skill_hits(text):
            window_required = False
            window_preferred = False
            aliases = SKILL_BY_CANONICAL[skill].exact_aliases
            for alias in aliases:
                for m in re.finditer(rf"(?:\b|_){re.escape(alias)}(?:\b|_)", text):
                    window = text[max(0, m.start() - 140) : min(len(text), m.end() + 140)]
                    if any(re.search(p, window) for p in REQUIRED_MARKERS):
                        window_required = True
                    if any(re.search(p, window) for p in PREFERRED_MARKERS):
                        window_preferred = True
            if window_required and not window_preferred:
                add(required, [skill])
            elif window_preferred:
                add(preferred, [skill])
            else:
                add(unclear, [skill])
    else:
        for kind, body in sections:
            hits = _skill_hits(_norm(body))
            if kind == "required":
                add(required, hits)
            elif kind == "preferred":
                add(preferred, hits)
            elif kind == "context":
                add(contextual, hits)
            else:
                add(unclear, hits)

    all_skills: List[str] = []
    for seq in (required, preferred, unclear, contextual):
        for s in seq:
            if s not in all_skills:
                all_skills.append(s)
    return {
        "must_have": required,
        "nice_to_have": preferred,
        "unclear": unclear,
        "contextual": contextual,
        "all": all_skills,
    }


def candidate_skill_set(variant: Dict[str, Any]) -> Dict[str, str]:
    """Map canonical skill -> match kind present in the variant corpus."""
    corpus = variant_corpus(variant)
    present_exact = set(_skill_hits(corpus))
    out: Dict[str, str] = {}
    for skill in SKILLS:
        if skill.canonical in present_exact:
            out[skill.canonical] = "exact"
            continue
        if any(rel in present_exact for rel in skill.related):
            # Supporting evidence only — not demonstrated direct experience.
            out[skill.canonical] = "related"
            continue
        out[skill.canonical] = "not_equivalent"
    return out


def classify_skill_match(jd_skill: str, candidate_map: Dict[str, str]) -> str:
    return candidate_map.get(jd_skill, "not_equivalent")


def analyze_seniority(job_text: str) -> Dict[str, Any]:
    text = _norm(job_text)
    explicit = {
        "junior": any(_word_hit(k, text) for k in ("junior", "jr")),
        "entry_level": any(k in text for k in ("entry level", "entry-level", "intern", "internship")),
        "graduate": any(k in text for k in ("fresh graduate", "graduate scheme", "new graduate")),
        "mid": any(k in text for k in ("mid-level", "mid level", "intermediate")),
        "senior": any(_word_hit(k, text) for k in ("senior", "sr")),
        "lead_staff_principal": any(
            k in text for k in ("principal", "staff engineer", "head of", "director")
        )
        or _word_hit("lead", text),
    }

    years_req: Optional[int] = None
    years_evidence: List[str] = []
    for m in re.finditer(r"(\d+)\+?\s*years?", text):
        window = text[max(0, m.start() - 80) : min(len(text), m.end() + 80)]
        if any(cue in window for cue in YEARS_IRRELEVANT_CUES) and not any(
            cue in window for cue in ("years of experience", "years experience", "years' experience")
        ):
            continue
        if not any(cue in window for cue in YEARS_CANDIDATE_CUES):
            continue
        n = int(m.group(1))
        years_evidence.append(window.strip())
        years_req = n if years_req is None else max(years_req, n)

    if explicit["lead_staff_principal"] or explicit["senior"]:
        level = "lead" if explicit["lead_staff_principal"] else "senior"
        source = "explicit_title"
        confidence = "high"
    elif explicit["mid"]:
        level = "mid"
        source = "explicit_title"
        confidence = "high"
    elif explicit["junior"] or explicit["entry_level"] or explicit["graduate"]:
        if explicit["entry_level"]:
            level = "entry_level"
        elif explicit["graduate"]:
            level = "graduate"
        else:
            level = "junior"
        source = "explicit_title"
        confidence = "high"
    elif years_req is not None:
        source = "years_requirement"
        if years_req >= 5:
            level = "senior"
        elif years_req >= 2:
            level = "mid"
        else:
            level = "junior"
        confidence = "medium"
    else:
        level = "unspecified"
        source = "unspecified"
        confidence = "low"

    return {
        "level": level,
        "years_requirement": years_req,
        "years_evidence": years_evidence[:3],
        "explicit": explicit,
        "source": source,
        "confidence": confidence,
        "flags": {
            "junior_friendly": level in {"junior", "entry_level", "graduate"},
            "mid_signal": level == "mid",
            "senior_heavy": level in {"senior", "lead"},
            "unspecified": level == "unspecified",
        },
    }


def _families_for_text(text: str) -> List[str]:
    n = _norm(text)
    found: List[str] = []
    for fam, aliases in ROLE_FAMILIES.items():
        if any(alias in n for alias in aliases):
            found.append(fam)
    return found


def analyze_role_alignment(job_text: str, variant: Dict[str, Any], role_hint: str = "") -> Dict[str, Any]:
    hay = f"{role_hint} {job_text}"
    jd_families = _families_for_text(hay)
    variant_roles = [str(r) for r in (variant.get("roles") or [])]
    variant_families: List[str] = []
    for role in variant_roles:
        variant_families.extend(_families_for_text(role))
    variant_families = list(dict.fromkeys(variant_families))

    relation = "unrelated"
    matched_roles: List[str] = []
    if not jd_families:
        relation = "unspecified"
    else:
        for role in variant_roles:
            rn = _norm(role)
            if rn and rn in _norm(hay):
                matched_roles.append(role)
        if matched_roles or set(jd_families) & set(variant_families):
            exact_title = bool(matched_roles)
            same_family = bool(set(jd_families) & set(variant_families))
            if exact_title and same_family:
                relation = "exact"
            elif same_family:
                relation = "close_family"
            else:
                relation = "adjacent"
        else:
            pairs = {frozenset({a, b}) for a in jd_families for b in variant_families}
            if pairs & CLOSE_FAMILIES:
                relation = "close_family"
            elif pairs & ADJACENT_FAMILIES:
                relation = "adjacent"
            elif variant_families:
                relation = "weak"
            else:
                relation = "unrelated"

    return {
        "relation": relation,
        "jd_families": jd_families,
        "variant_families": variant_families,
        "matched_roles": matched_roles,
        "target_roles": variant_roles,
    }


def _constraint_strength(window: str) -> ConstraintStrength:
    if any(re.search(p, window) for p in PREFERRED_MARKERS):
        return "preferred"
    if any(re.search(p, window) for p in REQUIRED_MARKERS):
        return "required"
    return "unclear"


def analyze_constraints(job_text: str) -> List[Dict[str, str]]:
    text = _norm(job_text)
    specs = [
        ("eligibility", r"\bus citizenship\b|\bcitizenship required\b|\bsecurity clearance\b|\bvisa\b"),
        ("location", r"\bon[- ]site only\b|\bno remote\b|\bmust relocate\b|\brelocation required\b"),
        ("education", r"\bphd\b|\bph\.d\b|\bmaster'?s\b|\bmasters degree\b"),
    ]
    findings: List[Dict[str, str]] = []
    for kind, pattern in specs:
        for m in re.finditer(pattern, text):
            window = text[max(0, m.start() - 90) : min(len(text), m.end() + 90)]
            findings.append(
                {
                    "kind": kind,
                    "span": m.group(0),
                    "strength": _constraint_strength(window),
                    "window": window.strip()[:180],
                }
            )
            break
    return findings


def analyze_job(
    job_text: str,
    variant: Dict[str, Any],
    role_hint: str = "",
    company_hint: str = "",
) -> Dict[str, Any]:
    jd_skills = extract_jd_skills(job_text)
    cand_map = candidate_skill_set(variant)
    required = jd_skills["must_have"]
    preferred = jd_skills["nice_to_have"]
    unclear = jd_skills["unclear"]

    def partition(skills: List[str]) -> Dict[str, List[str]]:
        buckets = {"exact": [], "related": [], "transferable": [], "not_equivalent": []}
        for s in skills:
            buckets[classify_skill_match(s, cand_map)].append(s)
        return buckets

    req_p = partition(required)
    pref_p = partition(preferred)
    unclear_p = partition(unclear)

    seniority = analyze_seniority(job_text)
    roles = analyze_role_alignment(job_text, variant, role_hint=role_hint)
    constraints = analyze_constraints(job_text)

    exact_req = req_p["exact"]
    missing_req = req_p["not_equivalent"]
    related_req = req_p["related"] + req_p["transferable"]

    must_cov = len(exact_req) / max(len(required), 1) if required else 0.55
    nice_cov = len(pref_p["exact"]) / max(len(preferred), 1) if preferred else 0.5
    score = must_cov * 55 + nice_cov * 10
    relation = roles["relation"]
    score += {"exact": 15, "close_family": 12, "adjacent": 8, "weak": 4, "unspecified": 8, "unrelated": 2}.get(relation, 5)
    level = seniority["level"]
    if level in {"junior", "entry_level", "graduate", "mid", "unspecified"}:
        score += 10 if level != "unspecified" else 8
    elif level in {"senior", "lead"}:
        score -= 8
    required_constraints = [c for c in constraints if c["strength"] == "required"]
    score -= min(12, 4 * len(required_constraints))
    score = int(round(max(0.0, min(100.0, score))))

    if score >= 72 and len(missing_req) <= 1 and level not in {"senior", "lead"}:
        verdict, verdict_label = "apply", "Strong apply"
    elif score >= 55 and len(missing_req) <= 3:
        verdict, verdict_label = "stretch", "Stretch — apply with honest gaps"
    elif score >= 40:
        verdict, verdict_label = "weak", "Weak fit — only apply if strategic"
    else:
        verdict, verdict_label = "skip", "Skip — low probability of interview"

    evidence = []
    for exp in (variant.get("experience") or [])[:4]:
        evidence.append({"type": "experience", "id": exp.get("id"), "label": f"{exp.get('role')} @ {exp.get('company')}"})
    for proj in (variant.get("projects") or [])[:6]:
        evidence.append({"type": "project", "id": proj.get("id"), "label": proj.get("name")})

    warnings: List[str] = []
    if seniority["source"] == "unspecified":
        warnings.append("No explicit seniority requirement — not assumed junior.")
    if related_req:
        warnings.append(
            "Related/transferable skills are supporting evidence only, not claimed as direct experience: "
            + ", ".join(related_req[:6])
        )
    for c in constraints:
        if c["strength"] == "required":
            warnings.append(f"Required constraint ({c['kind']}): {c['span']}")
        elif c["strength"] == "preferred":
            warnings.append(f"Preferred (not mandatory) constraint ({c['kind']}): {c['span']}")

    return {
        "ok": True,
        "score": score,
        "confidence": seniority["confidence"] if required else "medium",
        "verdict": verdict,
        "verdict_label": verdict_label,
        "role": (role_hint or "").strip(),
        "company": (company_hint or "").strip(),
        "seniority": seniority,
        "role_alignment": roles,
        "required_skills": required,
        "preferred_skills": preferred,
        "unclear_skills": unclear,
        "contextual_skills": jd_skills["contextual"],
        "candidate_skill_matches": {
            "exact_required": exact_req,
            "related_required": req_p["related"],
            "transferable_required": req_p["transferable"],
            "missing_required": missing_req,
            "exact_preferred": pref_p["exact"],
            "related_preferred": pref_p["related"],
        },
        "missing_skills": missing_req,
        "constraints": constraints,
        "warnings": warnings,
        "evidence": evidence,
        "recruiter_summary": (
            f"{verdict_label} ({score}/100). "
            f"Exact must-have coverage {len(exact_req)}/{len(required) or 0}. "
            f"Seniority: {level} ({seniority['source']}). "
            f"Role alignment: {relation}."
        ),
        "selected_variant": variant.get("key"),
    }


def build_tailoring_brief(analysis: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    matches = analysis.get("candidate_skill_matches") or {}
    return {
        "target_role": analysis.get("role") or "",
        "company": analysis.get("company") or "",
        "selected_variant": variant.get("key"),
        "required_skills": analysis.get("required_skills") or [],
        "matched_skills": matches.get("exact_required") or [],
        "related_skills": (matches.get("related_required") or []) + (matches.get("transferable_required") or []),
        "missing_skills": analysis.get("missing_skills") or [],
        "strongest_evidence": [e.get("label") for e in (analysis.get("evidence") or [])[:5]],
        "relevant_projects": [
            {"id": p.get("id"), "name": p.get("name")} for p in (variant.get("projects") or [])
        ],
        "relevant_experience": [
            {"id": e.get("id"), "role": e.get("role"), "company": e.get("company")}
            for e in (variant.get("experience") or [])
        ],
        "relevant_certifications": [
            {"id": c.get("id"), "text": c.get("text")} for c in (variant.get("certification_records") or [])
        ],
        "seniority_alignment": analysis.get("seniority") or {},
        "role_alignment": analysis.get("role_alignment") or {},
        "constraints": analysis.get("constraints") or [],
        "risks": analysis.get("warnings") or [],
        "tailoring_priorities": [
            "Lead with exact matched must-haves; never claim related skills as direct experience.",
            "Reorder canonical projects and experience by relevance; do not drop baseline entries.",
            "Rewrite existing bullets only; never invent metrics, employers, or tools.",
            "Put missing must-haves into gap_notes instead of fabricating coverage.",
        ],
        "score": analysis.get("score"),
        "verdict": analysis.get("verdict"),
        "canonical_candidate": {
            "experience_ids": [e.get("id") for e in variant.get("experience") or []],
            "project_ids": [p.get("id") for p in variant.get("projects") or []],
            "certification_ids": [c.get("id") for c in variant.get("certification_records") or []],
            "skill_categories": list((variant.get("skills_categories") or {}).keys()),
        },
    }


def format_hr_review_for_chat(analysis: Dict[str, Any]) -> str:
    if not analysis.get("ok"):
        return analysis.get("error") or "HR review failed."
    lines = [
        f"HR screen — {analysis.get('verdict_label')} ({analysis.get('score')}/100)",
        "",
        analysis.get("recruiter_summary") or "",
        f"Variant: {analysis.get('selected_variant')}",
        "",
        "Exact must-have matches:",
    ]
    exact = (analysis.get("candidate_skill_matches") or {}).get("exact_required") or []
    related = (analysis.get("candidate_skill_matches") or {}).get("related_required") or []
    missing = analysis.get("missing_skills") or []
    for s in exact or ["(none)"]:
        lines.append(f"  • {s}")
    if related:
        lines.append("")
        lines.append("Related (not claimed as direct):")
        for s in related:
            lines.append(f"  • {s}")
    if missing:
        lines.append("")
        lines.append("Must-have gaps:")
        for s in missing:
            lines.append(f"  • {s}")
    constraints = analysis.get("constraints") or []
    if constraints:
        lines.append("")
        lines.append("Constraints:")
        for c in constraints:
            lines.append(f"  • [{c.get('strength')}] {c.get('kind')}: {c.get('span')}")
    return "\n".join(lines).strip()
