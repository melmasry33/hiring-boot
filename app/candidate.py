"""Canonical candidate access.

Career facts come only from data/cv_variants.json. profile.json is identity
metadata (name, contact, links) — never a competing CV source.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from config import CV_VARIANTS_PATH, logger

VARIANT_KEYS = ("ai", "bi", "data_analyst", "data_scientist")

VARIANT_ALIASES = {
    "auto": "auto",
    "dataanalysis": "data_analyst",
    "data_analysis": "data_analyst",
    "dataanalyst": "data_analyst",
    "datascience": "data_scientist",
    "datascientist": "data_scientist",
    "machine_learning": "data_scientist",
    "ml": "data_scientist",
    "powerbi": "bi",
    "business_intelligence": "bi",
    "ai_ml": "ai",
    "llm": "ai",
    "agentic": "ai",
}

IDENTITY_FIELDS = (
    "name",
    "location",
    "phone",
    "email",
    "linkedin",
    "github",
    "datacamp",
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _slug(value: Any, fallback: str = "item") -> str:
    slug = _SLUG_RE.sub("_", _norm(value)).strip("_")
    return slug[:48] or fallback


def _load_raw_variants() -> Dict[str, Any]:
    if not CV_VARIANTS_PATH.exists():
        raise FileNotFoundError(f"Canonical CV file missing: {CV_VARIANTS_PATH}")
    data = json.loads(CV_VARIANTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("cv_variants.json must be an object keyed by variant")
    return data


def _assign_ids(variant_key: str, variant: Dict[str, Any]) -> Dict[str, Any]:
    """Attach stable per-variant IDs. Does not drop any baseline fields."""
    out = deepcopy(variant)
    out["key"] = variant_key

    used_exp: set = set()
    experience = []
    for i, exp in enumerate(out.get("experience") or []):
        if not isinstance(exp, dict):
            continue
        item = dict(exp)
        base = f"{variant_key}:exp:{_slug(item.get('company'))}:{_slug(item.get('role'))}"
        ident = base
        n = 2
        while ident in used_exp:
            ident = f"{base}:{n}"
            n += 1
        used_exp.add(ident)
        item["id"] = ident
        item["source_index"] = i
        item["bullets"] = [str(b) for b in (item.get("bullets") or [])]
        experience.append(item)
    out["experience"] = experience

    used_proj: set = set()
    projects = []
    for i, proj in enumerate(out.get("projects") or []):
        if not isinstance(proj, dict):
            continue
        item = dict(proj)
        base = f"{variant_key}:proj:{_slug(item.get('name'))}"
        ident = base
        n = 2
        while ident in used_proj:
            ident = f"{base}:{n}"
            n += 1
        used_proj.add(ident)
        item["id"] = ident
        item["source_index"] = i
        item["bullets"] = [str(b) for b in (item.get("bullets") or [])]
        projects.append(item)
    out["projects"] = projects

    certs = []
    for i, cert in enumerate(out.get("certifications") or []):
        text = str(cert)
        certs.append({"id": f"{variant_key}:cert:{i}:{_slug(text, 'cert')}", "text": text, "source_index": i})
    out["certification_records"] = certs
    out["certifications"] = [c["text"] for c in certs]

    education = [str(x) for x in (out.get("education") or []) if str(x).strip()]
    out["education"] = education
    training = [str(x) for x in (out.get("training") or []) if str(x).strip()]
    out["training"] = training
    languages = [str(x) for x in (out.get("languages") or []) if str(x).strip()]
    out["languages"] = languages

    cats = out.get("skills_categories") or {}
    if not isinstance(cats, dict):
        cats = {}
    out["skills_categories"] = {str(k): [str(v) for v in (vals or [])] for k, vals in cats.items()}
    out["skill_category_records"] = [
        {"id": f"{variant_key}:skillcat:{_slug(k, 'cat')}", "key": k, "skills": list(vals)}
        for k, vals in out["skills_categories"].items()
    ]
    return out


@lru_cache(maxsize=1)
def load_variants() -> Dict[str, Dict[str, Any]]:
    raw = _load_raw_variants()
    variants = {}
    for key in VARIANT_KEYS:
        if key not in raw:
            logger.warning("Canonical variants missing key %s", key)
            continue
        variants[key] = _assign_ids(key, raw[key])
    return variants


def clear_variant_cache() -> None:
    load_variants.cache_clear()


def get_variant(key: str) -> Dict[str, Any]:
    variants = load_variants()
    if key not in variants:
        raise KeyError(f"Unknown CV variant {key!r}. Known: {sorted(variants)}")
    return deepcopy(variants[key])


def list_variant_catalog() -> Dict[str, Any]:
    catalog = {}
    for key, variant in load_variants().items():
        catalog[key] = {
            "key": key,
            "label": variant.get("label"),
            "roles": list(variant.get("roles") or []),
            "headline": variant.get("headline"),
            "experience": [
                {"id": e["id"], "role": e.get("role"), "company": e.get("company"), "period": e.get("period")}
                for e in variant.get("experience") or []
            ],
            "projects": [
                {"id": p["id"], "name": p.get("name"), "tech": p.get("tech")}
                for p in variant.get("projects") or []
            ],
            "skill_categories": [
                {"id": c["id"], "key": c["key"]} for c in variant.get("skill_category_records") or []
            ],
            "certifications": [
                {"id": c["id"], "text": c["text"]} for c in variant.get("certification_records") or []
            ],
            "education_count": len(variant.get("education") or []),
            "training_count": len(variant.get("training") or []),
        }
    return catalog


def canonical_payload(variant_key: str) -> Dict[str, Any]:
    """Structured candidate data the LLM must see when selecting or writing."""
    variant = get_variant(variant_key)
    return {
        "candidate_variant": variant_key,
        "label": variant.get("label"),
        "roles": list(variant.get("roles") or []),
        "headline": variant.get("headline"),
        "summary": variant.get("summary"),
        "experience": [
            {
                "id": e["id"],
                "role": e.get("role"),
                "company": e.get("company"),
                "period": e.get("period"),
                "bullets": list(e.get("bullets") or []),
            }
            for e in variant.get("experience") or []
        ],
        "projects": [
            {
                "id": p["id"],
                "name": p.get("name"),
                "tech": p.get("tech"),
                "links": p.get("links"),
                "bullets": list(p.get("bullets") or []),
            }
            for p in variant.get("projects") or []
        ],
        "skills_categories": deepcopy(variant.get("skills_categories") or {}),
        "skill_category_records": deepcopy(variant.get("skill_category_records") or []),
        "certifications": [
            {"id": c["id"], "text": c["text"]} for c in variant.get("certification_records") or []
        ],
        "education": list(variant.get("education") or []),
        "training": list(variant.get("training") or []),
        "languages": list(variant.get("languages") or []),
        "additional_information": variant.get("additional_information") or "",
    }


def variant_corpus(variant: Dict[str, Any]) -> str:
    chunks: List[str] = [
        str(variant.get("headline") or ""),
        str(variant.get("summary") or ""),
        str(variant.get("additional_information") or ""),
    ]
    for role in variant.get("roles") or []:
        chunks.append(str(role))
    for values in (variant.get("skills_categories") or {}).values():
        chunks.extend(str(v) for v in values or [])
    for exp in variant.get("experience") or []:
        chunks.extend([str(exp.get("role") or ""), str(exp.get("company") or "")])
        chunks.extend(str(b) for b in exp.get("bullets") or [])
    for proj in variant.get("projects") or []:
        chunks.extend([str(proj.get("name") or ""), str(proj.get("tech") or ""), str(proj.get("links") or "")])
        chunks.extend(str(b) for b in proj.get("bullets") or [])
    chunks.extend(str(c) for c in variant.get("certifications") or [])
    chunks.extend(str(c) for c in variant.get("education") or [])
    chunks.extend(str(c) for c in variant.get("training") or [])
    chunks.extend(str(c) for c in variant.get("languages") or [])
    return _norm(" ".join(chunks))


# Title / role-family phrases. Order matters: AI titles beat incidental ML stack words.
_TITLE_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "bi",
        (
            "power bi developer",
            "business intelligence developer",
            "bi developer",
            "bi analyst",
            "reporting developer",
            "power bi analyst",
        ),
    ),
    (
        "ai",
        (
            "ai engineer",
            "llm engineer",
            "agentic ai",
            "genai engineer",
            "generative ai engineer",
            "ai/ml engineer",
            "ai ml engineer",
        ),
    ),
    (
        "data_scientist",
        (
            "data scientist",
            "machine learning engineer",
            "ml engineer",
            "applied ml engineer",
            "applied machine learning engineer",
            "applied scientist",
        ),
    ),
    (
        "data_analyst",
        (
            "data analyst",
            "business analyst",
            "product analyst",
            "operations analyst",
            "supply chain analyst",
            "product analytics",
            "operations analytics",
        ),
    ),
)

_BODY_TECH_HINTS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("bi", ("power bi", "business intelligence", "dax", "power query"), "BI tooling in JD body"),
    ("data_analyst", ("tableau", "looker", "excel dashboard"), "Analyst tooling in JD body"),
)


def _title_haystack(role_text: str, job_text: str) -> str:
    role = _norm(role_text)
    lines = [ln.strip() for ln in (job_text or "").splitlines() if ln.strip()]
    head = _norm(" ".join(lines[:10]))
    return f"{role} {head}".strip()


def _family_hits(hay: str) -> List[Tuple[str, str]]:
    hits: List[Tuple[str, str]] = []
    for key, phrases in _TITLE_FAMILIES:
        for phrase in phrases:
            if phrase in hay:
                hits.append((key, phrase))
                break
    return hits


def choose_variant(role_text: str, job_text: str, requested: str = "auto") -> Dict[str, Any]:
    """Deterministic variant choice. Title/role-family beats incidental tech."""
    reasons: List[str] = []
    requested_n = _norm(requested).replace("-", "_").replace(" ", "_")
    requested_n = VARIANT_ALIASES.get(requested_n, requested_n)
    variants = load_variants()
    if requested_n in variants:
        reasons.append(f"Explicit resume_variant={requested_n}")
        return {
            "selected_variant": requested_n,
            "reasons": reasons,
            "confidence": "high",
            "ambiguous": False,
        }

    title_hits = _family_hits(_title_haystack(role_text, job_text))
    if title_hits:
        key, phrase = title_hits[0]
        extra = [f"{k}:{p}" for k, p in title_hits[1:]]
        reasons.append(f"Role-family/title match: {phrase} -> {key}")
        if extra:
            reasons.append("Additional title signals: " + ", ".join(extra[:3]))
        return {
            "selected_variant": key,
            "reasons": reasons,
            "confidence": "high",
            "ambiguous": len(title_hits) > 1,
        }

    body = _norm(job_text)
    body_hits = _family_hits(body)
    if body_hits:
        key, phrase = body_hits[0]
        reasons.append(f"Role phrase in JD body: {phrase} -> {key}")
        return {
            "selected_variant": key,
            "reasons": reasons,
            "confidence": "medium",
            "ambiguous": len(body_hits) > 1,
        }

    for key, needles, reason in _BODY_TECH_HINTS:
        hits = [n for n in needles if n in body]
        if hits:
            reasons.append(f"{reason} (matched: {', '.join(hits[:4])}) — low confidence")
            return {
                "selected_variant": key,
                "reasons": reasons,
                "confidence": "low",
                "ambiguous": True,
            }

    reasons.append("No family-specific title; defaulting to ai variant with low confidence")
    return {
        "selected_variant": "ai",
        "reasons": reasons,
        "confidence": "low",
        "ambiguous": True,
    }


def select_variant(role_text: str, job_text: str, requested: str = "auto") -> Tuple[str, List[str]]:
    """Deterministic variant selection. The LLM cannot invent a variant."""
    choice = choose_variant(role_text, job_text, requested=requested)
    return choice["selected_variant"], list(choice["reasons"])


def experience_by_id(variant: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {e["id"]: e for e in variant.get("experience") or []}


def projects_by_id(variant: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {p["id"]: p for p in variant.get("projects") or []}


def certifications_by_id(variant: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {c["id"]: c for c in variant.get("certification_records") or []}


def skill_categories_by_id(variant: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {c["id"]: c for c in variant.get("skill_category_records") or []}


def load_identity(profile: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Contact-card fields only. Never used as career-fact source."""
    from store import load_profile

    raw = profile if profile is not None else load_profile()
    identity = {field: str((raw or {}).get(field) or "") for field in IDENTITY_FIELDS}
    if not identity.get("name"):
        logger.warning("Identity profile is missing candidate name")
    return identity
