"""Deterministic validators for selection, claims, email, and PDF text."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from candidate import certifications_by_id, experience_by_id, projects_by_id, variant_corpus
from matcher import SKILLS

HYPE_RE = re.compile(r"\b(thrilled|excited|passionate|perfect fit|great fit)\b", re.I)
MD_RE = re.compile(r"[#*_`]|^\s*[-•]", re.M)
METRIC_RE = re.compile(
    r"(?:\$\s?\d[\d,]*(?:\.\d+)?\s*[kmb]?|\b\d+(?:\.\d+)?\s*%|\b\d{1,3}(?:,\d{3})+\b|\b\d+\.\d+\b)",
    re.I,
)


class PipelineError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": False, "code": self.code, "error": self.message}


def validate_selection(selection: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    exp_ids = set(experience_by_id(variant))
    proj_ids = set(projects_by_id(variant))
    cert_ids = set(certifications_by_id(variant))
    cat_ids = set((variant.get("skills_categories") or {}).keys())

    raw_proj = list(selection.get("selected_project_ids") or [])
    raw_exp = list(selection.get("selected_experience_ids") or [])
    raw_cats = list(selection.get("selected_skill_categories") or [])
    raw_certs = list(selection.get("selected_certification_ids") or [])

    invalid = [i for i in raw_proj if i not in proj_ids]
    invalid += [i for i in raw_exp if i not in exp_ids]
    invalid += [i for i in raw_cats if i not in cat_ids]
    invalid += [i for i in raw_certs if i not in cert_ids]
    if invalid:
        code = "INVALID_PROJECT_ID" if any(i not in proj_ids for i in raw_proj) else "INVALID_ID"
        return {
            "ok": False,
            "code": code,
            "error": f"Unknown canonical IDs: {invalid}",
            "invalid_ids": invalid,
        }

    groups = {
        "project": raw_proj,
        "experience": raw_exp,
        "skill category": raw_cats,
        "certification": raw_certs,
    }
    duplicates = [
        f"{label} ID {value}" for label, values in groups.items()
        for value in set(values) if values.count(value) > 1
    ]
    if duplicates:
        return {"ok": False, "code": "DUPLICATE_SELECTION_ID", "error": "Duplicate canonical IDs: " + ", ".join(sorted(duplicates)), "duplicates": duplicates}
    return {"ok": True, "selection": {
        "selected_project_ids": raw_proj,
        "selected_experience_ids": raw_exp,
        "selected_skill_categories": raw_cats,
        "selected_certification_ids": raw_certs,
    }}


def _metrics(text: str) -> Set[str]:
    return {m.group(0).lower().replace(" ", "") for m in METRIC_RE.finditer(text or "")}


def _tech_mentions(text: str) -> Set[str]:
    n = (text or "").lower()
    found = set()
    for skill in SKILLS:
        for alias in skill.exact_aliases:
            if re.search(rf"(?:\b|_){re.escape(alias)}(?:\b|_)", n):
                found.add(skill.canonical)
                break
    return found


def _allowed_corpus(variant: Dict[str, Any], extra: str = "") -> str:
    return variant_corpus(variant) + " " + (extra or "")


def _unsupported_claims(text: str, variant: Dict[str, Any], local_source: str = "") -> List[str]:
    """Conservative deterministic checks for common high-risk factual claims.

    This is intentionally not semantic fact verification. It catches explicit
    employers, roles, certifications, years, team/client-management, and
    outcome claims when their phrase is absent from canonical evidence.
    """
    source = _allowed_corpus(variant, local_source).lower()
    value = (text or "").lower()
    issues: List[str] = []
    for pattern, label in (
        (r"\b(?:worked|working|employed)\s+(?:at|for)\s+([A-Z][\w.& -]{2,})", "employer"),
        (r"\bat\s+([A-Z][\w.& -]{2,})", "employer"),
        (r"\b(?:as|title of)\s+(?:a |an )?([A-Z][A-Za-z /-]{2,})", "job title"),
        (r"\b(?:certified|certification(?: in)?|certificate(?: in)?)\s+([A-Z][\w .+-]{2,})", "certification"),
    ):
        for match in re.finditer(pattern, text or ""):
            phrase = match.group(1).strip().lower()
            if phrase and phrase not in source:
                issues.append(f"unsupported {label} {match.group(1).strip()}")
    for match in re.finditer(r"\b(\d+)\+?\s+years?\b", value):
        if match.group(0) not in source:
            issues.append(f"unsupported years claim {match.group(0)}")
    for pattern, label in (
        (r"\b(?:led|managed|supervised)\s+(?:a\s+)?team\s+of\s+\d+", "team-size responsibility"),
        (r"\bmanaged\s+(?:enterprise|key|strategic)\s+clients?", "client-management responsibility"),
        (r"\b(?:increased|reduced|improved|saved|generated)\b", "achievement"),
    ):
        if re.search(pattern, value) and not re.search(pattern, source):
            issues.append(f"unsupported {label}")
    return issues


def validate_tailoring(tailoring: Dict[str, Any], variant: Dict[str, Any], job_text: str = "") -> Dict[str, Any]:
    exp_map = experience_by_id(variant)
    proj_map = projects_by_id(variant)
    headline = str(tailoring.get("headline") or "").strip()
    summary = str(tailoring.get("summary") or "").strip()
    bullets_by_exp = tailoring.get("rewritten_bullets_by_experience_id") or {}
    desc_by_proj = tailoring.get("rewritten_project_descriptions_by_project_id") or {}
    if not isinstance(bullets_by_exp, dict) or not isinstance(desc_by_proj, dict):
        return {"ok": False, "code": "UNSUPPORTED_CLAIM", "error": "Rewrite maps must be objects keyed by canonical ID"}

    issues: List[str] = []
    allowed = _allowed_corpus(variant)
    allowed_metrics = _metrics(allowed)
    allowed_tech = _tech_mentions(allowed)

    def check_text(label: str, text: str, local_source: str = "") -> None:
        source = allowed + " " + local_source
        source_metrics = _metrics(source)
        source_tech = _tech_mentions(source)
        for metric in _metrics(text) - source_metrics:
            issues.append(f"{label}: invented metric {metric}")
        for tech in _tech_mentions(text) - source_tech:
            issues.append(f"{label}: unsupported technology {tech}")
        for claim in _unsupported_claims(text, variant, local_source):
            issues.append(f"{label}: {claim}")

    check_text("headline", headline)
    check_text("summary", summary, variant.get("summary") or "")

    cleaned_exp: Dict[str, List[str]] = {}
    for ident, bullets in bullets_by_exp.items():
        if ident not in exp_map:
            issues.append(f"Unknown experience id {ident}")
            continue
        src = exp_map[ident]
        src_text = " ".join(src.get("bullets") or []) + " " + str(src.get("role") or "") + " " + str(src.get("company") or "")
        cleaned: List[str] = []
        for b in bullets or []:
            text = str(b).strip()
            if not text:
                continue
            check_text(f"experience {ident}", text, src_text)
            cleaned.append(text)
        if len(cleaned) != len(src.get("bullets") or []):
            issues.append(f"experience {ident}: bullet count must match canonical count")
        cleaned_exp[ident] = cleaned

    cleaned_proj: Dict[str, List[str]] = {}
    for ident, bullets in desc_by_proj.items():
        if ident not in proj_map:
            issues.append(f"Unknown project id {ident}")
            continue
        src = proj_map[ident]
        src_text = " ".join(
            [str(src.get("name") or ""), str(src.get("tech") or "")] + list(src.get("bullets") or [])
        )
        cleaned = []
        for b in bullets or []:
            text = str(b).strip()
            if not text:
                continue
            check_text(f"project {ident}", text, src_text)
            cleaned.append(text)
        if len(cleaned) != len(src.get("bullets") or []):
            issues.append(f"project {ident}: bullet count must match canonical count")
        cleaned_proj[ident] = cleaned

    if issues:
        invented = [i for i in issues if "invented metric" in i or "unsupported technology" in i or "Unknown" in i]
        code = "UNSUPPORTED_CLAIM" if invented else "UNSUPPORTED_CLAIM"
        return {"ok": False, "code": code, "error": "; ".join(issues[:8]), "issues": issues}

    return {
        "ok": True,
        "tailoring": {
            "headline": headline,
            "summary": summary,
            "rewritten_bullets_by_experience_id": cleaned_exp,
            "rewritten_project_descriptions_by_project_id": cleaned_proj,
        },
    }


def validate_email(
    email: Dict[str, Any],
    variant: Dict[str, Any],
    identity: Dict[str, str],
    job_text: str = "",
) -> Dict[str, Any]:
    subject = str(email.get("subject") or "").strip()
    body = str(email.get("body") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    issues: List[str] = []
    if not subject:
        issues.append("Missing email subject")
    elif len(subject) > 80:
        issues.append("Subject over 80 characters")
    words = len(body.split())
    if words < 90 or words > 220:
        issues.append(f"Email length {words} words — target 120–180")
    if MD_RE.search(body):
        issues.append("Email still has markdown/bullets — plain text only")
    if HYPE_RE.search(body):
        issues.append("Email uses hype phrases")
    # The JD can explain relevance but is never evidence that the candidate
    # possesses a skill or achievement.
    allowed = _allowed_corpus(variant)
    extra_metrics = _metrics(body) - _metrics(allowed)
    if extra_metrics:
        issues.append(f"Email invented metrics: {sorted(extra_metrics)[:5]}")
    extra_tech = _tech_mentions(body) - _tech_mentions(allowed)
    if extra_tech:
        issues.append(f"Email unsupported technologies: {sorted(extra_tech)[:5]}")
    issues.extend(f"Email {claim}" for claim in _unsupported_claims(body, variant))
    if issues:
        return {"ok": False, "code": "EMAIL_VALIDATION_FAILED", "error": "; ".join(issues), "issues": issues}
    accepted = dict(email)
    accepted["subject"] = subject
    accepted["body"] = body
    return {"ok": True, "email": accepted}


def extract_pdf_text(pdf_path: str) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(pdf_path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to validate generated PDFs") from exc
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def validate_pdf(pdf_path: str, cv_data: Dict[str, Any]) -> Dict[str, Any]:
    if not pdf_path or not Path(pdf_path).exists():
        return {"ok": False, "code": "PDF_CONTENT_MISMATCH", "error": "CV PDF path missing or file not found"}
    try:
        text = extract_pdf_text(pdf_path)
    except Exception as exc:
        return {"ok": False, "code": "PDF_CONTENT_MISMATCH", "error": f"Could not extract PDF text: {exc}"}

    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    issues: List[str] = []
    if "\ufffd" in text or "�" in text:
        issues.append("Broken replacement characters in PDF")

    def present(value: str, label: str, min_len: int = 8) -> None:
        token = str(value or "").strip()
        if not token:
            return
        key = re.sub(r"[^a-z0-9]+", "", token.lower())[:min_len if min_len > 12 else 18]
        if len(key) < 6:
            return
        if key not in compact:
            issues.append(f"Missing {label}: {token[:80]}")

    name = str(cv_data.get("name") or "").strip()
    present(name, "candidate name", min_len=10)
    present(str(cv_data.get("headline") or ""), "headline")
    summary = str(cv_data.get("summary") or "").strip()
    if summary:
        present(" ".join(summary.split()[:8]), "summary")

    for exp in cv_data.get("experience") or []:
        present(exp.get("company") or "", "experience company")
        present(exp.get("role") or "", "experience role")
    for proj in cv_data.get("projects") or []:
        present(proj.get("name") or "", "project")
    for edu in cv_data.get("education") or []:
        present(str(edu)[:40], "education")
    for cert in cv_data.get("certifications") or []:
        present(str(cert)[:30], "certification")
    for lang in cv_data.get("languages") or []:
        present(str(lang).split("(")[0].strip(), "language")

    if "PROFESSIONAL SUMMARY" not in text.upper() and cv_data.get("summary"):
        issues.append("Missing Professional Summary section")
    if cv_data.get("experience") and "EXPERIENCE" not in text.upper():
        issues.append("Missing Experience section")
    if cv_data.get("projects") and "PROJECTS" not in text.upper():
        issues.append("Missing Projects section")

    if issues:
        return {
            "ok": False,
            "code": "PDF_CONTENT_MISMATCH",
            "error": "; ".join(issues[:8]),
            "issues": issues,
            "chars": len(text),
        }
    return {"ok": True, "chars": len(text), "pages": text.count("\n") and None}
