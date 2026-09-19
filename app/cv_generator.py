import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from config import GENERATED_CVS_DIR, logger


# DejaVu is deliberately bundled by the Docker image. Helvetica/Times in the
# PDF standard fonts are Latin-1 only and turn Unicode punctuation into '?'.
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
FONT_ITALIC = FONT_DIR / "DejaVuSans-Oblique.ttf"
FONT_BOLD_ITALIC = FONT_DIR / "DejaVuSans-BoldOblique.ttf"


class PDFResume(FPDF):
    def header(self):
        # Intentional whitespace: the CV uses a strong header only on page 1.
        pass

    def footer(self):
        self.set_y(-10)
        self.set_font("DejaVu", "", 7.2)
        self.set_text_color(125, 130, 138)
        self.cell(0, 6, f"{self.page_no()}", align="C")


def clean_text(value: Any) -> str:
    """Normalize Unicode without destroying real candidate content."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    # Replace control characters, but keep tabs/newlines meaningful to callers.
    text = "".join(ch if (ch in "\n\t" or ord(ch) >= 32) else " " for ch in text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _ensure_fonts() -> None:
    missing = [str(p) for p in (FONT_REGULAR, FONT_BOLD, FONT_ITALIC, FONT_BOLD_ITALIC) if not p.exists()]
    if missing:
        raise RuntimeError(
            "Unicode CV fonts are missing from the image. Expected DejaVu fonts at "
            + ", ".join(missing)
        )


def _safe_filename(name: str) -> str:
    clean_name = re.sub(r"[^a-zA-Z0-9]+", "_", clean_text(name)).strip("_") or "Resume"
    return f"{clean_name}_CV.pdf"


def _flatten_skills(skills_categories: Dict[str, Any]) -> List[str]:
    skills: List[str] = []
    for values in skills_categories.values():
        if isinstance(values, list):
            for value in values:
                item = clean_text(value)
                if item and item not in skills:
                    skills.append(item)
    return skills


def _link_label(url: str) -> str:
    value = clean_text(url)
    value = re.sub(r"^https?://", "", value)
    return value.rstrip("/")


def generate_pdf_cv(cv_data: Dict[str, Any], filename: str = None) -> str:
    """
    Generate a polished, Unicode-safe, ATS-friendly 1–2 page CV.

    The previous generator intentionally used the standard Helvetica font.
    That font silently replaces Unicode punctuation/symbols with '?'. It also
    rendered a very small, sparse one-page layout. This generator keeps real
    profile content, uses a Unicode TTF, and gives the candidate enough space
    for experience, projects, skills, education and credentials.
    """
    _ensure_fonts()

    if not filename:
        filename = _safe_filename(cv_data.get("name", "Resume"))

    output_path = GENERATED_CVS_DIR / filename

    pdf = PDFResume(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(15, 14, 15)
    pdf.add_font("DejaVu", "", str(FONT_REGULAR))
    pdf.add_font("DejaVu", "B", str(FONT_BOLD))
    pdf.add_font("DejaVu", "I", str(FONT_ITALIC))
    pdf.add_font("DejaVu", "BI", str(FONT_BOLD_ITALIC))
    pdf.add_page()

    accent = (25, 76, 118)
    dark = (28, 34, 42)
    body = (48, 53, 61)
    muted = (100, 107, 117)
    rule = (210, 216, 224)

    margin = 15
    epw = pdf.epw

    def set_font(style: str = "", size: float = 9.0, color: Tuple[int, int, int] = body):
        pdf.set_font("DejaVu", style, size)
        pdf.set_text_color(*color)

    def draw_rule():
        pdf.set_draw_color(*rule)
        pdf.set_line_width(0.25)
        pdf.line(margin, pdf.get_y(), margin + epw, pdf.get_y())

    def section_header(title: str):
        if pdf.get_y() > 265:
            pdf.add_page()
        pdf.ln(1.2)
        set_font("B", 9.4, accent)
        pdf.cell(epw, 5.5, clean_text(title).upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        draw_rule()
        pdf.ln(2.0)

    def bullet(text: Any, indent: float = 4.6, size: float = 8.55, line_h: float = 3.95):
        value = clean_text(text)
        if not value:
            return
        # Keep the bullet and wrapped body aligned. Unicode font prevents '?'.
        set_font("", size, body)
        x = margin
        pdf.set_x(x)
        pdf.cell(4.0, line_h, "•", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_x(x + indent)
        pdf.multi_cell(epw - indent, line_h, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(0.35)

    def compact_row(label: str, value: Any):
        value_s = clean_text(value)
        if not value_s:
            return
        set_font("B", 8.15, dark)
        label_w = pdf.get_string_width(label) + 2
        pdf.set_x(margin)
        pdf.cell(label_w, 4.2, label, new_x=XPos.RIGHT, new_y=YPos.TOP)
        set_font("", 8.15, body)
        pdf.multi_cell(epw - label_w, 4.2, value_s, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ------------------------------ HEADER ---------------------------------
    name = clean_text(cv_data.get("name"))
    headline = clean_text(cv_data.get("headline"))
    location = clean_text(cv_data.get("location"))
    phone = clean_text(cv_data.get("phone"))
    email = clean_text(cv_data.get("email"))
    linkedin = _link_label(cv_data.get("linkedin", ""))
    github = _link_label(cv_data.get("github", ""))
    datacamp = _link_label(cv_data.get("datacamp", ""))

    set_font("B", 22, dark)
    pdf.cell(epw, 9, name, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    if headline:
        set_font("B", 10.6, accent)
        pdf.cell(epw, 5.4, headline[:80], new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    contacts = [item for item in [location, phone, email] if item]
    if contacts:
        set_font("", 8.15, muted)
        pdf.cell(epw, 4.5, "  •  ".join(contacts), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    links = [item for item in [linkedin, github, datacamp] if item]
    if links:
        set_font("", 7.7, muted)
        pdf.cell(epw, 4.3, "  •  ".join(links), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.ln(2.0)
    draw_rule()
    pdf.ln(2.5)

    # ------------------------------ SUMMARY --------------------------------
    summary = clean_text(cv_data.get("summary"))
    if summary:
        section_header("Professional Summary")
        set_font("", 9.0, body)
        pdf.multi_cell(epw, 4.3, summary, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1.3)

    # ------------------------------ EXPERIENCE ------------------------------
    experience = cv_data.get("experience") or []
    if experience:
        section_header("Experience")
        for exp in experience[:3]:
            if not isinstance(exp, dict):
                continue
            role = clean_text(exp.get("role"))
            company = clean_text(exp.get("company"))
            period = clean_text(exp.get("period"))
            title = role
            if company and company.lower() not in role.lower():
                title = f"{role} | {company}" if role else company

            set_font("B", 9.45, dark)
            pdf.cell(epw * 0.72, 5.0, title, new_x=XPos.RIGHT, new_y=YPos.TOP)
            set_font("I", 8.0, muted)
            pdf.cell(epw * 0.28, 5.0, period, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")

            for item in exp.get("bullets", [])[:4]:
                bullet(item)
            pdf.ln(1.0)

    # ------------------------------ PROJECTS --------------------------------
    projects = cv_data.get("projects") or []
    if projects:
        section_header("Selected Projects")
        for proj in projects[:5]:
            if isinstance(proj, str):
                bullet(proj)
                continue
            if not isinstance(proj, dict):
                continue

            pname = clean_text(proj.get("name"))
            tech = clean_text(proj.get("tech"))
            links_text = _link_label(proj.get("links", ""))

            title = pname or "Project"
            if tech:
                title = f"{title} | {tech}"

            set_font("B", 9.15, dark)
            pdf.multi_cell(epw, 4.8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if links_text:
                set_font("I", 7.35, muted)
                pdf.multi_cell(epw, 3.9, links_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            for item in proj.get("bullets", [])[:3]:
                bullet(item, indent=4.6, size=8.45, line_h=3.85)
            pdf.ln(0.7)

    # ------------------------------ SKILLS ----------------------------------
    skills_categories = cv_data.get("skills_categories") or {}
    if skills_categories:
        section_header("Technical Skills")
        for category, skills in skills_categories.items():
            if not isinstance(skills, list):
                continue
            values = [clean_text(s) for s in skills if clean_text(s)]
            if not values:
                continue
            compact_row(f"{clean_text(category)}: ", ", ".join(values))
            pdf.ln(0.35)
    else:
        top_skills = cv_data.get("top_skills") or []
        if top_skills:
            section_header("Technical Skills")
            compact_row("", ", ".join(clean_text(s) for s in top_skills if clean_text(s)))

    # ------------------------- EDUCATION / CERTS ---------------------------
    education = [clean_text(x) for x in (cv_data.get("education") or []) if clean_text(x)]
    certifications = [clean_text(x) for x in (cv_data.get("certifications") or []) if clean_text(x)]
    languages = [clean_text(x) for x in (cv_data.get("languages") or []) if clean_text(x)]
    military = clean_text(cv_data.get("military_service"))

    if education:
        section_header("Education")
        for item in education[:2]:
            bullet(item, size=8.5)

    if certifications:
        section_header("Certifications")
        # Keep the list readable across pages rather than squeezing it into a tiny block.
        for item in certifications[:7]:
            bullet(item, size=8.25, line_h=3.75)

    extras: List[str] = []
    if languages:
        extras.append("Languages: " + " | ".join(languages))
    if military:
        extras.append("Military Service: " + military)

    if extras:
        section_header("Additional Information")
        for item in extras:
            bullet(item, size=8.35, line_h=3.75)

    pdf.output(str(output_path))
    logger.info("Generated Unicode-safe CV PDF at: %s", output_path)
    return str(output_path)
