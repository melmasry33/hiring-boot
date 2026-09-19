import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from config import GENERATED_CVS_DIR, logger


FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
FONT_ITALIC = FONT_DIR / "DejaVuSans-Oblique.ttf"
FONT_BOLD_ITALIC = FONT_DIR / "DejaVuSans-BoldOblique.ttf"


class PDFResume(FPDF):
    def footer(self):
        self.set_y(-10)
        self.set_font("DejaVu", "", 7.2)
        self.set_text_color(125, 130, 138)
        self.cell(0, 6, f"{self.page_no()}", align="C")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u200b", "").replace("\ufeff", "")
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


def _link_label(url: str) -> str:
    value = clean_text(url)
    value = re.sub(r"^https?://", "", value)
    return value.rstrip("/")


def generate_pdf_cv(cv_data: Dict[str, Any], filename: str = None) -> str:
    """Render the canonical CV in the user's fixed ATS-safe section order.

    Tailoring happens before this function. This renderer does not decide what
    to delete; it prints all supplied profile content.
    """
    _ensure_fonts()

    if not filename:
        filename = _safe_filename(cv_data.get("name", "Resume"))

    output_path = GENERATED_CVS_DIR / filename

    pdf = PDFResume(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(15, 14, 15)
    for style, path in (
        ("", FONT_REGULAR),
        ("B", FONT_BOLD),
        ("I", FONT_ITALIC),
        ("BI", FONT_BOLD_ITALIC),
    ):
        pdf.add_font("DejaVu", style, str(path))
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
        pdf.ln(1.0)
        set_font("B", 9.6, accent)
        pdf.cell(epw, 5.5, title.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        draw_rule()
        pdf.ln(2.0)

    def bullet(text: Any, size: float = 8.55, line_h: float = 3.95):
        value = clean_text(text)
        if not value:
            return
        set_font("", size, body)
        x = margin
        pdf.set_x(x)
        pdf.cell(4.0, line_h, "•", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_x(x + 4.6)
        pdf.multi_cell(epw - 4.6, line_h, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(0.35)

    def entry_heading(left: str, right: str = ""):
        set_font("B", 9.45, dark)
        pdf.cell(epw * 0.72, 5.0, clean_text(left), new_x=XPos.RIGHT, new_y=YPos.TOP)
        if right:
            set_font("I", 8.0, muted)
            pdf.cell(epw * 0.28, 5.0, clean_text(right), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        else:
            pdf.ln(5.0)

    def subline(left: str, right: str = ""):
        set_font("I", 8.0, muted)
        pdf.cell(epw * 0.72, 4.0, clean_text(left), new_x=XPos.RIGHT, new_y=YPos.TOP)
        if right:
            pdf.cell(epw * 0.28, 4.0, clean_text(right), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        else:
            pdf.ln(4.0)

    def compact_row(label: str, value: Any):
        value_s = clean_text(value)
        if not value_s:
            return
        set_font("B", 8.1, dark)
        label_w = min(pdf.get_string_width(label) + 2, epw * 0.33)
        pdf.set_x(margin)
        pdf.cell(label_w, 4.2, label, new_x=XPos.RIGHT, new_y=YPos.TOP)
        set_font("", 8.1, body)
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

    set_font("B", 21.5, dark)
    pdf.cell(epw, 9, name, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    if headline:
        set_font("B", 10.4, accent)
        pdf.cell(epw, 5.2, headline, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    contacts = [item for item in (location, phone, email) if item]
    if contacts:
        set_font("", 8.0, muted)
        pdf.cell(epw, 4.3, "  |  ".join(contacts), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    links = [item for item in (linkedin, github, datacamp) if item]
    if links:
        set_font("", 7.55, muted)
        pdf.cell(epw, 4.1, "  |  ".join(links), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.ln(1.7)
    draw_rule()
    pdf.ln(2.3)

    # ------------------------------ SUMMARY --------------------------------
    summary = clean_text(cv_data.get("summary"))
    if summary:
        section_header("Professional Summary")
        set_font("", 9.0, body)
        pdf.multi_cell(epw, 4.25, summary, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1.0)

    # ------------------------------ EDUCATION ------------------------------
    education = [clean_text(x) for x in (cv_data.get("education") or []) if clean_text(x)]
    if education:
        section_header("Education")
        for item in education:
            bullet(item, size=8.5, line_h=3.85)

    # ------------------------------ EXPERIENCE -----------------------------
    experience = cv_data.get("experience") or []
    if experience:
        section_header("Experience")
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            role = clean_text(exp.get("role"))
            company = clean_text(exp.get("company"))
            period = clean_text(exp.get("period"))
            entry_heading(company or role, period)
            if company and role and company.lower() != role.lower():
                subline(role)
            for item in exp.get("bullets", []) or []:
                bullet(item)
            pdf.ln(0.8)

    # ------------------------------ PROJECTS -------------------------------
    projects = cv_data.get("projects") or []
    if projects:
        section_header("Projects")
        for proj in projects:
            if isinstance(proj, str):
                bullet(proj)
                continue
            if not isinstance(proj, dict):
                continue
            name_p = clean_text(proj.get("name")) or "Project"
            tech = clean_text(proj.get("tech"))
            links_text = _link_label(proj.get("links", ""))
            entry_heading(name_p)
            if tech:
                subline(tech)
            if links_text:
                set_font("", 7.4, muted)
                pdf.multi_cell(epw, 3.7, links_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            for item in proj.get("bullets", []) or []:
                bullet(item, size=8.45, line_h=3.8)
            pdf.ln(0.7)

    # -------------------------------- SKILLS -------------------------------
    skills_categories = cv_data.get("skills_categories") or {}
    if skills_categories:
        section_header("Skills")
        for category, skills in skills_categories.items():
            values = [clean_text(x) for x in (skills or []) if clean_text(x)]
            if values:
                compact_row(f"{clean_text(category)}: ", ", ".join(values))
                pdf.ln(0.25)

    # -------------------------- CERTIFICATIONS ----------------------------
    certifications = [clean_text(x) for x in (cv_data.get("certifications") or []) if clean_text(x)]
    if certifications:
        section_header("Certifications")
        for item in certifications:
            bullet(item, size=8.25, line_h=3.75)

    # ----------------------------- LANGUAGES -------------------------------
    languages = [clean_text(x) for x in (cv_data.get("languages") or []) if clean_text(x)]
    if languages:
        section_header("Languages")
        set_font("", 8.5, body)
        pdf.multi_cell(epw, 4.0, ", ".join(languages), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ---------------------- ADDITIONAL INFORMATION ------------------------
    military = clean_text(cv_data.get("military_service"))
    if military:
        section_header("Additional Information")
        set_font("", 8.5, body)
        pdf.multi_cell(epw, 4.0, f"Military Service: {military}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(str(output_path))
    logger.info("Generated complete Unicode-safe CV PDF at: %s", output_path)
    return str(output_path)
