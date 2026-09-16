import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from config import GENERATED_CVS_DIR, logger


class PDFResume(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def clean_text(text: str) -> str:
    """Ensure text is encodable in standard latin-1/helvetica for clean PDF output."""
    if not text:
        return ""
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2022": "-", "\u2026": "...",
        "\u200b": "", "\u00a0": " ", "—": "-", "–": "-",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf_cv(cv_data: Dict[str, Any], filename: str = None) -> str:
    """
    Generate a full, ATS-friendly, professional PDF resume tailored to the target role.
    """
    if not filename:
        # Use candidate name for clean, short filename
        clean_name = re.sub(r"[^a-zA-Z0-9]", "_", cv_data.get("name", "Resume")).strip("_")
        filename = f"{clean_name}_CV.pdf"

    output_path = GENERATED_CVS_DIR / filename

    pdf = PDFResume(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    margin = 16
    pdf.set_margins(margin, 14, margin)
    epw = pdf.epw  # Effective page width

    # 1. Header: Name
    name = clean_text(cv_data.get("name", "Mohamed Waleed Ezzat"))
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 40, 75)
    pdf.cell(epw, 8, name, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    # Headline (keep it short - max 60 chars)
    headline = clean_text(cv_data.get("headline", "AI/ML Engineer - Agentic AI & LLM Systems"))
    if headline and len(headline) > 60:
        headline = headline[:57] + "..."
    if headline:
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(65, 80, 100)
        pdf.cell(epw, 5.5, headline, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    # Contacts Line
    contacts = [
        clean_text(cv_data.get("location", "Banha, Egypt")),
        clean_text(cv_data.get("phone", "+20 128 090 8922")),
        clean_text(cv_data.get("email", "mwezzat16@gmail.com")),
        clean_text(cv_data.get("linkedin", "linkedin.com/in/mohamedwaleed-data")),
        clean_text(cv_data.get("github", "github.com/Mohamed-Elmasry16")),
    ]
    contacts_str = "  |  ".join([c for c in contacts if c])
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(epw, 5, contacts_str, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(1.5)

    # Divider line
    pdf.set_draw_color(190, 200, 215)
    pdf.set_line_width(0.4)
    pdf.line(margin, pdf.get_y(), margin + epw, pdf.get_y())
    pdf.ln(3)

    def section_header(title: str):
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(20, 40, 75)
        pdf.cell(epw, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        pdf.set_draw_color(210, 215, 225)
        pdf.set_line_width(0.2)
        pdf.line(margin, pdf.get_y(), margin + epw, pdf.get_y())
        pdf.ln(2)

    def print_bullet(text: str):
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(40, 40, 40)
        curr_y = pdf.get_y()
        pdf.set_x(margin)
        pdf.cell(4, 4.2, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_x(margin + 4)
        pdf.multi_cell(epw - 4, 4.2, clean_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 2. Professional Summary
    summary = clean_text(cv_data.get("summary", ""))
    if summary:
        section_header("PROFESSIONAL SUMMARY")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(35, 35, 35)
        pdf.multi_cell(epw, 4.4, summary, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2.5)

    # 3. Work Experience
    experience = cv_data.get("experience", [])
    if experience:
        section_header("EXPERIENCE")
        for exp in experience:
            role = clean_text(exp.get("role", ""))
            company = clean_text(exp.get("company", ""))
            period = clean_text(exp.get("period", ""))

            title_left = f"{role} - {company}" if company and company not in role else role
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(25, 25, 25)
            
            # Left title, right period
            pdf.set_x(margin)
            pdf.cell(epw * 0.75, 5, title_left, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "I", 8.5)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(epw * 0.25, 5, period, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")

            for b in exp.get("bullets", []):
                print_bullet(b)
            pdf.ln(1.5)

    # 4. Featured Projects
    projects = cv_data.get("projects", [])
    if projects:
        section_header("FEATURED PROJECTS")
        for proj in projects[:4]:
            if isinstance(proj, dict):
                p_name = clean_text(proj.get("name", ""))
                p_tech = clean_text(proj.get("tech", ""))
                p_links = clean_text(proj.get("links", ""))

                pdf.set_font("Helvetica", "B", 9.5)
                pdf.set_text_color(25, 25, 25)
                pdf.set_x(margin)
                pdf.cell(epw, 4.8, f"{p_name} | {p_tech}" if p_tech else p_name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                if p_links:
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(70, 100, 140)
                    pdf.set_x(margin)
                    pdf.cell(epw, 4, p_links, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                for b in proj.get("bullets", []):
                    print_bullet(b)
                pdf.ln(1.5)
            elif isinstance(proj, str):
                print_bullet(proj)
                pdf.ln(1)

    # 5. Technical Skills
    skills_cats = cv_data.get("skills_categories", {})
    if skills_cats:
        section_header("TECHNICAL SKILLS")
        for cat_name, skill_list in skills_cats.items():
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(30, 30, 30)
            label = clean_text(cat_name) + ": "
            pdf.set_x(margin)
            label_w = pdf.get_string_width(label) + 2
            pdf.cell(label_w, 4.4, label, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(epw - label_w, 4.4, clean_text(", ".join(skill_list)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
    elif cv_data.get("top_skills"):
        section_header("CORE SKILLS")
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(50, 50, 50)
        pdf.set_x(margin)
        pdf.multi_cell(epw, 4.4, clean_text(" - ".join(cv_data.get("top_skills"))), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # 6. Education & Certifications
    education = cv_data.get("education", [])
    certifications = cv_data.get("certifications", [])
    if education or certifications:
        section_header("EDUCATION & CERTIFICATIONS")
        for edu in education:
            print_bullet(edu)
        for cert in certifications[:4]:
            print_bullet(cert)

    pdf.output(str(output_path))
    logger.info(f"Generated complete CV PDF at: {output_path}")
    return str(output_path)
