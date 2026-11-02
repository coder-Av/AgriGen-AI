"""PDF report generation with ReportLab.

Telugu / Hindi need Unicode fonts. Download these (free, Google Fonts) into assets/fonts/:
  NotoSansTelugu-Regular.ttf       NotoSansDevanagari-Regular.ttf
Without them, report_language_ok() returns False and the app falls back to English PDFs.
"""
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import config
from modules.multilingual import t

FONT_FILES = {"Telugu": "NotoSansTelugu-Regular.ttf", "Hindi": "NotoSansDevanagari-Regular.ttf"}
GREEN = colors.HexColor("#2E7D32")


def report_language_ok(language: str) -> bool:
    return language == "English" or (config.FONT_DIR / FONT_FILES[language]).exists()


def _font(language: str) -> str:
    if language == "English":
        return "Helvetica"
    name = f"Noto-{language}"
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(name, str(config.FONT_DIR / FONT_FILES[language])))
    return name


def _p(text, style):
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)


def build_pdf(report: dict, language: str = "English") -> bytes:
    """report keys: profile(dict), image_findings(dict|None), observations, possible_causes,
    recommendations, precautions (lists), references (list)."""
    if not report_language_ok(language):
        language = "English"
    f = _font(language)
    bold = "Helvetica-Bold" if language == "English" else f
    title = ParagraphStyle("title", fontName=bold, fontSize=20, textColor=GREEN, spaceAfter=4)
    h = ParagraphStyle("h", fontName=bold, fontSize=13, textColor=GREEN, spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("b", fontName=f, fontSize=10.5, leading=15)
    small = ParagraphStyle("s", fontName=f, fontSize=8.5, leading=11, textColor=colors.grey)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="AgriGen AI Report")
    story = [_p("AgriGen AI - " + t(language, "reports"), title),
             _p(datetime.now().strftime("%d %b %Y, %H:%M"), small), Spacer(1, 6)]

    profile = {k: v for k, v in (report.get("profile") or {}).items() if v}
    if profile:
        story.append(_p(t(language, "rep_profile"), h))
        tbl = Table([[_p(k, body), _p(v, body)] for k, v in profile.items()], colWidths=[5 * cm, 12 * cm])
        tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                                 ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F8E9")),
                                 ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(tbl)

    img = report.get("image_findings")
    if img:
        story.append(_p(t(language, "rep_image"), h))
        conds = "; ".join(f"{c.get('name')} ({c.get('confidence')})" for c in img.get("possible_conditions", []))
        lines = [f"Crop: {img.get('crop_identified', '-')}", f"Possible conditions: {conds or '-'}",
                 f"Severity: {img.get('severity_estimate', '-')}"]
        syms = img.get("visible_symptoms") or []
        if syms:
            lines.append("Symptoms: " + "; ".join(map(str, syms)))
        for line in lines:
            story.append(_p(line, body))

    for key, label in (("observations", "rep_obs"), ("possible_causes", "rep_causes"),
                       ("recommendations", "rep_reco"), ("precautions", "rep_prec")):
        items = report.get(key) or []
        if items:
            story.append(_p(t(language, label), h))
            for i, item in enumerate(items, 1):
                story.append(_p(f"{i}. {item}", body))
                story.append(Spacer(1, 2))

    refs = report.get("references") or []
    if refs:
        story.append(_p(t(language, "rep_refs"), h))
        for r in refs:
            story.append(_p(f"- {r}", body))

    story += [Spacer(1, 14), _p(t(language, "disclaimer"), small)]
    doc.build(story)
    return buf.getvalue()
