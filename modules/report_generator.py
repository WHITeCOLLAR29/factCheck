"""Reporting & Analytics Generator (`modules/report_generator.py`).

Computes real verification analytics from completed `FactCheckReport` results (never classifying
missing results as verified, and gracefully handling empty reports) and exports standalone
PDF reports (via ReportLab with drawn vector product logo, executive summary, statistics,
claim-by-claim findings, corrected facts, source URLs, and methodology limitations),
CSV spreadsheets (via Pandas), and structured JSON files.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

import pandas as pd

try:
    from reportlab.graphics.shapes import Circle, Drawing, Polygon, Rect
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False

try:
    import fitz  # PyMuPDF fallback when reportlab is not installed in bare environment
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

from config.settings import VERDICT_DISPLAY_LABELS
from models.schemas import (
    ClaimImportance,
    ClaimVerificationResult,
    EvidenceRelationship,
    FactCheckReport,
    VerificationStatus,
)


@dataclass
class ReportAnalytics:
    """Computed real-time analytics from a completed FactCheckReport."""

    total_claims: int = 0
    verified_count: int = 0
    inaccurate_count: int = 0
    false_count: int = 0
    unverifiable_count: int = 0
    overall_truth_score: float = 0.0
    average_confidence: float = 0.0
    average_source_credibility: float = 0.0
    verification_distribution: Dict[str, int] = field(default_factory=dict)
    verification_percentages: Dict[str, float] = field(default_factory=dict)
    category_counts: Dict[str, int] = field(default_factory=dict)
    high_risk_claims: List[ClaimVerificationResult] = field(default_factory=list)
    conflicting_evidence_claims: List[ClaimVerificationResult] = field(
        default_factory=list
    )


def has_conflicting_evidence(result: ClaimVerificationResult) -> bool:
    """Determine whether a claim has conflicting evidence across its retrieved sources."""
    rels = {ev.relationship for ev in result.evidence}
    if (
        EvidenceRelationship.SUPPORTS in rels
        and EvidenceRelationship.CONTRADICTS in rels
    ):
        return True
    if (
        EvidenceRelationship.PARTIALLY_SUPPORTS in rels
        and EvidenceRelationship.CONTRADICTS in rels
    ):
        return True

    has_sup = any(s.direct_support for s in result.sources)
    has_con = any(s.direct_contradiction for s in result.sources)
    if has_sup and has_con:
        return True

    exp_lower = (result.explanation or "").lower()
    return any(
        kw in exp_lower
        for kw in ("conflicting", "sources disagree", "contradictory sources", "discrepancy between sources")
    )


def is_high_risk_claim(result: ClaimVerificationResult) -> bool:
    """Determine whether a claim represents a High-Risk finding.

    A claim is High-Risk if:
    - Its status is FALSE, OR
    - Its status is INACCURATE and it has High importance or >= 0.65 confidence, OR
    - It has High importance and conflicting evidence.
    """
    if result.status == VerificationStatus.FALSE:
        return True
    if result.status == VerificationStatus.INACCURATE and (
        result.claim.importance == ClaimImportance.HIGH or result.confidence >= 0.65
    ):
        return True
    if (
        result.claim.importance == ClaimImportance.HIGH
        and has_conflicting_evidence(result)
    ):
        return True
    return False


def compute_report_analytics(report: Optional[FactCheckReport]) -> ReportAnalytics:
    """Calculate real metrics strictly from completed verification results.

    Never classifies missing or unverified results as verified, and handles empty reports cleanly.
    """
    empty_dist = {
        "Verified": 0,
        "Inaccurate": 0,
        "False": 0,
        "Unverifiable": 0,
    }
    empty_pct = {k: 0.0 for k in empty_dist}

    if report is None or not report.results:
        return ReportAnalytics(
            verification_distribution=empty_dist,
            verification_percentages=empty_pct,
        )

    results = report.results
    total = len(results)

    verified_cnt = 0
    inaccurate_cnt = 0
    false_cnt = 0
    unverifiable_cnt = 0
    cat_counts: Dict[str, int] = {}
    high_risk: List[ClaimVerificationResult] = []
    conflicting: List[ClaimVerificationResult] = []

    for res in results:
        # Never classify missing or unknown statuses as verified
        if res.status == VerificationStatus.VERIFIED and (res.evidence or res.sources):
            verified_cnt += 1
        elif res.status == VerificationStatus.INACCURATE:
            inaccurate_cnt += 1
        elif res.status == VerificationStatus.FALSE:
            false_cnt += 1
        else:
            unverifiable_cnt += 1

        cat_name = res.claim.category.value
        cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1

        if is_high_risk_claim(res):
            high_risk.append(res)
        if has_conflicting_evidence(res):
            conflicting.append(res)

    dist = {
        "Verified": verified_cnt,
        "Inaccurate": inaccurate_cnt,
        "False": false_cnt,
        "Unverifiable": unverifiable_cnt,
    }
    pcts = {
        k: round((v / total) * 100.0, 1) if total > 0 else 0.0
        for k, v in dist.items()
    }

    evaluated = verified_cnt + inaccurate_cnt + false_cnt
    if evaluated > 0:
        truth_score = round(((verified_cnt * 1.0 + inaccurate_cnt * 0.5) / evaluated) * 100.0, 1)
    else:
        truth_score = 0.0

    avg_conf = round(sum(r.confidence for r in results) / total, 3)
    avg_cred = round(sum(r.source_credibility_avg for r in results) / total, 3)

    return ReportAnalytics(
        total_claims=total,
        verified_count=verified_cnt,
        inaccurate_count=inaccurate_cnt,
        false_count=false_cnt,
        unverifiable_count=unverifiable_cnt,
        overall_truth_score=truth_score,
        average_confidence=avg_conf,
        average_source_credibility=avg_cred,
        verification_distribution=dist,
        verification_percentages=pcts,
        category_counts=cat_counts,
        high_risk_claims=high_risk,
        conflicting_evidence_claims=conflicting,
    )


def filter_claim_results(
    results: List[ClaimVerificationResult],
    selected_statuses: Optional[List[str]] = None,
    selected_categories: Optional[List[str]] = None,
    min_confidence_pct: float = 0.0,
    max_confidence_pct: float = 100.0,
    search_keyword: str = "",
    only_high_risk: bool = False,
    only_conflicting: bool = False,
) -> List[ClaimVerificationResult]:
    """Filter claim verification results by status, category, confidence range, keyword, high-risk, or conflict."""
    if not results:
        return []

    kw = (search_keyword or "").strip().lower()
    filtered: List[ClaimVerificationResult] = []

    for res in results:
        ui_status = VERDICT_DISPLAY_LABELS.get(res.status.value, res.status.value)
        raw_status = res.status.value

        if selected_statuses is not None:
            if ui_status not in selected_statuses and raw_status not in selected_statuses:
                continue

        if selected_categories is not None:
            if res.claim.category.value not in selected_categories:
                continue

        conf_pct = round(res.confidence * 100.0, 1)
        if conf_pct < min_confidence_pct or conf_pct > max_confidence_pct:
            continue

        if only_high_risk and not is_high_risk_claim(res):
            continue

        if only_conflicting and not has_conflicting_evidence(res):
            continue

        if kw:
            source_haystack = " ".join(
                f"{s.title} {s.domain} {s.url} {s.snippet}" for s in res.sources
            ).lower()
            ev_haystack = " ".join(
                f"{e.source_title} {e.source_domain} {e.source_url} {e.evidence_snippet}"
                for e in res.evidence
            ).lower()
            combined = " ".join(
                [
                    res.claim_id.lower(),
                    res.claim.claim_text.lower(),
                    res.claim.entity_or_subject.lower(),
                    (res.claim.date_or_period or "").lower(),
                    (res.claim.numeric_value or "").lower(),
                    res.explanation.lower(),
                    (res.corrected_fact or "").lower(),
                    source_haystack,
                    ev_haystack,
                ]
            )
            if kw not in combined:
                continue

        filtered.append(res)

    return filtered


def build_claims_dataframe(results: List[ClaimVerificationResult]) -> pd.DataFrame:
    """Build a structured Pandas DataFrame for dashboard tables and CSV export from actual results."""
    columns = [
        "Claim ID",
        "Page",
        "Category",
        "Status",
        "Confidence (%)",
        "High Risk",
        "Conflicting Evidence",
        "Original Statement",
        "Corrected Information",
        "Explanation",
        "Supporting Sources",
        "Publication Dates",
        "Source URLs",
    ]
    if not results:
        return pd.DataFrame(columns=columns)

    rows = []
    for res in results:
        ui_status = VERDICT_DISPLAY_LABELS.get(res.status.value, res.status.value)

        if res.evidence:
            domains = ", ".join(
                f"{ev.source_domain} ({ev.relationship.value})" for ev in res.evidence[:4]
            )
            pub_dates = ", ".join(
                ev.source_date for ev in res.evidence[:4] if ev.source_date
            ) or "Not listed"
            urls = " | ".join(ev.source_url for ev in res.evidence[:4])
        else:
            domains = ", ".join(s.domain for s in res.sources[:4]) or "None"
            pub_dates = ", ".join(
                s.published_date for s in res.sources[:4] if s.published_date
            ) or "Not listed"
            urls = " | ".join(s.url for s in res.sources[:4])

        rows.append(
            {
                "Claim ID": res.claim_id,
                "Page": res.claim.page_number,
                "Category": res.claim.category.value,
                "Status": ui_status,
                "Confidence (%)": round(res.confidence * 100.0, 1),
                "High Risk": "Yes" if is_high_risk_claim(res) else "No",
                "Conflicting Evidence": "Yes" if has_conflicting_evidence(res) else "No",
                "Original Statement": res.original_claim or res.claim.claim_text,
                "Corrected Information": res.corrected_fact or "—",
                "Explanation": res.explanation,
                "Supporting Sources": domains,
                "Publication Dates": pub_dates,
                "Source URLs": urls,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _safe_xml(text: object) -> str:
    """Escape text safely for ReportLab XML Paragraph tags."""
    if text is None:
        return ""
    return escape(str(text))


def _verdict_hex_color(status: VerificationStatus) -> colors.Color:
    """Return ReportLab Color matching the Material 3 claim verification status."""
    mapping = {
        VerificationStatus.VERIFIED: colors.HexColor("#16845B"),
        VerificationStatus.INACCURATE: colors.HexColor("#B7791F"),
        VerificationStatus.FALSE: colors.HexColor("#C43D4D"),
        VerificationStatus.UNVERIFIABLE: colors.HexColor("#667085"),
    }
    return mapping.get(status, colors.HexColor("#667085"))


def _build_brand_logo_drawing() -> Drawing:
    """Create a vector Material 3 shield-check product logo badge for the PDF header."""
    d = Drawing(42, 42)
    # Rounded square badge in Material 3 Primary (#3659C9)
    d.add(
        Rect(
            0,
            0,
            40,
            40,
            rx=10,
            ry=10,
            fillColor=colors.HexColor("#3659C9"),
            strokeColor=None,
        )
    )
    # Inner shield emblem
    d.add(
        Polygon(
            [20, 33, 31, 28, 31, 17, 20, 8, 9, 17, 9, 28],
            fillColor=colors.HexColor("#E5ECFF"),
            strokeColor=None,
        )
    )
    # Center verification dot / check mark accent
    d.add(
        Circle(
            20,
            20,
            5,
            fillColor=colors.HexColor("#3659C9"),
            strokeColor=None,
        )
    )
    return d


def _draw_page_footer(canvas, doc) -> None:
    """Draw page number and product attribution footer on every PDF page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(
        0.6 * inch,
        0.4 * inch,
        "FactCheck AI – The Truth Layer for Every Document | Automated Evidence Verification Audit",
    )
    canvas.drawRightString(
        letter[0] - 0.6 * inch,
        0.4 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def _generate_pdf_report_pymupdf_fallback(
    report: FactCheckReport,
    analytics: ReportAnalytics,
) -> bytes:
    """Fallback multi-page PDF generator using PyMuPDF (fitz) if ReportLab is not yet installed in the environment."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 48.0

    def _write_block(text: str, fontsize: float = 10.0, bold: bool = False, color: tuple = (0.12, 0.15, 0.23)) -> None:
        nonlocal page, y
        lines = text.splitlines() or [text]
        for raw_line in lines:
            # Wrap long lines to ~88 chars
            words = raw_line.split(" ")
            current = ""
            wrapped: List[str] = []
            for w in words:
                if len(current) + len(w) + 1 > 88 and current:
                    wrapped.append(current)
                    current = w
                else:
                    current = f"{current} {w}".strip() if current else w
            if current:
                wrapped.append(current)
            for wl in (wrapped or [""]):
                if y > 735:
                    page = doc.new_page(width=612, height=792)
                    y = 48.0
                fontname = "hebo" if bold else "helv"
                page.insert_text((45, y), wl, fontsize=fontsize, fontname=fontname, color=color)
                y += fontsize + 4.5

    # Draw vector logo badge at top left
    page.draw_rect(fitz.Rect(45, 36, 75, 66), color=(0.21, 0.35, 0.79), fill=(0.21, 0.35, 0.79))
    page.insert_text((53, 56), "FC", fontsize=12, fontname="hebo", color=(1, 1, 1))
    page.insert_text((85, 50), "FactCheck AI – The Truth Layer", fontsize=16, fontname="hebo", color=(0.12, 0.15, 0.23))
    page.insert_text(
        (85, 64),
        f"Report ID: {report.report_id} | Date: {report.generated_at} | Model: {report.model_used}",
        fontsize=8.5,
        fontname="helv",
        color=(0.40, 0.44, 0.52),
    )
    y = 86.0

    _write_block("1. Executive Summary & Document Profile", fontsize=12, bold=True, color=(0.21, 0.35, 0.79))
    _write_block(
        f"Document Name: {report.document.filename} | Pages: {report.document.readable_page_count}/{report.document.page_count} | "
        f"Words: {report.document.total_words:,} | SHA-256: {report.document.document_hash}"
    )
    _write_block(f"Primary Topic: {report.document_topic}")
    _write_block(
        f"Truth Score: {analytics.overall_truth_score:.1f}% | Total Claims: {analytics.total_claims} | "
        f"Verified: {analytics.verified_count} | Inaccurate: {analytics.inaccurate_count} | "
        f"False: {analytics.false_count} | Unverifiable: {analytics.unverifiable_count} | "
        f"High-Risk: {len(analytics.high_risk_claims)} | Conflicting: {len(analytics.conflicting_evidence_claims)}"
    )
    y += 6

    _write_block("2. Verification Statistics", fontsize=12, bold=True, color=(0.21, 0.35, 0.79))
    _write_block(
        f"Mean Confidence: {analytics.average_confidence * 100:.0f}% | "
        f"Mean Source Credibility: {analytics.average_source_credibility * 100:.0f}%"
    )
    y += 6

    _write_block("3. Claim-by-Claim Verification Findings", fontsize=12, bold=True, color=(0.21, 0.35, 0.79))
    if not report.results:
        _write_block("No claim verification findings are present in this report.")
    else:
        for res in report.results:
            claim = res.claim
            ui_status = VERDICT_DISPLAY_LABELS.get(res.status.value, res.status.value).upper()
            _write_block(
                f"[{res.claim_id}] Page {claim.page_number} ({claim.category.value}) — {ui_status} "
                f"(Confidence: {res.confidence * 100:.0f}%)",
                bold=True,
            )
            _write_block(f"Original Statement: \"{res.original_claim or claim.claim_text}\"")
            if res.corrected_fact:
                _write_block(f"Corrected Information: {res.corrected_fact}", bold=True, color=(0.72, 0.47, 0.12))
            _write_block(f"Explanation: {res.explanation}")
            for ev in res.evidence[:4]:
                d_str = f" ({ev.source_date})" if ev.source_date else ""
                _write_block(
                    f"  Source [{ev.relationship.value}]: {ev.source_domain}{d_str} - {ev.source_title} | {ev.source_url}",
                    fontsize=8.5,
                    color=(0.21, 0.35, 0.79),
                )
            y += 6

    _write_block("4. Important Methodology & Limitations", fontsize=12, bold=True, color=(0.21, 0.35, 0.79))
    _write_block(
        "Verdicts are grounded strictly in live public web sources retrieved at the time of analysis. "
        "Claims lacking sufficient public evidence are classified as Unverifiable rather than Verified or False.",
        fontsize=8.5,
        color=(0.40, 0.44, 0.52),
    )

    out_bytes = doc.tobytes()
    doc.close()
    return out_bytes


def generate_pdf_report(report: FactCheckReport) -> bytes:
    """Generate a professionally formatted, standalone PDF verification report using ReportLab Platypus
    (with automatic PyMuPDF fallback if ReportLab is not yet installed in the active environment).
    """
    analytics = compute_report_analytics(report)
    if not _HAS_REPORTLAB:
        if _HAS_FITZ:
            return _generate_pdf_report_pymupdf_fallback(report, analytics)
        raise RuntimeError("Neither ReportLab nor PyMuPDF is installed for PDF generation.")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.65 * inch,
        title=f"FactCheck AI Report - {report.document.filename}",
        author="FactCheck AI – The Truth Layer",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        textColor=colors.HexColor("#20283B"),
        spaceAfter=2,
    )
    tagline_style = ParagraphStyle(
        "ReportTagline",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#667085"),
        spaceAfter=4,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#3659C9"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#20283B"),
        spaceAfter=5,
    )
    small_style = ParagraphStyle(
        "SmallTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#667085"),
        spaceAfter=3,
    )

    story = []

    # 1. Product Logo & Brand Header Table
    logo_drawing = _build_brand_logo_drawing()
    brand_text = [
        Paragraph("FactCheck AI", title_style),
        Paragraph(
            "<b>The Truth Layer for Every Document.</b> &mdash; AI-powered document verification backed by real-world evidence.",
            tagline_style,
        ),
        Paragraph(
            f"Report ID: <b>{_safe_xml(report.report_id)}</b> &nbsp;|&nbsp; "
            f"Analysis Date: <b>{_safe_xml(report.generated_at)}</b> &nbsp;|&nbsp; "
            f"Model: <b>{_safe_xml(report.model_used)}</b>",
            small_style,
        ),
    ]
    header_table = Table(
        [[logo_drawing, brand_text]],
        colWidths=[0.65 * inch, 6.65 * inch],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(header_table)
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.2,
            color=colors.HexColor("#DCE2F2"),
            spaceAfter=10,
        )
    )

    # 2. Document Information & Executive Summary
    story.append(Paragraph("1. Executive Summary & Document Profile", section_heading))
    story.append(
        Paragraph(
            f"<b>Document Name:</b> {_safe_xml(report.document.filename)} &nbsp;|&nbsp; "
            f"<b>Pages Analyzed:</b> {report.document.readable_page_count} of {report.document.page_count} &nbsp;|&nbsp; "
            f"<b>Word Count:</b> {report.document.total_words:,} &nbsp;|&nbsp; "
            f"<b>SHA-256:</b> <font name='Courier'>{_safe_xml(report.document.document_hash)}</font>",
            body_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Primary Document Topic:</b> {_safe_xml(report.document_topic)}",
            body_style,
        )
    )

    if analytics.total_claims == 0:
        exec_summary_text = (
            "No objectively verifiable factual claims were extracted from this document, or no claims "
            "were present in the analyzed pages. Missing results are never classified as verified."
        )
    else:
        exec_summary_text = (
            f"FactCheck AI evaluated <b>{analytics.total_claims} factual claim(s)</b> extracted from "
            f"<b>{_safe_xml(report.document.filename)}</b> against live web evidence. The document achieved an "
            f"overall Truth Score of <b>{analytics.overall_truth_score:.1f}%</b> with an average evidence "
            f"confidence of <b>{analytics.average_confidence * 100:.0f}%</b>. Among the analyzed statements, "
            f"<b>{analytics.verified_count}</b> ({analytics.verification_percentages['Verified']:.1f}%) were verified, "
            f"<b>{analytics.inaccurate_count}</b> ({analytics.verification_percentages['Inaccurate']:.1f}%) were partially inaccurate or outdated, "
            f"<b>{analytics.false_count}</b> ({analytics.verification_percentages['False']:.1f}%) were directly contradicted (False), and "
            f"<b>{analytics.unverifiable_count}</b> ({analytics.verification_percentages['Unverifiable']:.1f}%) were unverifiable from public sources. "
            f"The audit flagged <b>{len(analytics.high_risk_claims)} high-risk claim(s)</b> and "
            f"<b>{len(analytics.conflicting_evidence_claims)} claim(s) with conflicting evidence</b>."
        )
    story.append(Paragraph(exec_summary_text, body_style))

    if report.document.extraction_warnings:
        for warn in report.document.extraction_warnings:
            story.append(
                Paragraph(
                    f"<font color='#B7791F'><b>Document Extraction Note:</b> {_safe_xml(warn)}</font>",
                    small_style,
                )
            )
    story.append(Spacer(1, 6))

    # 3. Verification Statistics Table
    story.append(Paragraph("2. Verification Statistics", section_heading))
    summary_data = [
        [
            "Truth Score",
            "Total Claims",
            "Verified",
            "Inaccurate",
            "False",
            "Unverifiable",
            "High-Risk",
            "Conflicting",
        ],
        [
            f"{analytics.overall_truth_score:.1f}%",
            str(analytics.total_claims),
            str(analytics.verified_count),
            str(analytics.inaccurate_count),
            str(analytics.false_count),
            str(analytics.unverifiable_count),
            str(len(analytics.high_risk_claims)),
            str(len(analytics.conflicting_evidence_claims)),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[0.91 * inch] * 8)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3659C9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F4F6FC")),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#20283B")),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, 1), 10.5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2F2")),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # 4. Detailed Claim-by-Claim Findings
    story.append(
        Paragraph("3. Claim-by-Claim Verification Findings", section_heading)
    )

    if not report.results:
        story.append(
            Paragraph(
                "No claim verification findings are present in this report.",
                body_style,
            )
        )
    else:
        for res in report.results:
            claim = res.claim
            badge_color = _verdict_hex_color(res.status)
            ui_status_label = VERDICT_DISPLAY_LABELS.get(res.status.value, res.status.value).upper()
            risk_flag = " &nbsp;|&nbsp; <font color='#C43D4D'><b>[HIGH RISK]</b></font>" if is_high_risk_claim(res) else ""
            conflict_flag = " &nbsp;|&nbsp; <font color='#B7791F'><b>[CONFLICTING SOURCES]</b></font>" if has_conflicting_evidence(res) else ""

            claim_elements = []
            header_html = (
                f"<b>[{_safe_xml(res.claim_id)}] Page {claim.page_number} "
                f"&bull; {_safe_xml(claim.category.value)}</b> &nbsp;&mdash;&nbsp; "
                f"<font color='{badge_color.hexval()}'><b>{_safe_xml(ui_status_label)}</b></font> "
                f"(Confidence: {res.confidence * 100:.0f}% | "
                f"Source Authority: {res.source_credibility_avg * 100:.0f}%)"
                f"{risk_flag}{conflict_flag}"
            )
            claim_elements.append(Paragraph(header_html, body_style))
            claim_elements.append(
                Paragraph(
                    f"<b>Original Statement:</b> &ldquo;{_safe_xml(res.original_claim or claim.claim_text)}&rdquo;",
                    body_style,
                )
            )

            if claim.surrounding_context:
                claim_elements.append(
                    Paragraph(
                        f"<i>PDF Context (Page {claim.page_number}):</i> &ldquo;{_safe_xml(claim.surrounding_context)}&rdquo;",
                        small_style,
                    )
                )

            if res.corrected_fact:
                claim_elements.append(
                    Paragraph(
                        f"<font color='#B7791F'><b>Corrected Information:</b> {_safe_xml(res.corrected_fact)}</font>",
                        body_style,
                    )
                )

            claim_elements.append(
                Paragraph(
                    f"<b>Verification Explanation:</b> {_safe_xml(res.explanation)}",
                    body_style,
                )
            )

            # Supporting / Contradicting Sources with clickable URLs and publication dates
            if res.evidence:
                src_lines = []
                for s_idx, ev in enumerate(res.evidence[:5], start=1):
                    date_part = (
                        f" &bull; Published: {_safe_xml(ev.source_date)}"
                        if ev.source_date
                        else " &bull; Date unstated"
                    )
                    safe_url = _safe_xml(ev.source_url)
                    src_lines.append(
                        f"[{s_idx}] <b>{_safe_xml(ev.source_domain)}</b>{date_part} "
                        f"[<i>{_safe_xml(ev.relationship.value)}</i>]: "
                        f"{_safe_xml(ev.source_title)}<br/>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;<a href='{safe_url}' color='#3659C9'>{safe_url}</a>"
                    )
                claim_elements.append(
                    Paragraph(
                        "<b>Supporting &amp; Evaluated Sources:</b><br/>" + "<br/>".join(src_lines),
                        small_style,
                    )
                )
            elif res.sources:
                src_lines = []
                for s_idx, src in enumerate(res.sources[:5], start=1):
                    date_part = (
                        f" &bull; Published: {_safe_xml(src.published_date)}"
                        if src.published_date
                        else " &bull; Date unstated"
                    )
                    safe_url = _safe_xml(src.url)
                    src_lines.append(
                        f"[{s_idx}] <b>{_safe_xml(src.domain)}</b>{date_part} "
                        f"({_safe_xml(src.source_type)}): {_safe_xml(src.title)}<br/>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;<a href='{safe_url}' color='#3659C9'>{safe_url}</a>"
                    )
                claim_elements.append(
                    Paragraph(
                        "<b>Supporting &amp; Evaluated Sources:</b><br/>" + "<br/>".join(src_lines),
                        small_style,
                    )
                )
            else:
                claim_elements.append(
                    Paragraph(
                        "<i>Supporting Sources:</i> No external web sources matched this claim.",
                        small_style,
                    )
                )

            claim_elements.append(Spacer(1, 5))
            claim_elements.append(
                HRFlowable(
                    width="100%",
                    thickness=0.6,
                    color=colors.HexColor("#DCE2F2"),
                    spaceAfter=7,
                )
            )
            story.append(KeepTogether(claim_elements))

    # 5. Important Limitations & Methodology Disclaimer
    story.append(
        Paragraph("4. Important Methodology & Limitations", section_heading)
    )
    limitations_text = (
        "1. <b>Evidence-Bounded Verification:</b> Verdicts are grounded strictly in live public web sources "
        "retrieved at the time of analysis ({gen_at}). Private, paywalled, or offline internal records cannot be "
        "inspected.<br/>"
        "2. <b>Distinction of Unverifiable Claims:</b> Claims lacking sufficient independent public evidence are "
        "classified as <i>Unverifiable</i> rather than <i>Verified</i> or <i>False</i>.<br/>"
        "3. <b>Temporal Snapshot:</b> Factual accuracy reflects the reporting periods cited in the PDF and the "
        "publication dates of retrieved sources as of the analysis date.<br/>"
        "4. <b>Advisory Audit Use:</b> This report is an automated forensic decision-support tool and does not "
        "constitute formal legal, financial, or regulatory certification."
    ).format(gen_at=_safe_xml(report.generated_at))
    story.append(Paragraph(limitations_text, small_style))

    doc.build(story, onFirstPage=_draw_page_footer, onLaterPages=_draw_page_footer)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_csv_export(report: Optional[FactCheckReport]) -> bytes:
    """Convert FactCheckReport results into a UTF-8 CSV byte buffer using Pandas (handles empty reports)."""
    results = report.results if report and report.results else []
    df = build_claims_dataframe(results)
    return df.to_csv(index=False).encode("utf-8")


def generate_json_export(report: Optional[FactCheckReport]) -> bytes:
    """Serialize the FactCheckReport including analytics and canonical claim verification objects to JSON bytes."""
    analytics = compute_report_analytics(report)
    if report is None:
        payload = {
            "report_id": "EMPTY",
            "total_claims": 0,
            "claims": [],
        }
        return json.dumps(payload, indent=2).encode("utf-8")

    payload = {
        "report_id": report.report_id,
        "document_filename": report.document.filename,
        "document_topic": report.document_topic,
        "generated_at": report.generated_at,
        "model_used": report.model_used,
        "overall_truth_score": analytics.overall_truth_score,
        "statistics": {
            "total_claims": analytics.total_claims,
            "verified_count": analytics.verified_count,
            "inaccurate_count": analytics.inaccurate_count,
            "false_count": analytics.false_count,
            "unverifiable_count": analytics.unverifiable_count,
            "high_risk_count": len(analytics.high_risk_claims),
            "conflicting_evidence_count": len(analytics.conflicting_evidence_claims),
            "average_confidence": analytics.average_confidence,
            "average_source_credibility": analytics.average_source_credibility,
        },
        "claims": [res.to_canonical_dict() for res in report.results],
    }
    return json.dumps(payload, indent=2).encode("utf-8")
