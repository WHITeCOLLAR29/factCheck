"""Pages 1 & 2: Home / Dashboard and Document Analysis workspace components (Material Design 3)."""

from __future__ import annotations

from typing import Callable, List, Optional

import streamlit as st

from components.sidebar import SidebarSettings, navigate_to
from config.settings import CONFIG
from models.schemas import FactCheckReport
from modules.pdf_extractor import (
    ExtractedPDFContent,
    PDFExtractionError,
    extract_pdf_content,
)
from utils.helpers import format_file_size


def _process_uploaded_file(uploaded_file) -> Optional[ExtractedPDFContent]:
    """Extract and store PDF content in session state when a file is uploaded."""
    if uploaded_file is None:
        return st.session_state.get("extracted_pdf")

    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    cached_pdf: Optional[ExtractedPDFContent] = st.session_state.get("extracted_pdf")
    if (
        cached_pdf is not None
        and cached_pdf.metadata.filename == filename
        and cached_pdf.metadata.file_size_bytes == len(file_bytes)
    ):
        return cached_pdf

    try:
        with st.spinner(f"Reading and analyzing page layers in `{filename}`..."):
            extracted = extract_pdf_content(
                file_bytes=file_bytes,
                filename=filename,
            )
            st.session_state["extracted_pdf"] = extracted
            if st.session_state.get("last_file_hash") != extracted.metadata.document_hash:
                from utils.security import clear_sensitive_session_data

                clear_sensitive_session_data(st.session_state, keep_recent_history=True)
                st.session_state.pop("last_run_signature", None)
                st.session_state["extracted_pdf"] = extracted
                st.session_state["last_file_hash"] = extracted.metadata.document_hash
            return extracted
    except PDFExtractionError as pdf_err:
        st.error(f"**PDF Validation / Extraction Error:** {pdf_err}")
        return None
    except Exception as exc:
        st.error(f"**Unexpected error reading PDF:** {exc}")
        return None


def _render_extraction_diagnostics(extracted_pdf: ExtractedPDFContent) -> None:
    """Display explicit warnings for scanned, OCR-recovered, or blank pages so no page is silently ignored."""
    meta = extracted_pdf.metadata
    if not meta.extraction_warnings:
        return

    for warn_msg in meta.extraction_warnings:
        if warn_msg.startswith("Unreadable Scanned"):
            st.warning(f"⚠️ **{warn_msg}**")
        elif warn_msg.startswith("OCR Recovery"):
            st.info(f"🔎 **{warn_msg}**")
        else:
            st.caption(f"ℹ️ {warn_msg}")


def render_home_page(settings: SidebarSettings) -> None:
    """Render PAGE 1: HOME / DASHBOARD following Material Design 3 guidelines."""
    st.markdown(
        f"""
        <div class="m3-card" style="padding: 2rem 2.25rem; margin-bottom: 1.5rem;">
            <div style="display:inline-flex; align-items:center; gap:0.45rem; padding:0.3rem 0.85rem; border-radius:999px; background-color:var(--m3-primary-container); color:var(--m3-primary); font-size:0.8rem; font-weight:600; margin-bottom:0.9rem;">
                <span class="ms-icon" style="font-size:16px;">shield_lock</span>
                {CONFIG.APP_TAGLINE}
            </div>
            <h1 style="margin:0 0 0.5rem 0; font-size:2.1rem; font-weight:700; letter-spacing:-0.02em; color:var(--m3-text);">
                Your documents, verified with evidence.
            </h1>
            <p style="margin:0; font-size:1.05rem; color:var(--m3-muted); max-width:740px; line-height:1.55;">
                {CONFIG.APP_DESCRIPTION} Upload research reports, financial filings, policy briefs, or articles to automatically extract factual claims and validate them against authoritative live web sources.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_spec = st.columns([7, 5], gap="large")

    with col_upload:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                <span class="ms-icon" style="color:var(--m3-primary);">upload_file</span>
                <span style="font-size:1.05rem; font-weight:600;">Upload PDF Document</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drag and drop a PDF document here, or click to browse",
            type=["pdf"],
            accept_multiple_files=False,
            key="home_pdf_uploader",
        )
        extracted_pdf = _process_uploaded_file(uploaded_file)

        if extracted_pdf is not None:
            meta = extracted_pdf.metadata
            st.markdown(
                f"""
                <div class="m3-tonal-card" style="margin-top:0.75rem; display:flex; align-items:center; justify-content:space-between;">
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        <span class="ms-icon" style="color:var(--m3-success); font-size:24px;">task_alt</span>
                        <div>
                            <div style="font-weight:600; font-size:0.95rem;">{meta.filename}</div>
                            <div style="font-size:0.8rem; color:var(--m3-muted);">
                                {format_file_size(meta.file_size_bytes)} &bull; {meta.readable_page_count}/{meta.page_count} Readable Pages &bull; {meta.total_words:,} Words
                            </div>
                        </div>
                    </div>
                    <span class="m3-chip m3-chip-verified">Ready</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_extraction_diagnostics(extracted_pdf)

        btn_disabled = extracted_pdf is None
        if st.button(
            "Start Fact Checking",
            type="primary",
            disabled=btn_disabled,
            use_container_width=True,
        ):
            navigate_to("Document Analysis")
            st.rerun()

        if btn_disabled:
            st.caption("Upload a valid PDF file above to enable fact checking.")

    with col_spec:
        st.markdown(
            f"""
            <div class="m3-tonal-card" style="height:100%;">
                <div style="display:flex; align-items:center; gap:0.5rem; font-weight:600; font-size:0.98rem; margin-bottom:0.75rem;">
                    <span class="ms-icon" style="color:var(--m3-primary);">info</span>
                    Supported Document Specifications
                </div>
                <ul style="margin:0; padding-left:1.2rem; color:var(--m3-muted); font-size:0.88rem; line-height:1.75;">
                    <li><strong>File Format:</strong> Standard PDF documents (<code>.pdf</code>)</li>
                    <li><strong>Maximum File Size:</strong> Up to {CONFIG.MAX_PDF_SIZE_MB} MB per document</li>
                    <li><strong>Page Capacity:</strong> Up to {CONFIG.MAX_PDF_PAGES} pages analyzed per run</li>
                    <li><strong>Scanned PDF Support:</strong> Automatic Tesseract OCR fallback with explicit per-page readability reporting</li>
                    <li><strong>Security:</strong> Encrypted/password-protected PDFs are rejected without authorization</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Feature Cards: Extract, Verify, Report
    st.markdown(
        "<div style='font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:var(--m3-muted); margin-bottom:0.75rem;'>Verification Workflow</div>",
        unsafe_allow_html=True,
    )
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        st.markdown(
            """
            <div class="m3-feature-card">
                <div class="m3-icon-circle"><span class="ms-icon">manage_search</span></div>
                <div style="font-weight:600; font-size:1.05rem; margin-bottom:0.4rem;">1. Extract</div>
                <div style="font-size:0.88rem; color:var(--m3-muted); line-height:1.55;">
                    Isolates verifiable statistics, financial figures, historical dates, technical specifications, and scientific claims while preserving exact claim text and paragraph context.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div class="m3-feature-card">
                <div class="m3-icon-circle"><span class="ms-icon">travel_explore</span></div>
                <div style="font-weight:600; font-size:1.05rem; margin-bottom:0.4rem;">2. Verify</div>
                <div style="font-size:0.88rem; color:var(--m3-muted); line-height:1.55;">
                    Queries live internet sources via Tavily Search, evaluates domain authority, and cross-checks each claim using structured Gemini reasoning.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
            <div class="m3-feature-card">
                <div class="m3-icon-circle"><span class="ms-icon">assignment_turned_in</span></div>
                <div style="font-weight:600; font-size:1.05rem; margin-bottom:0.4rem;">3. Report</div>
                <div style="font-size:0.88rem; color:var(--m3-muted); line-height:1.55;">
                    Generates interactive visual analytics, source-linked claim cards, factual corrections, and downloadable PDF &amp; CSV audit reports.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Recent Analysis Section
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.75rem;">
            <span class="ms-icon" style="color:var(--m3-primary);">history</span>
            <span style="font-size:1.1rem; font-weight:600;">Recent Analysis</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recent_reports: List[FactCheckReport] = st.session_state.get("recent_reports", [])
    if not recent_reports:
        st.markdown(
            """
            <div class="m3-card" style="text-align:center; padding: 2rem;">
                <span class="ms-icon" style="font-size:32px; color:var(--m3-muted); margin-bottom:0.5rem;">folder_open</span>
                <div style="font-weight:600; font-size:0.98rem; margin-bottom:0.25rem;">No documents analyzed in this session yet</div>
                <div style="font-size:0.86rem; color:var(--m3-muted);">
                    Upload a PDF above and click <strong>Start Fact Checking</strong> to run your first verification audit.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for idx, rep in enumerate(recent_reports[:5]):
            r_col1, r_col2, r_col3, r_col4 = st.columns([4, 3, 3, 2])
            with r_col1:
                st.markdown(
                    f"**{rep.document.filename}**  \n"
                    f"<span style='font-size:0.8rem; color:var(--m3-muted);'>ID: {rep.report_id} &bull; {rep.generated_at}</span>",
                    unsafe_allow_html=True,
                )
            with r_col2:
                st.markdown(
                    f"<span style='font-size:0.85rem; color:var(--m3-muted);'>Truth Score:</span> **{rep.overall_truth_score:.1f}%** ({rep.total_claims} claims)",
                    unsafe_allow_html=True,
                )
            with r_col3:
                st.markdown(
                    f"<span class='m3-chip m3-chip-verified'>{rep.verified_count} Verified</span> "
                    f"<span class='m3-chip m3-chip-inaccurate'>{rep.partially_true_count} Inaccurate</span> "
                    f"<span class='m3-chip m3-chip-false'>{rep.refuted_count} False</span>",
                    unsafe_allow_html=True,
                )
            with r_col4:
                if st.button("Open Report", key=f"open_recent_{rep.report_id}_{idx}", use_container_width=True):
                    st.session_state["factcheck_report"] = rep
                    navigate_to("Analysis Dashboard")
                    st.rerun()


def render_document_analysis_page(
    settings: SidebarSettings,
    run_pipeline_callback: Callable[[ExtractedPDFContent, SidebarSettings], None],
) -> None:
    """Render PAGE 2: DOCUMENT ANALYSIS with metadata summary, page diagnostics, preview, and 5-step progress."""
    st.markdown(
        """
        <div style="margin-bottom:1.25rem;">
            <h2 style="margin:0 0 0.25rem 0; font-size:1.75rem; font-weight:700;">Document Analysis</h2>
            <p style="margin:0; color:var(--m3-muted); font-size:0.95rem;">
                Inspect extracted document paragraphs, review page readability diagnostics, and launch the live evidence audit.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    extracted_pdf: Optional[ExtractedPDFContent] = st.session_state.get("extracted_pdf")

    with st.expander(
        "Upload or Replace PDF Document" if extracted_pdf else "Upload a PDF Document to Begin",
        expanded=(extracted_pdf is None),
    ):
        uploaded_file = st.file_uploader(
            "Select a PDF file",
            type=["pdf"],
            accept_multiple_files=False,
            key="analysis_page_uploader",
        )
        if uploaded_file is not None:
            extracted_pdf = _process_uploaded_file(uploaded_file)

    if extracted_pdf is None:
        st.markdown(
            """
            <div class="m3-card" style="text-align:center; padding: 2.5rem 1.5rem;">
                <span class="ms-icon" style="font-size:36px; color:var(--m3-muted); margin-bottom:0.6rem;">description</span>
                <div style="font-weight:600; font-size:1.05rem; margin-bottom:0.35rem;">No PDF Document Loaded</div>
                <div style="font-size:0.9rem; color:var(--m3-muted); max-width:480px; margin:0 auto;">
                    Please upload a PDF document above or return to the Home page to select a file for verification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    meta = extracted_pdf.metadata
    _render_extraction_diagnostics(extracted_pdf)

    # Document Summary Metric Cards
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        st.markdown(
            f"""
            <div class="m3-kpi-card">
                <div class="m3-kpi-label"><span class="ms-icon" style="font-size:17px;">draft</span>Filename</div>
                <div class="m3-kpi-value" style="font-size:1.1rem; word-break:break-word;">{meta.filename}</div>
                <div class="m3-kpi-sub">SHA-256: {meta.document_hash[:12]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="m3-kpi-card">
                <div class="m3-kpi-label"><span class="ms-icon" style="font-size:17px;">hard_drive</span>File Size</div>
                <div class="m3-kpi-value">{format_file_size(meta.file_size_bytes)}</div>
                <div class="m3-kpi-sub">OCR Engine: {"Active" if meta.ocr_engine_available else "Unavailable"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        unreadable_cnt = len(meta.scanned_unreadable_pages)
        sub_page_info = (
            f"{unreadable_cnt} scanned page(s) unreadable"
            if unreadable_cnt > 0
            else f"{len(meta.ocr_pages)} OCR / {len(extracted_pdf.pages)} native"
        )
        st.markdown(
            f"""
            <div class="m3-kpi-card">
                <div class="m3-kpi-label"><span class="ms-icon" style="font-size:17px;">auto_stories</span>Page Count</div>
                <div class="m3-kpi-value">{len(extracted_pdf.pages)} / {meta.page_count}</div>
                <div class="m3-kpi-sub">{sub_page_info}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="m3-kpi-card">
                <div class="m3-kpi-label"><span class="ms-icon" style="font-size:17px;">notes</span>Word Count</div>
                <div class="m3-kpi-value">{meta.total_words:,}</div>
                <div class="m3-kpi-sub">{len(extracted_pdf.paragraphs)} paragraphs preserved</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    col_preview, col_action = st.columns([7, 5], gap="large")

    with col_preview:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.45rem; font-weight:600; font-size:1rem; margin-bottom:0.5rem;">
                <span class="ms-icon" style="color:var(--m3-primary);">preview</span>
                Extracted Paragraph Preview (by Page)
            </div>
            """,
            unsafe_allow_html=True,
        )
        page_labels = [
            f"Page {p_num}{' (OCR)' if p_num in meta.ocr_pages else ''}"
            for p_num, _ in extracted_pdf.pages
        ]
        selected_label = st.selectbox(
            "Select Page",
            options=page_labels,
            index=0,
            label_visibility="collapsed",
        )
        sel_idx = page_labels.index(selected_label)
        _, page_content = extracted_pdf.pages[sel_idx]
        st.text_area(
            "Page Text Content",
            value=page_content,
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )

    with col_action:
        st.markdown(
            f"""
            <div class="m3-card">
                <div style="display:flex; align-items:center; gap:0.5rem; font-weight:600; font-size:1.05rem; margin-bottom:0.65rem;">
                    <span class="ms-icon" style="color:var(--m3-primary);">fact_check</span>
                    Verification Configuration
                </div>
                <div style="font-size:0.88rem; color:var(--m3-muted); line-height:1.65; margin-bottom:0.75rem;">
                    <div>&bull; <strong>AI Model:</strong> <code>{settings.selected_model}</code></div>
                    <div>&bull; <strong>Claims Target:</strong> Up to <strong>{settings.max_claims} claims</strong></div>
                    <div>&bull; <strong>Search Engine:</strong> Tavily Live Web (<code>{settings.search_depth}</code>, {settings.max_sources_per_claim} sources/claim)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not settings.is_api_ready:
            st.warning(
                "API prerequisites missing: Please connect both your Google Gemini and Tavily Search API keys in the sidebar to enable document analysis."
            )

        confirm_run = st.checkbox(
            f"Confirm live AI extraction and web search for up to {settings.max_claims} claims",
            value=False,
            disabled=not settings.is_api_ready,
            help="Prevents accidental API quota consumption before starting the multi-step verification pipeline.",
        )

        analyze_disabled = (not settings.is_api_ready) or (not confirm_run)
        analyze_clicked = st.button(
            "Analyze Document",
            type="primary",
            disabled=analyze_disabled,
            use_container_width=True,
        )

        existing_report: Optional[FactCheckReport] = st.session_state.get("factcheck_report")
        if existing_report and existing_report.document.document_hash == meta.document_hash:
            if st.button(
                "View Existing Analysis Dashboard",
                type="secondary",
                use_container_width=True,
            ):
                navigate_to("Analysis Dashboard")
                st.rerun()

    if analyze_clicked:
        run_pipeline_callback(extracted_pdf, settings)
