"""FactCheck AI – The Truth Layer for Every Document.
Main Streamlit application entry point implementing a Google Material Design 3 (Material You)
5-page SaaS experience for PDF claim extraction, live Tavily web search, and Gemini verification.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import streamlit as st

from components.claim_cards import render_claim_cards
from components.dashboard import render_dashboard
from components.report_view import render_report_export_view
from components.sidebar import SidebarSettings, navigate_to, render_sidebar
from components.upload import render_document_analysis_page, render_home_page
from config.settings import CONFIG
from models.schemas import (
    ClaimEvidenceBundle,
    ClaimVerificationResult,
    ExtractedClaim,
    FactCheckReport,
    VerificationStatus,
)
from modules.claim_extractor import ClaimExtractor
from modules.fact_verifier import FactVerifier
from modules.pdf_extractor import ExtractedPDFContent
from modules.web_search import TavilySearchService, WebSearchError
from utils.helpers import inject_material3_css
from utils.llm_client import GeminiClient, LLMServiceError


def configure_page() -> None:
    """Configure Streamlit viewport and application title."""
    st.set_page_config(
        page_title=f"{CONFIG.APP_NAME} — {CONFIG.APP_TAGLINE}",
        page_icon=":material/verified_user:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _render_five_step_tracker(container, current_step: int, detail_msg: str) -> None:
    """Render the 5-stage Material Design 3 progress tracker during document analysis:
    1. Reading document.
    2. Extracting factual claims.
    3. Searching the web.
    4. Evaluating evidence.
    5. Preparing report.
    """
    steps = [
        "1. Reading document",
        "2. Extracting factual claims",
        "3. Searching the web",
        "4. Evaluating evidence",
        "5. Preparing report",
    ]
    rows_html = []
    for idx, label in enumerate(steps, start=1):
        if idx < current_step:
            cls = "m3-step-row m3-step-done"
            icon = "<span class='ms-icon' style='color:var(--m3-success);'>check_circle</span>"
            sub = "<span style='margin-left:auto; font-size:0.78rem; color:var(--m3-success); font-weight:600;'>Completed</span>"
        elif idx == current_step:
            cls = "m3-step-row m3-step-active"
            icon = "<span class='ms-icon' style='color:var(--m3-primary);'>sync</span>"
            sub = f"<span style='margin-left:auto; font-size:0.8rem;'>{detail_msg}</span>"
        else:
            cls = "m3-step-row"
            icon = "<span class='ms-icon' style='color:var(--m3-muted);'>radio_button_unchecked</span>"
            sub = "<span style='margin-left:auto; font-size:0.78rem; color:var(--m3-muted);'>Pending</span>"

        rows_html.append(
            f"<div class='{cls}'>{icon}<span>{label}</span>{sub}</div>"
        )

    container.markdown(
        "<div class='m3-card' style='margin-top:1rem;'>"
        "<div style='font-weight:600; font-size:1rem; margin-bottom:0.75rem;'>Verification Progress</div>"
        + "".join(rows_html)
        + "</div>",
        unsafe_allow_html=True,
    )


def run_verification_pipeline(
    extracted_pdf: ExtractedPDFContent,
    settings: SidebarSettings,
) -> None:
    """Execute the 5-step document verification pipeline with caching, controlled batching, and partial-result preservation."""
    if not settings.is_api_ready:
        st.error(
            "Missing API credentials. Please configure both Google Gemini and Tavily Search API keys in the sidebar."
        )
        return

    doc_hash = extracted_pdf.metadata.document_hash
    run_signature = (
        f"{doc_hash}:{settings.selected_model}:{settings.max_claims}:"
        f"{settings.search_depth}:{settings.max_sources_per_claim}"
    )

    # Prevent duplicate processing if an identical report for this document & settings is already in session
    existing_report: Optional[FactCheckReport] = st.session_state.get("factcheck_report")
    if (
        existing_report is not None
        and st.session_state.get("last_run_signature") == run_signature
    ):
        navigate_to("Analysis Dashboard")
        st.rerun()
        return

    tracker_slot = st.empty()
    progress_bar = st.progress(0.05)
    partial_results: List[ClaimVerificationResult] = []
    doc_topic: str = "General Document"

    try:
        # Step 1: Reading document
        _render_five_step_tracker(
            tracker_slot,
            current_step=1,
            detail_msg=f"Validated {len(extracted_pdf.pages)} pages ({extracted_pdf.metadata.total_words:,} words)",
        )
        llm_client = GeminiClient(
            api_key=settings.gemini_api_key or "",
            model_name=settings.selected_model,
        )
        search_service = TavilySearchService(
            api_key=settings.tavily_api_key,
            search_depth=settings.search_depth,
            max_results=settings.max_sources_per_claim,
        )
        progress_bar.progress(0.15)

        # Step 2: Extracting factual claims (with document-hash + model + max_claims session cache)
        extraction_cache: Dict[str, Tuple[str, List[ExtractedClaim]]] = (
            st.session_state.setdefault("gemini_extraction_cache", {})
        )
        ext_cache_key = f"{doc_hash}:{settings.selected_model}:{settings.max_claims}"

        if ext_cache_key in extraction_cache:
            doc_topic, claims = extraction_cache[ext_cache_key]
            _render_five_step_tracker(
                tracker_slot,
                current_step=2,
                detail_msg=f"Loaded {len(claims)} cached claims from session",
            )
            progress_bar.progress(0.38)
        else:
            extractor = ClaimExtractor(llm_client=llm_client)

            def on_extract_progress(frac: float, msg: str) -> None:
                progress_bar.progress(min(0.38, 0.15 + frac * 0.23))
                _render_five_step_tracker(tracker_slot, current_step=2, detail_msg=msg)

            doc_topic, claims = extractor.extract_claims(
                pages=extracted_pdf.pages,
                max_claims=settings.max_claims,
                progress_callback=on_extract_progress,
            )
            extraction_cache[ext_cache_key] = (doc_topic, claims)

        if not claims:
            tracker_slot.empty()
            progress_bar.empty()
            st.warning(
                "No verifiable factual claims were detected in this PDF. "
                "Please upload a document containing factual assertions, statistics, dates, or citations."
            )
            return

        # Step 3: Searching the live web independently for each extracted claim
        def on_search_progress(frac: float, msg: str) -> None:
            progress_bar.progress(min(0.68, 0.38 + frac * 0.30))
            _render_five_step_tracker(tracker_slot, current_step=3, detail_msg=msg)

        evidence_bundles: Dict[str, ClaimEvidenceBundle] = (
            search_service.search_claims_batch(
                claims=claims,
                progress_callback=on_search_progress,
            )
        )
        st.session_state["claim_evidence_bundles"] = evidence_bundles

        any_search_warnings = [
            f"{cid}: {b.search_warning}"
            for cid, b in evidence_bundles.items()
            if b.search_warning
        ]
        if any_search_warnings:
            st.warning(
                "⚠️ **Tavily Search API Warning:** Some search requests encountered errors:\n\n"
                + "\n".join(f"- {w}" for w in any_search_warnings)
            )

        # Step 4: Evaluating evidence & synthesizing verdicts in controlled batches, preserving partial results
        verifier = FactVerifier(
            llm_client=llm_client,
            search_service=search_service,
        )
        total_claims = len(claims)
        batch_size = 5

        for batch_start in range(0, total_claims, batch_size):
            batch_slice = claims[batch_start : batch_start + batch_size]
            for offset, claim in enumerate(batch_slice):
                idx = batch_start + offset + 1
                eval_frac = idx / max(total_claims, 1)
                progress_bar.progress(min(0.94, 0.68 + eval_frac * 0.26))
                _render_five_step_tracker(
                    tracker_slot,
                    current_step=4,
                    detail_msg=f"Evaluating evidence for {claim.claim_id} ({idx}/{total_claims})",
                )
                bundle = evidence_bundles[claim.claim_id]
                try:
                    res = verifier.verify_claim_with_bundle(bundle)
                except Exception as claim_err:
                    # Preserve all completed results if a later claim fails
                    res = ClaimVerificationResult(
                        claim_id=claim.claim_id,
                        original_claim=claim.claim_text,
                        status=VerificationStatus.UNVERIFIABLE,
                        corrected_fact=None,
                        explanation=f"Verification interrupted on this claim: {claim_err}",
                        confidence=0.0,
                        evidence=[],
                        claim=claim,
                        confidence_score=0.0,
                        source_credibility_avg=bundle.average_credibility,
                        sources=bundle.sources,
                    )
                partial_results.append(res)
                st.session_state["partial_verification_results"] = list(partial_results)

        # Step 5: Preparing report
        progress_bar.progress(0.97)
        _render_five_step_tracker(
            tracker_slot,
            current_step=5,
            detail_msg="Compiling truth score, charts, and PDF report...",
        )

        report: FactCheckReport = verifier._compile_report(
            results=partial_results,
            document_metadata=extracted_pdf.metadata,
            document_topic=doc_topic,
        )

        progress_bar.progress(1.0)
        st.session_state["factcheck_report"] = report
        st.session_state["last_run_signature"] = run_signature

        recent: List[FactCheckReport] = st.session_state.get("recent_reports", [])
        recent.insert(0, report)
        st.session_state["recent_reports"] = recent[:10]

        navigate_to("Analysis Dashboard")
        st.rerun()

    except (LLMServiceError, WebSearchError) as service_err:
        progress_bar.empty()
        if partial_results:
            verifier_fallback = FactVerifier(llm_client=llm_client, search_service=search_service)
            partial_report = verifier_fallback._compile_report(
                results=partial_results,
                document_metadata=extracted_pdf.metadata,
                document_topic=doc_topic,
            )
            st.session_state["factcheck_report"] = partial_report
            st.warning(
                f"Pipeline interrupted ({service_err}), but **{len(partial_results)} completed claim verification(s)** were preserved."
            )
        else:
            st.error(f"**Verification Service Error:** {service_err}")
    except Exception as exc:
        progress_bar.empty()
        st.error(f"**An unexpected error occurred during analysis:** {exc}")


def _render_no_report_empty_state(page_title: str) -> None:
    """Render a helpful Material Design 3 empty state when a user visits pages 3-5 before analyzing a PDF."""
    extracted_pdf: Optional[ExtractedPDFContent] = st.session_state.get("extracted_pdf")
    st.markdown(
        f"""
        <div style="margin-bottom:1.25rem;">
            <h2 style="margin:0 0 0.25rem 0; font-size:1.75rem; font-weight:700;">{page_title}</h2>
        </div>
        <div class="m3-card" style="text-align:center; padding:2.75rem 1.75rem;">
            <span class="ms-icon" style="font-size:40px; color:var(--m3-primary); margin-bottom:0.75rem;">pending_actions</span>
            <div style="font-size:1.15rem; font-weight:600; margin-bottom:0.4rem;">No Verification Results Available Yet</div>
            <div style="font-size:0.92rem; color:var(--m3-muted); max-width:520px; margin:0 auto 1.25rem auto; line-height:1.55;">
                {"You have uploaded <strong>" + extracted_pdf.metadata.filename + "</strong>, but it has not been analyzed yet." if extracted_pdf else "Upload a PDF document and run the fact-checking pipeline to populate your analytics dashboard, claim details, and downloadable report."}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        target_page = "Document Analysis" if extracted_pdf else "Home"
        btn_label = "Go to Document Analysis" if extracted_pdf else "Upload a PDF on Home"
        if st.button(btn_label, type="primary", use_container_width=True):
            navigate_to(target_page)
            st.rerun()


def main() -> None:
    """Main entry point orchestrating Material 3 theming, sidebar navigation, and the 5 pages."""
    configure_page()
    sidebar_settings = render_sidebar()
    inject_material3_css(sidebar_settings.theme_mode)

    active_page = sidebar_settings.active_page
    report: Optional[FactCheckReport] = st.session_state.get("factcheck_report")

    if active_page == "Home":
        render_home_page(sidebar_settings)

    elif active_page == "Document Analysis":
        render_document_analysis_page(
            settings=sidebar_settings,
            run_pipeline_callback=run_verification_pipeline,
        )

    elif active_page == "Analysis Dashboard":
        if report is None:
            _render_no_report_empty_state("Analysis Dashboard")
        else:
            if report.search_warnings:
                st.warning(
                    "⚠️ **Tavily Search API Warning during analysis:** "
                    + " | ".join(report.search_warnings)
                )
            render_dashboard(report, theme_mode=sidebar_settings.theme_mode)

    elif active_page == "Claim Details":
        if report is None:
            _render_no_report_empty_state("Claim Details")
        else:
            render_claim_cards(report.results)

    elif active_page == "Report":
        if report is None:
            _render_no_report_empty_state("Verification Audit Report")
        else:
            render_report_export_view(report)


if __name__ == "__main__":
    main()
