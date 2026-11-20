"""Page 5: Report component (`components/report_view.py`).

Renders the complete verification audit report view including:
- One-click standalone PDF report download (ReportLab), CSV download (Pandas), JSON download, and 'Analyze Another Document' action.
- Document summary and computed verification statistics (including High-Risk and Conflicting Evidence counts).
- Interactive filtering controls (status, category, confidence, keyword) over the full Claim-by-Claim Results ledger.
- Methodology & important limitations disclosure.
"""

from __future__ import annotations

import streamlit as st

from components.sidebar import navigate_to
from config.settings import VERDICT_DISPLAY_LABELS
from models.schemas import FactCheckReport, VerificationStatus
from modules.report_generator import (
    build_claims_dataframe,
    compute_report_analytics,
    filter_claim_results,
    generate_csv_export,
    generate_json_export,
    generate_pdf_report,
    has_conflicting_evidence,
    is_high_risk_claim,
)
from utils.helpers import format_file_size


def render_report_export_view(report: FactCheckReport) -> None:
    """Render PAGE 5: REPORT with executive summary, verification statistics, filterable claim report cards, and exports."""
    analytics = compute_report_analytics(report)

    st.markdown(
        """
        <div style="margin-bottom:1.15rem;">
            <h2 style="margin:0 0 0.25rem 0; font-size:1.75rem; font-weight:700;">Verification Audit Report</h2>
            <p style="margin:0; color:var(--m3-muted); font-size:0.94rem;">
                Comprehensive document verification summary, filterable claim-by-claim findings, and standalone PDF &amp; CSV report downloads.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    meta = report.document
    base_filename = meta.filename.rsplit(".", 1)[0].replace(" ", "_")

    # 1. Primary Export & Action Bar
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([3, 3, 2, 3], gap="medium")

    with btn_col1:
        try:
            pdf_bytes = generate_pdf_report(report)
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"FactCheck_Report_{base_filename}_{report.report_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF export error: {exc}")

    with btn_col2:
        csv_bytes = generate_csv_export(report)
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=f"FactCheck_Audit_{base_filename}_{report.report_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with btn_col3:
        json_bytes = generate_json_export(report)
        st.download_button(
            label="Export JSON",
            data=json_bytes,
            file_name=f"FactCheck_Data_{base_filename}_{report.report_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    with btn_col4:
        if st.button("Analyze Another Document", type="secondary", use_container_width=True):
            from utils.security import clear_sensitive_session_data

            clear_sensitive_session_data(st.session_state, keep_recent_history=True)
            st.session_state.pop("last_run_signature", None)
            navigate_to("Home")
            st.rerun()

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    # 2. Document Summary & Verification Statistics Split Card
    col_doc, col_stats = st.columns([6, 6], gap="large")

    with col_doc:
        st.markdown(
            f"""
            <div class="m3-card" style="height:100%;">
                <div style="display:flex; align-items:center; gap:0.5rem; font-weight:600; font-size:1.05rem; margin-bottom:0.85rem;">
                    <span class="ms-icon" style="color:var(--m3-primary);">description</span>
                    Document Summary
                </div>
                <div style="font-size:0.9rem; line-height:1.8; color:var(--m3-text);">
                    <div><strong>Document Name:</strong> {meta.filename}</div>
                    <div><strong>Primary Topic:</strong> {report.document_topic}</div>
                    <div><strong>File Size &amp; Length:</strong> {format_file_size(meta.file_size_bytes)} &bull; {meta.readable_page_count}/{meta.page_count} Pages ({meta.total_words:,} words)</div>
                    <div><strong>SHA-256 Fingerprint:</strong> <code>{meta.document_hash}</code></div>
                    <div><strong>Audit Report ID:</strong> <code>{report.report_id}</code></div>
                    <div><strong>Verification Model:</strong> <code>{report.model_used}</code></div>
                    <div><strong>Analysis Date:</strong> {report.generated_at}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_stats:
        st.markdown(
            f"""
            <div class="m3-card" style="height:100%;">
                <div style="display:flex; align-items:center; gap:0.5rem; font-weight:600; font-size:1.05rem; margin-bottom:0.85rem;">
                    <span class="ms-icon" style="color:var(--m3-primary);">analytics</span>
                    Verification Statistics
                </div>
                <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:0.65rem;">
                    <div class="m3-tonal-card" style="margin:0; padding:0.75rem 0.9rem;">
                        <div style="font-size:0.75rem; color:var(--m3-muted);">Truth Score</div>
                        <div style="font-size:1.35rem; font-weight:700; color:var(--m3-primary);">{analytics.overall_truth_score:.1f}%</div>
                    </div>
                    <div class="m3-tonal-card" style="margin:0; padding:0.75rem 0.9rem;">
                        <div style="font-size:0.75rem; color:var(--m3-muted);">Mean Confidence</div>
                        <div style="font-size:1.35rem; font-weight:700;">{analytics.average_confidence * 100:.0f}%</div>
                    </div>
                    <div class="m3-tonal-card" style="margin:0; padding:0.75rem 0.9rem;">
                        <div style="font-size:0.75rem; color:var(--m3-muted);">Total Claims</div>
                        <div style="font-size:1.35rem; font-weight:700;">{analytics.total_claims}</div>
                    </div>
                    <div class="m3-tonal-card" style="margin:0; padding:0.75rem 0.9rem;">
                        <div style="font-size:0.75rem; color:var(--m3-muted);">Verified / Inaccurate</div>
                        <div style="font-size:1.2rem; font-weight:700;">
                            <span style="color:var(--m3-success);">{analytics.verified_count}</span> /
                            <span style="color:var(--m3-warning);">{analytics.inaccurate_count}</span>
                        </div>
                    </div>
                    <div class="m3-tonal-card" style="margin:0; padding:0.75rem 0.9rem;">
                        <div style="font-size:0.75rem; color:var(--m3-muted);">False / Unverifiable</div>
                        <div style="font-size:1.2rem; font-weight:700;">
                            <span style="color:var(--m3-error);">{analytics.false_count}</span> /
                            <span style="color:var(--m3-neutral);">{analytics.unverifiable_count}</span>
                        </div>
                    </div>
                    <div class="m3-tonal-card" style="margin:0; padding:0.75rem 0.9rem;">
                        <div style="font-size:0.75rem; color:var(--m3-muted);">High-Risk / Conflict</div>
                        <div style="font-size:1.2rem; font-weight:700;">
                            <span style="color:var(--m3-error);">{len(analytics.high_risk_claims)}</span> /
                            <span style="color:var(--m3-warning);">{len(analytics.conflicting_evidence_claims)}</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # 3. Filterable Claim-by-Claim Results Table & Report Breakdown
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; font-weight:600; font-size:1.08rem; margin-bottom:0.65rem;">
            <span class="ms-icon" style="color:var(--m3-primary);">table_chart</span>
            Claim-by-Claim Verification Report
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not report.results:
        st.info("This report contains 0 verified claims.")
    else:
        rf1, rf2, rf3, rf4 = st.columns([2, 2, 2, 3], gap="medium")
        status_opts = ["Verified", "Inaccurate", "False", "Unverifiable"]
        with rf1:
            rep_statuses = st.multiselect(
                "Status",
                options=status_opts,
                default=status_opts,
                key="rep_status_filter",
            )
        cat_opts = sorted({r.claim.category.value for r in report.results})
        with rf2:
            rep_cats = st.multiselect(
                "Category",
                options=cat_opts,
                default=cat_opts,
                key="rep_cat_filter",
            )
        with rf3:
            rep_min_conf = st.slider(
                "Min Confidence (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                key="rep_conf_slider",
            )
        with rf4:
            rep_kw = st.text_input(
                "Search Report",
                placeholder="Filter by claim ID, text, source domain, or date...",
                key="rep_kw_input",
            )

        filtered_claims = filter_claim_results(
            results=report.results,
            selected_statuses=rep_statuses,
            selected_categories=rep_cats,
            min_confidence_pct=float(rep_min_conf),
            search_keyword=rep_kw,
        )

        df_report = build_claims_dataframe(filtered_claims)
        st.caption(f"Showing **{len(df_report)}** of **{len(report.results)}** claims in report.")
        st.dataframe(
            df_report,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

    # 4. Important Limitations & Epistemic Disclosure
    st.markdown(
        f"""
        <div class="m3-tonal-card">
            <div style="display:flex; align-items:center; gap:0.45rem; font-weight:600; font-size:0.95rem; margin-bottom:0.4rem;">
                <span class="ms-icon" style="color:var(--m3-primary);">policy</span>
                Important Methodology &amp; Verification Limitations
            </div>
            <ul style="margin:0; padding-left:1.2rem; font-size:0.85rem; color:var(--m3-muted); line-height:1.65;">
                <li><strong>Evidence-Bounded Scope:</strong> All verification verdicts are derived strictly from live public web sources retrieved on {report.generated_at}. Internal, proprietary, or paywalled records not indexed on the public web cannot be evaluated.</li>
                <li><strong>Conservative Unverifiable Handling:</strong> Missing or inconclusive search results are never classified as <em>Verified</em> or <em>False</em>; they are explicitly designated as <em>Unverifiable</em>.</li>
                <li><strong>Temporal Sensitivity:</strong> Statistical and financial findings reflect the reporting period stated in the PDF compared against source publication dates available at the time of analysis.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
