"""Page 3: Analysis Dashboard component (`components/dashboard.py`).

Calculates real metrics from completed verification results using `modules.report_generator.compute_report_analytics`
and renders Material Design 3 KPI cards, high-risk and conflicting-evidence spotlights,
3 interactive Plotly visualizations (Verdict Donut Chart, Claim Categories Bar Chart, and Confidence Distribution Chart),
and a multi-filterable Claim Results Table.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.sidebar import navigate_to
from config.settings import VERDICT_DISPLAY_LABELS, get_active_palette
from models.schemas import FactCheckReport
from modules.report_generator import (
    ReportAnalytics,
    build_claims_dataframe,
    compute_report_analytics,
    filter_claim_results,
)


def _build_findings_narrative(
    report: FactCheckReport,
    analytics: ReportAnalytics,
) -> str:
    """Generate an executive summary narrative from actual computed verification analytics."""
    total = analytics.total_claims
    if total == 0:
        return (
            "No verifiable factual claims were present in the completed verification results. "
            "Missing or empty results are never classified as verified."
        )

    score = analytics.overall_truth_score
    if score >= 80:
        assessment = "demonstrates high factual reliability"
    elif score >= 55:
        assessment = "shows mixed factual accuracy requiring targeted revisions"
    else:
        assessment = "contains significant factual discrepancies against live web evidence"

    return (
        f"Across <strong>{total} evaluated claim(s)</strong> on <em>{report.document_topic}</em>, "
        f"<strong>{report.document.filename}</strong> {assessment} with an overall Truth Score of "
        f"<strong>{score:.1f}%</strong>. Specifically, <strong>{analytics.verified_count}</strong> "
        f"({analytics.verification_percentages['Verified']:.1f}%) were verified by external sources, "
        f"<strong>{analytics.inaccurate_count}</strong> ({analytics.verification_percentages['Inaccurate']:.1f}%) "
        f"were partially inaccurate or outdated, <strong>{analytics.false_count}</strong> "
        f"({analytics.verification_percentages['False']:.1f}%) were contradicted by evidence, and "
        f"<strong>{analytics.unverifiable_count}</strong> ({analytics.verification_percentages['Unverifiable']:.1f}%) "
        f"were unverifiable. The engine flagged <strong>{len(analytics.high_risk_claims)} high-risk claim(s)</strong> "
        f"and <strong>{len(analytics.conflicting_evidence_claims)} claim(s) with conflicting evidence</strong>."
    )


def render_dashboard(report: FactCheckReport, theme_mode: str = "light") -> None:
    """Render PAGE 3: ANALYSIS DASHBOARD using real computed analytics and Plotly charts."""
    palette = get_active_palette(theme_mode)
    analytics = compute_report_analytics(report)

    # Top Header & Quick Actions
    h_col1, h_col2, h_col3 = st.columns([6, 2, 2])
    with h_col1:
        st.markdown(
            f"""
            <div style="margin-bottom:1rem;">
                <h2 style="margin:0 0 0.25rem 0; font-size:1.75rem; font-weight:700;">Analysis Dashboard</h2>
                <p style="margin:0; color:var(--m3-muted); font-size:0.92rem;">
                    Document: <strong>{report.document.filename}</strong> &bull; Report ID: <code>{report.report_id}</code> &bull; {report.generated_at}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with h_col2:
        if st.button("Inspect Claim Details", type="secondary", use_container_width=True):
            navigate_to("Claim Details")
            st.rerun()
    with h_col3:
        if st.button("Export Audit Report", type="primary", use_container_width=True):
            navigate_to("Report")
            st.rerun()

    # Row 1: Primary Verification KPI Cards (Total, Verified, Inaccurate, False, Unverifiable)
    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")

    with c1:
        st.markdown(
            f"""
            <div class="m3-kpi-card">
                <div class="m3-kpi-label">
                    <span class="ms-icon" style="font-size:18px; color:var(--m3-primary);">format_list_numbered</span>
                    Total Claims
                </div>
                <div class="m3-kpi-value">{analytics.total_claims}</div>
                <div class="m3-kpi-sub">Truth Score: <strong>{analytics.overall_truth_score:.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="m3-kpi-card" style="border-bottom: 3px solid var(--m3-success);">
                <div class="m3-kpi-label">
                    <span class="ms-icon" style="font-size:18px; color:var(--m3-success);">check_circle</span>
                    Verified
                </div>
                <div class="m3-kpi-value" style="color:var(--m3-success);">{analytics.verified_count}</div>
                <div class="m3-kpi-sub">{analytics.verification_percentages['Verified']:.1f}% of claims</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="m3-kpi-card" style="border-bottom: 3px solid var(--m3-warning);">
                <div class="m3-kpi-label">
                    <span class="ms-icon" style="font-size:18px; color:var(--m3-warning);">warning</span>
                    Inaccurate
                </div>
                <div class="m3-kpi-value" style="color:var(--m3-warning);">{analytics.inaccurate_count}</div>
                <div class="m3-kpi-sub">{analytics.verification_percentages['Inaccurate']:.1f}% of claims</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="m3-kpi-card" style="border-bottom: 3px solid var(--m3-error);">
                <div class="m3-kpi-label">
                    <span class="ms-icon" style="font-size:18px; color:var(--m3-error);">cancel</span>
                    False
                </div>
                <div class="m3-kpi-value" style="color:var(--m3-error);">{analytics.false_count}</div>
                <div class="m3-kpi-sub">{analytics.verification_percentages['False']:.1f}% of claims</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"""
            <div class="m3-kpi-card" style="border-bottom: 3px solid var(--m3-neutral);">
                <div class="m3-kpi-label">
                    <span class="ms-icon" style="font-size:18px; color:var(--m3-neutral);">help</span>
                    Unverifiable
                </div>
                <div class="m3-kpi-value" style="color:var(--m3-neutral);">{analytics.unverifiable_count}</div>
                <div class="m3-kpi-sub">{analytics.verification_percentages['Unverifiable']:.1f}% of claims</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Row 2: Secondary Risk & Quality KPI Cards (High-Risk Claims, Conflicting Evidence, Avg Confidence, Categories)
    r1, r2, r3, r4 = st.columns(4, gap="medium")
    with r1:
        st.markdown(
            f"""
            <div class="m3-kpi-card">
                <div class="m3-kpi-label">
                    <span class="ms-icon" style="font-size:18px; color:var(--m3-error);">gpp_maybe</span>
                    High-Risk Claims
                </div>
                <div class="m3-kpi-value" style="font-size:1.45rem; color:var(--m3-error);">{len(analytics.high_risk_claims)}</div>
                <div class="m3-kpi-sub">False or high-priority inaccurate claims</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f"""
            <div class="m3-kpi-card">
                <div class="m3-kpi-label">
                    <span class="ms-icon" style="font-size:18px; color:var(--m3-warning);">compare_arrows</span>
                    Conflicting Evidence
                </div>
                <div class="m3-kpi-value" style="font-size:1.45rem; color:var(--m3-warning);">{len(analytics.conflicting_evidence_claims)}</div>
                <div class="m3-kpi-sub">Claims where sources disagree</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            f"""
            <div class="m3-kpi-card">
                <div class="m3-kpi-label">
                    <span class="ms-icon" style="font-size:18px; color:var(--m3-primary);">verified</span>
                    Mean Evidence Confidence
                </div>
                <div class="m3-kpi-value" style="font-size:1.45rem;">{analytics.average_confidence * 100:.0f}%</div>
                <div class="m3-kpi-sub">Source Authority Avg: {analytics.average_source_credibility * 100:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r4:
        st.markdown(
            f"""
            <div class="m3-kpi-card">
                <div class="m3-kpi-label">
                    <span class="ms-icon" style="font-size:18px; color:var(--m3-secondary);">category</span>
                    Active Claim Categories
                </div>
                <div class="m3-kpi-value" style="font-size:1.45rem;">{len(analytics.category_counts)}</div>
                <div class="m3-kpi-sub">Distinct factual domains detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

    # Executive Document Findings Narrative Card
    findings_html = _build_findings_narrative(report, analytics)
    st.markdown(
        f"""
        <div class="m3-tonal-card">
            <div style="display:flex; align-items:center; gap:0.5rem; font-weight:600; font-size:1rem; margin-bottom:0.4rem;">
                <span class="ms-icon" style="color:var(--m3-primary);">insights</span>
                Overall Document Findings
            </div>
            <div style="font-size:0.93rem; line-height:1.6; color:var(--m3-text);">
                {findings_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analytics.total_claims == 0:
        st.info("No verifiable claims were found in this report.")
        return

    # High-Risk & Conflicting Evidence Alert Drawers (when present)
    if analytics.high_risk_claims or analytics.conflicting_evidence_claims:
        alert_col1, alert_col2 = st.columns(2, gap="medium")
        with alert_col1:
            with st.expander(
                f"High-Risk Claims Requiring Attention ({len(analytics.high_risk_claims)})",
                expanded=bool(analytics.high_risk_claims),
            ):
                if not analytics.high_risk_claims:
                    st.caption("No high-risk claims identified.")
                else:
                    for hr in analytics.high_risk_claims:
                        ui_lbl = VERDICT_DISPLAY_LABELS.get(hr.status.value, hr.status.value)
                        corr_note = f" — **Correction:** {hr.corrected_fact}" if hr.corrected_fact else ""
                        st.markdown(
                            f"- **[{hr.claim_id} • Page {hr.claim.page_number}] ({ui_lbl}):** "
                            f"“{hr.claim.claim_text}”{corr_note}"
                        )
        with alert_col2:
            with st.expander(
                f"Claims with Conflicting Evidence ({len(analytics.conflicting_evidence_claims)})",
                expanded=bool(analytics.conflicting_evidence_claims),
            ):
                if not analytics.conflicting_evidence_claims:
                    st.caption("No inter-source evidence conflicts detected.")
                else:
                    for cf in analytics.conflicting_evidence_claims:
                        ui_lbl = VERDICT_DISPLAY_LABELS.get(cf.status.value, cf.status.value)
                        st.markdown(
                            f"- **[{cf.claim_id} • Page {cf.claim.page_number}] ({ui_lbl}):** "
                            f"“{cf.claim.claim_text}”"
                        )

    df_all = build_claims_dataframe(report.results)

    chart_color_map = {
        "Verified": palette["success"],
        "Inaccurate": palette["warning"],
        "False": palette["error"],
        "Unverifiable": palette["neutral"],
    }

    # Plotly Visualizations Row: 1. Donut Chart, 2. Category Bar Chart, 3. Confidence Distribution Chart
    ch1, ch2, ch3 = st.columns([4, 4, 4], gap="medium")

    with ch1:
        st.markdown(
            """
            <div class="m3-card" style="padding:1rem 1.25rem 0.4rem 1.25rem; margin-bottom:0.25rem;">
                <div style="font-weight:600; font-size:0.98rem;">1. Verification Distribution</div>
                <div style="font-size:0.78rem; color:var(--m3-muted);">Status share across evaluated claims</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        verdict_counts = (
            df_all["Status"]
            .value_counts()
            .reindex(["Verified", "Inaccurate", "False", "Unverifiable"], fill_value=0)
            .reset_index()
        )
        verdict_counts.columns = ["Status", "Count"]
        verdict_counts = verdict_counts[verdict_counts["Count"] > 0]

        fig_donut = px.pie(
            verdict_counts,
            names="Status",
            values="Count",
            hole=0.60,
            color="Status",
            color_discrete_map=chart_color_map,
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Claims: %{value} (%{percent})<extra></extra>",
        )
        fig_donut.update_layout(
            margin=dict(t=10, b=15, l=10, r=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=palette["text"], family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with ch2:
        st.markdown(
            """
            <div class="m3-card" style="padding:1rem 1.25rem 0.4rem 1.25rem; margin-bottom:0.25rem;">
                <div style="font-weight:600; font-size:0.98rem;">2. Claims by Category</div>
                <div style="font-size:0.78rem; color:var(--m3-muted);">Horizontal breakdown by factual domain</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cat_counts_df = (
            df_all.groupby(["Category", "Status"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        fig_hbar = px.bar(
            cat_counts_df,
            x="Count",
            y="Category",
            color="Status",
            orientation="h",
            barmode="stack",
            color_discrete_map=chart_color_map,
        )
        fig_hbar.update_layout(
            margin=dict(t=10, b=15, l=10, r=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=palette["text"], family="Inter, sans-serif"),
            xaxis=dict(title="Claims", dtick=1, gridcolor=palette["surface_border"]),
            yaxis=dict(title=None),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_hbar, use_container_width=True)

    with ch3:
        st.markdown(
            """
            <div class="m3-card" style="padding:1rem 1.25rem 0.4rem 1.25rem; margin-bottom:0.25rem;">
                <div style="font-weight:600; font-size:0.98rem;">3. Confidence Distribution</div>
                <div style="font-size:0.78rem; color:var(--m3-muted);">Evidence confidence score per claim</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_conf = px.bar(
            df_all,
            x="Claim ID",
            y="Confidence (%)",
            color="Status",
            color_discrete_map=chart_color_map,
            hover_data=["Category", "Page", "High Risk"],
        )
        fig_conf.update_layout(
            margin=dict(t=10, b=15, l=10, r=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=palette["text"], family="Inter, sans-serif"),
            yaxis=dict(title="Confidence (%)", range=[0, 105], gridcolor=palette["surface_border"]),
            xaxis=dict(title=None),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Interactive Filtering & Claim Results Table
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; font-weight:600; font-size:1.05rem; margin-bottom:0.65rem;">
            <span class="ms-icon" style="color:var(--m3-primary);">filter_list</span>
            Filter &amp; Explore Claim Results Table
        </div>
        """,
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns([2, 2, 2, 3], gap="medium")
    status_options = ["Verified", "Inaccurate", "False", "Unverifiable"]
    with f1:
        sel_statuses = st.multiselect(
            "Verification Status",
            options=status_options,
            default=status_options,
            key="dash_status_filter",
        )
    cat_options = sorted(df_all["Category"].unique().tolist())
    with f2:
        sel_cats = st.multiselect(
            "Claim Category",
            options=cat_options,
            default=cat_options,
            key="dash_cat_filter",
        )
    with f3:
        min_conf_val = st.slider(
            "Minimum Confidence (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="dash_min_conf_slider",
        )
    with f4:
        search_q = st.text_input(
            "Search Keyword",
            placeholder="Filter by claim text, figure, entity, domain, or correction...",
            key="dash_search_input",
        )

    filtered_results = filter_claim_results(
        results=report.results,
        selected_statuses=sel_statuses,
        selected_categories=sel_cats,
        min_confidence_pct=float(min_conf_val),
        search_keyword=search_q,
    )
    df_filtered = build_claims_dataframe(filtered_results)

    st.caption(
        f"Displaying **{len(df_filtered)}** of **{len(df_all)}** verified claims matching your filters."
    )
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
    )
