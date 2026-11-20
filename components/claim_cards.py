"""Page 4: Claim Details component (`components/claim_cards.py`).

Displays every verified claim in an expandable Material Design 3 card showing:
- Claim ID, Original statement, Verification status, Category, Corrected information,
  Explanation, Confidence score, Supporting sources with clickable URLs,
  Publication date where available, and PDF page number.
Allows filtering by Verification status, Claim category, Confidence score, and Search keyword.
"""

from __future__ import annotations

from typing import List, Tuple

import streamlit as st

from config.settings import VERDICT_DISPLAY_LABELS
from models.schemas import ClaimVerificationResult, VerificationStatus
from modules.report_generator import (
    filter_claim_results,
    has_conflicting_evidence,
    is_high_risk_claim,
)


def _status_chip_meta(status: VerificationStatus) -> Tuple[str, str, str, str]:
    """Return (display_label, css_class, material_icon, hex_color) for a verification status."""
    mapping = {
        VerificationStatus.VERIFIED: (
            "Verified",
            "m3-chip-verified",
            "check_circle",
            "var(--m3-success)",
        ),
        VerificationStatus.INACCURATE: (
            "Inaccurate",
            "m3-chip-inaccurate",
            "warning",
            "var(--m3-warning)",
        ),
        VerificationStatus.FALSE: (
            "False",
            "m3-chip-false",
            "cancel",
            "var(--m3-error)",
        ),
        VerificationStatus.UNVERIFIABLE: (
            "Unverifiable",
            "m3-chip-unverifiable",
            "help",
            "var(--m3-neutral)",
        ),
    }
    return mapping.get(
        status,
        ("Unverifiable", "m3-chip-unverifiable", "help", "var(--m3-neutral)"),
    )


def _relationship_badge_meta(rel_str: str) -> Tuple[str, str]:
    """Return (display_label, chip_css_class) for an evidence source relationship."""
    norm = (rel_str or "").lower()
    if norm == "supports" or "directly supports" in norm:
        return ("Supports Claim", "m3-chip-verified")
    if norm == "contradicts" or "directly contradicts" in norm:
        return ("Contradicts Claim", "m3-chip-false")
    if norm == "partially_supports" or "partial" in norm:
        return ("Partially Supports", "m3-chip-inaccurate")
    return ("Contextual Evidence", "m3-chip-unverifiable")


def render_claim_cards(results: List[ClaimVerificationResult]) -> None:
    """Render PAGE 4: CLAIM DETAILS with 4-way filtering (status, category, confidence, keyword) and expandable cards."""
    st.markdown(
        """
        <div style="margin-bottom:1.1rem;">
            <h2 style="margin:0 0 0.25rem 0; font-size:1.75rem; font-weight:700;">Claim Details</h2>
            <p style="margin:0; color:var(--m3-muted); font-size:0.94rem;">
                Expand any claim card to inspect its verbatim claim statement, verification status, category, corrected information, explanation, confidence score, and supporting source URLs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results:
        st.markdown(
            """
            <div class="m3-card" style="text-align:center; padding:2.25rem;">
                <span class="ms-icon" style="font-size:34px; color:var(--m3-muted);">fact_check</span>
                <div style="font-weight:600; margin-top:0.5rem;">No Claims Available</div>
                <div style="font-size:0.88rem; color:var(--m3-muted);">Run a document analysis first to inspect claim details.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    search_warnings = [
        f"**{r.claim.claim_id}:** {r.search_warning}"
        for r in results
        if r.search_warning
    ]
    if search_warnings:
        st.warning(
            "⚠️ **Tavily Search API Warning:** Some live web searches encountered errors:\n\n"
            + "\n".join(f"- {w}" for w in search_warnings)
        )

    # 4-Way Filter Bar: Verification Status, Claim Category, Confidence, and Search Keyword
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 3], gap="medium")

    ui_status_options = ["Verified", "Inaccurate", "False", "Unverifiable"]
    with f_col1:
        selected_ui_statuses = st.multiselect(
            "Filter by Status",
            options=ui_status_options,
            default=ui_status_options,
            key="details_status_filter",
        )

    available_categories = sorted({r.claim.category.value for r in results})
    with f_col2:
        selected_categories = st.multiselect(
            "Filter by Category",
            options=available_categories,
            default=available_categories,
            key="details_category_filter",
        )

    with f_col3:
        min_confidence = st.slider(
            "Minimum Confidence (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="details_confidence_filter",
        )

    with f_col4:
        keyword_filter = st.text_input(
            "Search Keyword",
            placeholder="Filter by claim text, entity, date, number, domain, or explanation...",
            key="details_keyword_filter",
        )

    t_col1, t_col2, _ = st.columns([2, 2, 5])
    with t_col1:
        only_high_risk = st.checkbox("High-Risk Claims Only", value=False, key="details_high_risk_cb")
    with t_col2:
        only_conflicting = st.checkbox("Conflicting Evidence Only", value=False, key="details_conflict_cb")

    filtered_results = filter_claim_results(
        results=results,
        selected_statuses=selected_ui_statuses,
        selected_categories=selected_categories,
        min_confidence_pct=float(min_confidence),
        search_keyword=keyword_filter,
        only_high_risk=only_high_risk,
        only_conflicting=only_conflicting,
    )

    st.caption(
        f"Showing **{len(filtered_results)}** of **{len(results)}** verified claims."
    )

    for idx, res in enumerate(filtered_results):
        claim = res.claim
        ui_label, chip_class, icon_name, bar_color = _status_chip_meta(res.status)
        conf_pct = round(res.confidence * 100)
        auth_pct = round(res.source_credibility_avg * 100)
        high_risk = is_high_risk_claim(res)
        conflicting = has_conflicting_evidence(res)

        short_claim = (
            claim.claim_text[:100] + "..."
            if len(claim.claim_text) > 100
            else claim.claim_text
        )
        expander_header = (
            f"[{res.claim_id} • Page {claim.page_number}] {ui_label.upper()} ({conf_pct}% Confidence) — {short_claim}"
        )

        default_open = idx == 0 or res.status in (
            VerificationStatus.FALSE,
            VerificationStatus.INACCURATE,
        )

        with st.expander(expander_header, expanded=default_open):
            if res.search_warning:
                st.warning(f"⚠️ **Live Web Search Warning:** {res.search_warning}")

            risk_badge_html = (
                "<span class='m3-chip m3-chip-false'><span class='ms-icon' style='font-size:15px;'>gpp_maybe</span>High-Risk Claim</span>"
                if high_risk
                else ""
            )
            conflict_badge_html = (
                "<span class='m3-chip m3-chip-inaccurate'><span class='ms-icon' style='font-size:15px;'>compare_arrows</span>Conflicting Evidence</span>"
                if conflicting
                else ""
            )

            # Top Metadata Badges & Confidence Indicator Bar
            st.markdown(
                f"""
                <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:0.75rem; margin-bottom:1rem;">
                    <div style="display:flex; flex-wrap:wrap; align-items:center; gap:0.5rem;">
                        <span class="m3-chip {chip_class}">
                            <span class="ms-icon" style="font-size:16px;">{icon_name}</span>
                            {ui_label} ({res.status.value})
                        </span>
                        <span class="m3-chip m3-chip-category">
                            <span class="ms-icon" style="font-size:15px;">category</span>
                            {claim.category.value}
                        </span>
                        <span class="m3-chip m3-chip-unverifiable">
                            Importance: {claim.importance.value}
                        </span>
                        <span class="m3-chip m3-chip-unverifiable">
                            PDF Page {claim.page_number}
                        </span>
                        {risk_badge_html}
                        {conflict_badge_html}
                    </div>
                    <div style="min-width:210px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.78rem; font-weight:600; color:var(--m3-muted);">
                            <span>Confidence Score</span>
                            <span style="color:var(--m3-text);">{conf_pct}% (Authority: {auth_pct}%)</span>
                        </div>
                        <div class="m3-meter-track">
                            <div class="m3-meter-fill" style="width:{conf_pct}%; background-color:{bar_color};"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Structured Claim Attributes Row
            attr_pills = [f"<span><strong>Claim ID:</strong> <code>{res.claim_id}</code></span>"]
            if claim.entity_or_subject:
                attr_pills.append(
                    f"<span><strong>Entity / Subject:</strong> {claim.entity_or_subject}</span>"
                )
            if claim.date_or_period:
                attr_pills.append(
                    f"<span><strong>Date / Period:</strong> {claim.date_or_period}</span>"
                )
            if claim.numeric_value:
                unit_str = f" {claim.numeric_unit}" if claim.numeric_unit else ""
                attr_pills.append(
                    f"<span><strong>Numeric Value:</strong> <code>{claim.numeric_value}{unit_str}</code></span>"
                )
            attr_bar_html = (
                "<div style='display:flex; flex-wrap:wrap; gap:1.1rem; font-size:0.83rem; color:var(--m3-muted); margin-top:0.55rem; padding-top:0.55rem; border-top:1px solid var(--m3-border);'>"
                + "".join(attr_pills)
                + "</div>"
            )

            context_html = (
                f"<div style='margin-top:0.5rem; font-size:0.83rem; color:var(--m3-muted);'>"
                f"<strong>Surrounding Document Context (Page {claim.page_number}):</strong> &ldquo;{claim.surrounding_context}&rdquo;</div>"
                if claim.surrounding_context
                else ""
            )

            st.markdown(
                f"""
                <div class="m3-tonal-card" style="margin-bottom:0.85rem;">
                    <div style="font-size:0.76rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:var(--m3-muted); margin-bottom:0.25rem;">
                        Original Statement
                    </div>
                    <div style="font-size:1.02rem; font-weight:600; color:var(--m3-text); line-height:1.5;">
                        {res.original_claim or claim.claim_text}
                    </div>
                    {context_html}
                    {attr_bar_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Corrected Information
            if res.corrected_fact:
                st.markdown(
                    f"""
                    <div style="background-color:var(--m3-warning-bg); border-left:4px solid var(--m3-warning); border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:0.85rem;">
                        <div style="display:flex; align-items:center; gap:0.4rem; font-weight:600; font-size:0.86rem; color:var(--m3-warning); margin-bottom:0.25rem;">
                            <span class="ms-icon" style="font-size:18px;">edit_note</span>
                            Corrected Information (Evidence-Backed)
                        </div>
                        <div style="font-size:0.92rem; color:var(--m3-text); line-height:1.55;">
                            {res.corrected_fact}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Explanation & Key Findings
            st.markdown(
                f"""
                <div style="margin-bottom:0.85rem;">
                    <div style="font-weight:600; font-size:0.9rem; margin-bottom:0.3rem; display:flex; align-items:center; gap:0.4rem;">
                        <span class="ms-icon" style="font-size:18px; color:var(--m3-primary);">psychology</span>
                        Evidence-Based Explanation
                    </div>
                    <div style="font-size:0.92rem; color:var(--m3-text); line-height:1.6;">
                        {res.explanation}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if res.key_evidence_points:
                st.markdown("**Key Evidence Highlights:**")
                for pt in res.key_evidence_points:
                    st.markdown(f"- {pt}")

            st.divider()

            # Supporting Sources with Clickable URLs & Publication Dates
            queries_display = (
                " | ".join(f"<code>{q}</code>" for q in res.executed_queries[:2])
                if res.executed_queries
                else f"<code>{claim.search_query}</code>"
            )
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; margin-bottom:0.65rem;">
                    <div style="display:flex; align-items:center; gap:0.4rem; font-weight:600; font-size:0.92rem;">
                        <span class="ms-icon" style="font-size:18px; color:var(--m3-primary);">public</span>
                        Supporting &amp; Evaluated Sources ({len(res.sources)})
                    </div>
                    <div style="font-size:0.8rem; color:var(--m3-muted);">
                        Search Queries: {queries_display}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not res.sources:
                st.caption("No public web sources were returned for this claim.")
            else:
                for s_idx, src in enumerate(res.sources, start=1):
                    pub_date_label = (
                        f"Published: {src.published_date}"
                        if src.published_date
                        else src.recency_note
                    )
                    snippet_text = (
                        src.snippet[:380] + "..."
                        if len(src.snippet) > 380
                        else src.snippet
                    )
                    rel_label, rel_cls = _relationship_badge_meta(
                        src.relationship if src.relationship != "context" else src.alignment_signal
                    )
                    ev_cls_chip = (
                        "m3-chip-verified"
                        if src.evidence_class == "Primary Evidence"
                        else "m3-chip-unverifiable"
                    )

                    st.markdown(
                        f"""
                        <div style="border:1px solid var(--m3-border); border-radius:14px; padding:0.85rem 1rem; margin-bottom:0.55rem; background-color:var(--m3-surface);">
                            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:0.5rem; flex-wrap:wrap;">
                                <div style="font-weight:600; font-size:0.9rem;">
                                    [{s_idx}] <a href="{src.url}" target="_blank" rel="noopener noreferrer" style="color:var(--m3-primary); text-decoration:none;">{src.title}</a>
                                </div>
                                <div style="display:flex; flex-wrap:wrap; gap:0.35rem;">
                                    <span class="m3-chip {rel_cls}" style="font-size:0.7rem;">
                                        {rel_label}
                                    </span>
                                    <span class="m3-chip {ev_cls_chip}" style="font-size:0.7rem;">
                                        {src.evidence_class}
                                    </span>
                                    <span class="m3-chip m3-chip-category" style="font-size:0.7rem;">
                                        {src.source_type} ({src.credibility_score * 100:.0f}%)
                                    </span>
                                </div>
                            </div>
                            <div style="font-size:0.78rem; color:var(--m3-muted); margin:0.25rem 0 0.45rem 0;">
                                <strong>{src.domain}</strong> &bull; {pub_date_label} &bull; Relevance: {src.relevance_score * 100:.0f}% &bull;
                                <a href="{src.url}" target="_blank" rel="noopener noreferrer" style="color:var(--m3-secondary);">{src.url[:70]}{'...' if len(src.url) > 70 else ''}</a>
                            </div>
                            <div style="font-size:0.84rem; color:var(--m3-text); line-height:1.5; border-left:2px solid var(--m3-border); padding-left:0.65rem;">
                                &ldquo;{snippet_text}&rdquo;
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
