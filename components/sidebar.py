"""Material Design 3 Sidebar navigation, theme switcher, API connection status, and settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import streamlit as st

from config.settings import CONFIG, get_secret
from utils.security import mask_secret

NAV_PAGES: List[str] = [
    "Home",
    "Document Analysis",
    "Analysis Dashboard",
    "Claim Details",
    "Report",
]


@dataclass
class SidebarSettings:
    """Resolved runtime parameters, theme mode, and API credentials from the global sidebar."""

    active_page: str
    theme_mode: str
    gemini_api_key: Optional[str]
    tavily_api_key: Optional[str]
    selected_model: str
    max_claims: int
    search_depth: str
    max_sources_per_claim: int

    @property
    def is_api_ready(self) -> bool:
        """Return True when both Gemini and Tavily credentials are present."""
        return bool(self.gemini_api_key and self.tavily_api_key)


def navigate_to(page_name: str) -> None:
    """Programmatically switch the active navigation page in session state."""
    if page_name in NAV_PAGES:
        st.session_state["active_page"] = page_name


def render_sidebar() -> SidebarSettings:
    """Render the Material 3 global sidebar with navigation, theme toggle, API status, and help."""
    if "active_page" not in st.session_state or st.session_state["active_page"] not in NAV_PAGES:
        st.session_state["active_page"] = "Home"
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "light"

    with st.sidebar:
        # 1. Product Logo & Brand Identity
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.75rem; padding: 0.25rem 0 0.75rem 0;">
                <div style="width:40px; height:40px; border-radius:12px; background-color:var(--m3-primary); color:var(--m3-on-primary); display:flex; align-items:center; justify-content:center; font-weight:700;">
                    <span class="ms-icon" style="font-size:22px;">verified_user</span>
                </div>
                <div>
                    <div style="font-size:1.15rem; font-weight:700; line-height:1.2; color:var(--m3-text);">FactCheck AI</div>
                    <div style="font-size:0.75rem; color:var(--m3-muted);">The Truth Layer</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Primary Navigation (Clean labels, no emojis)
        st.markdown(
            "<div style='font-size:0.74rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:var(--m3-muted); margin: 0.5rem 0 0.35rem 0;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        current_idx = NAV_PAGES.index(st.session_state["active_page"])
        selected_page = st.radio(
            "Workspace Navigation",
            options=NAV_PAGES,
            index=current_idx,
            label_visibility="collapsed",
        )
        if selected_page != st.session_state["active_page"]:
            st.session_state["active_page"] = selected_page
            st.rerun()

        st.divider()

        # 3. Appearance / Theme Toggle
        st.markdown(
            "<div style='font-size:0.74rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:var(--m3-muted); margin-bottom:0.35rem;'>Appearance</div>",
            unsafe_allow_html=True,
        )
        dark_enabled = st.toggle(
            "Dark Theme (Material You)",
            value=(st.session_state["theme_mode"] == "dark"),
            help="Switch between Material 3 Light (#F4F6FC) and Dark (#191C24) palettes.",
        )
        new_theme = "dark" if dark_enabled else "light"
        if new_theme != st.session_state["theme_mode"]:
            st.session_state["theme_mode"] = new_theme
            st.rerun()

        st.divider()

        # 4. API Connection Status & Credentials
        env_gemini = get_secret("GEMINI_API_KEY")
        env_tavily = get_secret("TAVILY_API_KEY")

        override_gemini = st.session_state.get("override_gemini_key", "")
        override_tavily = st.session_state.get("override_tavily_key", "")

        resolved_gemini = get_secret("GEMINI_API_KEY", override_gemini)
        resolved_tavily = get_secret("TAVILY_API_KEY", override_tavily)

        gemini_ok = bool(resolved_gemini)
        tavily_ok = bool(resolved_tavily)

        st.markdown(
            "<div style='font-size:0.74rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:var(--m3-muted); margin-bottom:0.5rem;'>API Connection Status</div>",
            unsafe_allow_html=True,
        )

        gemini_chip_cls = "m3-chip-verified" if gemini_ok else "m3-chip-false"
        gemini_icon = "check_circle" if gemini_ok else "error"
        gemini_state_lbl = "Connected" if gemini_ok else "Missing Key"

        tavily_chip_cls = "m3-chip-verified" if tavily_ok else "m3-chip-false"
        tavily_icon = "check_circle" if tavily_ok else "error"
        tavily_state_lbl = "Connected" if tavily_ok else "Missing Key"

        st.markdown(
            f"""
            <div class="m3-tonal-card" style="padding: 0.85rem 1rem; margin-bottom: 0.65rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    <span style="font-size:0.83rem; font-weight:600;">Google Gemini</span>
                    <span class="m3-chip {gemini_chip_cls}">
                        <span class="ms-icon" style="font-size:15px;">{gemini_icon}</span>{gemini_state_lbl}
                    </span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.83rem; font-weight:600;">Tavily Search</span>
                    <span class="m3-chip {tavily_chip_cls}">
                        <span class="ms-icon" style="font-size:15px;">{tavily_icon}</span>{tavily_state_lbl}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Configure API Keys & Parameters", expanded=not (gemini_ok and tavily_ok)):
            if env_gemini and env_tavily:
                st.caption(
                    f"Environment keys active (`{mask_secret(env_gemini)}` / `{mask_secret(env_tavily)}`). "
                    "Enter below only to override for this session."
                )

            input_gemini = st.text_input(
                "Gemini API Key",
                value=override_gemini,
                type="password",
                placeholder="AIza...",
                help="Google Gemini API key for the official google-genai SDK.",
            )
            input_tavily = st.text_input(
                "Tavily Search API Key",
                value=override_tavily,
                type="password",
                placeholder="tvly-...",
                help="Tavily API key for live web evidence search.",
            )
            st.session_state["override_gemini_key"] = input_gemini
            st.session_state["override_tavily_key"] = input_tavily

            resolved_gemini = get_secret("GEMINI_API_KEY", input_gemini)
            resolved_tavily = get_secret("TAVILY_API_KEY", input_tavily)

            selected_model = st.selectbox(
                "Gemini Model",
                options=CONFIG.AVAILABLE_GEMINI_MODELS,
                index=CONFIG.AVAILABLE_GEMINI_MODELS.index(
                    st.session_state.get("selected_model", CONFIG.DEFAULT_GEMINI_MODEL)
                )
                if st.session_state.get("selected_model") in CONFIG.AVAILABLE_GEMINI_MODELS
                else 0,
                key="sidebar_selected_model",
            )
            st.session_state["selected_model"] = selected_model

            max_claims = st.slider(
                "Max Claims to Extract",
                min_value=CONFIG.MIN_CLAIMS_LIMIT,
                max_value=CONFIG.MAX_CLAIMS_LIMIT,
                value=int(st.session_state.get("max_claims", CONFIG.DEFAULT_MAX_CLAIMS)),
                step=1,
                key="sidebar_max_claims",
            )
            st.session_state["max_claims"] = max_claims

            search_depth = st.selectbox(
                "Web Search Depth",
                options=["advanced", "basic"],
                index=0 if st.session_state.get("search_depth", "advanced") == "advanced" else 1,
                key="sidebar_search_depth",
            )
            st.session_state["search_depth"] = search_depth

            max_sources_per_claim = st.slider(
                "Evidence Sources per Claim",
                min_value=3,
                max_value=8,
                value=int(
                    st.session_state.get(
                        "max_sources_per_claim", CONFIG.DEFAULT_SEARCH_RESULTS_PER_CLAIM
                    )
                ),
                step=1,
                key="sidebar_max_sources",
            )
            st.session_state["max_sources_per_claim"] = max_sources_per_claim

            st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
            if st.button(
                "Test API Connections",
                type="secondary",
                use_container_width=True,
                key="sidebar_test_api_btn",
            ):
                from modules.web_search import TavilySearchService, WebSearchError
                from utils.llm_client import GeminiClient, LLMServiceError

                with st.spinner("Testing Gemini & Tavily API connections..."):
                    # Test Gemini
                    if not resolved_gemini:
                        st.error("Gemini: Missing `GEMINI_API_KEY`.")
                    else:
                        try:
                            g_client = GeminiClient(
                                api_key=resolved_gemini,
                                model_name=selected_model,
                            )
                            g_ok, g_msg = g_client.test_connection()
                            if g_ok:
                                st.success(g_msg)
                            else:
                                st.error(g_msg)
                        except LLMServiceError as g_err:
                            st.error(str(g_err))

                    # Test Tavily
                    if not resolved_tavily:
                        st.error("Tavily: Missing `TAVILY_API_KEY`.")
                    else:
                        try:
                            t_client = TavilySearchService(
                                api_key=resolved_tavily,
                                search_depth="basic",
                                max_results=2,
                            )
                            t_ok, t_msg = t_client.test_connection()
                            if t_ok:
                                st.success(t_msg)
                            else:
                                st.error(t_msg)
                        except WebSearchError as t_err:
                            st.error(str(t_err))

        # 5. Help & About Section
        with st.expander("Help & About FactCheck AI", expanded=False):
            st.markdown(
                f"""
                **{CONFIG.APP_NAME}**  
                *{CONFIG.APP_TAGLINE}*

                {CONFIG.APP_DESCRIPTION}

                **How it works:**
                1. **Upload:** Provide any text-based PDF document up to {CONFIG.MAX_PDF_SIZE_MB} MB.
                2. **Extract:** Gemini identifies concrete statistics, dates, and factual assertions.
                3. **Verify:** Tavily retrieves live web sources scored by domain authority.
                4. **Report:** Export evidence-backed PDF and CSV audit reports.

                `Version {CONFIG.APP_VERSION}`
                """
            )

    return SidebarSettings(
        active_page=st.session_state["active_page"],
        theme_mode=st.session_state["theme_mode"],
        gemini_api_key=resolved_gemini,
        tavily_api_key=resolved_tavily,
        selected_model=selected_model,
        max_claims=max_claims,
        search_depth=search_depth,
        max_sources_per_claim=max_sources_per_claim,
    )
