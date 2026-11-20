"""Configuration, Material Design 3 color tokens, and environment settings for FactCheck AI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration parameters."""

    APP_NAME: str = "FactCheck AI"
    APP_TAGLINE: str = "The Truth Layer for Every Document."
    APP_DESCRIPTION: str = (
        "AI-powered document verification backed by real-world evidence."
    )
    APP_VERSION: str = "2.1.0"

    # Gemini Model Options (official google-genai models)
    DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
    AVAILABLE_GEMINI_MODELS: List[str] = field(
        default_factory=lambda: [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ]
    )

    # PDF Processing Limits
    MAX_PDF_SIZE_MB: int = 25
    MAX_PDF_PAGES: int = 75
    CHUNK_CHAR_SIZE: int = 6000
    CHUNK_CHAR_OVERLAP: int = 400

    # Claim Extraction & Verification Defaults
    DEFAULT_MAX_CLAIMS: int = 10
    MIN_CLAIMS_LIMIT: int = 3
    MAX_CLAIMS_LIMIT: int = 30
    DEFAULT_SEARCH_RESULTS_PER_CLAIM: int = 5
    DEFAULT_SEARCH_DEPTH: str = "advanced"
    LLM_TEMPERATURE: float = 0.05
    API_TIMEOUT_SECONDS: int = 45


# Material Design 3 Light & Dark Theme Tokens
M3_LIGHT_PALETTE: Dict[str, str] = {
    "primary": "#3659C9",
    "on_primary": "#FFFFFF",
    "primary_container": "#E5ECFF",
    "on_primary_container": "#122B7A",
    "secondary": "#6677B8",
    "secondary_container": "#EEF1FA",
    "background": "#F4F6FC",
    "surface": "#FFFFFF",
    "surface_tonal": "#ECF0FA",
    "surface_border": "#DCE2F2",
    "text": "#20283B",
    "muted_text": "#667085",
    "success": "#16845B",
    "success_container": "#E6F5EE",
    "warning": "#B7791F",
    "warning_container": "#FEF5E7",
    "error": "#C43D4D",
    "error_container": "#FCEBEC",
    "neutral": "#667085",
    "neutral_container": "#EEF1F6",
}

M3_DARK_PALETTE: Dict[str, str] = {
    "primary": "#B5C6FF",
    "on_primary": "#142766",
    "primary_container": "#28396B",
    "on_primary_container": "#DDE4FF",
    "secondary": "#C0C9EC",
    "secondary_container": "#2B3142",
    "background": "#191C24",
    "surface": "#222631",
    "surface_tonal": "#2A2F3D",
    "surface_border": "#343A4A",
    "text": "#E2E5F0",
    "muted_text": "#9BA3B8",
    "success": "#34D399",
    "success_container": "#15392E",
    "warning": "#FBBF24",
    "warning_container": "#3B2F16",
    "error": "#F87171",
    "error_container": "#3E1F24",
    "neutral": "#9BA3B8",
    "neutral_container": "#2C3240",
}

# Canonical status -> UI Display Labels (Verified, Inaccurate, False, Unverifiable)
VERDICT_DISPLAY_LABELS: Dict[str, str] = {
    "VERIFIED": "Verified",
    "INACCURATE": "Inaccurate",
    "FALSE": "False",
    "UNVERIFIABLE": "Unverifiable",
    "Verified": "Verified",
    "Partially True": "Inaccurate",
    "Inaccurate": "Inaccurate",
    "Refuted": "False",
    "False": "False",
    "Unverifiable": "Unverifiable",
}

# Material 3 semantic colors for charts & badges
VERDICT_COLORS: Dict[str, str] = {
    "VERIFIED": "#16845B",
    "INACCURATE": "#B7791F",
    "FALSE": "#C43D4D",
    "UNVERIFIABLE": "#667085",
    "Verified": "#16845B",
    "Partially True": "#B7791F",
    "Inaccurate": "#B7791F",
    "Refuted": "#C43D4D",
    "False": "#C43D4D",
    "Unverifiable": "#667085",
}


# Domain Credibility Reference Lists
HIGH_AUTHORITY_DOMAINS: List[str] = [
    ".gov",
    ".edu",
    ".mil",
    ".int",
    "who.int",
    "un.org",
    "worldbank.org",
    "imf.org",
    "oecd.org",
    "europa.eu",
    "nature.com",
    "science.org",
    "nejm.org",
    "thelancet.com",
    "bmj.com",
    "ieee.org",
    "acm.org",
    "pubmed.ncbi.nlm.nih.gov",
    "arxiv.org",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "nytimes.com",
    "economist.com",
    "pewresearch.org",
    "ourworldindata.org",
    "statista.com",
]

MEDIUM_AUTHORITY_DOMAINS: List[str] = [
    "wikipedia.org",
    "britannica.com",
    "cnbc.com",
    "forbes.com",
    "theguardian.com",
    "washingtonpost.com",
    "npr.org",
    "pbs.org",
    "scientificamerican.com",
    "mit.edu",
    "harvard.edu",
    "stanford.edu",
    "oxford.ac.uk",
    "techcrunch.com",
    "arstechnica.com",
    "wired.com",
    "investopedia.com",
]

LOW_CREDIBILITY_INDICATORS: List[str] = [
    "blogspot.",
    "wordpress.com",
    "medium.com",
    "substack.com",
    "reddit.com",
    "quora.com",
    "tiktok.com",
    "facebook.com",
    "x.com",
    "twitter.com",
]


def get_active_palette(theme_mode: str = "light") -> Dict[str, str]:
    """Return the Material Design 3 color palette for the active theme ('light' or 'dark')."""
    if str(theme_mode).lower() == "dark":
        return M3_DARK_PALETTE
    return M3_LIGHT_PALETTE


def get_secret(key_name: str, session_override: Optional[str] = None) -> Optional[str]:
    """Safely retrieve an API key with priority:
    1. Explicit session override (if entered in the sidebar).
    2. Streamlit secrets (`st.secrets[key_name]`).
    3. Environment variable fallback (`os.environ.get(key_name)`).

    Never logs or exposes secret values.
    """
    if session_override and session_override.strip():
        return session_override.strip()

    # 1. Read from Streamlit secrets first
    try:
        secret_val = st.secrets[key_name]
        if secret_val is not None:
            cleaned = str(secret_val).strip()
            if cleaned and not cleaned.startswith("your-"):
                return cleaned
    except Exception:
        # Gracefully fall back if secrets.toml is absent, invalid, or key is missing
        pass

    # 2. Safe fallback to environment variables
    env_val = os.environ.get(key_name)
    if env_val and env_val.strip():
        return env_val.strip()

    return None


CONFIG = AppConfig()
