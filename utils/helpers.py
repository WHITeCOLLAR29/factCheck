"""Helper utilities for text chunking, URL normalization, formatting, and Material Design 3 CSS injection."""

from __future__ import annotations

import uuid
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import streamlit as st

from config.settings import get_active_palette


def extract_domain(url: str) -> str:
    """Extract clean root domain from a URL without 'www.' prefix."""
    if not url:
        return "unknown"
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        netloc = parsed.netloc.lower().strip()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or "unknown"
    except Exception:
        return "unknown"


def chunk_pages_for_extraction(
    pages: List[Tuple[int, str]],
    max_chunk_chars: int = 6500,
) -> List[str]:
    """Combine page texts into page-annotated chunks suitable for LLM claim extraction."""
    chunks: List[str] = []
    current_buffer: List[str] = []
    current_length = 0

    for page_num, page_text in pages:
        cleaned = page_text.strip()
        if not cleaned:
            continue

        annotated_page = f"\n--- [PAGE {page_num}] ---\n{cleaned}\n"

        if len(annotated_page) > max_chunk_chars:
            if current_buffer:
                chunks.append("".join(current_buffer))
                current_buffer = []
                current_length = 0

            step = max_chunk_chars - 300
            for start in range(0, len(cleaned), step):
                segment = cleaned[start : start + max_chunk_chars]
                chunks.append(f"\n--- [PAGE {page_num}] ---\n{segment}\n")
            continue

        if current_length + len(annotated_page) > max_chunk_chars and current_buffer:
            chunks.append("".join(current_buffer))
            current_buffer = [annotated_page]
            current_length = len(annotated_page)
        else:
            current_buffer.append(annotated_page)
            current_length += len(annotated_page)

    if current_buffer:
        chunks.append("".join(current_buffer))

    return chunks


def generate_report_id() -> str:
    """Generate a short human-readable report identifier."""
    short_uuid = uuid.uuid4().hex[:8].upper()
    return f"FC-{short_uuid}"


def format_file_size(size_bytes: int) -> str:
    """Format byte size into human-readable KB or MB string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def normalize_claim_text_for_dedup(claim_text: str) -> str:
    """Create a normalized lowercase alphanumeric key to prevent duplicate claims across chunks."""
    alnum = "".join(ch.lower() for ch in claim_text if ch.isalnum() or ch.isspace())
    return " ".join(alnum.split())


def inject_material3_css(theme_mode: str = "light") -> Dict[str, str]:
    """Inject Google Material Design 3 (Material You) CSS variables, typography, and component styles."""
    p = get_active_palette(theme_mode)

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0&display=swap');

    :root {{
        --m3-primary: {p['primary']};
        --m3-on-primary: {p['on_primary']};
        --m3-primary-container: {p['primary_container']};
        --m3-on-primary-container: {p['on_primary_container']};
        --m3-secondary: {p['secondary']};
        --m3-secondary-container: {p['secondary_container']};
        --m3-bg: {p['background']};
        --m3-surface: {p['surface']};
        --m3-surface-tonal: {p['surface_tonal']};
        --m3-border: {p['surface_border']};
        --m3-text: {p['text']};
        --m3-muted: {p['muted_text']};
        --m3-success: {p['success']};
        --m3-success-bg: {p['success_container']};
        --m3-warning: {p['warning']};
        --m3-warning-bg: {p['warning_container']};
        --m3-error: {p['error']};
        --m3-error-bg: {p['error_container']};
        --m3-neutral: {p['neutral']};
        --m3-neutral-bg: {p['neutral_container']};
    }}

    /* Global App Surface & Typography */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--m3-text);
    }}

    .stApp {{
        background-color: var(--m3-bg) !important;
        color: var(--m3-text) !important;
    }}

    section[data-testid="stSidebar"] {{
        background-color: var(--m3-surface) !important;
        border-right: 1px solid var(--m3-border) !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: var(--m3-text);
    }}

    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1220px !important;
    }}

    /* Material Symbols Icon Utility */
    .ms-icon {{
        font-family: 'Material Symbols Rounded';
        font-weight: normal;
        font-style: normal;
        font-size: 20px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        vertical-align: middle;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
    }}

    /* Material 3 Surface Cards */
    .m3-card {{
        background-color: var(--m3-surface);
        border: 1px solid var(--m3-border);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
        margin-bottom: 1rem;
    }}

    .m3-tonal-card {{
        background-color: var(--m3-surface-tonal);
        border: 1px solid var(--m3-border);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    }}

    .m3-feature-card {{
        background-color: var(--m3-surface);
        border: 1px solid var(--m3-border);
        border-radius: 18px;
        padding: 1.35rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
    }}

    .m3-icon-circle {{
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background-color: var(--m3-primary-container);
        color: var(--m3-primary);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.85rem;
    }}

    /* Material 3 KPI Stat Cards */
    .m3-kpi-card {{
        background-color: var(--m3-surface);
        border: 1px solid var(--m3-border);
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
    }}

    .m3-kpi-label {{
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--m3-muted);
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.35rem;
    }}

    .m3-kpi-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--m3-text);
        line-height: 1.2;
    }}

    .m3-kpi-sub {{
        font-size: 0.78rem;
        color: var(--m3-muted);
        margin-top: 0.25rem;
    }}

    /* Material 3 Status Pills / Badges */
    .m3-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }}

    .m3-chip-verified {{
        background-color: var(--m3-success-bg);
        color: var(--m3-success);
    }}

    .m3-chip-inaccurate {{
        background-color: var(--m3-warning-bg);
        color: var(--m3-warning);
    }}

    .m3-chip-false {{
        background-color: var(--m3-error-bg);
        color: var(--m3-error);
    }}

    .m3-chip-unverifiable {{
        background-color: var(--m3-neutral-bg);
        color: var(--m3-neutral);
    }}

    .m3-chip-category {{
        background-color: var(--m3-secondary-container);
        color: var(--m3-secondary);
    }}

    /* Pill-shaped Material 3 Buttons */
    div.stButton > button, div.stDownloadButton > button {{
        border-radius: 999px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.35rem !important;
        border: 1px solid var(--m3-border) !important;
        transition: all 0.15s ease !important;
    }}

    div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {{
        background-color: var(--m3-primary) !important;
        color: var(--m3-on-primary) !important;
        border-color: var(--m3-primary) !important;
    }}

    div.stButton > button[kind="secondary"], div.stDownloadButton > button[kind="secondary"] {{
        background-color: var(--m3-surface) !important;
        color: var(--m3-text) !important;
    }}

    /* File Uploader Dropzone Container */
    [data-testid="stFileUploader"] {{
        background-color: var(--m3-surface);
        border: 1.5px dashed var(--m3-secondary);
        border-radius: 20px;
        padding: 1.1rem 1.25rem;
    }}

    /* Expanders & Containers */
    [data-testid="stExpander"] {{
        background-color: var(--m3-surface) !important;
        border: 1px solid var(--m3-border) !important;
        border-radius: 16px !important;
        overflow: hidden;
        margin-bottom: 0.75rem !important;
    }}

    /* Stepper / Progress Tracker */
    .m3-step-row {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.65rem 0.9rem;
        border-radius: 12px;
        margin-bottom: 0.45rem;
        background-color: var(--m3-surface-tonal);
        border: 1px solid var(--m3-border);
        font-size: 0.9rem;
    }}

    .m3-step-done {{
        border-left: 4px solid var(--m3-success);
    }}

    .m3-step-active {{
        border-left: 4px solid var(--m3-primary);
        background-color: var(--m3-primary-container);
        color: var(--m3-on-primary-container);
        font-weight: 600;
    }}

    /* Confidence Meter Bar */
    .m3-meter-track {{
        width: 100%;
        height: 8px;
        background-color: var(--m3-neutral-bg);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 0.35rem;
    }}

    .m3-meter-fill {{
        height: 100%;
        border-radius: 999px;
    }}

    /* Responsive adjustments for mobile viewports */
    @media (max-width: 768px) {{
        .block-container {{
            padding-top: 1.1rem !important;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
        }}
        .m3-kpi-value {{
            font-size: 1.4rem;
        }}
        .m3-card, .m3-tonal-card, .m3-feature-card {{
            padding: 1.05rem;
            border-radius: 16px;
        }}
        .m3-step-row {{
            flex-wrap: wrap;
            gap: 0.45rem;
            padding: 0.55rem 0.75rem;
        }}
        .m3-chip {{
            font-size: 0.72rem;
            padding: 0.2rem 0.6rem;
        }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return p
