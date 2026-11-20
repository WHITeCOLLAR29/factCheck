"""Security utilities for file validation, untrusted text neutralization, session cleanup, and secret masking."""

from __future__ import annotations

import hashlib
import re
from typing import Any, MutableMapping, Optional, Tuple

from config.settings import CONFIG

PDF_MAGIC_BYTES = b"%PDF-"

# Session state keys holding document or verification payloads that must be cleared on reset / new analysis
SENSITIVE_SESSION_KEYS: Tuple[str, ...] = (
    "extracted_pdf",
    "factcheck_report",
    "last_file_hash",
    "claim_evidence_bundles",
    "partial_verification_results",
    "cached_extracted_claims",
    "tavily_query_cache",
    "gemini_extraction_cache",
)


def clear_sensitive_session_data(
    session_state: MutableMapping[str, Any],
    keep_recent_history: bool = False,
) -> None:
    """Purge uploaded PDF bytes, extracted paragraphs, cached queries, and active reports from session state."""
    for key in SENSITIVE_SESSION_KEYS:
        session_state.pop(key, None)
    if not keep_recent_history:
        session_state.pop("recent_reports", None)


def validate_pdf_upload(
    file_bytes: bytes,
    filename: str,
    max_size_mb: int = CONFIG.MAX_PDF_SIZE_MB,
) -> Tuple[bool, str]:
    """Validate that uploaded bytes represent a genuine, appropriately sized PDF file."""
    if not file_bytes:
        return False, "The uploaded file is empty (0 bytes)."

    if not filename or not filename.lower().endswith(".pdf"):
        return False, "Unsupported file format. Please upload a valid `.pdf` document."

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return (
            False,
            f"File size ({size_mb:.2f} MB) exceeds the maximum allowed limit of {max_size_mb} MB.",
        )

    header_window = file_bytes[:1024]
    if PDF_MAGIC_BYTES not in header_window:
        return (
            False,
            "Invalid PDF header signature. The file is corrupted or is not a genuine PDF document.",
        )

    return True, "Valid PDF file."


def mask_secret(secret: Optional[str]) -> str:
    """Return a safely masked representation of an API key for UI display."""
    if not secret:
        return "Not Configured"
    cleaned = secret.strip()
    if len(cleaned) <= 8:
        return "••••••••"
    return f"{cleaned[:4]}{'•' * 12}{cleaned[-4:]}"


def scrub_secrets_from_message(
    message: str,
    secrets_to_redact: Optional[list[Optional[str]]] = None,
) -> str:
    """Remove any accidental API keys or tokens from exception messages before showing or logging."""
    if not message:
        return "An unknown error occurred."

    sanitized = message
    if secrets_to_redact:
        for secret in secrets_to_redact:
            if secret and len(secret.strip()) >= 6:
                sanitized = sanitized.replace(secret.strip(), "[REDACTED_KEY]")

    sanitized = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "[REDACTED_GEMINI_KEY]", sanitized)
    sanitized = re.sub(r"tvly-[0-9A-Za-z\-_]{15,}", "[REDACTED_TAVILY_KEY]", sanitized)
    sanitized = re.sub(
        r"(key|token|api_key|apikey)=([0-9A-Za-z\-_]{12,})",
        r"\1=[REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


def sanitize_text_input(text: str, max_length: int = 100000) -> str:
    """Clean extracted text by removing null bytes and non-printable control characters while preserving paragraphs."""
    if not text:
        return ""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned[:max_length].strip()


def neutralize_untrusted_document_text(text: str) -> str:
    """Neutralize boundary-escape tags in untrusted PDF text so document content cannot break XML delimiters."""
    if not text:
        return ""
    neutralized = text.replace("</UNTRUSTED_PDF_CONTENT>", "[END_TAG_ESCAPED]")
    neutralized = neutralized.replace("<UNTRUSTED_PDF_CONTENT>", "[START_TAG_ESCAPED]")
    return neutralized


def neutralize_untrusted_search_result(text: str) -> str:
    """Neutralize untrusted web search titles/snippets so search results never act as prompt instructions."""
    if not text:
        return ""
    cleaned = sanitize_text_input(text, max_length=3500)
    cleaned = cleaned.replace("</UNTRUSTED_WEB_EVIDENCE>", "[WEB_END_TAG_ESCAPED]")
    cleaned = cleaned.replace("<UNTRUSTED_WEB_EVIDENCE>", "[WEB_START_TAG_ESCAPED]")
    cleaned = cleaned.replace("</UNTRUSTED_PDF_CONTENT>", "[PDF_END_TAG_ESCAPED]")
    return cleaned


def compute_sha256(data: bytes) -> str:
    """Compute a short SHA-256 fingerprint for caching and report auditing."""
    return hashlib.sha256(data).hexdigest()[:16]
