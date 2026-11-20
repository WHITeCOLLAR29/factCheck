"""MODULE 2: AI-driven extraction of objectively verifiable factual claims from PDF documents.

Uses Google Gemini with Pydantic schema validation, paragraph-preserving chunking,
cross-chunk deduplication, verbatim claim verification, and untrusted input sandboxing.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from typing import Callable, Dict, List, Optional, Tuple

from config.settings import CONFIG
from models.schemas import (
    ClaimExtractionList,
    ClaimImportance,
    ExtractedClaim,
)
from utils.helpers import normalize_claim_text_for_dedup
from utils.llm_client import GeminiClient
from utils.prompts import (
    CLAIM_EXTRACTION_SYSTEM_INSTRUCTION,
    build_claim_extraction_prompt,
)

# Minimum character length or factual signal check to skip trivial headers/footers
_FACTUAL_SIGNAL_PATTERN = re.compile(
    r"(\d+|%|\$|€|£|¥|\b(?:million|billion|trillion|percent|founded|established|launched|"
    r"acquired|published|enacted|signed|invented|discovered|measured|recorded|capacity|"
    r"frequency|voltage|watt|bandwidth|latency|population|gdp|inflation|revenue|profit|"
    r"century|decade|year|january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b)",
    re.IGNORECASE,
)


def build_paragraph_aware_chunks(
    pages: List[Tuple[int, str]],
    max_chunk_chars: int = CONFIG.CHUNK_CHAR_SIZE,
    overlap_paragraphs: int = 1,
) -> List[str]:
    """Split page-indexed document text into paragraph-preserving chunks annotated with `[PAGE N]`.

    Never splits mid-sentence or mid-paragraph unless a single paragraph exceeds `max_chunk_chars`.
    Includes slight paragraph overlap so claims spanning section boundaries retain context.
    """
    # Flatten into a sequence of (page_num, paragraph_text)
    annotated_Units: List[Tuple[int, str]] = []
    for page_num, page_text in pages:
        raw_paras = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        for para in raw_paras:
            annotated_Units.append((page_num, para))

    if not annotated_Units:
        return []

    chunks: List[str] = []
    current_items: List[Tuple[int, str]] = []
    current_chars = 0

    for page_num, para in annotated_Units:
        entry = f"[PAGE {page_num}]\n{para}\n\n"
        entry_len = len(entry)

        # If a single paragraph is larger than max_chunk_chars, flush current and slice on sentence boundaries
        if entry_len > max_chunk_chars:
            if current_items:
                chunks.append(
                    "".join(f"[PAGE {p}]\n{t}\n\n" for p, t in current_items).strip()
                )
                current_items = []
                current_chars = 0

            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf: List[str] = []
            buf_len = 0
            for sent in sentences:
                if buf_len + len(sent) + 20 > max_chunk_chars and buf:
                    chunks.append(f"[PAGE {page_num}]\n" + " ".join(buf))
                    buf = [sent]
                    buf_len = len(sent)
                else:
                    buf.append(sent)
                    buf_len += len(sent) + 1
            if buf:
                chunks.append(f"[PAGE {page_num}]\n" + " ".join(buf))
            continue

        if current_chars + entry_len > max_chunk_chars and current_items:
            chunks.append(
                "".join(f"[PAGE {p}]\n{t}\n\n" for p, t in current_items).strip()
            )
            # Carry over last `overlap_paragraphs` for context continuity
            carry = current_items[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
            current_items = list(carry) + [(page_num, para)]
            current_chars = sum(len(f"[PAGE {p}]\n{t}\n\n") for p, t in current_items)
        else:
            current_items.append((page_num, para))
            current_chars += entry_len

    if current_items:
        chunks.append(
            "".join(f"[PAGE {p}]\n{t}\n\n" for p, t in current_items).strip()
        )

    return chunks


def _chunk_has_factual_signal(chunk_text: str) -> bool:
    """Quick pre-filter to avoid wasting LLM tokens on chunks that are just table-of-contents dots or short boilerplate."""
    stripped = re.sub(r"\[PAGE \d+\]", "", chunk_text).strip()
    if len(stripped) < 35:
        return False
    # Even if no regex keyword matches, substantial prose paragraphs (>= 120 chars) may hold historical/scientific claims
    if len(stripped) >= 120:
        return True
    return bool(_FACTUAL_SIGNAL_PATTERN.search(stripped))


def _is_duplicate_claim(
    candidate: ExtractedClaim,
    existing_claims: List[ExtractedClaim],
    seen_norm_keys: set[str],
    similarity_threshold: float = 0.86,
) -> bool:
    """Detect exact or near-duplicate overlapping claims across chunks."""
    norm_cand = normalize_claim_text_for_dedup(candidate.claim_text)
    if not norm_cand or norm_cand in seen_norm_keys:
        return True

    for existing in existing_claims:
        norm_exist = normalize_claim_text_for_dedup(existing.claim_text)
        # Check substring containment for overlapping chunk extractions
        if norm_cand in norm_exist or norm_exist in norm_cand:
            return True
        # Check sequence similarity ratio
        ratio = difflib.SequenceMatcher(None, norm_cand, norm_exist).ratio()
        if ratio >= similarity_threshold:
            return True

    return False


def _find_best_page_and_context(
    claim: ExtractedClaim,
    pages_dict: Dict[int, str],
    valid_page_numbers: List[int],
) -> Tuple[int, str]:
    """Ensure the claim's page_number exists in the PDF and resolve surrounding paragraph context from actual PDF text."""
    page_num = claim.page_number if claim.page_number in pages_dict else valid_page_numbers[0]

    # Search for verbatim substring match across pages to confirm exact page & paragraph context
    claim_snippet = " ".join(claim.claim_text.split())[:60].lower()

    # Check reported page first, then all pages
    search_order = [page_num] + [p for p in valid_page_numbers if p != page_num]
    for p_num in search_order:
        page_text = pages_dict.get(p_num, "")
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        for para in paragraphs:
            norm_para = " ".join(para.split()).lower()
            if claim_snippet and claim_snippet in norm_para:
                return p_num, para

    # Fallback: use LLM's surrounding_context if non-empty, otherwise the first matching paragraph on `page_num`
    if claim.surrounding_context and claim.surrounding_context.strip():
        return page_num, " ".join(claim.surrounding_context.split())

    fallback_page_text = pages_dict.get(page_num, "")
    first_para = fallback_page_text.split("\n\n")[0].strip() if fallback_page_text else claim.claim_text
    return page_num, first_para[:600]


def _importance_weight(importance: ClaimImportance) -> int:
    """Map ClaimImportance enum to numeric sort priority."""
    mapping = {
        ClaimImportance.HIGH: 3,
        ClaimImportance.MEDIUM: 2,
        ClaimImportance.LOW: 1,
    }
    return mapping.get(importance, 2)


class ClaimExtractor:
    """Extracts, validates, deduplicates, and numbers factual claims from PDF text using Gemini."""

    def __init__(self, llm_client: GeminiClient) -> None:
        self.llm_client = llm_client

    def extract_claims(
        self,
        pages: List[Tuple[int, str]],
        max_claims: int = CONFIG.DEFAULT_MAX_CLAIMS,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[str, List[ExtractedClaim]]:
        """Extract up to `max_claims` objectively verifiable factual claims across document chunks.

        Returns:
            Tuple of (document_topic, list_of_extracted_claims)
        """
        if not pages:
            return "Empty Document", []

        pages_dict: Dict[int, str] = {p_num: p_txt for p_num, p_txt in pages}
        valid_page_numbers = [p_num for p_num, _ in pages]

        all_chunks = build_paragraph_aware_chunks(
            pages=pages,
            max_chunk_chars=CONFIG.CHUNK_CHAR_SIZE,
            overlap_paragraphs=1,
        )

        # Filter out trivial/empty chunks to save tokens
        eligible_chunks = [c for c in all_chunks if _chunk_has_factual_signal(c)]
        if not eligible_chunks:
            return "No Verifiable Factual Content", []

        # Cap total chunks sent to LLM to control token usage while sampling across the full document
        max_chunks_cap = 8
        if len(eligible_chunks) <= max_chunks_cap:
            selected_chunks = eligible_chunks
        else:
            # Uniformly sample across beginning, middle, and end chunks of long documents
            step = len(eligible_chunks) / max_chunks_cap
            selected_chunks = [
                eligible_chunks[min(len(eligible_chunks) - 1, int(i * step))]
                for i in range(max_chunks_cap)
            ]

        claims_per_chunk = max(
            3, math.ceil((max_claims * 1.35) / len(selected_chunks))
        )

        collected_claims: List[ExtractedClaim] = []
        seen_normalized: set[str] = set()
        detected_topic: str = "General Document"

        for idx, chunk_text in enumerate(selected_chunks, start=1):
            if progress_callback:
                progress_callback(
                    idx / len(selected_chunks),
                    f"Analyzing document chunk {idx} of {len(selected_chunks)}...",
                )

            prompt = build_claim_extraction_prompt(
                annotated_text=chunk_text,
                max_claims=claims_per_chunk,
            )

            extraction_result: ClaimExtractionList = (
                self.llm_client.generate_structured(
                    system_instruction=CLAIM_EXTRACTION_SYSTEM_INSTRUCTION,
                    user_prompt=prompt,
                    response_schema=ClaimExtractionList,
                )
            )

            if idx == 1 and extraction_result.document_topic:
                detected_topic = extraction_result.document_topic

            for claim in extraction_result.claims:
                # Reject questions if any slipped through
                if claim.claim_text.strip().endswith("?"):
                    continue

                if _is_duplicate_claim(claim, collected_claims, seen_normalized):
                    continue

                # Anchor page number and surrounding paragraph context against actual PDF pages
                verified_page, verified_context = _find_best_page_and_context(
                    claim=claim,
                    pages_dict=pages_dict,
                    valid_page_numbers=valid_page_numbers,
                )
                claim.page_number = verified_page
                claim.surrounding_context = verified_context

                if not claim.entity_or_subject and claim.entities:
                    claim.entity_or_subject = claim.entities[0]

                norm_key = normalize_claim_text_for_dedup(claim.claim_text)
                seen_normalized.add(norm_key)
                collected_claims.append(claim)

            # Early exit optimization: if we already have >= 1.5x max_claims high-importance claims, stop making extra API calls
            high_imp_count = sum(
                1 for c in collected_claims if c.importance == ClaimImportance.HIGH
            )
            if len(collected_claims) >= int(max_claims * 1.5) and high_imp_count >= max_claims:
                break

        # Sort by importance (High > Medium > Low), check_worthiness descending, then page_number ascending
        collected_claims.sort(
            key=lambda c: (
                -_importance_weight(c.importance),
                -c.check_worthiness,
                c.page_number,
            )
        )

        final_claims = collected_claims[:max_claims]

        # Re-order selected claims in document page sequence and assign continuous sequential IDs across chunks
        final_claims.sort(key=lambda c: c.page_number)
        for i, claim in enumerate(final_claims, start=1):
            claim.claim_id = f"CLM-{i:03d}"

        return detected_topic, final_claims

    @staticmethod
    def to_structured_json(
        document_topic: str,
        claims: List[ExtractedClaim],
    ) -> str:
        """Serialize extracted claims into validated structured JSON string."""
        payload = ClaimExtractionList(
            document_topic=document_topic,
            claims=claims,
        )
        return payload.model_dump_json(indent=2)
