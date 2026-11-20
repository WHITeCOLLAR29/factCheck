"""Live Web Search Component for FactCheck AI (`modules/web_search.py`).

Independently searches the live internet for every extracted factual claim using the Tavily Search API,
retrieves supporting or contradicting evidence, caches repeated queries during a session to conserve
API credits, deduplicates and filters irrelevant results, and preserves the exact claim-to-evidence mapping.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from tavily import TavilyClient
    _HAS_TAVILY = True
except ImportError:
    TavilyClient = None  # type: ignore[assignment,misc]
    _HAS_TAVILY = False

from config.settings import CONFIG, get_secret
from models.schemas import (
    ClaimCategory,
    ClaimEvidenceBundle,
    EvidenceSource,
    ExtractedClaim,
)
from modules.source_evaluator import (
    Canonicalize_url,
    compute_average_credibility,
    evaluate_and_filter_sources_for_claim,
)
from utils.security import (
    neutralize_untrusted_search_result,
    sanitize_text_input,
    scrub_secrets_from_message,
)

logger = logging.getLogger(__name__)

# Process-level fallback cache when running outside an active Streamlit runtime context
_PROCESS_QUERY_CACHE: Dict[str, List[dict]] = {}


class WebSearchError(Exception):
    """Raised when the Tavily Search API fails unrecoverably or is misconfigured."""


def _get_session_query_cache() -> Dict[str, List[dict]]:
    """Return a session-scoped dictionary for caching Tavily search queries."""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            if "tavily_query_cache" not in st.session_state:
                st.session_state["tavily_query_cache"] = {}
            return st.session_state["tavily_query_cache"]
    except Exception:
        pass
    return _PROCESS_QUERY_CACHE


def _authority_hint_for_category(category: ClaimCategory) -> str:
    """Return a concise primary-source search hint tailored to the claim's category."""
    hints = {
        ClaimCategory.STATISTICAL: "official statistics report data",
        ClaimCategory.POPULATION_DEMOGRAPHIC: "census official statistics World Bank OECD",
        ClaimCategory.FINANCIAL: "SEC filing annual report financial statement earnings",
        ClaimCategory.COMPANY_ORG: "official press release annual report filing",
        ClaimCategory.TECHNICAL_PRODUCT: "official specifications datasheet documentation",
        ClaimCategory.SCIENTIFIC: "peer-reviewed study journal publication",
        ClaimCategory.HISTORICAL: "historical record archive primary source",
        ClaimCategory.POLICY_LEGAL: "official legislation government regulation statute",
        ClaimCategory.GENERAL: "official report primary source",
    }
    return hints.get(category, "official source report")


def generate_claim_search_queries(claim: ExtractedClaim) -> List[str]:
    """Generate targeted, non-redundant search queries from an ExtractedClaim following the 4-part strategy:
    1. Search the independently verifiable fact + relevant date/reporting period (primary high-signal query).
    2. Search the exact claim statement (concise verbatim core).
    3. Search for authoritative primary sources (government statistics, SEC filings, product docs, or academic papers).
    """
    queries: List[str] = []
    seen_lower: set[str] = set()

    def _add_query(q: str) -> None:
        cleaned = " ".join(q.split()).strip()
        # Strip double quotes that might over-constrain search engines if too long
        if len(cleaned) >= 6 and cleaned.lower() not in seen_lower:
            seen_lower.add(cleaned.lower())
            queries.append(cleaned[:380])

    # 1. Independently verifiable fact query combining entity, metric/number, search_query, and date/reporting period
    base_fact_query = claim.search_query
    if claim.date_or_period and claim.date_or_period.lower() not in base_fact_query.lower():
        base_fact_query = f"{base_fact_query} {claim.date_or_period}"
    if (
        claim.entity_or_subject
        and claim.entity_or_subject.lower() not in base_fact_query.lower()
    ):
        base_fact_query = f"{claim.entity_or_subject} {base_fact_query}"
    _add_query(base_fact_query)

    # 2. Authoritative primary-source query (combines core entity + numeric value + date + category authority hint)
    auth_parts: List[str] = []
    if claim.entity_or_subject:
        auth_parts.append(claim.entity_or_subject)
    elif claim.entities:
        auth_parts.extend(claim.entities[:2])

    if claim.numeric_value:
        unit_str = f" {claim.numeric_unit}" if claim.numeric_unit else ""
        auth_parts.append(f"{claim.numeric_value}{unit_str}")

    if claim.date_or_period:
        auth_parts.append(claim.date_or_period)

    auth_parts.append(_authority_hint_for_category(claim.category))
    auth_query = " ".join(auth_parts).strip()
    if len(auth_query.split()) >= 4:
        _add_query(auth_query)

    # 3. Exact claim search (trimmed to core 18 words so search engines match verbatim statements)
    exact_words = claim.claim_text.split()
    exact_core = " ".join(exact_words[:18])
    _add_query(exact_core)

    return queries


def _extract_genuine_published_date(item: dict) -> Optional[str]:
    """Extract publication date strictly as provided by the search result without ever fabricating a date."""
    for field_name in ("published_date", "publishedDate", "date"):
        val = item.get(field_name)
        if val is not None:
            val_str = str(val).strip()
            if val_str and val_str.lower() not in ("none", "null", "n/a", "unknown", ""):
                return sanitize_text_input(val_str, max_length=64)
    return None


class TavilySearchService:
    """Reusable live web search client powered by the Tavily Search API.

    Features:
    - Secure API key resolution from `st.secrets`, environment variables, or explicit override.
    - Multi-facet search query generation per claim.
    - Credit-conscious two-stage search execution (runs secondary authoritative query only if primary query yields insufficient high-authority evidence).
    - Session-level query caching to avoid repeated API calls.
    - Request timeouts and exponential backoff retry handling.
    - Strict untrusted-data sanitization and URL deduplication.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: str = CONFIG.DEFAULT_SEARCH_DEPTH,
        max_results: int = CONFIG.DEFAULT_SEARCH_RESULTS_PER_CLAIM,
        timeout_seconds: int = CONFIG.API_TIMEOUT_SECONDS,
    ) -> None:
        resolved_key = get_secret("TAVILY_API_KEY", session_override=api_key)
        if not resolved_key:
            raise WebSearchError(
                "Tavily API Key is missing. Please configure `TAVILY_API_KEY` in `.streamlit/secrets.toml`, environment variables, or the sidebar."
            )
        if not _HAS_TAVILY:
            raise WebSearchError(
                "The `tavily-python` package is not installed. Please run `pip install -r requirements.txt`."
            )

        self._api_key = resolved_key
        self.search_depth = (
            search_depth if search_depth in ("basic", "advanced") else "advanced"
        )
        self.max_results = max(2, min(max_results, 10))
        self.timeout_seconds = max(10, min(timeout_seconds, 90))

        try:
            self.client = TavilyClient(api_key=self._api_key)
        except Exception as exc:
            safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
            raise WebSearchError(
                f"Failed to initialize Tavily Search client: {safe_msg}"
            ) from exc

    def test_connection(self) -> tuple[bool, str]:
        """Verify Tavily Search API connectivity with a minimal basic query without exposing credentials."""
        try:
            response = self.client.search(
                query="fact check test",
                search_depth="basic",
                max_results=1,
                include_answer=False,
                include_raw_content=False,
            )
            if isinstance(response, dict) and "results" in response:
                return True, "Connected to Tavily Search API."
            return False, "Tavily returned an unexpected response."
        except Exception as exc:
            safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
            return False, f"Tavily connection failed: {safe_msg}"

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    def _execute_raw_query(self, query: str) -> List[dict]:
        """Execute a single Tavily search call with timeout and session cache lookup."""
        cache = _get_session_query_cache()
        cache_key = f"{self.search_depth}:{self.max_results}:{query.strip().lower()}"
        if cache_key in cache:
            return cache[cache_key]

        search_kwargs = {
            "query": query[:380],
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": (self.search_depth == "advanced"),
            "timeout": self.timeout_seconds,
        }

        try:
            response = self.client.search(**search_kwargs)
        except TypeError:
            # Compatibility fallback if an older tavily-python version does not accept `timeout` keyword
            search_kwargs.pop("timeout", None)
            response = self.client.search(**search_kwargs)

        raw_results = (
            response.get("results", [])
            if isinstance(response, dict)
            else []
        )
        if isinstance(raw_results, list):
            cache[cache_key] = raw_results
            return raw_results
        return []

    def _parse_raw_results(self, raw_items: List[dict]) -> List[EvidenceSource]:
        """Convert raw Tavily result dicts into sanitized EvidenceSource objects without inventing URLs or dates."""
        parsed_sources: List[EvidenceSource] = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            raw_url = (item.get("url") or "").strip()
            # Never invent URLs; require a valid http/https URL
            if not raw_url or not raw_url.lower().startswith(("http://", "https://")):
                continue

            title = neutralize_untrusted_search_result(
                item.get("title") or "Untitled Web Source"
            )[:300]
            snippet = neutralize_untrusted_search_result(
                item.get("content") or ""
            )[:1400]

            # If advanced search returned raw_content, keep an extended excerpt so verification doesn't rely solely on a short snippet
            raw_extended = item.get("raw_content")
            extended_excerpt: Optional[str] = None
            if isinstance(raw_extended, str) and len(raw_extended.strip()) > len(snippet):
                extended_excerpt = neutralize_untrusted_search_result(raw_extended)[:2400]

            # Require either a snippet or extended content so empty placeholder pages are ignored
            if not snippet and not extended_excerpt:
                continue

            raw_score = item.get("score", 0.5)
            try:
                relevance_score = max(0.0, min(1.0, float(raw_score)))
            except (TypeError, ValueError):
                relevance_score = 0.5

            pub_date = _extract_genuine_published_date(item)

            parsed_sources.append(
                EvidenceSource(
                    title=title or "Untitled Web Source",
                    url=raw_url,
                    snippet=snippet or (extended_excerpt[:400] if extended_excerpt else ""),
                    raw_content_excerpt=extended_excerpt,
                    published_date=pub_date,
                    relevance_score=relevance_score,
                )
            )

        return parsed_sources

    def search_for_claim(self, claim: ExtractedClaim) -> ClaimEvidenceBundle:
        """Search the internet independently for a single ExtractedClaim and return a structured ClaimEvidenceBundle.

        Strategy:
        1. Generate targeted queries (verifiable fact + date, authoritative primary-source query, exact claim).
        2. Execute the primary query.
        3. Evaluate and filter results. If fewer than 2 credible/primary sources are found and a second authoritative query is available, execute at most 1 follow-up query to locate primary sources without wasting API credits.
        4. Deduplicate URLs, filter irrelevant items, and preserve the claim-to-evidence mapping.
        """
        queries = generate_claim_search_queries(claim)
        if not queries:
            return ClaimEvidenceBundle(
                claim_id=claim.claim_id,
                claim=claim,
                generated_queries=[],
                executed_query="",
                sources=[],
                search_warning="Could not generate a valid search query from this claim.",
            )

        cache = _get_session_query_cache()
        primary_query = queries[0]
        primary_cache_key = f"{self.search_depth}:{self.max_results}:{primary_query.strip().lower()}"
        was_cached = primary_cache_key in cache

        collected_raw_sources: List[EvidenceSource] = []
        executed_queries: List[str] = []
        warning_messages: List[str] = []

        # Stage 1: Execute Primary Verifiable-Fact Query
        try:
            raw_items = self._execute_raw_query(primary_query)
            executed_queries.append(primary_query)
            collected_raw_sources.extend(self._parse_raw_results(raw_items))
        except Exception as exc:
            safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
            logger.warning(
                "Tavily search error for claim %s ('%s'): %s",
                claim.claim_id,
                primary_query[:60],
                safe_msg,
            )
            warning_messages.append(f"Tavily Search API warning: {safe_msg}")

        evaluated = evaluate_and_filter_sources_for_claim(
            sources=collected_raw_sources,
            claim=claim,
        )

        # Stage 2: Credit-aware fallback to Authoritative Primary-Source Query
        # Only run a 2nd query if Stage 1 yielded < 2 sources OR 0 primary/direct-signal sources
        has_strong_evidence = (
            len(evaluated) >= 2
            and any(
                s.evidence_class == "Primary Evidence"
                or s.direct_support
                or s.direct_contradiction
                or s.credibility_score >= 0.80
                for s in evaluated
            )
        )

        if not has_strong_evidence and len(queries) > 1 and not warning_messages:
            secondary_query = queries[1]
            try:
                raw_items_2 = self._execute_raw_query(secondary_query)
                executed_queries.append(secondary_query)
                collected_raw_sources.extend(self._parse_raw_results(raw_items_2))
                evaluated = evaluate_and_filter_sources_for_claim(
                    sources=collected_raw_sources,
                    claim=claim,
                )
            except Exception as exc:
                safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
                logger.warning(
                    "Secondary Tavily search error for claim %s: %s",
                    claim.claim_id,
                    safe_msg,
                )
                warning_messages.append(f"Secondary search warning: {safe_msg}")

        # Cap final sources per claim to `self.max_results`
        final_sources = evaluated[: self.max_results]
        avg_cred = compute_average_credibility(final_sources)
        has_primary = any(
            s.evidence_class == "Primary Evidence" for s in final_sources
        )

        combined_warning = " | ".join(warning_messages) if warning_messages else None

        return ClaimEvidenceBundle(
            claim_id=claim.claim_id,
            claim=claim,
            generated_queries=queries,
            executed_query=" ; ".join(executed_queries) if executed_queries else primary_query,
            sources=final_sources,
            average_credibility=avg_cred,
            has_primary_evidence=has_primary,
            search_warning=combined_warning,
            from_cache=was_cached,
        )

    def search_claims_batch(
        self,
        claims: List[ExtractedClaim],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, ClaimEvidenceBundle]:
        """Search each extracted claim individually and return an ordered mapping of `claim_id -> ClaimEvidenceBundle`."""
        evidence_map: Dict[str, ClaimEvidenceBundle] = {}
        total = len(claims)

        for idx, claim in enumerate(claims, start=1):
            if progress_callback:
                progress_callback(
                    idx / max(total, 1),
                    f"Searching live web evidence for {claim.claim_id} ({idx}/{total})...",
                )
            bundle = self.search_for_claim(claim)
            evidence_map[claim.claim_id] = bundle

        return evidence_map

    def search_evidence_for_claim(
        self,
        search_query: str,
        fallback_claim_text: str = "",
        claim: Optional[ExtractedClaim] = None,
    ) -> List[EvidenceSource]:
        """Compatibility helper that returns evaluated `List[EvidenceSource]` for a claim or query."""
        if claim is not None:
            bundle = self.search_for_claim(claim)
            if bundle.search_warning and not bundle.sources:
                raise WebSearchError(bundle.search_warning)
            return bundle.sources

        effective_query = (search_query or fallback_claim_text).strip()
        if not effective_query:
            return []

        try:
            raw_items = self._execute_raw_query(effective_query)
        except Exception as exc:
            safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
            raise WebSearchError(f"Tavily Search API error: {safe_msg}") from exc

        parsed = self._parse_raw_results(raw_items)
        return evaluate_and_filter_sources_for_claim(parsed, claim=None)[
            : self.max_results
        ]
