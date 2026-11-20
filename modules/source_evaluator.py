"""Evidence & Source Evaluation Module (`modules/source_evaluator.py`).

Evaluates retrieved web search sources across 6 forensic dimensions:
1. Domain authority and granular source type (Government/Statistical, Official Filings, Product Docs, Academic/Scientific, Reputable News, Credible Secondary, User-Generated).
2. Primary versus Secondary evidence classification.
3. Nuanced authority verification (does NOT blindly assume every .gov, .edu, or official-looking domain is automatically authoritative or correct—penalizes student blogs, user forums, comments, or mismatched topics).
4. Relevance to the specific claim (entity, numeric, and keyword overlap).
5. Date and recency alignment relative to the claim's reporting period.
6. Preliminary direct support versus direct contradiction signal analysis.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from config.settings import (
    HIGH_AUTHORITY_DOMAINS,
    LOW_CREDIBILITY_INDICATORS,
    MEDIUM_AUTHORITY_DOMAINS,
)
from models.schemas import EvidenceSource, ExtractedClaim
from utils.helpers import extract_domain

# Stopwords excluded when computing claim-to-source lexical relevance
_STOPWORDS: set[str] = {
    "the", "and", "for", "that", "this", "with", "from", "were", "was", "are",
    "has", "have", "had", "been", "will", "would", "could", "should", "into",
    "over", "under", "between", "among", "their", "there", "about", "which",
    "when", "where", "while", "after", "before", "during", "through", "than",
    "more", "most", "some", "such", "only", "also", "very", "much", "many",
}

# Specialized domain classification sets
_GOV_STATISTICAL_DOMAINS: Tuple[str, ...] = (
    "bls.gov", "bea.gov", "census.gov", "cdc.gov", "fda.gov", "nih.gov",
    "epa.gov", "energy.gov", "eia.gov", "nasa.gov", "noaa.gov", "federalreserve.gov",
    "treasury.gov", "who.int", "un.org", "worldbank.org", "imf.org", "oecd.org",
    "ec.europa.eu", "eurostat.ec.europa.eu", "ons.gov.uk", "statcan.gc.ca",
)

_FILINGS_FINANCIAL_DOMAINS: Tuple[str, ...] = (
    "sec.gov", "edgar", "investor.", "ir.", "annualreports.com",
    "companieshouse.gov.uk", "sedarplus.ca",
)

_ACADEMIC_SCIENTIFIC_DOMAINS: Tuple[str, ...] = (
    "nature.com", "science.org", "nejm.org", "thelancet.com", "bmj.com",
    "jamanetwork.com", "cell.com", "pnas.org", "ieee.org", "acm.org",
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "arxiv.org",
    "sciencedirect.com", "springer.com", "wiley.com", "oup.com",
    "cambridge.org", "plos.org", "frontiersin.org", "ourworldindata.org",
    "pewresearch.org",
)

_REPUTABLE_NEWS_DOMAINS: Tuple[str, ...] = (
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "bloomberg.com",
    "ft.com", "wsj.com", "nytimes.com", "economist.com", "washingtonpost.com",
    "theguardian.com", "npr.org", "pbs.org", "cnbc.com", "forbes.com",
    "scientificamerican.com", "arstechnica.com", "techcrunch.com", "wired.com",
)

# URL path patterns indicating non-authoritative or user-hosted content even on .edu/.gov/official domains
_NON_AUTHORITATIVE_URL_PATTERNS: re.Pattern[str] = re.compile(
    r"(/~[a-z0-9_\-]+|/people/|/student[s]?/|/users/|/forum[s]?/|/community/|"
    r"/discussion[s]?/|/comment[s]?/|/opinion/|/editorial/|/blog[s]?/|/wiki/|"
    r"/sandbox/|/personal/|/club[s]?/|/courses/)",
    re.IGNORECASE,
)

# Keywords indicating explicit refutation or contradiction in a source
_CONTRADICTION_KEYWORDS: re.Pattern[str] = re.compile(
    r"\b(false|fake|hoax|myth|debunk(?:ed|s)?|refut(?:ed|es)|incorrect|inaccurate|"
    r"misleading|untrue|fabricated|baseless|contrary to claims|no evidence that|"
    r"did not|never happened|overstated|exaggerated)\b",
    re.IGNORECASE,
)


def Canonicalize_url(url: str) -> str:
    """Normalize a URL for deduplication (strip fragments and common tracking query parameters)."""
    if not url or not url.strip():
        return ""
    raw = url.strip()
    try:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")
        # Keep query string only if it doesn't consist purely of utm_ tracking params
        if parsed.query:
            q_parts = [
                p for p in parsed.query.split("&")
                if not p.lower().startswith(("utm_", "fbclid=", "gclid=", "ref="))
            ]
            clean_query = "&".join(q_parts)
        else:
            clean_query = ""
        base = f"{scheme}://{netloc}{path}"
        return f"{base}?{clean_query}" if clean_query else base
    except Exception:
        return raw.rstrip("/")


def classify_source_type_and_evidence_class(
    url: str,
    title: str = "",
    claim: Optional[ExtractedClaim] = None,
) -> Tuple[str, str, str, float, str]:
    """Evaluate a URL's domain, path structure, and context.

    Returns:
        (clean_domain, source_type, evidence_class, base_authority_score, credibility_tier)

    Crucially does NOT assume every .gov, .edu, or official-looking domain is automatically authoritative:
    checks URL paths for personal pages (`/~user`), student projects, blogs, opinions, or forums.
    """
    domain = extract_domain(url)
    if domain == "unknown":
        return (
            domain,
            "User-Generated / Unverified",
            "Secondary Evidence",
            0.35,
            "Unclassified Web",
        )

    url_lower = url.lower()
    title_lower = (title or "").lower()
    has_non_auth_path = bool(_NON_AUTHORITATIVE_URL_PATTERNS.search(url_lower))

    # 0. Check low-credibility / user-generated platforms first
    for low_ind in LOW_CREDIBILITY_INDICATORS:
        if low_ind.lower() in domain:
            return (
                domain,
                "User-Generated / Unverified",
                "Secondary Evidence",
                0.35,
                "User-Generated / Blog",
            )

    # 1. Official Company Filings & Financial Reports (SEC EDGAR, Investor Relations, 10-K/10-Q)
    is_filing = any(ind in url_lower or ind in domain for ind in _FILINGS_FINANCIAL_DOMAINS) or any(
        kw in title_lower for kw in ("form 10-k", "form 10-q", "8-k", "annual report", "earnings release", "sec filing")
    )
    if is_filing and not has_non_auth_path:
        return (
            domain,
            "Official Filing & Financial Report",
            "Primary Evidence",
            0.93,
            "High Authority (Primary Filing)",
        )

    # 2. Government & Official Statistical Agencies
    is_gov_tld = (
        domain.endswith((".gov", ".mil", ".int", ".gov.uk", ".gc.ca", ".gov.au", ".gov.in", ".europa.eu"))
        or any(domain == d or domain.endswith("." + d) for d in _GOV_STATISTICAL_DOMAINS)
    )
    if is_gov_tld:
        if has_non_auth_path:
            # Penalize forum, comment, or blog paths on government domains
            return (
                domain,
                "Credible Secondary Source",
                "Secondary Evidence",
                0.68,
                "Government Hosted (Non-Primary Path)",
            )
        return (
            domain,
            "Government & Official Statistics",
            "Primary Evidence",
            0.91,
            "High Authority (Official / Statistical)",
        )

    # 3. Academic Research & Scientific Publications (including nuanced .edu check)
    is_academic_publisher = any(
        domain == d or domain.endswith("." + d) for d in _ACADEMIC_SCIENTIFIC_DOMAINS
    )
    is_edu_tld = domain.endswith((".edu", ".ac.uk", ".edu.au"))

    if is_academic_publisher:
        if has_non_auth_path or "/blog" in url_lower or "opinion" in url_lower:
            return (
                domain,
                "Academic & Scientific Publication",
                "Secondary Evidence",
                0.76,
                "Scientific Commentary / Blog",
            )
        return (
            domain,
            "Academic & Scientific Publication",
            "Primary Evidence",
            0.90,
            "High Authority (Peer-Reviewed / Research)",
        )

    if is_edu_tld:
        # Do NOT assume every .edu page is primary peer-reviewed science!
        # Personal pages (/~username), student clubs, course syllabi, or blogs get downgraded.
        if has_non_auth_path:
            return (
                domain,
                "Credible Secondary Source",
                "Secondary Evidence",
                0.58,
                "Academic Hosted (Personal / Course Page)",
            )
        return (
            domain,
            "Academic & Scientific Publication",
            "Secondary Evidence",
            0.80,
            "Academic Institution (.edu)",
        )

    # 4. Official Product Documentation & Technical Datasheets
    is_product_doc = (
        url_lower.startswith(("https://docs.", "https://developer.", "https://support.", "https://specs."))
        or any(seg in url_lower for seg in ("/docs/", "/documentation/", "/datasheet", "/specifications", "/whitepaper", "/manual"))
    )
    # Also check if domain matches the claim's primary entity/company
    entity_slug = ""
    if claim and claim.entity_or_subject:
        entity_slug = "".join(
            ch.lower() for ch in claim.entity_or_subject.split()[0] if ch.isalnum()
        )
    is_entity_first_party = bool(entity_slug and len(entity_slug) >= 4 and entity_slug in domain)

    if (is_product_doc or is_entity_first_party) and not has_non_auth_path:
        return (
            domain,
            "Official Product Documentation",
            "Primary Evidence",
            0.85 if is_product_doc else 0.78,
            "Official Documentation / First-Party",
        )

    # 5. Reputable News Organizations
    is_reputable_news = any(
        domain == d or domain.endswith("." + d) for d in _REPUTABLE_NEWS_DOMAINS
    )
    if is_reputable_news:
        if "/opinion/" in url_lower or "/editorial/" in url_lower or "/letters/" in url_lower:
            return (
                domain,
                "Reputable News Organization",
                "Secondary Evidence",
                0.66,
                "News Opinion / Editorial",
            )
        return (
            domain,
            "Reputable News Organization",
            "Secondary Evidence",
            0.82,
            "Established Media / Wire Service",
        )

    # 6. Check remaining HIGH / MEDIUM authority domain lists from settings
    for high_d in HIGH_AUTHORITY_DOMAINS:
        hl = high_d.lower().lstrip(".")
        if domain == hl or domain.endswith("." + hl):
            return (
                domain,
                "Credible Secondary Source",
                "Secondary Evidence",
                0.84,
                "High Authority Reference",
            )

    for med_d in MEDIUM_AUTHORITY_DOMAINS:
        ml = med_d.lower().lstrip(".")
        if domain == ml or domain.endswith("." + ml):
            return (
                domain,
                "Credible Secondary Source",
                "Secondary Evidence",
                0.75,
                "Established Reference",
            )

    if domain.endswith(".org"):
        return (
            domain,
            "Credible Secondary Source",
            "Secondary Evidence",
            0.68,
            "Organization (.org)",
        )

    return (
        domain,
        "Credible Secondary Source",
        "Secondary Evidence",
        0.58,
        "Standard Web",
    )


def evaluate_domain_credibility(url: str) -> Tuple[str, float, str]:
    """Backward-compatible helper returning (clean_domain, credibility_score, credibility_tier)."""
    domain, _, _, score, tier = classify_source_type_and_evidence_class(url)
    return domain, score, tier


def _extract_meaningful_tokens(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric tokens >= 3 chars, excluding stopwords."""
    raw_tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return {t for t in raw_tokens if len(t) >= 3 and t not in _STOPWORDS}


def _extract_numbers(text: str) -> set[str]:
    """Extract normalized numeric strings (e.g., '14.2', '2023', '35', '4.85') from text."""
    matches = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)
    return {m.replace(",", "") for m in matches}


def evaluate_relevance_to_claim(
    source: EvidenceSource,
    claim: ExtractedClaim,
) -> float:
    """Compute a composite relevance score in [0.0, 1.0] measuring how closely the source addresses the claim."""
    combined_source_text = " ".join(
        filter(None, [source.title, source.snippet, source.raw_content_excerpt])
    )
    if not combined_source_text.strip():
        return 0.0

    claim_tokens = _extract_meaningful_tokens(claim.claim_text)
    if claim.entity_or_subject:
        claim_tokens |= _extract_meaningful_tokens(claim.entity_or_subject)

    source_tokens = _extract_meaningful_tokens(combined_source_text)
    if not claim_tokens or not source_tokens:
        return round(source.relevance_score, 3)

    # 1. Lexical token overlap with the claim
    overlap_count = len(claim_tokens & source_tokens)
    lexical_ratio = min(1.0, overlap_count / max(len(claim_tokens) * 0.65, 1.0))

    # 2. Primary entity match bonus
    entity_bonus = 0.0
    if claim.entity_or_subject:
        ent_tokens = _extract_meaningful_tokens(claim.entity_or_subject)
        if ent_tokens and (ent_tokens & source_tokens):
            entity_bonus = 0.15

    # 3. Date or number presence bonus
    claim_nums = _extract_numbers(claim.claim_text)
    source_nums = _extract_numbers(combined_source_text)
    num_bonus = 0.10 if (claim_nums and (claim_nums & source_nums)) else 0.0

    # Combine search engine score (40%) with lexical/entity relevance (60%)
    composite = (source.relevance_score * 0.40) + (lexical_ratio * 0.45) + entity_bonus + num_bonus
    return round(max(0.0, min(1.0, composite)), 3)


def evaluate_date_and_recency(
    source: EvidenceSource,
    claim: ExtractedClaim,
) -> Tuple[str, float]:
    """Evaluate the source's publication date/recency relative to the claim's date_or_period.

    Never fabricates a publication date if `source.published_date` is None.
    Returns (recency_note, credibility_adjustment_delta).
    """
    claim_years = re.findall(r"\b(19\d{2}|20\d{2})\b", (claim.date_or_period or "") + " " + claim.claim_text)
    target_year: Optional[int] = max((int(y) for y in claim_years), default=None)

    pub_date = (source.published_date or "").strip()
    pub_years = re.findall(r"\b(19\d{2}|20\d{2})\b", pub_date)
    pub_year: Optional[int] = int(pub_years[0]) if pub_years else None

    if pub_year is not None and target_year is not None:
        if pub_year < target_year:
            return (
                f"Predates claim period ({pub_year} vs {target_year})",
                -0.10,
            )
        if pub_year == target_year or pub_year == target_year + 1:
            return (
                f"Contemporaneous with claim period ({pub_year})",
                +0.04,
            )
        return (
            f"Published after claim period ({pub_year})",
            +0.02,
        )

    if pub_year is not None:
        return (f"Published in {pub_year}", +0.02)

    # Check if the snippet itself explicitly mentions the target year even when metadata date is absent
    combined_text = (source.snippet or "") + " " + (source.raw_content_excerpt or "")
    if target_year and str(target_year) in combined_text:
        return (f"References target year {target_year} (publication date unstated)", 0.0)

    return ("Publication date not reported by source", 0.0)


def evaluate_support_or_contradiction(
    source: EvidenceSource,
    claim: ExtractedClaim,
) -> Tuple[str, bool, bool]:
    """Analyze whether the source content directly supports or contradicts the claim.

    Note: Search snippets are treated as preliminary signals; extended content (`raw_content_excerpt`)
    is prioritized when available.

    Returns:
        (alignment_signal_label, direct_support_bool, direct_contradiction_bool)
    """
    # Prefer stronger extended content over short snippet when available
    analysis_text = (
        f"{source.title}. {source.raw_content_excerpt}"
        if source.raw_content_excerpt and len(source.raw_content_excerpt) > len(source.snippet)
        else f"{source.title}. {source.snippet}"
    )
    if not analysis_text.strip():
        return ("Contextual / Neutral", False, False)

    claim_tokens = _extract_meaningful_tokens(claim.claim_text)
    source_tokens = _extract_meaningful_tokens(analysis_text)

    # Require basic subject/entity overlap before declaring direct support or contradiction
    entity_tokens = (
        _extract_meaningful_tokens(claim.entity_or_subject)
        if claim.entity_or_subject
        else claim_tokens
    )
    has_subject_match = bool(entity_tokens and (entity_tokens & source_tokens))
    if not has_subject_match:
        return ("Contextual / Neutral", False, False)

    claim_nums = _extract_numbers(claim.claim_text)
    if claim.numeric_value:
        claim_nums |= _extract_numbers(claim.numeric_value)
    source_nums = _extract_numbers(analysis_text)

    has_contradiction_marker = bool(_CONTRADICTION_KEYWORDS.search(analysis_text))

    # Case 1: Explicit refutation language or conflicting numeric values for the same subject
    if has_contradiction_marker:
        return ("Directly Contradicts", False, True)

    if claim_nums:
        matching_nums = claim_nums & source_nums
        if matching_nums and len(matching_nums) == len(claim_nums):
            return ("Directly Supports", True, False)
        if matching_nums:
            return ("Partial / Nuanced Match", True, False)
        # If the source discusses the exact subject + metric tokens but cites different numbers
        token_overlap_ratio = len(claim_tokens & source_tokens) / max(len(claim_tokens), 1)
        if token_overlap_ratio >= 0.55 and source_nums and not matching_nums:
            return ("Directly Contradicts", False, True)
    else:
        # Non-numeric historical/technical/company claim
        token_overlap_ratio = len(claim_tokens & source_tokens) / max(len(claim_tokens), 1)
        if token_overlap_ratio >= 0.55:
            return ("Directly Supports", True, False)

    return ("Contextual / Neutral", False, False)


def evaluate_and_filter_sources_for_claim(
    sources: List[EvidenceSource],
    claim: Optional[ExtractedClaim] = None,
    min_relevance_threshold: float = 0.22,
) -> List[EvidenceSource]:
    """Deduplicate URLs, evaluate authority/primary-vs-secondary/relevance/recency/alignment,
    remove irrelevant results, and rank sources for a claim.
    """
    seen_canonical_urls: set[str] = set()
    evaluated_sources: List[EvidenceSource] = []

    for src in sources:
        canon_url = Canonicalize_url(src.url)
        if not canon_url or canon_url in seen_canonical_urls:
            continue
        seen_canonical_urls.add(canon_url)

        domain, source_type, evidence_class, base_auth, tier = (
            classify_source_type_and_evidence_class(
                url=src.url,
                title=src.title,
                claim=claim,
            )
        )
        src.domain = domain
        src.source_type = source_type
        src.evidence_class = evidence_class
        src.credibility_tier = tier

        if claim is not None:
            relevance = evaluate_relevance_to_claim(src, claim)
            src.relevance_score = relevance

            # Filter out irrelevant search results
            if relevance < min_relevance_threshold:
                continue

            recency_note, recency_delta = evaluate_date_and_recency(src, claim)
            src.recency_note = recency_note

            alignment_label, is_support, is_contradict = (
                evaluate_support_or_contradiction(src, claim)
            )
            src.alignment_signal = alignment_label
            src.direct_support = is_support
            src.direct_contradiction = is_contradict

            # Adjust credibility score so even a .gov/.edu domain with low claim relevance or outdated date is not overrated
            relevance_adjustment = (relevance - 0.5) * 0.14
            final_cred = max(
                0.15,
                min(0.98, base_auth + recency_delta + relevance_adjustment),
            )
            src.credibility_score = round(final_cred, 3)
        else:
            src.credibility_score = round(base_auth, 3)

        evaluated_sources.append(src)

    # Sort prioritizing Primary Evidence, Direct Support/Contradiction signal, Credibility, and Relevance
    def _rank_key(s: EvidenceSource) -> Tuple[int, int, float]:
        is_primary = 1 if s.evidence_class == "Primary Evidence" else 0
        has_direct_signal = 1 if (s.direct_support or s.direct_contradiction) else 0
        quality_score = (s.credibility_score * 0.58) + (s.relevance_score * 0.42)
        return (is_primary, has_direct_signal, quality_score)

    evaluated_sources.sort(key=_rank_key, reverse=True)
    return evaluated_sources


def enrich_and_rank_sources(
    sources: List[EvidenceSource],
    claim: Optional[ExtractedClaim] = None,
) -> List[EvidenceSource]:
    """Enrich sources with domain credibility and claim-level evaluation metrics."""
    return evaluate_and_filter_sources_for_claim(
        sources=sources,
        claim=claim,
        min_relevance_threshold=0.20 if claim is not None else 0.0,
    )


def compute_average_credibility(sources: List[EvidenceSource]) -> float:
    """Calculate the average credibility score across a list of evidence sources."""
    if not sources:
        return 0.0
    total = sum(s.credibility_score for s in sources)
    return round(total / len(sources), 3)
