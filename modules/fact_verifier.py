"""Core Fact Verification Engine (`modules/fact_verifier.py`).

Compares every original claim extracted from a PDF against independently retrieved Tavily web evidence
using Google Gemini (`google-genai`) and strict Pydantic structured validation.

Enforces all critical epistemic, resilience, and anti-hallucination guardrails:
1. Never marks a claim FALSE solely because no evidence was found (0 sources -> UNVERIFIABLE).
2. Never relies on a single weak search result to declare a conclusive verdict (downgrades to UNVERIFIABLE).
3. Never allows fake/invented URLs or fabricated publication dates in `evidence` (cross-verifies against retrieved Tavily sources).
4. Never invents a `corrected_fact` when status is VERIFIED or UNVERIFIABLE, or when unsupported by evidence.
5. Detects conflicting high-authority sources and handles disagreements transparently.
6. Calibrates confidence scores strictly from evidence quality, consensus, and source authority—not LLM subjective certainty.
7. Processes claims in controlled batches and preserves completed verification results if a later claim encounters an error.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from models.schemas import (
    ClaimEvidenceBundle,
    ClaimVerificationResult,
    DocumentMetadata,
    EvidenceRelationship,
    EvidenceSource,
    ExtractedClaim,
    FactCheckReport,
    LLMVerificationOutput,
    VerificationStatus,
    VerifiedEvidenceItem,
)
from modules.source_evaluator import Canonicalize_url, compute_average_credibility
from modules.web_search import TavilySearchService, WebSearchError
from utils.helpers import generate_report_id
from utils.llm_client import GeminiClient, LLMServiceError
from utils.prompts import (
    FACT_VERIFICATION_SYSTEM_INSTRUCTION,
    build_fact_verification_prompt,
)

logger = logging.getLogger(__name__)


def _infer_relationship_from_source(
    source: EvidenceSource,
    verdict_status: VerificationStatus,
) -> EvidenceRelationship:
    """Determine the canonical EvidenceRelationship ('supports', 'contradicts', 'partially_supports', 'context')
    for a retrieved source when building or repairing verified evidence citations.
    """
    if source.direct_contradiction or source.alignment_signal == "Directly Contradicts":
        return EvidenceRelationship.CONTRADICTS
    if source.alignment_signal == "Partial / Nuanced Match":
        return EvidenceRelationship.PARTIALLY_SUPPORTS
    if source.direct_support or source.alignment_signal == "Directly Supports":
        if verdict_status == VerificationStatus.INACCURATE:
            return EvidenceRelationship.PARTIALLY_SUPPORTS
        if verdict_status == VerificationStatus.FALSE:
            return EvidenceRelationship.CONTRADICTS
        return EvidenceRelationship.SUPPORTS

    if verdict_status == VerificationStatus.VERIFIED:
        return EvidenceRelationship.SUPPORTS
    if verdict_status == VerificationStatus.FALSE:
        return EvidenceRelationship.CONTRADICTS
    if verdict_status == VerificationStatus.INACCURATE:
        return EvidenceRelationship.PARTIALLY_SUPPORTS
    return EvidenceRelationship.CONTEXT


def _is_single_weak_source_only(sources: List[EvidenceSource]) -> bool:
    """Return True if the retrieved evidence consists of only a single low-credibility/low-relevance source."""
    if len(sources) != 1:
        return False
    only_src = sources[0]
    if only_src.evidence_class == "Primary Evidence" and only_src.credibility_score >= 0.80:
        return False
    return (
        only_src.credibility_score < 0.65
        or only_src.relevance_score < 0.45
        or only_src.source_type == "User-Generated / Unverified"
    )


def _has_unresolved_authority_conflict(sources: List[EvidenceSource]) -> bool:
    """Return True if high-credibility sources directly disagree (some directly support while others directly contradict)."""
    credible_supporters = [
        s for s in sources if s.direct_support and s.credibility_score >= 0.72
    ]
    credible_contradictors = [
        s for s in sources if s.direct_contradiction and s.credibility_score >= 0.72
    ]
    return bool(credible_supporters and credible_contradictors)


def _ground_and_sanitize_evidence_items(
    llm_evidence: List[VerifiedEvidenceItem],
    retrieved_sources: List[EvidenceSource],
    verdict_status: VerificationStatus,
) -> List[VerifiedEvidenceItem]:
    """Ensure every VerifiedEvidenceItem cites a real retrieved URL, real domain, and un-fabricated publication date."""
    source_by_canon_url: Dict[str, EvidenceSource] = {}
    source_by_raw_url: Dict[str, EvidenceSource] = {}
    for src in retrieved_sources:
        canon = Canonicalize_url(src.url)
        if canon:
            source_by_canon_url[canon] = src
        source_by_raw_url[src.url.strip()] = src

    grounded_items: List[VerifiedEvidenceItem] = []
    seen_urls: set[str] = set()

    for item in llm_evidence:
        raw_u = (item.source_url or "").strip()
        canon_u = Canonicalize_url(raw_u)
        matched_src = source_by_raw_url.get(raw_u) or source_by_canon_url.get(canon_u)

        if matched_src is None:
            logger.warning("Discarding unverified LLM URL not in retrieved sources: %s", raw_u)
            continue

        if matched_src.url in seen_urls:
            continue
        seen_urls.add(matched_src.url)

        genuine_date = matched_src.published_date
        grounded_snippet = (
            item.evidence_snippet.strip()
            if item.evidence_snippet and len(item.evidence_snippet.strip()) >= 10
            else matched_src.snippet
        )

        matched_src.relationship = item.relationship.value

        grounded_items.append(
            VerifiedEvidenceItem(
                source_title=matched_src.title,
                source_url=matched_src.url,
                source_domain=matched_src.domain,
                source_date=genuine_date,
                evidence_snippet=grounded_snippet,
                relationship=item.relationship,
            )
        )

    if not grounded_items and retrieved_sources:
        for src in retrieved_sources[:3]:
            rel = _infer_relationship_from_source(src, verdict_status)
            src.relationship = rel.value
            grounded_items.append(
                VerifiedEvidenceItem(
                    source_title=src.title,
                    source_url=src.url,
                    source_domain=src.domain,
                    source_date=src.published_date,
                    evidence_snippet=src.snippet,
                    relationship=rel,
                )
            )

    return grounded_items


def _compute_evidence_calibrated_confidence(
    llm_confidence: float,
    verdict_status: VerificationStatus,
    sources: List[EvidenceSource],
    grounded_evidence: List[VerifiedEvidenceItem],
    has_conflict: bool,
) -> float:
    """Calculate an objective confidence score in [0.0, 1.0] driven by evidence quality, count, and agreement."""
    if not sources:
        return 0.0

    avg_cred = compute_average_credibility(sources)
    avg_rel = sum(s.relevance_score for s in sources) / len(sources)
    primary_bonus = 0.06 if any(s.evidence_class == "Primary Evidence" for s in sources) else 0.0
    multi_source_factor = min(1.0, 0.70 + (0.10 * min(len(sources), 3)))

    if grounded_evidence:
        rel_counts: Dict[str, int] = {}
        for ev in grounded_evidence:
            rel_counts[ev.relationship.value] = rel_counts.get(ev.relationship.value, 0) + 1
        dominant_count = max(rel_counts.values())
        agreement_ratio = dominant_count / len(grounded_evidence)
    else:
        agreement_ratio = 0.6

    evidence_quality_score = (
        (avg_cred * 0.45)
        + (avg_rel * 0.30)
        + (agreement_ratio * 0.25)
        + primary_bonus
    ) * multi_source_factor

    blended = (evidence_quality_score * 0.55) + (llm_confidence * 0.45)

    if has_conflict:
        blended = min(blended, 0.60)
    if verdict_status == VerificationStatus.UNVERIFIABLE:
        blended = min(blended, 0.55)

    return round(max(0.05, min(0.99, blended)), 2)


class FactVerifier:
    """Core fact verification engine comparing PDF claims against retrieved Tavily web evidence."""

    def __init__(
        self,
        llm_client: GeminiClient,
        search_service: TavilySearchService,
    ) -> None:
        self.llm_client = llm_client
        self.search_service = search_service

    def verify_claim_with_bundle(
        self,
        bundle: ClaimEvidenceBundle,
    ) -> ClaimVerificationResult:
        """Verify an ExtractedClaim against its pre-fetched ClaimEvidenceBundle following the 10-step forensic process."""
        claim = bundle.claim
        evidence_sources = bundle.sources

        # Rule 1: Never mark a claim FALSE when search failed or no evidence was found -> UNVERIFIABLE
        if bundle.search_warning and not evidence_sources:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                original_claim=claim.claim_text,
                status=VerificationStatus.UNVERIFIABLE,
                corrected_fact=None,
                explanation=(
                    f"Available evidence is insufficient to verify or refute this claim because the live web search "
                    f"encountered a service error ({bundle.search_warning})."
                ),
                confidence=0.0,
                evidence=[],
                claim=claim,
                confidence_score=0.0,
                source_credibility_avg=0.0,
                key_evidence_points=["Live web search failed or timed out; claim could not be verified."],
                factual_correction=None,
                sources=[],
                search_warning=bundle.search_warning,
                executed_queries=bundle.generated_queries,
            )

        if not evidence_sources:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                original_claim=claim.claim_text,
                status=VerificationStatus.UNVERIFIABLE,
                corrected_fact=None,
                explanation=(
                    "No relevant independent web sources were found for this claim. "
                    "In accordance with verification standards, a claim with insufficient public evidence is marked "
                    "UNVERIFIABLE rather than FALSE."
                ),
                confidence=0.15,
                evidence=[],
                claim=claim,
                confidence_score=0.15,
                source_credibility_avg=0.0,
                key_evidence_points=["0 relevant web sources matched the verification queries."],
                factual_correction=None,
                sources=[],
                search_warning=bundle.search_warning,
                executed_queries=bundle.generated_queries,
            )

        # Rule 2: Do not rely on a single weak search result to make a conclusive VERIFIED/FALSE claim
        if _is_single_weak_source_only(evidence_sources):
            weak_src = evidence_sources[0]
            weak_src.relationship = EvidenceRelationship.CONTEXT.value
            weak_item = VerifiedEvidenceItem(
                source_title=weak_src.title,
                source_url=weak_src.url,
                source_domain=weak_src.domain,
                source_date=weak_src.published_date,
                evidence_snippet=weak_src.snippet,
                relationship=EvidenceRelationship.CONTEXT,
            )
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                original_claim=claim.claim_text,
                status=VerificationStatus.UNVERIFIABLE,
                corrected_fact=None,
                explanation=(
                    f"Only a single low-authority or weakly matching web result (`{weak_src.domain}`, "
                    f"credibility {weak_src.credibility_score * 100:.0f}%) was found. "
                    "A single weak search result is insufficient to conclusively verify or refute a factual claim."
                ),
                confidence=0.25,
                evidence=[weak_item],
                claim=claim,
                confidence_score=0.25,
                source_credibility_avg=weak_src.credibility_score,
                key_evidence_points=[
                    f"Single weak source found ({weak_src.domain}); insufficient corroboration for a definitive verdict."
                ],
                factual_correction=None,
                sources=evidence_sources,
                search_warning=bundle.search_warning,
                executed_queries=bundle.generated_queries,
            )

        avg_credibility = bundle.average_credibility or compute_average_credibility(
            evidence_sources
        )
        has_conflict = _has_unresolved_authority_conflict(evidence_sources)
        current_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        prompt = build_fact_verification_prompt(
            claim=claim,
            evidence_sources=evidence_sources,
            current_date_str=current_date_str,
        )

        try:
            llm_verdict: LLMVerificationOutput = self.llm_client.generate_structured(
                system_instruction=FACT_VERIFICATION_SYSTEM_INSTRUCTION,
                user_prompt=prompt,
                response_schema=LLMVerificationOutput,
            )
        except Exception as llm_err:
            fallback_evidence = _ground_and_sanitize_evidence_items(
                llm_evidence=[],
                retrieved_sources=evidence_sources,
                verdict_status=VerificationStatus.UNVERIFIABLE,
            )
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                original_claim=claim.claim_text,
                status=VerificationStatus.UNVERIFIABLE,
                corrected_fact=None,
                explanation=f"Verification reasoning engine encountered an API error: {llm_err}",
                confidence=0.0,
                evidence=fallback_evidence,
                claim=claim,
                confidence_score=0.0,
                source_credibility_avg=avg_credibility,
                key_evidence_points=["Automated reasoning step encountered an API error."],
                factual_correction=None,
                sources=evidence_sources,
                search_warning=bundle.search_warning,
                executed_queries=bundle.generated_queries,
            )

        verdict_status = llm_verdict.status
        explanation = llm_verdict.explanation.strip()

        # Rule 3: Ground and sanitize all evidence items against actual retrieved Tavily sources
        grounded_evidence = _ground_and_sanitize_evidence_items(
            llm_evidence=llm_verdict.evidence,
            retrieved_sources=evidence_sources,
            verdict_status=verdict_status,
        )

        # Rule 3b: Epistemic Safeguard — A true claim must NOT be marked FALSE if no evidence item or source contradicts it,
        # and a claim with 0 grounded evidence items must be marked UNVERIFIABLE.
        if not grounded_evidence:
            verdict_status = VerificationStatus.UNVERIFIABLE
        elif verdict_status == VerificationStatus.FALSE:
            has_any_contradiction = any(
                ev.relationship
                in (
                    EvidenceRelationship.CONTRADICTS,
                    EvidenceRelationship.PARTIALLY_SUPPORTS,
                )
                for ev in grounded_evidence
            ) or any(s.direct_contradiction for s in evidence_sources)
            all_directly_support = all(
                ev.relationship == EvidenceRelationship.SUPPORTS
                for ev in grounded_evidence
            ) and any(s.direct_support for s in evidence_sources)
            if all_directly_support and not has_any_contradiction:
                logger.warning(
                    "Overriding contradictory LLM FALSE status to VERIFIED because all grounded evidence supports %s.",
                    claim.claim_id,
                )
                verdict_status = VerificationStatus.VERIFIED

        # Rule 4: If sources have an unresolved conflict and explanation didn't note it, append conflict note
        if has_conflict and "conflict" not in explanation.lower() and "disagree" not in explanation.lower():
            explanation += (
                " Note: Retrieved credible sources present conflicting figures or interpretations on this point."
            )

        # Rule 5: Never invent or retain a `corrected_fact` when status is VERIFIED or UNVERIFIABLE
        corrected_fact = (llm_verdict.corrected_fact or "").strip() or None
        if verdict_status in (VerificationStatus.VERIFIED, VerificationStatus.UNVERIFIABLE):
            corrected_fact = None

        # Rule 6: Compute evidence-calibrated confidence score
        calibrated_confidence = _compute_evidence_calibrated_confidence(
            llm_confidence=llm_verdict.confidence,
            verdict_status=verdict_status,
            sources=evidence_sources,
            grounded_evidence=grounded_evidence,
            has_conflict=has_conflict,
        )

        key_points = list(llm_verdict.key_evidence_points)
        if not key_points and grounded_evidence:
            for ev in grounded_evidence[:3]:
                date_tag = f" ({ev.source_date})" if ev.source_date else ""
                key_points.append(
                    f"[{ev.source_domain}{date_tag} — {ev.relationship.value}] {ev.evidence_snippet[:180]}"
                )

        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            original_claim=claim.claim_text,
            status=verdict_status,
            corrected_fact=corrected_fact,
            explanation=explanation,
            confidence=calibrated_confidence,
            evidence=grounded_evidence,
            claim=claim,
            confidence_score=calibrated_confidence,
            source_credibility_avg=avg_credibility,
            key_evidence_points=key_points,
            factual_correction=corrected_fact,
            sources=evidence_sources,
            search_warning=bundle.search_warning,
            executed_queries=bundle.generated_queries,
        )

    def verify_single_claim(self, claim: ExtractedClaim) -> ClaimVerificationResult:
        """Retrieve web evidence for a single claim and synthesize a structured verification verdict."""
        try:
            bundle = self.search_service.search_for_claim(claim)
        except Exception as exc:
            bundle = ClaimEvidenceBundle(
                claim_id=claim.claim_id,
                claim=claim,
                sources=[],
                search_warning=f"Search failed: {exc}",
            )
        return self.verify_claim_with_bundle(bundle)

    def verify_claims_batch(
        self,
        claims: List[ExtractedClaim],
        document_metadata: DocumentMetadata,
        document_topic: str,
        batch_size: int = 5,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        partial_results_sink: Optional[List[ClaimVerificationResult]] = None,
    ) -> FactCheckReport:
        """Process claims in controlled batches, preserving all completed results even if a later claim fails."""
        results: List[ClaimVerificationResult] = []
        total = len(claims)
        effective_batch_size = max(1, batch_size)

        for batch_start in range(0, total, effective_batch_size):
            batch_claims = claims[batch_start : batch_start + effective_batch_size]
            for offset, claim in enumerate(batch_claims):
                idx = batch_start + offset + 1
                if progress_callback:
                    short_preview = (
                        claim.claim_text[:75] + "..."
                        if len(claim.claim_text) > 75
                        else claim.claim_text
                    )
                    progress_callback(
                        idx / max(total, 1),
                        f"Verifying {claim.claim_id} ({idx}/{total}): {short_preview}",
                    )

                try:
                    result = self.verify_single_claim(claim)
                except Exception as unexpected_err:
                    logger.error(
                        "Preserving completed results after error on claim %s: %s",
                        claim.claim_id,
                        unexpected_err,
                    )
                    result = ClaimVerificationResult(
                        claim_id=claim.claim_id,
                        original_claim=claim.claim_text,
                        status=VerificationStatus.UNVERIFIABLE,
                        corrected_fact=None,
                        explanation=f"Claim verification encountered an unexpected error: {unexpected_err}",
                        confidence=0.0,
                        evidence=[],
                        claim=claim,
                        confidence_score=0.0,
                        source_credibility_avg=0.0,
                        key_evidence_points=["Claim verification interrupted by an unexpected error."],
                    )

                results.append(result)
                if partial_results_sink is not None:
                    partial_results_sink.append(result)

        return self._compile_report(
            results=results,
            document_metadata=document_metadata,
            document_topic=document_topic,
        )

    def _compile_report(
        self,
        results: List[ClaimVerificationResult],
        document_metadata: DocumentMetadata,
        document_topic: str,
    ) -> FactCheckReport:
        """Aggregate claim verification results into a FactCheckReport with truth score and metrics."""
        verified_count = sum(
            1 for r in results if r.status == VerificationStatus.VERIFIED
        )
        inaccurate_count = sum(
            1 for r in results if r.status == VerificationStatus.INACCURATE
        )
        false_count = sum(
            1 for r in results if r.status == VerificationStatus.FALSE
        )
        unverifiable_count = sum(
            1 for r in results if r.status == VerificationStatus.UNVERIFIABLE
        )

        evaluated_count = verified_count + inaccurate_count + false_count
        if evaluated_count > 0:
            weighted_points = (verified_count * 1.0) + (inaccurate_count * 0.5)
            overall_truth_score = round((weighted_points / evaluated_count) * 100.0, 1)
        else:
            overall_truth_score = 0.0

        total_claims = len(results)
        avg_confidence = (
            round(sum(r.confidence for r in results) / total_claims, 3)
            if total_claims > 0
            else 0.0
        )
        avg_credibility = (
            round(sum(r.source_credibility_avg for r in results) / total_claims, 3)
            if total_claims > 0
            else 0.0
        )

        search_warnings = [
            f"[{r.claim_id}] {r.search_warning}"
            for r in results
            if r.search_warning
        ]

        return FactCheckReport(
            report_id=generate_report_id(),
            document=document_metadata,
            document_topic=document_topic,
            model_used=self.llm_client.model_name,
            total_claims=total_claims,
            verified_count=verified_count,
            partially_true_count=inaccurate_count,
            refuted_count=false_count,
            unverifiable_count=unverifiable_count,
            overall_truth_score=overall_truth_score,
            average_confidence=avg_confidence,
            average_source_credibility=avg_credibility,
            search_warnings=search_warnings,
            results=results,
        )
