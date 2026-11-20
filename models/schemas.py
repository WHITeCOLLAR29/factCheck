"""Pydantic schemas for strict validation of claims, search evidence, and verification verdicts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class VerificationStatus(str, Enum):
    """Canonical verification verdicts for a factual claim."""

    VERIFIED = "VERIFIED"
    INACCURATE = "INACCURATE"
    FALSE = "FALSE"
    UNVERIFIABLE = "UNVERIFIABLE"

    # Backward-compatible aliases so existing code referencing PARTIALLY_TRUE or REFUTED seamlessly maps to INACCURATE and FALSE
    PARTIALLY_TRUE = "INACCURATE"
    REFUTED = "FALSE"

    @classmethod
    def _missing_(cls, value: object) -> Optional["VerificationStatus"]:
        """Allow case-insensitive and legacy status string normalization."""
        if not isinstance(value, str):
            return None
        norm = value.strip().upper()
        mapping = {
            "VERIFIED": cls.VERIFIED,
            "TRUE": cls.VERIFIED,
            "INACCURATE": cls.INACCURATE,
            "PARTIALLY TRUE": cls.INACCURATE,
            "PARTIALLY_TRUE": cls.INACCURATE,
            "MISLEADING": cls.INACCURATE,
            "OUTDATED": cls.INACCURATE,
            "FALSE": cls.FALSE,
            "REFUTED": cls.FALSE,
            "UNVERIFIABLE": cls.UNVERIFIABLE,
            "INSUFFICIENT": cls.UNVERIFIABLE,
        }
        return mapping.get(norm)


class EvidenceRelationship(str, Enum):
    """Relationship of a cited web evidence item to the original PDF claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    PARTIALLY_SUPPORTS = "partially_supports"
    CONTEXT = "context"

    @classmethod
    def _missing_(cls, value: object) -> Optional["EvidenceRelationship"]:
        if not isinstance(value, str):
            return None
        norm = value.strip().lower()
        if "contradict" in norm or "refute" in norm:
            return cls.CONTRADICTS
        if "partial" in norm or "nuance" in norm:
            return cls.PARTIALLY_SUPPORTS
        if "support" in norm or "confirm" in norm or "corroborat" in norm:
            return cls.SUPPORTS
        return cls.CONTEXT


class ClaimCategory(str, Enum):
    """Categories of factual claims extracted from documents."""

    STATISTICAL = "Statistical & Percentages"
    FINANCIAL = "Financial & Economic"
    HISTORICAL = "Historical & Dates"
    SCIENTIFIC = "Scientific & Medical"
    TECHNICAL_PRODUCT = "Technical & Product Specs"
    COMPANY_ORG = "Company & Organizational"
    POPULATION_DEMOGRAPHIC = "Population & Macroeconomic"
    POLICY_LEGAL = "Policy, Legal & Regulatory"
    GENERAL = "General Factual"


class ClaimImportance(str, Enum):
    """Importance priority level of an extracted factual claim."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ExtractedClaim(BaseModel):
    """A single factual claim extracted from a PDF document with full forensic metadata."""

    claim_id: str = Field(
        default="C001",
        description="Unique identifier for the claim (e.g., C001).",
    )
    claim_text: str = Field(
        ...,
        min_length=8,
        description="Exact, unmodified factual claim text as stated in the PDF document.",
    )
    category: ClaimCategory = Field(
        default=ClaimCategory.GENERAL,
        description="Domain category of the factual claim.",
    )
    importance: ClaimImportance = Field(
        default=ClaimImportance.HIGH,
        description="Importance level of verifying this claim (High, Medium, Low).",
    )
    page_number: int = Field(
        default=1,
        ge=1,
        description="1-based source page number where the claim appears in the PDF.",
    )
    surrounding_context: str = Field(
        default="",
        description="Surrounding paragraph context from the document where the claim appears.",
    )
    date_or_period: Optional[str] = Field(
        default=None,
        description="Relevant date, year, quarter, or time period mentioned in the claim, if present.",
    )
    entity_or_subject: str = Field(
        default="",
        description="Primary entity, company, product, person, or subject the claim is about.",
    )
    numeric_value: Optional[str] = Field(
        default=None,
        description="Extracted numeric figure(s) when applicable (e.g., '42.5', '1.8 billion').",
    )
    numeric_unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement or currency when applicable (e.g., '%', 'USD', 'km/h', 'MW').",
    )
    check_worthiness: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Score from 0.0 to 1.0 indicating how objective and check-worthy this claim is.",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="List of key named entities, numbers, dates, or organizations in the claim.",
    )
    search_query: str = Field(
        ...,
        min_length=4,
        description="Neutral, high-signal web search query designed to find authoritative evidence for or against this claim.",
    )

    @property
    def original_quote(self) -> str:
        """Backward-compatible alias returning surrounding_context or claim_text."""
        return self.surrounding_context or self.claim_text

    @field_validator("claim_text", "search_query")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace only.")
        return cleaned


class ClaimExtractionList(BaseModel):
    """Structured output schema returned by Gemini during claim extraction."""

    document_topic: str = Field(
        default="General Document",
        description="Brief 1-sentence topic summary of the analyzed text.",
    )
    claims: List[ExtractedClaim] = Field(
        default_factory=list,
        description="List of distinct, objectively verifiable factual claims extracted from the text.",
    )


class EvidenceSource(BaseModel):
    """A single live web search result evaluated for credibility, relevance, recency, and stance."""

    title: str = Field(default="Untitled Source", description="Page title of the web source.")
    url: str = Field(..., description="Full URL of the web source (never fabricated).")
    domain: str = Field(default="", description="Extracted root domain (e.g., bls.gov, sec.gov, reuters.com).")
    snippet: str = Field(default="", description="Relevant text excerpt from the source.")
    raw_content_excerpt: Optional[str] = Field(
        default=None,
        description="Extended page content when available from advanced search, preferred over short snippets.",
    )
    published_date: Optional[str] = Field(
        default=None,
        description="Publication date strictly as reported by the source (None if unavailable; never fabricated).",
    )
    source_type: str = Field(
        default="Credible Secondary Source",
        description=(
            "Source classification: 'Government & Official Statistics', 'Official Filing & Financial Report', "
            "'Official Product Documentation', 'Academic & Scientific Publication', 'Reputable News Organization', "
            "'Credible Secondary Source', or 'User-Generated / Unverified'."
        ),
    )
    evidence_class: str = Field(
        default="Secondary Evidence",
        description="Classification as 'Primary Evidence' or 'Secondary Evidence'.",
    )
    relevance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Calculated claim-to-source relevance score normalized to [0.0, 1.0].",
    )
    credibility_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Evaluated composite credibility score in [0.0, 1.0] (not blindly 1.0 for .gov/.edu).",
    )
    credibility_tier: str = Field(
        default="Standard Web",
        description="Tier label summarizing domain authority and evidence quality.",
    )
    recency_note: str = Field(
        default="Date unspecified",
        description="Evaluation of source date and recency relative to the claim's reporting period.",
    )
    alignment_signal: str = Field(
        default="Contextual / Neutral",
        description="Preliminary evidence alignment signal: 'Directly Supports', 'Directly Contradicts', 'Partial / Nuanced Match', or 'Contextual / Neutral'.",
    )
    direct_support: bool = Field(
        default=False,
        description="True if source content directly corroborates the claim's entity and figures/dates.",
    )
    direct_contradiction: bool = Field(
        default=False,
        description="True if source content directly contradicts the claim's figures/dates or includes explicit refutation.",
    )
    relationship: str = Field(
        default="context",
        description="Verification relationship assigned to this source ('supports', 'contradicts', 'partially_supports', 'context').",
    )


class ClaimEvidenceBundle(BaseModel):
    """Preserves the exact mapping between an extracted claim, the search queries used, and retrieved evidence."""

    claim_id: str
    claim: ExtractedClaim
    generated_queries: List[str] = Field(
        default_factory=list,
        description="Targeted search queries generated for this claim (exact, verifiable fact, primary source, date/period).",
    )
    executed_query: str = Field(
        default="",
        description="Primary search query executed against Tavily.",
    )
    sources: List[EvidenceSource] = Field(
        default_factory=list,
        description="Deduplicated, relevance-filtered, and evaluated web evidence sources for this claim.",
    )
    average_credibility: float = Field(default=0.0, ge=0.0, le=1.0)
    has_primary_evidence: bool = Field(default=False)
    search_warning: Optional[str] = Field(
        default=None,
        description="Warning message if the Tavily search API encountered an error or timeout.",
    )
    from_cache: bool = Field(
        default=False,
        description="True if results were served from the session query cache.",
    )


class VerifiedEvidenceItem(BaseModel):
    """Canonical structured evidence citation item inside a verified claim result."""

    source_title: str = Field(
        ...,
        description="Exact title of the retrieved web source.",
    )
    source_url: str = Field(
        ...,
        description="Exact URL from the provided web evidence (never invented or modified).",
    )
    source_domain: str = Field(
        default="",
        description="Root domain of the source (e.g., sec.gov, bls.gov, reuters.com).",
    )
    source_date: Optional[str] = Field(
        default=None,
        description="Publication date of the source if available in the retrieved evidence, or null.",
    )
    evidence_snippet: str = Field(
        ...,
        description="Relevant factual quote or excerpt from the retrieved source.",
    )
    relationship: EvidenceRelationship = Field(
        default=EvidenceRelationship.SUPPORTS,
        description="How this source relates to the claim: 'supports', 'contradicts', 'partially_supports', or 'context'.",
    )


class LLMVerificationOutput(BaseModel):
    """Structured JSON schema returned by Gemini and validated by Pydantic during fact verification.

    Matches the exact required canonical structure:
    {
      "claim_id": "C001",
      "original_claim": "...",
      "status": "INACCURATE",
      "corrected_fact": "...",
      "explanation": "...",
      "confidence": 0.95,
      "evidence": [
        {
          "source_title": "...",
          "source_url": "...",
          "source_domain": "...",
          "source_date": "...",
          "evidence_snippet": "...",
          "relationship": "contradicts"
        }
      ]
    }
    """

    claim_id: str = Field(
        default="C001",
        description="Unique claim identifier matching the input claim (e.g., C001).",
    )
    original_claim: str = Field(
        default="",
        description="Exact original claim text from the PDF.",
    )
    status: VerificationStatus = Field(
        ...,
        description="Verification verdict: 'VERIFIED', 'INACCURATE', 'FALSE', or 'UNVERIFIABLE'.",
    )
    corrected_fact: Optional[str] = Field(
        default=None,
        description=(
            "When status is INACCURATE or FALSE and evidence provides the true figure/date/context, "
            "state the corrected fact strictly backed by the retrieved sources. Otherwise null."
        ),
    )
    explanation: str = Field(
        ...,
        min_length=15,
        description="Clear, objective explanation comparing the original claim against the retrieved evidence, dates, units, and source agreement.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Evidence-based confidence score between 0.0 and 1.0 reflecting source quality and consensus.",
    )
    evidence: List[VerifiedEvidenceItem] = Field(
        default_factory=list,
        description="List of supporting or contradicting evidence citations strictly drawn from the retrieved sources.",
    )
    key_evidence_points: List[str] = Field(
        default_factory=list,
        description="2 to 4 concise bullet points summarizing specific figures, dates, or conflicts across the sources.",
    )

    @property
    def confidence_score(self) -> float:
        """Backward-compatible property alias for `confidence`."""
        return self.confidence

    @property
    def factual_correction(self) -> Optional[str]:
        """Backward-compatible property alias for `corrected_fact`."""
        return self.corrected_fact


class ClaimVerificationResult(BaseModel):
    """Complete verification record combining the canonical verified output, claim metadata, and evaluated sources."""

    claim_id: str = Field(default="C001")
    original_claim: str = Field(default="")
    status: VerificationStatus
    corrected_fact: Optional[str] = None
    explanation: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: List[VerifiedEvidenceItem] = Field(default_factory=list)

    # Extended fields for UI dashboard, analytics, and audit traceability
    claim: ExtractedClaim
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_credibility_avg: float = Field(default=0.0, ge=0.0, le=1.0)
    key_evidence_points: List[str] = Field(default_factory=list)
    factual_correction: Optional[str] = None
    sources: List[EvidenceSource] = Field(default_factory=list)
    search_warning: Optional[str] = Field(default=None)
    executed_queries: List[str] = Field(default_factory=list)
    verified_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    def to_canonical_dict(self) -> dict:
        """Return the exact canonical verification dictionary format."""
        return {
            "claim_id": self.claim_id,
            "original_claim": self.original_claim or self.claim.claim_text,
            "status": self.status.value,
            "corrected_fact": self.corrected_fact,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "evidence": [
                {
                    "source_title": ev.source_title,
                    "source_url": ev.source_url,
                    "source_domain": ev.source_domain,
                    "source_date": ev.source_date,
                    "evidence_snippet": ev.evidence_snippet,
                    "relationship": ev.relationship.value,
                }
                for ev in self.evidence
            ],
        }


class DocumentMetadata(BaseModel):
    """Metadata and page diagnostics about the uploaded PDF document."""

    filename: str
    file_size_bytes: int
    page_count: int
    readable_page_count: int = 0
    total_characters: int
    total_words: int
    document_hash: str
    title: Optional[str] = None
    author: Optional[str] = None
    ocr_engine_available: bool = False
    ocr_pages: List[int] = Field(default_factory=list)
    scanned_unreadable_pages: List[int] = Field(default_factory=list)
    empty_pages: List[int] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)
    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )


class FactCheckReport(BaseModel):
    """Top-level report model encapsulating document metadata, summary metrics, and all claim results."""

    report_id: str
    document: DocumentMetadata
    document_topic: str = "General Document"
    model_used: str
    total_claims: int
    verified_count: int
    partially_true_count: int
    refuted_count: int
    unverifiable_count: int
    overall_truth_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Weighted accuracy score from 0 to 100 across verifiable claims.",
    )
    average_confidence: float = Field(ge=0.0, le=1.0)
    average_source_credibility: float = Field(ge=0.0, le=1.0)
    search_warnings: List[str] = Field(default_factory=list)
    results: List[ClaimVerificationResult] = Field(default_factory=list)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )
