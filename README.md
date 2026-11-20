# FactCheck AI – The Truth Layer for Every Document

**FactCheck AI** is a production-quality, AI-powered PDF fact-checking web application built with **Python 3.11+**, **Streamlit**, Google's **Material Design 3 (Material You)** UI architecture, the official **Google GenAI SDK (`google-genai`)**, **Tavily Search API**, **PyMuPDF**, **Pydantic v2**, **Plotly**, and **ReportLab**.

It extracts objectively verifiable factual claims from uploaded PDFs, searches the live internet for independent supporting or contradicting evidence, evaluates source credibility and recency, verifies each claim with strict anti-hallucination guardrails, and generates downloadable PDF, CSV, and JSON verification audit reports.

---

## 1. Project Overview

Organizations, analysts, journalists, and researchers routinely review dense PDF documents—financial filings, macroeconomic reports, technical whitepapers, policy briefs, and academic preprints—that contain dozens of quantitative and historical assertions. Manually verifying every figure, reporting period, unit, and citation is slow and error-prone.

**FactCheck AI** serves as **The Truth Layer for Every Document**, automating the forensic verification pipeline end-to-end:
`PDF Upload → Paragraph-Aware Text & OCR Extraction → Gemini Structured Claim Identification → Live Tavily Web Search → Multi-Dimensional Source Evaluation → 10-Step Evidence Verification → Interactive Plotly Analytics → ReportLab PDF & CSV Audit Export`.

---

## 2. Key Features

- **Google Material Design 3 (Material You) 5-Page SaaS Experience**:
  - **Page 1 — Home / Dashboard**: Hero overview, drag-and-drop PDF dropzone, document specifications, workflow cards (*Extract, Verify, Report*), and session Recent Analysis history.
  - **Page 2 — Document Analysis**: File metadata cards (filename, file size, readable page count, word/paragraph count, SHA-256 fingerprint), page-by-page paragraph preview, OCR diagnostics, prerequisite checks, confirmation gate, and a live 5-step progress tracker.
  - **Page 3 — Analysis Dashboard**: Real computed metrics (Total Claims, Verified, Inaccurate, False, Unverifiable, High-Risk Claims, Conflicting Evidence, Mean Confidence), 3 interactive Plotly charts (Verdict Distribution Donut, Claims by Category Horizontal Bar, Confidence Distribution Bar), and a multi-filterable results table.
  - **Page 4 — Claim Details**: 4-way filtering (by status, category, minimum confidence %, and keyword) plus High-Risk/Conflict toggles; expandable Material 3 claim cards showing verbatim statements, surrounding PDF context, extracted attributes (entity, date/period, numeric value & unit), evidence-backed corrected facts, confidence meters, and relationship-tagged (`supports`, `contradicts`, `partially_supports`, `context`) web sources.
  - **Page 5 — Report**: Standalone in-memory **PDF Report** download (ReportLab with vector logo, executive summary, statistics, claim-by-claim findings, source URLs, and methodology limitations), **CSV** export (Pandas), **JSON** export, and one-click session cleanup (*Analyze Another Document*).
- **Forensic PDF & Scanned Document Handling (`PyMuPDF` + `Tesseract OCR`)**:
  - Preserves 1-based page numbers and paragraph boundaries (`PageParagraph`).
  - Detects encrypted/password-protected, corrupted, or oversized PDFs and rejects them safely.
  - Detects scanned image-only pages, attempts optional Tesseract OCR recovery, and explicitly reports any unreadable or blank pages without ever silently discarding them.
- **Nuanced Source Credibility & Evidence Evaluation**:
  - Classifies sources into *Government & Official Statistics*, *Official Filing & Financial Report*, *Official Product Documentation*, *Academic & Scientific Publication*, *Reputable News Organization*, *Credible Secondary Source*, and *User-Generated / Unverified* (`Primary Evidence` vs. `Secondary Evidence`).
  - Does **not** blindly assume every `.gov` or `.edu` URL is authoritative—inspects URL paths to downgrade personal pages (`/~user`), student blogs, forums, or opinion columns.
- **Strict Epistemic & Anti-Hallucination Guardrails**:
  - Never marks a claim `FALSE` solely because no public evidence was found (`0 sources → UNVERIFIABLE`).
  - Never relies on a single weak/low-authority search result for a conclusive verdict.
  - Cross-checks every LLM-cited URL and publication date against actual retrieved Tavily results, discarding any invented URLs and preventing fabricated dates.
  - Sandboxes both `<UNTRUSTED_PDF_CONTENT>` and `<UNTRUSTED_WEB_EVIDENCE>` against prompt-injection attacks.

---

## 3. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Runtime** | Python 3.11+ | Core application language |
| **Frontend UI** | Streamlit (`>=1.38.0`) + Material Design 3 CSS | 5-page SaaS dashboard with Light & Dark themes |
| **LLM SDK** | Official `google-genai` (`from google import genai`) | Gemini 2.5 Flash / Pro structured JSON generation |
| **Live Web Search** | `tavily-python` (`TavilyClient`) | Independent real-time web evidence retrieval |
| **PDF & OCR** | `PyMuPDF` (`fitz`), `pytesseract`, `Pillow` | Digital PDF parsing & scanned page OCR fallback |
| **Validation** | `Pydantic v2` (`>=2.7.0`) | Strict schema validation for claims, sources & verdicts |
| **Analytics & Data** | `Plotly` (`>=5.22.0`), `Pandas` (`>=2.2.0`) | Interactive visualizations & CSV tabular exports |
| **PDF Export** | `ReportLab` (`>=4.2.0`) | Publication-grade downloadable PDF audit reports |
| **Resilience & QA** | `tenacity`, `pytest` | Exponential backoff retries & automated unit/E2E testing |

---

## 4. Architecture

```text
factcheck-ai/
│
├── app.py                              # Main Streamlit entrypoint & 5-step pipeline orchestrator
├── requirements.txt                    # Python package dependencies
├── packages.txt                        # OS-level packages (libmupdf-dev, tesseract-ocr)
├── README.md                           # Project documentation & deployment guide
├── .gitignore                          # Git exclusions (.env, secrets.toml, __pycache__, venv)
│
├── config/
│   ├── __init__.py
│   └── settings.py                     # AppConfig, Material 3 Light/Dark palettes, domain lists, get_secret()
│
├── models/
│   ├── __init__.py
│   └── schemas.py                      # Pydantic v2 models (ExtractedClaim, EvidenceSource, ClaimVerificationResult, FactCheckReport)
│
├── modules/
│   ├── __init__.py
│   ├── pdf_extractor.py                # PyMuPDF paragraph extraction, scanned page detection & Tesseract OCR
│   ├── claim_extractor.py              # Paragraph-aware chunking, Gemini claim extraction & cross-chunk deduplication
│   ├── web_search.py                   # Multi-facet Tavily search client with session caching & credit conservation
│   ├── source_evaluator.py             # 6-dimension source authority, relevance, recency & support/contradiction evaluator
│   ├── fact_verifier.py                # 10-step Gemini fact verification engine with anti-hallucination grounding
│   └── report_generator.py             # Real-time analytics computation, 4-way filtering, ReportLab PDF, CSV & JSON exporters
│
├── utils/
│   ├── __init__.py
│   ├── llm_client.py                   # Official google-genai wrapper with Pydantic validation & malformed-JSON retries
│   ├── prompts.py                      # System instructions, domain checklists & <UNTRUSTED_*> prompt sandboxing
│   ├── security.py                     # PDF magic-byte validation, secret masking/scrubbing & session state cleanup
│   └── helpers.py                      # Domain extraction, file size formatting & Material Design 3 CSS injection
│
├── components/
│   ├── __init__.py
│   ├── sidebar.py                      # Global sidebar navigation, Material You theme toggle, API status & settings
│   ├── upload.py                       # Page 1 (Home) & Page 2 (Document Analysis) UI components
│   ├── dashboard.py                    # Page 3 (Analysis Dashboard) KPI cards, Plotly charts & filterable table
│   ├── claim_cards.py                  # Page 4 (Claim Details) 4-way filter bar & expandable Material 3 claim cards
│   └── report_view.py                  # Page 5 (Report) summary, statistics, PDF/CSV/JSON downloads & session reset
│
├── tests/
│   ├── test_pdf_and_claims.py          # Unit tests for PDF extraction, OCR diagnostics & claim identification
│   ├── test_web_search_and_evaluator.py# Unit tests for Tavily query generation, caching & source evaluator
│   ├── test_fact_verifier.py           # Unit tests for verification statuses, grounding & epistemic rules
│   ├── test_reporting_and_analytics.py # Unit tests for real analytics, filtering & PDF/CSV/JSON exports
│   └── test_end_to_end_qa.py           # Pytest E2E suite covering all 10 required document QA scenarios
│
└── .streamlit/
    ├── config.toml                     # Streamlit server & Material 3 default theme configuration
    └── secrets.toml.example            # Template for GEMINI_API_KEY and TAVILY_API_KEY
```

---

## 5. Installation Steps

```bash
# 1. Navigate to the project directory
cd factcheck-ai

# 2. Create and activate a Python 3.11+ virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional for local OCR on scanned PDFs)*: Install the Tesseract binary on your OS:
- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng libmupdf-dev`

---

## 6. Environment Variables & API Key Setup

FactCheck AI requires two API keys:
1. **`GEMINI_API_KEY`**: Obtain from [Google AI Studio](https://aistudio.google.com/).
2. **`TAVILY_API_KEY`**: Obtain from [Tavily Search](https://tavily.com/).

You can provide these keys in any of three secure ways (listed in order of resolution priority):

1. **Runtime Sidebar Override**: Enter keys in the masked password fields under **Configure API Keys & Parameters** in the sidebar (stored in session memory only).
2. **Environment Variables** (or a local `.env` file, which is git-ignored):
   ```bash
   export GEMINI_API_KEY="AIzaSy..."
   export TAVILY_API_KEY="tvly-..."
   ```
3. **Streamlit Secrets (`.streamlit/secrets.toml`)**:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Then edit `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   TAVILY_API_KEY = "tvly-..."
   ```

---

## 7. Local Execution Instructions

### Run the Streamlit Web Application

```bash
cd factcheck-ai
streamlit run app.py
```

The application will open at `http://localhost:8501`.

### Run the Automated Test Suite (`pytest` / `unittest`)

```bash
cd factcheck-ai
python3 -m pytest tests/ -v
```

---

## 8. Deployment Instructions (Streamlit Community Cloud)

1. **Push to GitHub**:
   - Initialize a Git repository in `factcheck-ai/` and push to GitHub.
   - Verify that `.streamlit/secrets.toml` and `.env` are excluded by `.gitignore`.
2. **Create App on Streamlit Community Cloud**:
   - Sign in to [share.streamlit.io](https://share.streamlit.io/) and click **Create app**.
   - Select your GitHub repository, branch, and set **Main file path** to `app.py`.
   - Streamlit Community Cloud automatically reads `requirements.txt` (Python packages) and `packages.txt` (`libmupdf-dev`, `tesseract-ocr`, `tesseract-ocr-eng`).
3. **Configure Cloud Secrets**:
   - Before or after launching, open **App Settings → Secrets** and paste:
     ```toml
     GEMINI_API_KEY = "your-actual-google-gemini-api-key"
     TAVILY_API_KEY = "tvly-your-actual-tavily-api-key"
     ```
4. **Save & Deploy**:
   - Click **Save** and wait for the container build to finish. Verify that both **Google Gemini** and **Tavily Search** show green **Connected** badges in the left sidebar.

---

## 9. Known Limitations

1. **Public Web Evidence Boundary**: Verification relies strictly on publicly indexed web sources accessible via the Tavily Search API at the time of analysis. Claims referencing private internal company metrics, paywalled subscription databases, or confidential records cannot be corroborated externally and will be classified as `UNVERIFIABLE`.
2. **Scanned PDF Image Quality**: While Tesseract OCR is integrated as an automatic fallback for scanned pages, severely degraded, handwritten, or low-resolution (<150 DPI) scans may yield incomplete text. Unreadable pages are explicitly listed in the UI and PDF audit report.
3. **Document Length & Rate Limits**: To maintain interactive responsiveness and respect API rate limits/quotas, analysis is capped at `75 pages` / `25 MB` per PDF and up to `30 claims` per run, sampling uniformly across long documents.
4. **Advisory Nature**: FactCheck AI is an automated evidence-retrieval and decision-support tool; its outputs do not constitute formal legal, financial, or regulatory audit certification.
