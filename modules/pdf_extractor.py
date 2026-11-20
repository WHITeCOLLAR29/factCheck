"""MODULE 1: PDF Extraction engine using PyMuPDF (fitz) with paragraph preservation,
scanned page detection, and optional Tesseract OCR fallback.
"""

from __future__ import annotations

import io
import logging
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from config.settings import CONFIG
from models.schemas import DocumentMetadata
from utils.security import compute_sha256, sanitize_text_input, validate_pdf_upload

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when a PDF is invalid, encrypted, corrupted, or completely unreadable."""


@dataclass
class PageParagraph:
    """A discrete paragraph extracted from a specific PDF page."""

    page_number: int
    paragraph_index: int
    text: str


@dataclass
class PageExtractionDiagnostic:
    """Per-page diagnostic status produced during PDF extraction."""

    page_number: int
    status: str  # "native_text", "ocr_recovered", "scanned_unreadable", "empty"
    word_count: int
    image_count: int
    note: str = ""


@dataclass
class ExtractedPDFContent:
    """Container holding document metadata, page-indexed text, paragraph blocks, and diagnostics."""

    metadata: DocumentMetadata
    pages: List[Tuple[int, str]]  # List of (1-based page_number, page_text_with_paragraphs)
    paragraphs: List[PageParagraph] = field(default_factory=list)
    page_diagnostics: List[PageExtractionDiagnostic] = field(default_factory=list)
    full_text: str = ""


def is_tesseract_ocr_available() -> bool:
    """Check whether both the `tesseract` system binary and `pytesseract`/`PIL` are installed."""
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        return False


def _extract_paragraphs_from_page(page: fitz.Page) -> List[str]:
    """Extract ordered text paragraphs from a PyMuPDF Page using block coordinates.

    Preserves paragraph boundaries so claims can be paired with their surrounding paragraph context.
    """
    raw_blocks = page.get_text("blocks", sort=True) or []
    paragraphs: List[str] = []

    for blk in raw_blocks:
        # Block tuple: (x0, y0, x1, y1, "lines\nin\nblock", block_no, block_type)
        if len(blk) >= 7 and blk[6] != 0:
            # block_type != 0 means image block, not text
            continue
        raw_text = str(blk[4] or "").strip()
        if not raw_text:
            continue

        # Join intra-paragraph soft line wraps while keeping intentional line breaks
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not lines:
            continue

        # Merge hyphenated line wraps (e.g., "verifi-\ncation" -> "verification")
        merged_lines: List[str] = []
        for line in lines:
            if merged_lines and merged_lines[-1].endswith("-") and len(merged_lines[-1]) > 2:
                merged_lines[-1] = merged_lines[-1][:-1] + line
            else:
                merged_lines.append(line)

        para_text = sanitize_text_input(" ".join(merged_lines))
        if para_text:
            paragraphs.append(para_text)

    return paragraphs


def _attempt_ocr_on_page(page: fitz.Page, dpi: int = 220) -> Optional[List[str]]:
    """ Attempt OCR on a scanned PDF page using PyMuPDF's built-in Tesseract bridge or pytesseract.

    Returns a list of extracted paragraphs if successful, or None if OCR is unavailable or fails.
    """
    # 1. Try PyMuPDF native get_textpage_ocr if Tesseract is on PATH
    if shutil.which("tesseract") is not None:
        try:
            tp = page.get_textpage_ocr(flags=0, dpi=dpi, full=True)
            ocr_raw = page.get_text("text", textpage=tp) or ""
            cleaned = sanitize_text_input(ocr_raw)
            if cleaned and len(cleaned.split()) >= 3:
                paras = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
                return paras if paras else [cleaned]
        except Exception as exc:
            logger.debug("PyMuPDF get_textpage_ocr fallback to pytesseract: %s", exc)

    # 2. Fallback to rendering pixmap -> PIL Image -> pytesseract
    if not is_tesseract_ocr_available():
        return None

    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_bytes = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_bytes))
        ocr_text = pytesseract.image_to_string(pil_img) or ""
        cleaned = sanitize_text_input(ocr_text)
        if not cleaned:
            return []
        paras = [
            " ".join(p.split())
            for p in cleaned.split("\n\n")
            if p.strip()
        ]
        return paras
    except Exception as exc:
        logger.warning("Tesseract OCR execution failed on page: %s", exc)
        return None


def _page_looks_scanned(page: fitz.Page, native_word_count: int) -> Tuple[bool, int]:
    """Determine if a page with little/no native text is a scanned image page.

    Returns (is_scanned_candidate, image_count).
    """
    images = page.get_images(full=True) or []
    image_count = len(images)

    # Also check image blocks inside get_text("dict") in case of inline XObjects/drawings
    if image_count == 0:
        try:
            page_dict = page.get_text("dict") or {}
            for blk in page_dict.get("blocks", []):
                if blk.get("type") == 1:
                    image_count += 1
        except Exception:
            pass

    is_scanned = (native_word_count < 8) and (image_count > 0)
    return is_scanned, image_count


def extract_pdf_content(
    file_bytes: bytes,
    filename: str,
    max_pages: int = CONFIG.MAX_PDF_PAGES,
    max_size_mb: int = CONFIG.MAX_PDF_SIZE_MB,
    enable_ocr: bool = True,
) -> ExtractedPDFContent:
    """Validate and extract page-indexed text, paragraph context, and scanned-page diagnostics from PDF bytes.

    Requirements enforced:
    - Rejects unsupported, corrupted, empty, or excessively large files.
    - Refuses to process password-protected/encrypted files without authorization.
    - Preserves 1-based page numbers and paragraph context.
    - Detects empty or scanned pages, attempts optional Tesseract OCR, and explicitly reports any pages that could not be read without silently discarding them.
    """
    is_valid, validation_msg = validate_pdf_upload(
        file_bytes=file_bytes,
        filename=filename,
        max_size_mb=max_size_mb,
    )
    if not is_valid:
        raise PDFExtractionError(validation_msg)

    doc_hash = compute_sha256(file_bytes)

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError(
            f"Unable to open PDF document '{filename}'. The file is corrupted or malformed."
        ) from exc

    try:
        # Never process encrypted or password-protected PDFs without authorization
        if doc.is_encrypted or doc.needs_pass:
            raise PDFExtractionError(
                f"'{filename}' is encrypted or password-protected. "
                "FactCheck AI does not process encrypted files without authorization. Please upload an unlocked PDF."
            )

        total_doc_pages = len(doc)
        if total_doc_pages == 0:
            raise PDFExtractionError(f"The uploaded PDF '{filename}' contains 0 pages.")

        if total_doc_pages > max_pages:
            raise PDFExtractionError(
                f"The uploaded PDF has {total_doc_pages} pages, exceeding the maximum allowed limit of {max_pages} pages."
            )

        ocr_available = is_tesseract_ocr_available() or (shutil.which("tesseract") is not None)

        extracted_pages: List[Tuple[int, str]] = []
        all_paragraphs: List[PageParagraph] = []
        diagnostics: List[PageExtractionDiagnostic] = []
        ocr_pages: List[int] = []
        scanned_unreadable_pages: List[int] = []
        empty_pages: List[int] = []
        warnings: List[str] = []

        total_chars = 0
        total_words = 0

        for page_idx in range(total_doc_pages):
            page_num = page_idx + 1
            page = doc.load_page(page_idx)

            native_paragraphs = _extract_paragraphs_from_page(page)
            native_text = "\n\n".join(native_paragraphs)
            native_words = len(native_text.split())

            is_scanned, img_count = _page_looks_scanned(page, native_words)

            # Case 1: Page has sufficient native text
            if native_words >= 5 and not is_scanned:
                extracted_pages.append((page_num, native_text))
                for p_i, p_txt in enumerate(native_paragraphs, start=1):
                    all_paragraphs.append(
                        PageParagraph(
                            page_number=page_num,
                            paragraph_index=p_i,
                            text=p_txt,
                        )
                    )
                total_chars += len(native_text)
                total_words += native_words
                diagnostics.append(
                    PageExtractionDiagnostic(
                        page_number=page_num,
                        status="native_text",
                        word_count=native_words,
                        image_count=img_count,
                        note="Extracted native digital text.",
                    )
                )
                continue

            # Case 2: Page looks like a scanned image (little/no native text, contains image(s))
            if is_scanned or (native_words == 0 and img_count > 0):
                ocr_paragraphs: Optional[List[str]] = None
                if enable_ocr and ocr_available:
                    ocr_paragraphs = _attempt_ocr_on_page(page)

                if ocr_paragraphs:
                    ocr_text = "\n\n".join(ocr_paragraphs)
                    ocr_word_count = len(ocr_text.split())
                    if ocr_word_count >= 3:
                        ocr_pages.append(page_num)
                        extracted_pages.append((page_num, ocr_text))
                        for p_i, p_txt in enumerate(ocr_paragraphs, start=1):
                            all_paragraphs.append(
                                PageParagraph(
                                    page_number=page_num,
                                    paragraph_index=p_i,
                                    text=p_txt,
                                )
                            )
                        total_chars += len(ocr_text)
                        total_words += ocr_word_count
                        diagnostics.append(
                            PageExtractionDiagnostic(
                                page_number=page_num,
                                status="ocr_recovered",
                                word_count=ocr_word_count,
                                image_count=img_count,
                                note="Scanned page text recovered via Tesseract OCR.",
                            )
                        )
                        continue

                # If OCR was unavailable or produced no readable words, record explicitly
                scanned_unreadable_pages.append(page_num)
                reason = (
                    "Tesseract OCR is not installed on this system."
                    if not ocr_available
                    else "OCR could not recognize legible text from the scanned image."
                )
                diagnostics.append(
                    PageExtractionDiagnostic(
                        page_number=page_num,
                        status="scanned_unreadable",
                        word_count=native_words,
                        image_count=img_count,
                        note=f"Page {page_num} appears to be a scanned image and could not be read ({reason})",
                    )
                )
                continue

            # Case 3: Page has a few words (1-4 words, e.g., cover title or divider)
            if native_words > 0:
                extracted_pages.append((page_num, native_text))
                for p_i, p_txt in enumerate(native_paragraphs, start=1):
                    all_paragraphs.append(
                        PageParagraph(
                            page_number=page_num,
                            paragraph_index=p_i,
                            text=p_txt,
                        )
                    )
                total_chars += len(native_text)
                total_words += native_words
                diagnostics.append(
                    PageExtractionDiagnostic(
                        page_number=page_num,
                        status="native_text",
                        word_count=native_words,
                        image_count=img_count,
                        note="Short text page.",
                    )
                )
            else:
                # Case 4: Truly blank page (0 words, 0 images)
                empty_pages.append(page_num)
                diagnostics.append(
                    PageExtractionDiagnostic(
                        page_number=page_num,
                        status="empty",
                        word_count=0,
                        image_count=0,
                        note=f"Page {page_num} is blank (no text or images).",
                    )
                )

        # Build explicit user-facing diagnostic warnings so unreadable pages are NEVER silently discarded
        if ocr_pages:
            page_list_str = ", ".join(str(p) for p in ocr_pages)
            warnings.append(
                f"OCR Recovery: Page(s) {page_list_str} were detected as scanned images and processed via Tesseract OCR."
            )

        if scanned_unreadable_pages:
            page_list_str = ", ".join(str(p) for p in scanned_unreadable_pages)
            ocr_hint = (
                "Install `tesseract-ocr` (included in `packages.txt`) to enable automatic OCR on scanned pages."
                if not ocr_available
                else "Ensure scanned images are clear and at least 200 DPI."
            )
            warnings.append(
                f"Unreadable Scanned Page(s): Page(s) {page_list_str} contain scanned images without a digital text layer and could not be read. {ocr_hint}"
            )

        if empty_pages:
            page_list_str = ", ".join(str(p) for p in empty_pages)
            warnings.append(
                f"Blank Page(s): Page(s) {page_list_str} contained no extractable text or images."
            )

        # If the entire PDF yielded no readable text at all, raise an actionable error listing the unreadable pages
        if not extracted_pages or total_words < 3:
            if scanned_unreadable_pages:
                unreadable_str = ", ".join(str(p) for p in scanned_unreadable_pages)
                ocr_status_msg = (
                    "Tesseract OCR is currently unavailable in this environment."
                    if not ocr_available
                    else "Tesseract OCR attempted extraction but found no legible text."
                )
                raise PDFExtractionError(
                    f"Could not extract readable text from '{filename}'. "
                    f"Page(s) {unreadable_str} appear to be scanned image-only pages. {ocr_status_msg}"
                )
            raise PDFExtractionError(
                f"No readable text could be extracted from '{filename}'. "
                f"All {total_doc_pages} page(s) were empty or contained no text layer."
            )

        pdf_meta = doc.metadata or {}
        doc_title = (pdf_meta.get("title") or "").strip() or None
        doc_author = (pdf_meta.get("author") or "").strip() or None

        metadata = DocumentMetadata(
            filename=filename,
            file_size_bytes=len(file_bytes),
            page_count=total_doc_pages,
            readable_page_count=len(extracted_pages),
            total_characters=total_chars,
            total_words=total_words,
            document_hash=doc_hash,
            title=doc_title,
            author=doc_author,
            ocr_engine_available=ocr_available,
            ocr_pages=ocr_pages,
            scanned_unreadable_pages=scanned_unreadable_pages,
            empty_pages=empty_pages,
            extraction_warnings=warnings,
        )

        return ExtractedPDFContent(
            metadata=metadata,
            pages=extracted_pages,
            paragraphs=all_paragraphs,
            page_diagnostics=diagnostics,
            full_text="\n\n".join(txt for _, txt in extracted_pages),
        )
    finally:
        doc.close()
