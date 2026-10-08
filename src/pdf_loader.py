"""
Load PDF files and extract text per page with metadata.
"""

from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.utils import setup_logging

logger = setup_logging()


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or has no usable text."""


def load_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Read a PDF and return one record per page with text, page number, and source filename.

    Returns:
        List of dicts: {"text": str, "page": int, "source": str}
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("PDF not found: %s", path)
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise PDFLoadError(f"Not a PDF file: {path.name}")

    source_name = path.name
    pages_out: list[dict[str, Any]] = []

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        logger.error("Invalid PDF %s: %s", source_name, exc)
        raise PDFLoadError(f"Could not read PDF '{source_name}'. It may be corrupted.") from exc
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", source_name, exc)
        raise PDFLoadError(f"Failed to open PDF '{source_name}'.") from exc

    if len(reader.pages) == 0:
        raise PDFLoadError(f"PDF '{source_name}' has no pages.")

    for i, page in enumerate(reader.pages):
        page_num = i + 1
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Could not extract text from %s page %s: %s", source_name, page_num, exc)
            raw = ""

        text = raw.strip()
        if not text:
            logger.debug("Empty page: %s page %s", source_name, page_num)
            continue

        pages_out.append(
            {
                "text": text,
                "page": page_num,
                "source": source_name,
            }
        )

    if not pages_out:
        raise PDFLoadError(
            f"No extractable text found in '{source_name}'. "
            "The PDF may be scanned images without a text layer."
        )

    logger.info("Loaded PDF %s: %s pages with text", source_name, len(pages_out))
    return pages_out


def load_multiple_pdfs(file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load several PDFs and concatenate page records."""
    all_pages: list[dict[str, Any]] = []
    for fp in file_paths:
        all_pages.extend(load_pdf(fp))
    logger.info("Total pages loaded from %s files: %s", len(file_paths), len(all_pages))
    return all_pages
