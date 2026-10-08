"""Tests for PDF loading."""

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.pdf_loader import PDFLoadError, load_pdf


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_pdf(Path("nonexistent_file_xyz.pdf"))


def test_not_pdf_raises(tmp_path: Path):
    bad = tmp_path / "doc.txt"
    bad.write_text("hello")
    with pytest.raises(PDFLoadError):
        load_pdf(bad)


def test_empty_pdf_no_text_raises(tmp_path: Path):
    pdf_path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    with pytest.raises(PDFLoadError):
        load_pdf(pdf_path)
