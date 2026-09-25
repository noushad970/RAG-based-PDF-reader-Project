"""
Simple Local RAG - Document Loader Module

RAG Concept:
Extracts text from PDF, TXT, and MD files.
For PDFs:
1. Extracts digital text page-by-page using PyMuPDF.
2. If a page is scanned (image-only / no selectable text), automatically runs
   local OCR (RapidOCR) to transcribe the handwritten/printed text into searchable knowledge.
"""

from pathlib import Path
from typing import List, Dict, Optional
import pymupdf

# Supported file formats
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Lazy-loaded OCR instance
_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except ImportError:
            _ocr_engine = False
    return _ocr_engine


def extract_text_from_pdf(file_path: Path) -> List[Dict[str, any]]:
    """
    Extracts text page-by-page from a PDF document.
    Uses direct text extraction first; falls back to OCR for scanned pages.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of dictionaries with 'filename', 'text', 'page', and 'total_pages'.
    """
    pages_data: List[Dict[str, any]] = []

    try:
        doc = pymupdf.open(str(file_path))
        num_pages = len(doc)

        for page_idx in range(num_pages):
            page = doc[page_idx]
            # 1. Attempt standard digital text extraction
            page_text = (page.get_text() or "").strip()

            # 2. If page has no selectable text (scanned image), run OCR fallback
            if len(page_text) < 15:
                ocr = get_ocr_engine()
                if ocr:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img_bytes = pix.tobytes("png")
                        ocr_result, _ = ocr(img_bytes)
                        if ocr_result:
                            ocr_text = "\n".join([line[1] for line in ocr_result]).strip()
                            if ocr_text:
                                page_text = ocr_text
                    except Exception as ocr_err:
                        print(f"OCR warning on {file_path.name} page {page_idx + 1}: {ocr_err}")

            if page_text:
                pages_data.append({
                    "filename": file_path.name,
                    "text": page_text,
                    "page": page_idx + 1,
                    "total_pages": num_pages
                })

    except Exception as e:
        print(f"Warning: Failed to extract text from PDF '{file_path.name}': {e}")

    return pages_data


def load_single_document(file_path: Path) -> List[Dict[str, any]]:
    """
    Loads text from a single file based on its extension.

    Returns:
        List of page/section dictionaries for this file.
    """
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    if ext in {".txt", ".md"}:
        try:
            text = file_path.read_text(encoding="utf-8").strip()
            if text:
                return [{
                    "filename": file_path.name,
                    "text": text,
                    "page": 1,
                    "total_pages": 1
                }]
        except Exception as e:
            print(f"Warning: Failed to read text file '{file_path.name}': {e}")

    return []


def load_documents(directory: Path) -> List[Dict[str, any]]:
    """
    Scans the specified directory and loads all supported documents (.pdf, .txt, .md).

    Returns:
        List of document sections with 'filename', 'text', 'page', and 'total_pages'.
    """
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return []

    all_sections: List[Dict[str, any]] = []

    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        doc_sections = load_single_document(file_path)
        all_sections.extend(doc_sections)

    return all_sections
