"""
Simple Local RAG - Text Chunking Module

RAG Concept:
Chunking splits large text (including multi-page PDFs) into smaller, overlapping segments.
For PDF documents, we keep track of the source page number for every chunk so that
retrieved answers cite the exact page in the PDF.
"""

from typing import List, Dict
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(
    text: str,
    source: str,
    page: int = 1,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, any]]:
    """
    Splits a piece of text (e.g. from a PDF page or text file) into overlapping character chunks.

    Args:
        text: String text to split.
        source: Filename (e.g. "report.pdf").
        page: Page number where this text originated.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Shared characters between consecutive chunks.

    Returns:
        List of chunk dictionaries.
    """
    chunks: List[Dict[str, any]] = []

    if len(text) <= chunk_size:
        return [{
            "text": text,
            "source": source,
            "page": page,
            "chunk_id": 0
        }]

    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than chunk_overlap.")

    chunk_id = 0
    start = 0
    total_length = len(text)

    while start < total_length:
        end = start + chunk_size
        chunk_slice = text[start:end].strip()

        if chunk_slice:
            chunks.append({
                "text": chunk_slice,
                "source": source,
                "page": page,
                "chunk_id": chunk_id
            })
            chunk_id += 1

        start += step

    return chunks


def chunk_documents(
    sections: List[Dict[str, any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, any]]:
    """
    Chunks all loaded document sections (pages).

    Args:
        sections: Output from load_documents() containing 'filename', 'text', 'page'.

    Returns:
        Flattened list of all chunks across all documents.
    """
    all_chunks: List[Dict[str, any]] = []
    global_chunk_id = 0

    for sec in sections:
        source = sec.get("filename", "unknown")
        page = sec.get("page", 1)
        text = sec.get("text", "")

        sec_chunks = chunk_text(
            text=text,
            source=source,
            page=page,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Assign unique global sequential chunk_ids across the index
        for c in sec_chunks:
            c["global_chunk_id"] = global_chunk_id
            global_chunk_id += 1
            all_chunks.append(c)

    return all_chunks
