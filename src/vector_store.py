"""
Simple Local RAG - FAISS Vector Store Module

RAG Concept:
What is FAISS?
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
and clustering of dense vectors.

Why Vector Normalization & Cosine Similarity?
Cosine similarity measures the angle between two vectors, regardless of their magnitude:
    Cosine Similarity(A, B) = (A · B) / (||A|| * ||B||)

When vectors are L2-normalized (meaning their Euclidean length ||A|| = 1):
    Cosine Similarity(A, B) = A · B (Inner Product / Dot Product)

FAISS provides `IndexFlatIP` (Inner Product). By normalizing all vectors to unit length
before adding them to the index and before searching, `IndexFlatIP` returns EXACT
Cosine Similarity scores (values between -1.0 and 1.0).

Index & Metadata Link:
FAISS stores ONLY numerical vectors and integer IDs (0, 1, 2, ...).
It does NOT store the original text or filenames.
Therefore, we save a parallel metadata file (`metadata.json`) where:
    FAISS vector at index 0  <--->  metadata[0] (source, chunk_id, text)
    FAISS vector at index 1  <--->  metadata[1] (source, chunk_id, text)
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from src.config import FAISS_INDEX_PATH, METADATA_PATH, DATA_DIR


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes 1D or 2D vectors to unit length (L2 norm = 1.0).

    Args:
        vectors: NumPy array of shape (D,) or (N, D).

    Returns:
        L2-normalized float32 NumPy array.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm > 0:
            return (vectors / norm).astype(np.float32)
        return vectors.astype(np.float32)

    # 2D matrix normalization along each row
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1e-10, norms)
    return (vectors / norms).astype(np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Creates and populates a FAISS IndexFlatIP index with normalized vectors.

    Args:
        vectors: 2D NumPy array of shape (N, D).

    Returns:
        Populated FAISS index.
    """
    num_vectors, dimension = vectors.shape

    # IndexFlatIP computes exact inner product (dot product)
    index = faiss.IndexFlatIP(dimension)

    # Normalize vectors so inner product equals cosine similarity
    normalized_vecs = normalize_vectors(vectors)

    # Add vectors to FAISS index
    index.add(normalized_vecs)

    return index


def save_index_and_metadata(
    index: faiss.Index,
    metadata: List[Dict[str, any]],
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = METADATA_PATH
) -> None:
    """
    Saves the FAISS index binary file and the metadata JSON file to disk.

    Args:
        index: FAISS index object.
        metadata: List of chunk metadata dicts matching the vector order.
        index_path: Destination path for faiss.index.
        metadata_path: Destination path for metadata.json.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(index_path))

    # Save metadata JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_index_and_metadata(
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = METADATA_PATH
) -> Tuple[Optional[faiss.Index], Optional[List[Dict[str, any]]]]:
    """
    Loads the FAISS index and metadata from disk if they exist.

    Returns:
        Tuple of (FAISS index, metadata list) or (None, None) if not found.
    """
    if not index_path.exists() or not metadata_path.exists():
        return None, None

    try:
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata
    except Exception as e:
        print(f"Error loading index or metadata: {e}")
        return None, None


def search_index(
    index: faiss.Index,
    query_vector: np.ndarray,
    top_k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Searches the FAISS index for the top_k most similar vectors.

    Args:
        index: Loaded FAISS index.
        query_vector: 1D NumPy array representing the question embedding.
        top_k: Number of nearest neighbors to retrieve.

    Returns:
        Tuple of (similarity_scores, chunk_indices).
        similarity_scores: 1D array of cosine similarity floats (higher = more similar).
        chunk_indices: 1D array of integer indices corresponding to metadata rows.
    """
    # Normalize query vector to unit length for cosine similarity
    normalized_query = normalize_vectors(query_vector)

    # Reshape 1D vector (D,) to 2D matrix (1, D) as required by FAISS search API
    query_matrix = np.expand_dims(normalized_query, axis=0).astype(np.float32)

    # Perform similarity search
    # scores: shape (1, top_k)
    # indices: shape (1, top_k)
    scores, indices = index.search(query_matrix, top_k)

    return scores[0], indices[0]
