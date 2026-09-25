"""
Simple Local RAG - Embeddings Module

RAG Concept:
What is an Embedding?
An embedding model converts human text into a dense numerical vector (a list of floating-point numbers).
Texts with similar meanings are mapped to vectors that point in similar directions in multi-dimensional space.

For nomic-embed-text:
Nomic uses asymmetric task prefixes to distinguish between indexing documents and searching queries:
- Query prefix: "search_query: "
- Document prefix: "search_document: "
"""

from typing import List, Dict
import numpy as np
import ollama
from src.config import EMBEDDING_MODEL, OLLAMA_HOST


def get_ollama_client() -> ollama.Client:
    """
    Initializes and returns the official Ollama client pointing to our configured host.
    """
    return ollama.Client(host=OLLAMA_HOST)


def embed_text(
    text: str,
    model: str = EMBEDDING_MODEL,
    is_query: bool = False
) -> np.ndarray:
    """
    Generates an embedding vector for a single string of text.

    Args:
        text: Input text string.
        model: Name of the Ollama embedding model.
        is_query: If True and using nomic model, prefixes with 'search_query: '.
                  If False, prefixes with 'search_document: '.

    Returns:
        1D NumPy array of float32 values representing the embedding vector.
    """
    client = get_ollama_client()

    # Apply nomic-embed-text task prefixes for optimal asymmetric retrieval
    formatted_text = text
    if "nomic" in model.lower():
        prefix = "search_query: " if is_query else "search_document: "
        if not text.startswith(prefix):
            formatted_text = prefix + text

    try:
        response = client.embeddings(model=model, prompt=formatted_text)
        vector = response["embedding"]
        return np.array(vector, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate embedding with model '{model}'. "
            f"Ensure Ollama is running and run 'ollama pull {model}'.\nError: {e}"
        )


def embed_documents(
    chunks: List[Dict[str, any]],
    model: str = EMBEDDING_MODEL
) -> np.ndarray:
    """
    Generates embedding vectors for a list of document chunks.

    Args:
        chunks: List of chunk dictionaries containing a 'text' key.
        model: Name of the Ollama embedding model.

    Returns:
        2D NumPy array of shape (num_chunks, embedding_dimension).
    """
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    vectors: List[np.ndarray] = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        if idx % 100 == 0 or idx == total or idx == 1:
            print(f"Generating embedding [{idx}/{total}] for '{chunk.get('source', 'unknown')}' (Page {chunk.get('page', 1)})...")
        vec = embed_text(chunk["text"], model=model, is_query=False)
        vectors.append(vec)

    # Stack all 1D vectors into a single 2D matrix (N, D) for FAISS
    return np.vstack(vectors).astype(np.float32)
