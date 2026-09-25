"""
Simple Local RAG - Retriever Module

RAG Concept:
This is the "R" (Retrieval) in Retrieval-Augmented Generation!

Why must we use the same embedding model for queries and documents?
An embedding model defines a specific geometric vector space.
Using the exact same model ensures query and document vectors exist in the SAME space.

In addition, smart retrieval checks for specific page mentions (e.g. "page 5")
to ensure direct page queries retrieve the requested page's chunks.
"""

import re
from typing import List, Dict, Tuple, Optional
import faiss
from src.config import TOP_K, EMBEDDING_MODEL
from src.embeddings import embed_text
from src.vector_store import search_index


def extract_page_number_from_query(query: str) -> Optional[int]:
    """
    Detects if the query is asking about a specific page (e.g., 'page 5', 'on page 12').
    """
    match = re.search(r'\bpage\s*(\d+)\b', query, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def retrieve(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, any]],
    top_k: int = TOP_K,
    embedding_model: str = EMBEDDING_MODEL
) -> Tuple[List[Dict[str, any]], Dict[str, any]]:
    """
    Retrieves the top_k most relevant document chunks for a given text query.

    Args:
        query: User question string.
        index: Loaded FAISS index.
        metadata: List of chunk metadata dictionaries.
        top_k: Number of chunks to retrieve.
        embedding_model: Name of the embedding model to use.

    Returns:
        Tuple of (results_list, debug_info_dict).
    """
    # 1. Embed the question using search_query mode
    query_vector = embed_text(query, model=embedding_model, is_query=True)

    # Check if a specific page was asked for in the query
    target_page = extract_page_number_from_query(query)

    # Request a slightly larger candidate pool from FAISS to allow re-ranking / page boosting
    candidate_k = min(max(top_k * 3, 10), index.ntotal)
    if candidate_k <= 0:
        return [], {"query_vector": query_vector, "raw_scores": [], "raw_indices": []}

    scores, indices = search_index(index, query_vector, top_k=candidate_k)

    retrieved_map: Dict[int, Dict[str, any]] = {}

    for score, idx in zip(scores, indices):
        if idx == -1 or idx >= len(metadata):
            continue

        chunk_meta = metadata[idx].copy()
        chunk_meta["similarity"] = float(score)
        chunk_meta["index_id"] = int(idx)
        retrieved_map[idx] = chunk_meta

    # If user explicitly asked for a specific page, ensure chunks from that page are prioritized
    if target_page is not None:
        for idx, chunk in enumerate(metadata):
            if chunk.get("page") == target_page:
                if idx not in retrieved_map:
                    chunk_copy = chunk.copy()
                    chunk_copy["similarity"] = 0.95  # Direct page match boost
                    chunk_copy["index_id"] = idx
                    retrieved_map[idx] = chunk_copy
                else:
                    retrieved_map[idx]["similarity"] = max(retrieved_map[idx]["similarity"], 0.95)

    # Sort candidates by similarity score descending
    sorted_candidates = sorted(retrieved_map.values(), key=lambda x: x.get("similarity", 0.0), reverse=True)

    # Pick top_k
    final_results = sorted_candidates[:top_k]

    debug_info = {
        "query_vector": query_vector,
        "raw_scores": scores.tolist() if hasattr(scores, "tolist") else list(scores),
        "raw_indices": indices.tolist() if hasattr(indices, "tolist") else list(indices),
        "target_page_filter": target_page
    }

    return final_results, debug_info
