"""
Simple Local RAG - Configuration Module

This module centralizes all configurable parameters for the RAG pipeline.
Centralizing configuration ensures we do not hardcode model names,
paths, or hyperparameters across multiple files.
"""

from pathlib import Path

# ==============================================================================
# 1. Ollama & Models Configuration
# ==============================================================================
# Ollama service endpoint (default local server)
OLLAMA_HOST = "http://127.0.0.1:11434"

# Embedding Model:
# Converts text chunks and questions into high-dimensional vectors.
# "nomic-embed-text" produces 768-dimensional embeddings and runs very fast locally.
EMBEDDING_MODEL = "nomic-embed-text"

# LLM Chat/Generation Model:
# Generates the final natural language answer from the retrieved context.
# Suitable lightweight choices for 8GB GPU: "qwen2.5-coder:latest", "qwen2.5:3b", "qwen2.5:1.5b"
LLM_MODEL = "qwen2.5-coder:latest"

# ==============================================================================
# 2. Chunking Hyperparameters
# ==============================================================================
# CHUNK_SIZE: Maximum number of characters in a single chunk.
# 1000 characters (~150-200 words) captures complete paragraphs & algorithm definitions.
CHUNK_SIZE = 1000

# CHUNK_OVERLAP: Number of overlapping characters between consecutive chunks.
CHUNK_OVERLAP = 200

# ==============================================================================
# 3. Retrieval Parameters
# ==============================================================================
# TOP_K: Number of most similar chunks retrieved from FAISS for each question.
TOP_K = 4


# ==============================================================================
# 4. Storage & Directory Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"

# FAISS index binary file
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"

# Metadata JSON file (maps FAISS vector ID -> chunk text & source document)
METADATA_PATH = DATA_DIR / "metadata.json"
