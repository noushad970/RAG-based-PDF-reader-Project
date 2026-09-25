"""
Simple Local RAG - Orchestration Pipeline (RAG Module)

RAG Concept:
This file connects all the individual components into a single, transparent pipeline:

          ┌────────────────────────────────────────────────┐
          │                  USER QUESTION                 │
          └───────────────────────┬────────────────────────┘
                                  │
                       [1. Question Embedding]
                                  │ (same model as document chunks)
                                  ▼
          ┌────────────────────────────────────────────────┐
          │              EMBEDDING VECTOR Q                │
          └───────────────────────┬────────────────────────┘
                                  │
                         [2. FAISS Retrieval]
                                  │ (Cosine Similarity Search)
                                  ▼
          ┌────────────────────────────────────────────────┐
          │             TOP-K RELEVANT CHUNKS              │
          └───────────────────────┬────────────────────────┘
                                  │
                        [3. Build Context]
                                  │ (Structure chunks into prompt)
                                  ▼
          ┌────────────────────────────────────────────────┐
          │            AUGMENTED PROMPT WITH CONTEXT       │
          └───────────────────────┬────────────────────────┘
                                  │
                         [4. LLM Generation]
                                  │ (Local Ollama Model)
                                  ▼
          ┌────────────────────────────────────────────────┐
          │              FINAL GROUNDED ANSWER             │
          │             + CITATIONS / SOURCES              │
          └────────────────────────────────────────────────┘
"""

from typing import Dict, Any, Optional
from src.config import TOP_K, EMBEDDING_MODEL, LLM_MODEL, FAISS_INDEX_PATH, METADATA_PATH
from src.vector_store import load_index_and_metadata
from src.retriever import retrieve
from src.generator import build_context, construct_prompt, generate_answer


class RAGPipeline:
    """
    Transparent RAG pipeline coordinator that demonstrates each stage explicitly.
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL
    ):
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.index = None
        self.metadata = None
        self._load_store()

    def _load_store(self) -> bool:
        """
        Loads the pre-built FAISS index and chunk metadata.
        """
        self.index, self.metadata = load_index_and_metadata(FAISS_INDEX_PATH, METADATA_PATH)
        return self.index is not None and self.metadata is not None

    def is_indexed(self) -> bool:
        """
        Checks if the FAISS index and metadata exist and contain data.
        """
        if self.index is None or self.metadata is None:
            self._load_store()
        return self.index is not None and self.index.ntotal > 0

    def query(self, question: str) -> Dict[str, Any]:
        """
        Executes the complete RAG query pipeline for a user question.

        Args:
            question: The user's query string.

        Returns:
            Dictionary containing:
            - "question": Original user question
            - "answer": LLM generated answer
            - "retrieved_chunks": List of retrieved chunk dictionaries
            - "context": Formatted context string passed to LLM
            - "prompt": Full prompt string sent to LLM
            - "debug_info": Vector dimensions, previews, and raw scores
        """
        if not self.is_indexed():
            raise FileNotFoundError(
                "No FAISS index found. Please run 'python main.py ingest' first."
            )

        # -------------------------------------------------------------
        # STEP 1 & 2: Embed Question & Retrieve Top-K Chunks from FAISS
        # -------------------------------------------------------------
        # The question is embedded using the exact same model used for the
        # documents, placing query and documents into the same vector space.
        retrieved_chunks, debug_meta = retrieve(
            query=question,
            index=self.index,
            metadata=self.metadata,
            top_k=self.top_k,
            embedding_model=self.embedding_model
        )

        # -------------------------------------------------------------
        # STEP 3: Build Grounded Context
        # -------------------------------------------------------------
        # We format the retrieved chunks with clear source attribution
        # so the LLM knows exactly which document provided each fact.
        context_str = build_context(retrieved_chunks)
        full_prompt = construct_prompt(question, context_str)

        # -------------------------------------------------------------
        # STEP 4: Call LLM for Generation
        # -------------------------------------------------------------
        # Send prompt with context to Ollama local LLM
        answer = generate_answer(
            query=question,
            context=context_str,
            model=self.llm_model
        )

        # -------------------------------------------------------------
        # STEP 5: Package Results and Debug Telemetry
        # -------------------------------------------------------------
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "context": context_str,
            "prompt": full_prompt,
            "debug_info": debug_meta
        }
