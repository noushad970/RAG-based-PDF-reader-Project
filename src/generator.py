"""
Simple Local RAG - Generator Module

RAG Concept:
This is the "AG" (Augmented Generation) in Retrieval-Augmented Generation!

Why Augment the Prompt?
By providing retrieved context snippets (including filenames and page numbers),
the LLM can read the exact source material and formulate an accurate answer.
"""

from typing import List, Dict
import ollama
from src.config import LLM_MODEL, OLLAMA_HOST


def build_context(retrieved_chunks: List[Dict[str, any]]) -> str:
    """
    Formats retrieved chunks into a clean, structured context string for the LLM.

    Args:
        retrieved_chunks: List of chunk metadata dictionaries.

    Returns:
        Structured context string.
    """
    if not retrieved_chunks:
        return "No relevant context found."

    context_blocks = []
    for chunk in retrieved_chunks:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", 1)
        chunk_id = chunk.get("chunk_id", 0)
        text = chunk.get("text", "").strip()

        block = f"[DOCUMENT: {source} | PAGE: {page} | CHUNK: {chunk_id}]\n{text}"
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


def construct_prompt(query: str, context: str) -> str:
    """
    Builds the complete augmented prompt sent to the LLM.

    Args:
        query: User's question.
        context: Retrieved documents context string.

    Returns:
        The full prompt string.
    """
    prompt = f"""You are an intelligent, precise AI document assistant.

Instructions:
1. Answer the user's question clearly and accurately using the provided context below.
2. Synthesize facts, author names, definitions, policies, tables, and page information found in the context.
3. If the user asks about authors, titles, specific pages, or topics that are mentioned in the context, describe them clearly and cite the relevant page/source.
4. Only say "I don't know based on the provided documents." if the provided context contains absolutely no relevant information to answer the question.
5. Do NOT invent facts that are not present or implied by the context.

=== PROVIDED CONTEXT ===
{context}

=== USER QUESTION ===
{query}

=== ANSWER ==="""
    return prompt


def generate_answer(
    query: str,
    context: str,
    model: str = LLM_MODEL
) -> str:
    """
    Sends the augmented prompt to Ollama and returns the generated text answer.

    Args:
        query: User's question.
        context: Retrieved context text.
        model: Ollama model name.

    Returns:
        Generated answer string from the LLM.
    """
    client = ollama.Client(host=OLLAMA_HOST)
    prompt = construct_prompt(query, context)

    try:
        response = client.generate(model=model, prompt=prompt)
        return response["response"].strip()
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate answer with model '{model}'. "
            f"Ensure Ollama is running and run 'ollama pull {model}'.\nError: {e}"
        )
