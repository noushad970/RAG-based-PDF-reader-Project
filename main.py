"""
Simple Local RAG - Command Line Interface (CLI)

Entry point for the Simple Local RAG application.

Supported Commands:
- python main.py setup    : Verify Ollama service and required models.
- python main.py ingest   : Load documents, chunk, embed, and build FAISS index.
- python main.py chat     : Interactive terminal Q&A session.
- python main.py ask "..." : Single question answering from CLI.
- python main.py debug    : Detailed step-by-step pipeline inspection.
"""

import sys
import argparse
from typing import List, Dict
import numpy as np
import ollama

# Configure console encoding safely for Windows command prompt / PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf8"}:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


from src.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_HOST,
    DOCUMENTS_DIR,
    DATA_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)
from src.document_loader import load_documents
from src.chunker import chunk_documents
from src.embeddings import embed_documents
from src.vector_store import (
    build_faiss_index,
    save_index_and_metadata,
    load_index_and_metadata
)
from src.rag import RAGPipeline


# ==============================================================================
# Helper / Health Check Functions
# ==============================================================================
def check_ollama_status() -> Dict[str, any]:
    """
    Checks if Ollama service is reachable and retrieves installed models.
    """
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models_response = client.list()
        installed_models = []
        for m in models_response.get("models", []):
            name = m.get("model") or m.get("name")
            if name:
                installed_models.append(name)
        return {
            "running": True,
            "models": installed_models,
            "error": None
        }
    except Exception as e:
        return {
            "running": False,
            "models": [],
            "error": str(e)
        }


def is_model_installed(target_model: str, installed_models: List[str]) -> bool:
    """
    Checks if target_model is present in installed_models, handling tag prefixes.
    E.g. 'nomic-embed-text' matches 'nomic-embed-text:latest'.
    """
    target_clean = target_model.split(":")[0]
    for installed in installed_models:
        if installed == target_model or installed.startswith(f"{target_clean}:"):
            return True
    return False


# ==============================================================================
# CLI Command Implementations
# ==============================================================================
def run_setup():
    """
    Checks Ollama environment and model availability.
    """
    print("=" * 40)
    print("      OLLAMA & ENVIRONMENT SETUP")
    print("=" * 40)

    status = check_ollama_status()
    if not status["running"]:
        print("Ollama Service: NOT RUNNING [X]")
        print("\nOllama is not running or not reachable at " + OLLAMA_HOST)
        print("Please start the Ollama application or service and try again.")
        return

    print("Ollama Service: RUNNING [OK]\n")

    installed = status["models"]

    # 1. Check Embedding Model
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    if is_model_installed(EMBEDDING_MODEL, installed):
        print("Status: INSTALLED [OK]\n")
    else:
        print("Status: NOT INSTALLED [X]")
        print("Please run:")
        print(f"  ollama pull {EMBEDDING_MODEL}\n")

    # 2. Check LLM Chat Model
    print(f"Chat Model: {LLM_MODEL}")
    if is_model_installed(LLM_MODEL, installed):
        print("Status: INSTALLED [OK]\n")
    else:
        print("Status: NOT INSTALLED [X]")
        print("Please run:")
        print(f"  ollama pull {LLM_MODEL}\n")

    print("=" * 40)


def run_ingest():
    """
    Runs the full indexing pipeline:
    Load Documents -> Chunk -> Embed -> Build FAISS Index -> Save Metadata.
    """
    print("=" * 40)
    print("             RAG INGESTION")
    print("=" * 40)

    # Pre-check Ollama status
    status = check_ollama_status()
    if not status["running"]:
        print("\nOllama is not running. Please start Ollama and try again.")
        return

    if not is_model_installed(EMBEDDING_MODEL, status["models"]):
        print(f"\nRequired embedding model '{EMBEDDING_MODEL}' is not installed.")
        print("Please run:")
        print(f"  ollama pull {EMBEDDING_MODEL}")
        return

    # Step 1: Load documents
    print("\nLoading documents from:", DOCUMENTS_DIR)
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print(f"No supported documents (.txt, .md) found in '{DOCUMENTS_DIR}'.")
        print(f"Please add your documents to '{DOCUMENTS_DIR}/' and re-run ingest.")
        return
    print(f"Found {len(documents)} document(s) [OK]")

    # Step 2: Chunk documents
    print(f"\nCreating chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        print("No text chunks could be created.")
        return
    print(f"Created {len(chunks)} chunk(s) [OK]")

    # Step 3: Generate embeddings
    print(f"\nGenerating embeddings using '{EMBEDDING_MODEL}'...")
    try:
        vectors = embed_documents(chunks, model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"\nEmbedding generation failed: {e}")
        return
    print(f"Generated embeddings shape: {vectors.shape} [OK]")

    # Step 4: Build FAISS index
    print("\nBuilding FAISS index (Inner Product / Cosine Similarity)...")
    try:
        index = build_faiss_index(vectors)
    except Exception as e:
        print(f"FAISS index creation failed: {e}")
        return
    print(f"FAISS index created with {index.ntotal} vector(s) [OK]")

    # Step 5: Save index and metadata
    print("\nSaving FAISS index and metadata...")
    save_index_and_metadata(index, chunks, FAISS_INDEX_PATH, METADATA_PATH)
    print(f"Index saved to: {FAISS_INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")

    print("\n" + "=" * 40)
    print("Ingestion complete! You can now run:")
    print("  python main.py chat")
    print("=" * 40)



def format_sources(retrieved_chunks: List[Dict[str, any]]) -> str:
    """
    Formats retrieved sources for clean terminal output.
    """
    if not retrieved_chunks:
        return "No sources used."

    lines = []
    for i, c in enumerate(retrieved_chunks, start=1):
        source = c.get("source", "Unknown")
        chunk_id = c.get("chunk_id", "Unknown")
        score = c.get("similarity", 0.0)
        lines.append(f"{i}. {source} (Chunk: {chunk_id}, Cosine Similarity: {score:.4f})")
    return "\n".join(lines)


def run_single_query(question: str, pipeline: RAGPipeline, debug: bool = False):
    """
    Processes a single question and displays the response.
    """
    if not question.strip():
        print("Please enter a non-empty question.")
        return

    try:
        result = pipeline.query(question)
    except Exception as e:
        print(f"\nQuery error: {e}")
        return

    if debug:
        print("\n" + "=" * 50)
        print("                 DEBUG MODE")
        print("=" * 50)

        print("\nUSER QUESTION")
        print("-" * 30)
        print(result["question"])

        debug_info = result["debug_info"]
        q_vec = debug_info.get("query_vector", np.array([]))
        dim = len(q_vec)
        preview = [round(float(v), 4) for v in q_vec[:5]]

        print("\nQUESTION EMBEDDING")
        print("-" * 30)
        print(f"Model: {pipeline.embedding_model}")
        print(f"Dimension: {dim}")
        print(f"Vector preview (first 5 elements): {preview} ...")

        print(f"\nRETRIEVAL (Top K: {len(result['retrieved_chunks'])})")
        print("-" * 30)
        for i, chunk in enumerate(result["retrieved_chunks"], start=1):
            print(f"Result {i}")
            print(f"Source: {chunk.get('source')}")
            print(f"Chunk ID: {chunk.get('chunk_id')}")
            print(f"Cosine Similarity Score: {chunk.get('similarity', 0.0):.4f}")
            print(f"Text Snippet: {chunk.get('text')[:120]}...\n")

        print("RETRIEVED CONTEXT PASSED TO PROMPT")
        print("-" * 30)
        print(result["context"])

        print("\nFULL LLM PROMPT")
        print("-" * 30)
        print(result["prompt"])

        print("\nFINAL ANSWER")
        print("-" * 30)
        print(result["answer"])

        print("\nSOURCES")
        print("-" * 30)
        print(format_sources(result["retrieved_chunks"]))
        print("=" * 50 + "\n")
    else:
        print("\n" + "=" * 40)
        print("ANSWER")
        print("=" * 40)
        print(result["answer"])

        print("\n" + "=" * 40)
        print("SOURCES")
        print("=" * 40)
        print(format_sources(result["retrieved_chunks"]))
        print()


def run_chat(debug: bool = False):
    """
    Runs interactive chat loop in terminal.
    """
    print("=" * 40)
    print("       SIMPLE LOCAL RAG CHAT")
    if debug:
        print("           [DEBUG MODE ON]")
    print("=" * 40)
    print("Type 'exit' or 'quit' to end the session.\n")

    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    if not pipeline.is_indexed():
        print("No FAISS index found.")
        print("Please build the index first by running:")
        print("  python main.py ingest\n")
        return

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue

            if question.lower() in {"exit", "quit", "q"}:
                print("\nGoodbye!")
                break

            print("\nSearching documents and generating answer...")
            run_single_query(question, pipeline, debug=debug)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Simple Local RAG System for Learning"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: setup
    subparsers.add_parser("setup", help="Check Ollama service and models status")

    # Command: ingest
    subparsers.add_parser("ingest", help="Ingest documents and build FAISS index")

    # Command: chat
    chat_parser = subparsers.add_parser("chat", help="Start interactive Q&A chat")
    chat_parser.add_argument("--debug", action="store_true", help="Enable debug telemetry output")

    # Command: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a single question directly")
    ask_parser.add_argument("question", type=str, help="The question to ask")
    ask_parser.add_argument("--debug", action="store_true", help="Enable debug telemetry output")

    # Command: debug
    debug_parser = subparsers.add_parser("debug", help="Ask a question with full RAG pipeline debug view")
    debug_parser.add_argument("question", nargs="?", type=str, default=None, help="Optional single question")

    # Command: web
    web_parser = subparsers.add_parser("web", help="Launch the local Web UI interface in your browser")
    web_parser.add_argument("--port", type=int, default=8000, help="Port to run web server on (default: 8000)")
    web_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")

    args = parser.parse_args()

    if args.command == "setup":
        run_setup()
    elif args.command == "ingest":
        run_ingest()
    elif args.command == "chat":
        run_chat(debug=args.debug)
    elif args.command == "web":
        import socket
        import uvicorn

        host = args.host
        port = args.port

        def is_port_in_use(p, h):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex((h, p)) == 0

        if is_port_in_use(port, host):
            for p in range(port + 1, port + 10):
                if not is_port_in_use(p, host):
                    print(f"\n[!] Note: Port {port} was in use. Switching to port {p}.\n")
                    port = p
                    break

        print("=" * 55)
        print(f"  Launching Local PDF RAG Web UI on http://{host}:{port}")
        print("=" * 55)
        uvicorn.run("src.server:app", host=host, port=port, reload=False)

    elif args.command == "ask":
        pipeline = RAGPipeline()
        run_single_query(args.question, pipeline, debug=args.debug)
    elif args.command == "debug":
        if args.question:
            pipeline = RAGPipeline()
            run_single_query(args.question, pipeline, debug=True)
        else:
            run_chat(debug=True)
    else:
        # Default behavior when no subcommand is specified
        parser.print_help()



if __name__ == "__main__":
    main()
