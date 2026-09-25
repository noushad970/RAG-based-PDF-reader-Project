"""
Simple Local RAG - FastAPI Web Server Module

Provides a REST API for uploading PDFs, querying the RAG pipeline,
and managing documents directly from a local web interface.
"""

import shutil
import time
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import (
    DOCUMENTS_DIR,
    DATA_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from src.document_loader import load_documents, SUPPORTED_EXTENSIONS
from src.chunker import chunk_documents
from src.embeddings import embed_documents
from src.vector_store import (
    build_faiss_index,
    save_index_and_metadata,
    load_index_and_metadata
)
from src.rag import RAGPipeline


app = FastAPI(title="Local PDF RAG Studio", version="1.0.0")

# Static files directory
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# Shared pipeline instance
_pipeline: Optional[RAGPipeline] = None


def get_pipeline(force_reload: bool = False) -> RAGPipeline:
    global _pipeline
    if _pipeline is None or force_reload:
        _pipeline = RAGPipeline(
            top_k=TOP_K,
            embedding_model=EMBEDDING_MODEL,
            llm_model=LLM_MODEL
        )
    return _pipeline


def run_ingestion_internal() -> dict:
    """
    Executes the ingestion pipeline on the documents/ folder.
    """
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        # If no documents, clear any existing index
        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()
        get_pipeline(force_reload=True)
        return {"documents_count": 0, "chunks_count": 0, "vectors_count": 0}

    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vectors = embed_documents(chunks, model=EMBEDDING_MODEL)
    index = build_faiss_index(vectors)
    save_index_and_metadata(index, chunks, FAISS_INDEX_PATH, METADATA_PATH)

    # Reload the pipeline with the new index
    get_pipeline(force_reload=True)

    # Unique source file count
    unique_files = len({c.get("source") for c in chunks})

    return {
        "documents_count": unique_files,
        "chunks_count": len(chunks),
        "vectors_count": index.ntotal
    }


# ==============================================================================
# API Endpoints
# ==============================================================================
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Web UI is loading...</h1>")


@app.get("/api/status")
async def get_status():
    """
    Returns Ollama health, installed models, and index statistics.
    """
    import ollama
    from src.config import OLLAMA_HOST

    ollama_ok = False
    installed_models = []
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models_resp = client.list()
        for m in models_resp.get("models", []):
            name = m.get("model") or m.get("name")
            if name:
                installed_models.append(name)
        ollama_ok = True
    except Exception:
        ollama_ok = False

    pipeline = get_pipeline()
    is_indexed = pipeline.is_indexed()
    total_vectors = pipeline.index.ntotal if is_indexed and pipeline.index else 0

    return {
        "ollama_running": ollama_ok,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "installed_models": installed_models,
        "is_indexed": is_indexed,
        "total_vectors": total_vectors,
        "documents_dir": str(DOCUMENTS_DIR)
    }


@app.get("/api/documents")
async def list_documents():
    """
    Returns a list of all documents stored in the documents/ directory.
    """
    docs = []
    if DOCUMENTS_DIR.exists():
        for f in DOCUMENTS_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                size_kb = round(f.stat().st_size / 1024, 2)
                docs.append({
                    "filename": f.name,
                    "extension": f.suffix.lower(),
                    "size_kb": size_kb
                })
    return {"documents": docs}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF or text document to documents/ and automatically re-indexes.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = DOCUMENTS_DIR / file.filename

    # Save uploaded file
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Trigger automatic ingestion
    stats = run_ingestion_internal()

    return {
        "message": f"Successfully uploaded and indexed '{file.filename}'",
        "filename": file.filename,
        "stats": stats
    }


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """
    Deletes a document from documents/ and re-indexes.
    """
    file_path = DOCUMENTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    file_path.unlink()
    stats = run_ingestion_internal()
    return {"message": f"Deleted '{filename}'", "stats": stats}


@app.post("/api/reindex")
async def trigger_reindex():
    """
    Triggers a manual re-indexing of all files in documents/.
    """
    stats = run_ingestion_internal()
    return {"message": "Re-indexing complete", "stats": stats}


@app.post("/api/ask")
async def ask_question(request: AskRequest):
    """
    Runs the RAG query pipeline for a user question.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = get_pipeline()
    if not pipeline.is_indexed():
        raise HTTPException(
            status_code=400,
            detail="No document index found. Please upload a PDF or run ingestion first."
        )

    start_time = time.time()
    try:
        # Override top_k if specified
        if request.top_k and request.top_k > 0:
            pipeline.top_k = request.top_k

        result = pipeline.query(question)
        duration_ms = int((time.time() - start_time) * 1000)

        # Prepare serializable debug info
        q_vec = result["debug_info"].get("query_vector")
        vec_preview = [round(float(v), 4) for v in q_vec[:5]] if q_vec is not None and len(q_vec) > 0 else []

        return {
            "question": result["question"],
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
            "context": result["context"],
            "prompt": result["prompt"],
            "duration_ms": duration_ms,
            "debug": {
                "embedding_model": pipeline.embedding_model,
                "llm_model": pipeline.llm_model,
                "vector_dimension": len(q_vec) if q_vec is not None else 0,
                "vector_preview": vec_preview,
                "top_k": len(result["retrieved_chunks"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
