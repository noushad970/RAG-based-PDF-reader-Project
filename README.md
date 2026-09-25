# Simple Local RAG System for Learning & PDF Analyzer Studio

A transparent, lightweight, framework-free **Retrieval-Augmented Generation (RAG)** system and **Local Web UI** built in Python using **Ollama**, **FAISS**, **pypdf**, and **NumPy**.

Upload multi-page **PDF**, **TXT**, or **MD** documents, index them into a local FAISS vector store, and ask questions through either a **modern web interface** or the **terminal**.

---

## Table of Contents
1. [Web UI Interface](#web-ui-interface)
2. [What is RAG?](#what-is-rag)
3. [Indexing vs. Querying Pipelines](#indexing-vs-querying-pipelines)
4. [Architecture Overview](#architecture-overview)
5. [Deep Dive: The 12 Core Components](#deep-dive-the-12-core-components)
6. [Project Structure](#project-structure)
7. [Prerequisites & Model Setup](#prerequisites--model-setup)
8. [Quickstart Commands](#quickstart-commands)
9. [Debug Mode Walkthrough](#debug-mode-walkthrough)
10. [How to Learn This Project (Code Walkthrough)](#how-to-learn-this-project)

---

## Web UI Interface

You can run the interactive web application locally on your machine:

```bash
python main.py web
# or
python app.py
```
Open **`http://127.0.0.1:8000`** in your browser.

### Key Web Features:
- **Drag & Drop PDF Upload**: Handles large and multi-page PDFs with live chunking & embedding progress.
- **Page-Level Grounded Citations**: Shows the exact source file, page number, chunk ID, and Cosine Similarity match percentage.
- **Interactive Retrieval Tuning**: Adjust the Top-K retrieval slider in real time.
- **Inspect RAG Telemetry**: Expandable pipeline inspector showing the 768-dim query vector preview, retrieved chunks, and the full prompt sent to Ollama.
- **100% Private & Offline**: All computations run locally on your machine via FAISS and Ollama.


---

## What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**.

Large Language Models (LLMs) have broad general knowledge from their training data, but they:
1. **Hallucinate**: They can invent plausible-sounding facts when they do not know the answer.
2. **Lack Private Knowledge**: They do not know your private notes, internal company wikis, or recent university policies.
3. **Have Context Limits**: You cannot dump hundreds of PDF books into a single prompt without hitting performance, token, or cost limits.

**RAG solves this in three simple steps:**
1. **Retrieve**: When a user asks a question, search your local document collection for the 2–3 most relevant paragraphs.
2. **Augment**: Paste those retrieved paragraphs into a prompt along with the user's question.
3. **Generate**: Ask the local LLM to answer the question using **only** the provided context.

---

## Indexing vs. Querying Pipelines

A RAG system consists of two separate workflows:

### 1. The Indexing Pipeline (Preparation Phase)
Run once when you add or change documents (`python main.py ingest`):

```text
documents/ (.txt, .md)
       │
       ▼
1. Document Loader (Extract raw text)
       │
       ▼
2. Chunker (Split into overlapping chunks)
       │
       ▼
3. Embedding Model (Convert text chunks into 768-dim vectors)
       │
       ▼
4. FAISS Vector Store + metadata.json (Save vectors and text mapping to disk)
```

### 2. The Querying Pipeline (Inference Phase)
Run whenever you ask a question (`python main.py chat` or `python main.py ask`):

```text
User Question ("What is the minimum attendance?")
       │
       ▼
1. Embed Question (Using the EXACT same embedding model)
       │
       ▼
2. FAISS Similarity Search (Cosine similarity against indexed vectors)
       │
       ▼
3. Top-K Chunks (Retrieve the best matching paragraphs)
       │
       ▼
4. Context Construction (Format chunks into structured text)
       │
       ▼
5. LLM Prompt Augmentation (System instruction + Context + Question)
       │
       ▼
6. Local Ollama LLM (Synthesizes grounded answer)
       │
       ▼
Final Answer + Source Citations
```

---

## Architecture Overview

```text
Documents ──► Text Extraction ──► Chunking ──► Embedding Model ──► FAISS Index & Metadata
                                                                           │
                                                                           ▼
User Question ────────► Question Embedding (Same Model) ────────► Similarity Search
                                                                           │
                                                                           ▼
Answer + Sources ◄───── Local Ollama LLM ◄───── Augmented Prompt ◄───── Top-K Chunks
```

---

## Deep Dive: The 12 Core Components

| Component | What It Is | Why It Is Important |
| :--- | :--- | :--- |
| **1. Documents** | Raw `.txt` and `.md` files stored in `documents/`. | Your knowledge base containing the ground-truth facts. |
| **2. Text Extraction** | Reading plain text from disk using standard UTF-8 encoding. | Converts raw files into clean Python strings. |
| **3. Chunking** | Splitting long text into smaller segments (e.g., 500 chars) with overlap (100 chars). | Keeps context focused so embeddings are sharp and don't split key sentences at edges. |
| **4. Embeddings** | Numerical vector representations generated by a neural model (`nomic-embed-text`). | Converts qualitative human language into geometric vectors. |
| **5. Vectors** | A list of floating-point numbers (e.g., 768 dimensions for nomic). | Represents semantic meaning in high-dimensional space. |
| **6. FAISS** | Facebook AI Similarity Search library. | Highly optimized C++ / Python vector index for fast nearest-neighbor lookups. |
| **7. Similarity Search** | Computing Cosine Similarity (dot product of L2-normalized vectors). | Mathematically finds chunks pointing in the closest direction to the query. |
| **8. Top-K Retrieval** | Selecting the top $K$ (default: 3) highest-scoring chunks. | Limits LLM input to only the most relevant passages, eliminating noise. |
| **9. Context** | The structured string assembled from retrieved chunks and file citations. | The factual evidence passed to the LLM. |
| **10. LLM** | Local language model (e.g. `qwen2.5-coder:latest` or `qwen2.5:3b`). | The reasoning engine that reads the context and writes the answer. |
| **11. Generation** | Producing the final response adhering strictly to context. | Formulates human-readable sentences without hallucinating. |
| **12. Sources** | Attribution metadata showing filename, chunk ID, and similarity score. | Transparency and auditability: you can verify where the answer came from. |

---

## Project Structure

```text
.
├── documents/                  # Place your .txt and .md files here
│   ├── attendance.txt          # Example attendance policy
│   └── examination.md          # Example exam rules
│
├── data/                       # Generated during ingestion (ignored by git)
│   ├── faiss.index             # FAISS vector index binary
│   └── metadata.json           # Chunk text and source document mapping
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration (models, paths, chunk sizes)
│   ├── document_loader.py      # Reads .txt and .md files
│   ├── chunker.py              # Splits text into overlapping chunks
│   ├── embeddings.py           # Calls Ollama to convert text to vectors
│   ├── vector_store.py         # FAISS index creation, search, and disk persistence
│   ├── retriever.py            # Embeds question & queries FAISS for Top-K chunks
│   ├── generator.py            # Constructs prompt and calls Ollama LLM
│   └── rag.py                  # Connects the end-to-end RAG pipeline
│
├── main.py                     # CLI entry point (setup, ingest, chat, ask, debug)
├── requirements.txt            # Minimal dependencies (faiss-cpu, numpy, ollama)
├── README.md                   # Complete learning documentation
└── .gitignore                  # Git ignore rules
```

---

## Prerequisites & Model Setup

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Download Ollama Models
Ensure the Ollama application is running, then pull the lightweight models:

```bash
# 1. Embedding model (274 MB)
ollama pull nomic-embed-text

# 2. Chat LLM model (fits easily in 8GB VRAM)
ollama pull qwen2.5-coder:latest
# or a smaller 3B / 1.5B model:
# ollama pull qwen2.5:3b
# ollama pull qwen2.5:1.5b
```

> [!NOTE]
> If you pull a different LLM model (e.g. `qwen2.5:3b`), simply update `LLM_MODEL = "qwen2.5:3b"` inside `src/config.py`.

---

## Quickstart Commands

### Step 1: Check Environment and Models
```bash
python main.py setup
```
**Output:**
```text
========================================
      OLLAMA & ENVIRONMENT SETUP
========================================
Ollama Service: RUNNING [OK]

Embedding Model: nomic-embed-text
Status: INSTALLED [OK]

Chat Model: qwen2.5-coder:latest
Status: INSTALLED [OK]

========================================
```

### Step 2: Ingest Documents and Build Vector Index
```bash
python main.py ingest
```
**Output:**
```text
========================================
             RAG INGESTION
========================================

Loading documents from: documents
Found 2 document(s) [OK]

Creating chunks (size=500, overlap=100)...
Created 2 chunk(s) [OK]

Generating embeddings using 'nomic-embed-text'...
Generating embedding [1/2] for chunk from 'attendance.txt'...
Generating embedding [2/2] for chunk from 'examination.md'...
Generated embeddings shape: (2, 768) [OK]

Building FAISS index (Inner Product / Cosine Similarity)...
FAISS index created with 2 vector(s) [OK]

Saving FAISS index and metadata...
Index saved to: data\faiss.index
Metadata saved to: data\metadata.json

========================================
Ingestion complete! You can now run:
  python main.py chat
========================================
```

### Step 3: Interactive Chat Mode
```bash
python main.py chat
```
**Example Session:**
```text
========================================
       SIMPLE LOCAL RAG CHAT
========================================
Type 'exit' or 'quit' to end the session.

Question: What is the minimum attendance requirement?

Searching documents and generating answer...

========================================
ANSWER
========================================
The minimum attendance requirement is 75%.

========================================
SOURCES
========================================
1. attendance.txt (Chunk: 0, Cosine Similarity: 0.7387)
2. examination.md (Chunk: 0, Cosine Similarity: 0.7324)

Question: What happens if attendance is below 75%?

Searching documents and generating answer...

========================================
ANSWER
========================================
Students with attendance below 75% may not be eligible to sit for the final examination.

========================================
SOURCES
========================================
1. attendance.txt (Chunk: 0, Cosine Similarity: 0.7207)
2. examination.md (Chunk: 0, Cosine Similarity: 0.6132)

Question: What is the university's hostel fee?

Searching documents and generating answer...

========================================
ANSWER
========================================
I don't know based on the provided documents.

========================================
SOURCES
========================================
1. attendance.txt (Chunk: 0, Cosine Similarity: 0.5812)
2. examination.md (Chunk: 0, Cosine Similarity: 0.5554)

Question: exit

Goodbye!
```

### Step 4: Ask a Single Question Directly
```bash
python main.py ask "What is the minimum attendance requirement?"
```

---

## Debug Mode Walkthrough

Run:
```bash
python main.py debug "What happens if attendance is below 75%?"
```

This reveals every internal step:
1. **User Question**: Raw query string.
2. **Question Embedding**: Model name, vector dimension (768), and a 5-element float preview.
3. **Retrieval**: Top-K retrieved chunks with exact Cosine Similarity scores.
4. **Retrieved Context**: The exact formatted text snippet passed to the prompt.
5. **Full LLM Prompt**: The combined system instructions + grounded context + user question.
6. **Final Answer & Sources**: The model's grounded synthesis and file citations.

---

## How to Learn This Project

Read the source files in this exact order to understand the full RAG pipeline:

```text
1. src/config.py
2. src/document_loader.py
3. src/chunker.py
4. src/embeddings.py
5. src/vector_store.py
6. src/retriever.py
7. src/generator.py
8. src/rag.py
9. main.py
```

### 1. [`src/config.py`](file:///c:/Users/abnou/OneDrive/Desktop/PDF%20Analyzer%20using%20RAG/src/config.py)
* **What it does**: Centralizes all model names, file paths, chunk sizes, and retrieval limits.
* **Why it exists**: Prevents magic numbers and hardcoded model strings scattered across files.
* **Input**: None (constants).
* **Output**: Configuration constants imported by other modules.
* **Connects to**: Every other module in `src/`.

### 2. [`src/document_loader.py`](file:///c:/Users/abnou/OneDrive/Desktop/PDF%20Analyzer%20using%20RAG/src/document_loader.py)
* **What it does**: Scans the `documents/` directory and reads supported `.txt` and `.md` files.
* **Why it exists**: Extracts plain text strings from files on disk.
* **Input**: Directory path (`documents/`).
* **Output**: `List[{"filename": "attendance.txt", "text": "..."}]`.
* **Connects to**: Passes documents to `chunker.py`.

### 3. [`src/chunker.py`](file:///c:/Users/abnou/OneDrive/Desktop/PDF%20Analyzer%20using%20RAG/src/chunker.py)
* **What it does**: Slices document text into fixed-size character windows with overlapping margins.
* **Why it exists**: LLMs and embedding models operate best on small, focused passages rather than full documents. Overlap prevents splitting key concepts across chunk boundaries.
* **Input**: Document dictionaries from `document_loader.py`.
* **Output**: `List[{"text": "...", "source": "attendance.txt", "chunk_id": 0}]`.
* **Connects to**: Passes chunks to `embeddings.py` and `vector_store.py`.

### 4. [`src/embeddings.py`](file:///c:/Users/abnou/OneDrive/Desktop/PDF%20Analyzer%20using%20RAG/src/embeddings.py)
* **What it does**: Calls Ollama's embeddings API (`nomic-embed-text`) to convert text into NumPy float32 arrays.
* **Why it exists**: Neural embeddings convert semantic meaning into geometry so similar texts point in similar directions.
* **Input**: Text string or list of chunk dictionaries.
* **Output**: 1D array (for single query) or 2D NumPy array of shape `(N, 768)` (for chunks).
* **Connects to**: Vectors are passed to `vector_store.py` (indexing) and `retriever.py` (querying).

### 5. [`src/vector_store.py`](file:///c:/Users/abnou/OneDrive/Desktop/PDF%20Analyzer%20using%20RAG/src/vector_store.py)
* **What it does**: L2-normalizes vectors, creates a FAISS `IndexFlatIP` index, and persists `faiss.index` + `metadata.json`.
* **Why it exists**: FAISS enables lightning-fast cosine similarity lookups. Normalizing vectors allows dot-product (`IndexFlatIP`) to equal exact cosine similarity.
* **Input**: 2D embedding matrix and chunk metadata.
* **Output**: Saved files on disk and FAISS search results `(similarity_scores, chunk_indices)`.
* **Connects to**: Used by `ingest` to save the index and `retriever.py` to load and search it.

### 6. [`src/retriever.py`](file:///c:/Users/abnou/OneDrive/Desktop/PDF%20Analyzer%20using%20RAG/src/retriever.py)
* **What it does**: Embeds the user's question using the **same** model and queries FAISS for the Top-K most similar chunks.
* **Why it exists**: This is the **Retrieval** step of RAG. Both questions and chunks must live in the same embedding space.
* **Input**: User query string + loaded FAISS index + metadata list.
* **Output**: Top matching chunks with similarity scores and debug information.
* **Connects to**: Passes retrieved chunks to `generator.py`.

### 7. [`src/generator.py`](file:///c:/Users/abnou/OneDrive/Desktop/PDF%20Analyzer%20using%20RAG/src/generator.py)
* **What it does**: Formats retrieved chunks into a structured context string and calls the local Ollama LLM (`qwen2.5-coder:latest`).
* **Why it exists**: This is the **Augmented Generation** step of RAG. Grounding the LLM with verified context prevents hallucinations.
* **Input**: User question + retrieved chunks.
* **Output**: Final synthesized answer string.
* **Connects to**: Used by `rag.py` to produce the final output.

### 8. [`src/rag.py`](file:///c:/Users/abnou/OneDrive/Desktop/PDF%20Analyzer%20using%20RAG/src/rag.py)
* **What it does**: Houses the `RAGPipeline` class that orchestrates the entire workflow step-by-step without framework black-boxes.
* **Why it exists**: Provides a clean, reusable API that links retrieval, context formatting, and generation.
* **Input**: User query.
* **Output**: Complete result dictionary containing answer, sources, prompt, and debug telemetry.
* **Connects to**: Called by the CLI commands in `main.py`.

### 9. [`main.py`](file:///c:/Users/abnou/OneDrive/Desktop/PDF%20Analyzer%20using%20RAG/main.py)
* **What it does**: Command-line interface with subcommands: `setup`, `ingest`, `chat`, `ask`, and `debug`.
* **Why it exists**: The user-facing terminal tool to operate the application.
* **Input**: CLI arguments (`sys.argv`).
* **Output**: Formatted terminal output and interactive REPL.
