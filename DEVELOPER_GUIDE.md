# Developer Guide

This document is for developers who want to understand the codebase, extend functionality, or run tests in the **Quick Hopper Hybrid RAG System**.

---

## 1. Project Directory Structure

```text
quick-hopper/
├── .env                         # Local environment configuration (e.g. GROQ_API_KEY)
├── .env.example                 # Template for environment configuration
├── app.py                       # Main application entry point & Gradio UI
├── hybriidrag.py                # Core Hybrid RAG Engine and Memory classes
├── feedback_model.py            # Local offline negative feedback analyzer
├── pharma_query_bank_300.csv    # Training dataset for feedback analyzer (580 rows)
├── verify_enhancements.py       # E2E test suite using unittest
├── TEST_INFRA.md                # Description of testing methodology & Feature list
├── ARCHITECTURE.md              # Technical design and architecture diagrams
├── USER_GUIDE.md                # Setup, configuration, and operation instructions
├── TROUBLESHOOTING.md           # Troubleshooting common errors and fallbacks
├── memory_index.faiss           # Generated conversational memory vector index
├── memory_store.json            # Generated conversational memory JSON store
└── rag_logs.jsonl               # Generated structured logs (query/response/feedback)
```

---

## 2. Key Code Modules & API Reference

### A. `hybriidrag.py`
Contains the core RAG retrieval logic and indices management.

#### 1. `HybridRagEngine`
- `ingest_file(file_path)`: Extracted text from PDF incrementally, splits it into chunks, encodes the new text chunks into embeddings using the SentenceTransformer model, and appends them to the memory store. Rebuilds the FAISS and BM25 indices. Returns `True` if successfully processed, `False` if already processed.
- `retrieve_hybrid(query, top_k=8, k_rrf=60)`: Computes dense distances using FAISS and sparse rankings using BM25. Combines them using Reciprocal Rank Fusion (RRF). Returns a list of retrieved chunks with metadata and RRF scores.
- `reset()`: Cleans up processed files, chunks, and clears indices.

#### 2. `FaissMemoryStore`
- `add_memory(user_query, assistant_answer)`: Formats and encodes the QA interaction, registers it in the memory FAISS index (`memory_index.faiss`), and appends the raw text metadata into `memory_store.json`.
- `search_memory(query, top_k=2)`: Queries the memory FAISS index for relevant past conversation turns, returning them to inject as context.

---

### B. `feedback_model.py`
Implements the zero-cost local feedback classifier.

#### 1. `FeedbackAnalyzerModel`
- `train(csv_path)`: Loads training queries and weak spots. Computes embeddings for the queries. Populates a local FAISS index (`faiss.IndexFlatL2`) representing known weak spot centers. Returns statistics.
- `analyze(query, response=None, sources=None)`: First does a semantic similarity search in the FAISS index. If confidence exceeds the threshold, returns the matching weak spot's description and suggestions. If confidence is low, falls back to regex-based local heuristics (`_heuristic_negative(query)`).
- `_heuristic_negative(query)`: Rule-based heuristics matching key pharma phrases like "side effect", "adverse reaction", "versus", "better than", etc., to safety or comparison issues.

---

### C. `app.py`
Wires together the Gradio frontend and orchestrates RAG generation.

- `generate_response(message, history)`: Triggered by submitting a query. Searches conversational memory, retrieves document context (or fallbacks to defaults), builds the LLM prompt, calls the Groq streaming chat API, writes log entries to disk, and updates the chat.
- `handle_like(x: gr.LikeData)`: Listens to 👍/👎 clicks in the Gradio chat component, calls `update_log_feedback()` to record the interaction's rating on disk.
- `run_feedback_analysis()`: Parses `rag_logs.jsonl` to count 👎-rated interactions per document, passes the bad queries through `feedback_analyzer.analyze()` to diagnose root causes offline, and updates the analysis summary table.
- `export_negative_report()`: Exports the diagnosed negative feedback report into a markdown file (`negative_feedback_report.md`).

---

## 3. How to Run Tests

The E2E test suite validates all required features (Ingestion, Attribution, Feedback Buttons, Log Storage, and Corner Cases).

### Run the verification test suite:
```bash
python verify_enhancements.py
```

All tests mock out the PDF extraction (`pdfplumber.open`) and file writing/logging, allowing the suite to run locally without internet or filesystem side-effects, while verifying correct internal logic.
