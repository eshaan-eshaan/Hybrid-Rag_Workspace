---
title: Hybrid Rag Workspace
emoji: 💬
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
python_version: 3.10
app_file: app.py
pinned: false
---

# Quick Hopper - Hybrid RAG System

## 1. Project Description
Quick Hopper is an advanced Retrieval-Augmented Generation (RAG) system with a Gradio web interface. It goes beyond standard RAG pipelines by implementing **Hybrid Retrieval** (combining Dense semantic search with Sparse keyword search) and a **Vector-based Conversational Memory** system. This allows the application to accurately answer user queries based on uploaded PDF documents or a default built-in knowledge base, while maintaining context across long conversations without exceeding LLM context limits.

## 2. What Problem It Solves
1. **Missed Keyword / Semantic Matches**: Dense retrieval (vectors) is great for semantic meaning but often fails at exact keyword/acronym matching. Sparse retrieval (BM25) is great for exact keywords but fails at semantic understanding. This system solves this by using a **Hybrid Search** approach combined with **Reciprocal Rank Fusion (RRF)** to get the best of both worlds.
2. **Context Window Exhaustion**: In long conversations, appending the entire chat history to the LLM prompt causes token limits to be exceeded and increases API costs. This system solves this by storing conversational memory as vectors in a FAISS database, retrieving only the most relevant historical context for the current query.
3. **Handling Missing Documents**: If a user doesn't upload a document, standard RAGs break. This system falls back gracefully to a built-in `DEFAULT_CHUNKS` knowledge base.

## 3. Tech Stack
- **Frontend / UI:** [Gradio](https://gradio.app/)
- **LLM Provider:** [Groq API](https://groq.com/) (Llama 3 / Mixtral models for ultra-fast inference)
- **Embeddings:** `SentenceTransformer` (`all-MiniLM-L6-v2`)
- **Dense Vector Database:** [FAISS](https://faiss.ai/) (Facebook AI Similarity Search)
- **Sparse Retrieval:** `rank_bm25` (BM25Okapi)
- **Document Parsing:** `pdfplumber`
- **Text Splitting:** `langchain_text_splitters.RecursiveCharacterTextSplitter`

## 4. Product Requirements Document (PRD)
**Goal:** Build a high-performance, cost-effective, and context-aware chat application that answers questions based on uploaded documents.
**Core Features:**
- Upload PDF documents and extract text accurately.
- Automatically chunk text into manageable sizes (1000 characters, 200 overlap).
- Hybrid search architecture (FAISS + BM25).
- RRF Ranking to merge dense and sparse search scores.
- Dynamic conversational memory stored via vectors.
- Responsive Chat UI with visual indicators for retrieved chunks and memory chunks.

## 5. Architecture Overview
The system architecture separates document processing, search logic, memory management, and response generation into modular components. 
It relies on two parallel FAISS indexes:
1. **Document Index (Volatile):** Stores chunks of the currently uploaded PDF.
2. **Memory Index (Persistent):** Stores past user queries and assistant responses on disk (`memory_index.faiss` and `memory_store.json`).

## 6. How it works - The system is built in multiple layers
1. **Ingestion Layer:** Takes PDF files, reads text via `pdfplumber`, splits it into chunks using LangChain's RecursiveCharacterTextSplitter.
2. **Embedding & Indexing Layer:** Uses `all-MiniLM-L6-v2` to convert text chunks into dense vectors. Simultaneously tokenizes the text for the BM25 sparse index.
3. **Retrieval Layer (Hybrid):** When a user asks a question, the query is embedded for Dense Search (Cosine Similarity) and tokenized for Sparse Search (BM25). Both return a list of top documents.
4. **Ranking Layer:** Applies Reciprocal Rank Fusion (RRF) to seamlessly merge and re-rank the outputs from FAISS and BM25 into a definitive top-K list.
5. **Memory Layer:** The query is also embedded and searched against the persistent FAISS memory store to fetch previous relevant conversation turns, preventing context bloat.
6. **Generation Layer:** The Groq LLM receives a prompt containing the System Instruction, the Retrieved RAG Context, the Retrieved Memory Context, and the Current Query, generating the final output.

## 7. Complete Pipeline
1. **Upload:** User uploads `document.pdf`.
2. **Process:** Text extracted $\rightarrow$ Split into chunks $\rightarrow$ Chunks embedded $\rightarrow$ Added to FAISS & BM25.
3. **Query:** User types "What are the types of RAG?".
4. **Search (Document):** Query embedded $\rightarrow$ FAISS returns top matches $\rightarrow$ Query tokenized $\rightarrow$ BM25 returns top matches.
5. **Rank:** RRF combines FAISS and BM25 lists.
6. **Search (Memory):** Query embedded $\rightarrow$ FAISS Memory Store returns relevant past chats.
7. **Generate:** Groq LLM parses Context + Memory + Query $\rightarrow$ Generates response.
8. **Save:** User Query and LLM Response are embedded and added to the FAISS Memory Store.

## 8. Workflow
1. User starts the Gradio web server via `python app.py`.
2. Groq API key is loaded from `.env`.
3. The interface opens at `http://127.0.0.1:7860`.
4. (Optional) User uploads a PDF. The UI confirms chunks are processed.
5. User enters a query in the chat box.
6. System retrieves relevant text, retrieves memory, and streams back the LLM response.
7. Accordions in the UI show exactly which chunks were retrieved for transparency.

## 9. DFD Diagram (Data Flow Diagram)
```mermaid
graph TD
    User([User]) -->|Uploads PDF| UI[Gradio UI]
    UI -->|Text Extraction| PDF[PDF Plumber]
    PDF -->|Raw Text| Splitter[Text Splitter]
    Splitter -->|Text Chunks| Encoder[Sentence Transformer]
    Splitter -->|Text Chunks| BM25[BM25 Indexer]
    Encoder -->|Vector Embeddings| FAISS_Doc[(FAISS Document DB)]
    
    User -->|Sends Query| QueryProcessor[Query Processor]
    QueryProcessor -->|Embeds Query| Encoder
    QueryProcessor -->|Tokenizes Query| BM25
    
    Encoder -->|Query Vector| FAISS_Doc
    Encoder -->|Query Vector| FAISS_Mem[(FAISS Memory DB)]
    
    FAISS_Doc -->|Dense Results| RRF[RRF Ranker]
    BM25 -->|Sparse Results| RRF
    
    FAISS_Mem -->|Relevant Past Context| PromptBuilder[Prompt Builder]
    RRF -->|Ranked Document Context| PromptBuilder
    QueryProcessor -->|Raw Query| PromptBuilder
    
    PromptBuilder -->|Final Prompt| LLM[Groq LLM]
    LLM -->|Generated Answer| UI
    LLM -->|Answer| FAISS_Mem
```

## 10. What Each Component Does
- **`app.py`**: The main entry point. Houses the Gradio UI setup, the `FaissMemoryStore` class, and the orchestration functions (`process_pdf`, `retrieve_hybrid_internal`, `generate_response`).
- **`hybriidrag.py`**: An alternative/legacy implementation script containing sandbox functions for the RAG pipeline. It mirrors much of the logic in `app.py` but is meant for backend testing and prototyping without the Gradio UI.
- **`FaissMemoryStore` Class**: A custom class that manages reading, writing, and querying the persistent FAISS index (`memory_index.faiss`) and the JSON metadata file (`memory_store.json`) for chat history.
- **`.env`**: Stores the `GROQ_API_KEY` locally and securely so it doesn't need to be hardcoded or pasted into the UI.
- **`memory_index.faiss` & `memory_store.json`**: Persistent storage files that remember conversation history across server restarts.

## 11. What we have built so far
- Set up a clean, robust, and scalable **Gradio UI** avoiding frontend complexities.
- Integrated **Groq** for high-speed inference.
- Implemented **Hybrid RAG** logic using FAISS (dense) and BM25 (sparse).
- Integrated **Reciprocal Rank Fusion (RRF)** for optimal retrieval accuracy.
- Added a self-managing **Vector Memory Store** that persists chat history locally.
- Extracted hardcoded API keys into a secure `.env` file loaded via backend logic.
- Optimized text splitting (`chunk_size=1000`, `chunk_overlap=200`, `top_k=8`) to ensure lists and long contexts are retrieved completely without truncation errors.
- Built a fallback mechanism using `DEFAULT_CHUNKS` when no PDF is uploaded.

## 12. Project File Structure
```text
quick-hopper/
│
├── .env                     # Contains GROQ_API_KEY
├── .env.example             # Template for environment variables
├── .git/                    # Git repository data
├── app.py                   # Main Application and Gradio UI
├── hybriidrag.py            # Backend Hybrid RAG logic and testing sandbox
├── HybriidRag.ipynb         # Jupyter Notebook used for initial experiments/prototyping
├── memory_index.faiss       # Persistent FAISS database for conversational memory
├── memory_store.json        # Persistent JSON metadata for conversational memory
└── README.md                # This comprehensive project documentation
```
