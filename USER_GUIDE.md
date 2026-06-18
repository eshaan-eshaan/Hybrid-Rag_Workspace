# User & Setup Guide

This guide details how to install dependencies, configure environment variables, launch the application, and navigate the Gradio UI.

---

## 1. Prerequisites & Installation

Ensure you have Python 3.9+ installed.

### Install Required Dependencies:
```bash
pip install -r requirements.txt
```
*(If a `requirements.txt` is not present, install them manually)*:
```bash
pip install gradio pdfplumber faiss-cpu numpy sentence-transformers groq rank-bm25 langchain-text-splitters
```

---

## 2. Configuration

1. Locate or create a file named `.env` in the root folder of the project.
2. Add your **Groq API Key**:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

*Note: The application will load this key automatically on startup. Do not share or commit this key to version control.*

---

## 3. Running the Application

Launch the Gradio web server by executing:
```bash
python app.py
```

Upon starting:
1. It downloads or loads the cached local SentenceTransformer model (`all-MiniLM-L6-v2`).
2. It trains the local Offline Feedback Analyzer on `pharma_query_bank_300.csv`.
3. It binds to port `8080`.
4. It opens in your browser at: **`http://127.0.0.1:8080`**

---

## 4. UI Walkthrough & Operation

### A. RAG Chat Portal
- **Engine Settings (Left Sidebar)**:
  - **PDF Document Source(s)**: Drag & drop one or multiple PDF files. Multiple files are indexed incrementally without resetting previously loaded files.
  - **Reset Memory & Index**: Wipes the current PDF index and memory index to start a fresh chat session.
  - **Log Details / Ingestion Status**: Shows real-time statistics (number of pages, characters, vector chunks) and success/error logs.
- **Q&A Portal (Main Panel)**:
  - Ask questions about the uploaded documents in the query textbox and press Enter or click **Search & Generate**.
  - Rate responses immediately using the **👍 (Thumbs Up)** or **👎 (Thumbs Down)** buttons at the bottom right of each chatbot bubble.

### B. Transparency Accordions
- **RAG Source Documents**: Expand this to see the exact text chunks retrieved from your PDFs, along with their source filename, page number, and RRF score.
- **FAISS Memory**: Expand this to view the historical QA turns retrieved from conversational memory.
- **Saved RAG Interaction Logs**: Displays structured interaction logs containing timestamps, queries, answers, sources, and votes. Includes a **Structured Table** view and a **Raw JSON Logs** viewer.

### C. Negative Feedback Analyzer Accordion
Located at the bottom of the chat portal:
1. Click **🚀 Run Negative Feedback Analysis** to parse log files.
2. View the **Document-wise Negative Feedback Summary** table to see which uploaded documents are producing the most incorrect answers.
3. Review the **Per-Interaction Root Cause Analysis** cards, diagnosing *why* each query failed and recommending specific additions to fix your documents.
4. Click **📥 Export Report (.md)** to save the full feedback report to disk as `negative_feedback_report.md`.
