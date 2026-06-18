# Troubleshooting Guide

This guide covers common errors, warning conditions, and performance bottlenecks in the **Quick Hopper Hybrid RAG System**.

---

## 1. API Key Errors (Groq API)

### Symptoms:
- Chatbot returns generic error messages like `"Authentication Error: Missing GROQ_API_KEY"`.
- Server prints warnings on boot indicating no API key was found.

### Solutions:
1. Ensure your `.env` file is in the root directory (not inside a subdirectory) and format it exactly as:
   ```env
   GROQ_API_KEY=gsk_your_key
   ```
2. Check for wrapping quotes. The loader handles single/double quotes, but it is best to provide the raw token.
3. If running in containerized or cloud environments, expose `GROQ_API_KEY` as an environment variable directly.

---

## 2. Ingestion Failures & Scanned PDFs

### Symptoms:
- Uploading a PDF outputs `"Errors occurred during ingestion: empty.pdf"`.
- Log details print: `"No text could be extracted or chunked from the PDF. Is it a scanned PDF with no OCR?"`.

### Root Cause:
The PDF lacks an active text stream (e.g., scanned images of documents). The extraction engines (`fitz`/`pypdf`/`pdfplumber`) cannot read images.

### Solutions:
1. Run OCR (Optical Character Recognition) on the PDF before uploading (using tools like Adobe Acrobat or `tesseract`).
2. Verify the document is not password-protected. Password-protected PDFs will fail parsing.

---

## 3. Slow Startup (SentenceTransformer Loading)

### Symptoms:
- Running `python app.py` hangs on `"Initializing SentenceTransformer model (all-MiniLM-L6-v2)..."`.
- The terminal displays HF download bars or warnings about HF_TOKEN.

### Solutions:
1. On the first launch, the system downloads the `all-MiniLM-L6-v2` weights from Hugging Face Hub (approx. 90MB). Ensure you have an active internet connection.
2. Subsequent startups will load directly from the local cache and complete in under 2 seconds.
3. If you are offline, ensure the weights are pre-downloaded to your Hugging Face cache directory (typically `~/.cache/huggingface/hub`).

---

## 4. Log Directory Write Permissions

### Symptoms:
- Clicks on 👍/👎 do not update the structured logs.
- System crashes or outputs `PermissionError` when generating or exporting reports.

### Solutions:
1. By default, logs write to `C:\Users\eshaa\Documents\antigravity\quick-hopper\rag_logs.jsonl`.
2. Ensure the user running `python app.py` has full write permissions to this path.
3. If run in a multi-user environment, check that no other process is holding a lock on `rag_logs.jsonl` or `negative_feedback_report.md`.
