import os
import re
import json
import time
import uuid
from datetime import datetime, timezone
import gradio as gr
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Import RAG classes and default chunks from hybriidrag
from hybriidrag import FaissMemoryStore, HybridRagEngine, DEFAULT_CHUNKS
from feedback_model import FeedbackAnalyzerModel

print("Initializing SentenceTransformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")

# Initialize the CSV-trained local feedback analyzer (pharma dataset)
CSV_DATASET_PATH = os.path.join(os.path.dirname(__file__), "pharma_query_bank_300.csv")
feedback_analyzer = FeedbackAnalyzerModel(embed_model=model)
if os.path.exists(CSV_DATASET_PATH):
    print(f"Training local feedback analyzer on {CSV_DATASET_PATH}...")
    train_stats = feedback_analyzer.train(CSV_DATASET_PATH)
    print(f"Feedback analyzer trained: {train_stats}")
else:
    print(f"WARNING: CSV dataset not found at {CSV_DATASET_PATH}. Local analyzer will not be available.")

def load_env_api_key():
    # Try environment variable
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    # Try reading .env file in the workspace
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("GROQ_API_KEY="):
                    val = line.strip().split("=", 1)[1]
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    return val.strip()
    return ""

CONTEXT_FILE_PATH = os.path.join(os.path.dirname(__file__), "user_context.json")

def load_saved_contexts():
    default_system = "You answer only from retrieved document context. Use memory and history only to understand the question better."
    if os.path.exists(CONTEXT_FILE_PATH):
        try:
            with open(CONTEXT_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("personal_context", ""), data.get("system_context", default_system)
        except Exception:
            pass
    return "", default_system

def save_contexts(personal_context="", system_context=""):
    # Defensive type checking in case of out-of-sync Gradio client events
    if not isinstance(personal_context, str):
        personal_context = str(personal_context) if personal_context is not None else ""
    if not isinstance(system_context, str):
        system_context = str(system_context) if system_context is not None else ""
    try:
        with open(CONTEXT_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "personal_context": personal_context,
                "system_context": system_context
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving contexts: {e}")

# Global memory store
memory_store = FaissMemoryStore(embed_model=model)

LOG_FILE_PATH = r"C:\Users\eshaa\Documents\antigravity\quick-hopper\rag_logs.jsonl"

# Global app state
class AppState:
    def __init__(self):
        self.engine = HybridRagEngine(model)
        self.interaction_ids = []
        self.stats = {
            "page_count": 0,
            "word_count": 0,
            "chunk_count": 0,
            "status": "No documents uploaded. System initialized with default knowledge base."
        }
        memory_store.clear()

    def reset(self):
        self.engine.reset()
        self.interaction_ids = []
        self.stats = {
            "page_count": 0,
            "word_count": 0,
            "chunk_count": 0,
            "status": "No documents uploaded. System initialized with default knowledge base."
        }
        memory_store.clear()

state = AppState()

# Append log entry to local JSONL
def append_log_entry(interaction_id, query, response, sources):
    entry = {
        "interaction_id": interaction_id,
        "query": query,
        "response": response,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sources": sources,
        "feedback": "None"
    }
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error appending log entry: {e}")

# Update feedback in local JSONL
def update_log_feedback(uuid_str, feedback_value):
    if not os.path.exists(LOG_FILE_PATH):
        return
        
    try:
        updated_lines = []
        found = False
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("interaction_id") == uuid_str:
                    entry["feedback"] = feedback_value
                    found = True
                updated_lines.append(entry)
                
        if found:
            with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
                for entry in updated_lines:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error updating feedback in log: {e}")

# Format logged interactions as markdown table
def get_formatted_logs():
    if not os.path.exists(LOG_FILE_PATH):
        return "### Saved RAG Interaction Logs\n\n*No logs generated yet.*"
        
    try:
        entries = []
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Filter by interaction_ids of current session
                    if entry.get("interaction_id") in state.interaction_ids:
                        entries.append(entry)
        
        if not entries:
            return "### Saved RAG Interaction Logs\n\n*No logs generated in this session yet.*"
            
        md = "### Saved RAG Interaction Logs\n\n"
        md += "| Timestamp | Query | Response | Sources | Feedback |\n"
        md += "|---|---|---|---|---|\n"
        
        for entry in reversed(entries):
            ts = entry.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    ts = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            q = entry.get("query", "")
            r = entry.get("response", "")
            r_clean = r.split("**Sources:**")[0].strip()
            r_clean = r_clean.replace("\n", " ").replace("\r", "")
            if len(r_clean) > 80:
                r_clean = r_clean[:77] + "..."
            
            srcs = ", ".join(entry.get("sources", []))
            fb = entry.get("feedback", "None")
            if fb == "thumbs_up":
                fb_str = "Up"
            elif fb == "thumbs_down":
                fb_str = "Down"
            else:
                fb_str = "None"
                
            md += f"| {ts} | {q} | {r_clean} | {srcs} | {fb_str} |\n"
            
        return md
    except Exception as e:
        return f"Error loading logs: {str(e)}"

# Read raw JSON log file content
def get_raw_json_logs():
    if not os.path.exists(LOG_FILE_PATH):
        return "[]"
    try:
        objects = []
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Filter by interaction_ids of current session
                    if entry.get("interaction_id") in state.interaction_ids:
                        objects.append(entry)
        return json.dumps(objects, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Error reading JSON logs: {str(e)}"}, indent=2)

def refresh_logs_callback():
    md = get_formatted_logs()
    raw_json = get_raw_json_logs()
    return md, raw_json

# ─────────────────────────────────────────────────────────────────────────────
# NEGATIVE FEEDBACK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def load_negative_entries():
    """Load all thumbs_down log entries for the current session."""
    if not os.path.exists(LOG_FILE_PATH):
        return []
    entries = []
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if obj.get("feedback") == "thumbs_down" and obj.get("interaction_id") in state.interaction_ids:
                        entries.append(obj)
    except Exception as e:
        print(f"Error loading negative entries: {e}")
    return entries


def build_doc_summary_table(neg_entries):
    """Build a markdown table: per-document negative feedback count + queries."""
    if not neg_entries:
        return (
            "### Document-wise Negative Feedback Summary\n\n"
            "*No thumbs-down entries found in logs yet.*\n\n"
            "Give some Down ratings in the chat first, then click **Run Analysis**."
        )

    # Group by document sources
    doc_map = {}  # doc_name -> {count, queries}
    for entry in neg_entries:
        sources = entry.get("sources", [])
        query = entry.get("query", "")
        # A negative response may cite multiple docs — blame all of them
        if not sources:
            sources = ["Unknown / No Source"]
        for src in sources:
            # Normalize: strip page info for grouping
            doc_name = src.split(" (Page ")[0].strip()
            if doc_name not in doc_map:
                doc_map[doc_name] = {"count": 0, "queries": []}
            doc_map[doc_name]["count"] += 1
            if query not in doc_map[doc_name]["queries"]:
                doc_map[doc_name]["queries"].append(query)

    # Sort by negative count descending
    sorted_docs = sorted(doc_map.items(), key=lambda x: x[1]["count"], reverse=True)

    md = "### Document-wise Negative Feedback Summary\n\n"
    md += f"**Total Down Interactions Found**: `{len(neg_entries)}`\n\n"
    md += "| # | Document | Down Count | Failed Queries |\n"
    md += "|---|---|---|---|\n"
    for i, (doc, info) in enumerate(sorted_docs, 1):
        queries_preview = "; ".join(info["queries"][:3])
        if len(info["queries"]) > 3:
            queries_preview += f" *(+{len(info['queries'])-3} more)*"
        md += f"| {i} | **{doc}** | `{info['count']}` | {queries_preview} |\n"

    return md


def ai_root_cause_analysis(query, response, sources):
    """Analyze why a response got thumbs-down.
    Uses only the local CSV-trained pharma model (zero API cost)."""
    src_str = ", ".join(sources) if sources else "Unknown source"

    # Check if the query or sources are related to the legacy NCERT Chemistry solutions
    is_legacy_chemistry = (
        any("ncert" in s.lower() or "chem" in s.lower() for s in sources)
        or "ncert" in query.lower()
        or "chemistry" in query.lower()
        or "colligative" in query.lower()
        or "molarity" in query.lower()
        or "molality" in query.lower()
    )
    if is_legacy_chemistry:
        return (
            "**Negative feedback analysis disabled for legacy NCERT Chemistry content.**\n\n"
            "The active feedback analyzer has been retrained on the pharma query bank. "
            "Please upload a pharma document or ask pharma-related questions to analyze feedback."
        )

    # Use local CSV-trained model
    if feedback_analyzer.trained:
        try:
            result = feedback_analyzer.analyze(query, response)
            ws_type = result.get('weak_spot_type', 'none')
            from feedback_model import WEAK_SPOT_LABELS
            ws_label = WEAK_SPOT_LABELS.get(ws_type, ws_type)

            output = (
                f"**Root Cause**: {result.get('root_cause', 'N/A')}\n\n"
                f"**Weak Spot Type**: {ws_label}\n\n"
                f"**Error Type**: {result.get('error_type', 'N/A')}\n\n"
                f"**Missing Information**: {result.get('missing_info', 'N/A')}\n\n"
                f"**Document Improvement Suggestion**: {result.get('suggestion', 'N/A')}\n\n"
                f"**Confidence**: {int(result.get('confidence', 0) * 100)}%\n\n"
            )
            if result.get('company'):
                output += f"*Company*: {result['company']}"
                if result.get('product'):
                    output += f"  |  *Product*: {result['product']}"
                output += "\n\n"
            output += (
                f"*Cited Sources*: {src_str}\n\n"
                f"*Powered by local pharma-trained model (zero API cost)*"
            )
            return output
        except Exception as e:
            pass  # Fall through to basic heuristic

    # Basic heuristic fallback (model not trained)
    r_lower = response.lower()
    if "not available" in r_lower or "not in the" in r_lower or "cannot find" in r_lower:
        cause = "The answer was not found in the document context"
        suggestion = f"Add a dedicated section covering '{query}' with factual details."
    elif len(response.strip()) < 100:
        cause = "The response was too short / insufficient"
        suggestion = f"Expand the document with more detailed content related to '{query}'."
    else:
        cause = "Possible vagueness or inaccuracy in retrieved content"
        suggestion = f"Review and enrich the relevant sections with clearer information about '{query}'."

    return (
        f"**Root Cause**: {cause}\n\n"
        f"**Document Improvement Suggestion**: {suggestion}\n\n"
        f"*Cited Sources*: {src_str}\n\n"
        f"*Local analysis (train pharma CSV for better results)*"
    )


def run_feedback_analysis():
    """Main analysis runner called by the Gradio button."""
    neg_entries = load_negative_entries()


    # ── Summary table (always shown)
    summary_md = build_doc_summary_table(neg_entries)

    if not neg_entries:
        detail_md = "*No thumbs-down ratings given in this session yet.*"
        return summary_md, detail_md

    # Per-entry detailed analysis cards
    detail_md = "### Per-Interaction Root Cause Analysis\n\n"
    detail_md += (
        "> Each card below shows one Down interaction. "
        "The AI analyzes why the answer failed and what to add to the document.\n\n"
    )

    for i, entry in enumerate(neg_entries, 1):
        query = entry.get("query", "")
        response = entry.get("response", "")
        sources = entry.get("sources", [])
        ts = entry.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                pass

        # Get AI analysis for this entry
        analysis = ai_root_cause_analysis(query, response, sources)

        # Strip citations from response preview
        resp_preview = response.split("**Sources:**")[0].strip()
        if len(resp_preview) > 300:
            resp_preview = resp_preview[:297] + "..."

        detail_md += f"---\n\n"
        detail_md += f"#### Case #{i} — {ts}\n\n"
        detail_md += f"**User Query**: {query}\n\n"
        detail_md += f"**Sources Cited**: {', '.join(sources) if sources else 'None'}\n\n"
        detail_md += f"**AI Response** *(preview)*:\n> {resp_preview.replace(chr(10), ' ')}\n\n"
        detail_md += f"**Analysis**:\n\n{analysis}\n\n"

    return summary_md, detail_md


def export_negative_report():
    """Export a Markdown report of all negative feedback entries."""
    neg_entries = load_negative_entries()
    if not neg_entries:
        return "No thumbs-down entries to export."


    lines = ["# Negative Feedback Report\n", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"]
    lines.append(build_doc_summary_table(neg_entries))
    lines.append("\n\n---\n\n")

    for i, entry in enumerate(neg_entries, 1):
        query = entry.get("query", "")
        response = entry.get("response", "")
        sources = entry.get("sources", [])
        analysis = ai_root_cause_analysis(query, response, sources)
        lines.append(f"## Case #{i}\n")
        lines.append(f"**Query**: {query}\n")
        lines.append(f"**Sources**: {', '.join(sources)}\n")
        lines.append(f"**Analysis**:\n{analysis}\n\n")

    report_text = "\n".join(lines)

    # Save to file
    report_path = os.path.join(os.path.dirname(LOG_FILE_PATH), "negative_feedback_report.md")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return f"Report exported to `{report_path}`\n\n" + report_text
    except Exception as e:
        return f"Could not save file: {e}\n\n" + report_text

# Helper function to extract text and build chunks from multiple PDFs
def process_pdf(pdf_files):
    if not pdf_files:
        status_md = """### Ingestion Status
- **System Status**: Ready (Waiting for files)
- **Loaded Files**: None
- **Total Pages**: 0
- **Total Words**: 0
- **Processed Sections**: 0
"""
        return (
            status_md,
            "No documents uploaded. System initialized with default knowledge base."
        )
    
    if not isinstance(pdf_files, list):
        pdf_files = [pdf_files]
        
    success_count = 0
    skipped_count = 0
    errors = []
    
    for f in pdf_files:
        try:
            basename = os.path.basename(f.name)
            if basename in state.engine.processed_files:
                skipped_count += 1
                continue
            
            state.engine.ingest_file(f.name)
            success_count += 1
        except Exception as e:
            errors.append(f"{os.path.basename(f.name)}: {str(e)}")
            
    total_pages = sum(meta["pages"] for meta in state.engine.doc_metadata.values())
    total_words = sum(meta.get("words", 0) for meta in state.engine.doc_metadata.values())
    total_chunks = len(state.engine.chunks)
    
    if errors:
        error_msg = "; ".join(errors)
        status_text_val = f"Errors occurred during ingestion: {error_msg}"
    else:
        status_text_val = f"Successfully ingested files. {total_chunks} chunks indexed in FAISS."
        
    if total_chunks > 0:
        status_md = f"""### Ingestion Status
- **System Status**: Active (Documents Loaded)
- **Loaded Files**: {", ".join(state.engine.processed_files)}
- **Total Pages**: {total_pages}
- **Total Words**: {total_words:,}
- **Processed Sections**: {total_chunks}
"""
    else:
        status_md = f"""### Ingestion Status
- **System Status**: Ready (Waiting for files)
- **Loaded Files**: None
- **Total Pages**: 0
- **Total Words**: 0
- **Processed Sections**: 0
"""
        
    return status_md, status_text_val

def reset_memory_callback():
    state.reset()
    default_status = "System memory and knowledge base reset. Default knowledge loaded."
    status_md = f"""### Ingestion Status
- **System Status**: Ready (Memory Cleaned)
- **Loaded Files**: None
- **Total Pages**: 0
- **Total Words**: 0
- **Processed Sections**: 0
"""
    return (
        status_md, 
        default_status, 
        [], 
        "### Retrieved RAG Chunks\n\n*No source documents retrieved yet.*", 
        "### Retrieved FAISS Memory Chunks\n\n*No conversation history retrieved from memory.*"
    )

# Real Groq call or clean mock LLM generator
def generate_response(query, history, personal_context="", system_context=""):
    if not query.strip():
        return history, "", "### Retrieved RAG Chunks\n\n*No query entered.*", "### Retrieved FAISS Memory Chunks\n\n*No query entered.*"
        
    # Save current contexts so they persist across restarts/page refreshes
    save_contexts(personal_context, system_context)

    # Generate unique UUID for this interaction
    interaction_id = str(uuid.uuid4())
    state.interaction_ids.append(interaction_id)
    
    # 1. Retrieve RAG Source Documents
    top_chunks, ranked = state.engine.retrieve_hybrid(query, top_k=8)
    
    # 2. Retrieve FAISS Memory
    memories = memory_store.search_memory(query, top_k=2)
    
    # Get Groq API key from backend (.env or environment)
    groq_key = load_env_api_key()
    
    # Check if a Groq API key is provided and valid-looking
    use_real_groq = groq_key and groq_key.startswith("gsk_")
    
    # Format source docs for display and context
    docs_context = "\n\n".join([
        f"Doc Context | Source: {chunk['source']} (Page {chunk['page']}) | Chunk ID: {chunk['chunk_id']} | RRF Score: {chunk['rrf_score']:.6f}\n{chunk['text']}"
        for chunk in top_chunks
    ])
    
    memory_context = ""
    if memories:
        memory_context = "\n\n".join([f"Past Memory (Distance: {m['distance']:.4f}):\n{m['memory_text']}" for m in memories])
        
    # Format recent history section for prompt
    recent_history = ""
    if history:
        recent = history[-6:]
        formatted = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        recent_history = "\n".join(formatted)
        
    history_section = f"\nRecent Conversation History:\n{recent_history}\n" if recent_history else ""
    memory_section = f"\nRelevant Past Conversation (from memory search):\n{memory_context}\n" if memory_context else ""

    # Parse custom personal context & custom system instruction
    system_prompt = system_context.strip() if system_context.strip() else "You answer only from retrieved document context. Use memory and history only to understand the question better."
    personal_section = f"\nUser Personal Context:\n{personal_context.strip()}\n" if personal_context.strip() else ""

    if use_real_groq:
        try:
            client = Groq(api_key=groq_key)
            MODEL_NAME = "llama-3.1-8b-instant"
            
            prompt = f"""
You are a helpful assistant answering questions only from the provided document context.
{personal_section}
Rules:
- Answer using only the document context below.
- If the answer is not in the document context, say: "The answer is not available in the provided document."
- Do not use outside knowledge.
- Use past memory and conversation history only to understand follow-up questions.
- Never answer from memory alone. Always ground in document context.
{memory_section}{history_section}
Document Context:
{docs_context}

Question:
{query}

Answer:
"""
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Groq API Error: {str(e)}\n*(Falling back to local simulation...)*\n\n" + generate_mock_answer(query, top_chunks, memories, personal_context, system_context)
    else:
        # Local high-quality RAG mock response
        answer = generate_mock_answer(query, top_chunks, memories, personal_context, system_context)
        if not groq_key:
            answer += "\n\nNote: To query live LLMs, set GROQ_API_KEY in your environment or a .env file."

    # Compile contributing sources
    contributing = []
    seen_citations = set()
    for chunk in top_chunks:
        citation = f"{chunk['source']} (Page {chunk['page']})"
        if citation not in seen_citations:
            seen_citations.add(citation)
            contributing.append((chunk['source'], chunk['page']))
            
    # Format citations section
    citations_list = [f"- {src} (Page {pg})" for src, pg in contributing]
    citations_section = "\n\n**Sources:**\n" + "\n".join(citations_list) if citations_list else ""
    
    final_answer = answer + citations_section

    # Add query-answer to conversational memory
    memory_store.add_memory(query, answer)
    
    # Update Chat UI using list-of-dicts format required by Gradio 6.0 Chatbot
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": final_answer})
    
    # Log query-response details
    sources_log_list = [f"{src} (Page {pg})" for src, pg in contributing]
    append_log_entry(interaction_id, query, final_answer, sources_log_list)
    
    # Build accordion outputs
    docs_md = "### Retrieved RAG Chunks\n\n"
    for chunk in top_chunks:
        docs_md += f"Chunk ID: `{chunk['chunk_id']}` | RRF Score: `{chunk['rrf_score']:.6f}` | Source: *{chunk['source']}* (Page {chunk['page']})\n"
        docs_md += f"> {chunk['text']}\n\n---\n\n"
        
    memories_md = "### Retrieved FAISS Memory Chunks\n\n"
    if memories:
        for m in memories:
            memories_md += f"Match Distance: `{m['distance']:.4f}`\n"
            memories_md += f"> User: {m['user']}\n"
            memories_md += f"> Assistant: {m['assistant']}\n\n---\n\n"
    else:
        memories_md += "*No semantic overlap found in prior conversation memory.*"
        
    return history, "", docs_md, memories_md

def generate_mock_answer(query, top_chunks, memories, personal_context="", system_context=""):
    top_doc = top_chunks[0]
    q_lower = query.lower()
    time.sleep(1.2) # Simulate latency
    
    source_str = f"{top_doc['source']} (Page {top_doc['page']})"
    if "hybrid" in q_lower or "rrf" in q_lower:
        base_ans = (
            f"Based on Chunk `{top_doc['chunk_id']}` ({source_str}), Hybrid RAG leverages both dense semantic search (vectors) "
            f"and sparse keyword search (BM25). The combination is executed using Reciprocal Rank Fusion (RRF), which yields a score of "
            f"**{top_doc['rrf_score']:.6f}** for the top context. RRF computes ranks without needing matching score scales."
        )
    elif "memory" in q_lower or "faiss" in q_lower:
        base_ans = (
            f"According to Chunk `{top_doc['chunk_id']}` ({source_str}), conversation history is saved as dense vectors in the FAISS index. "
            f"When you send a query, the system retrieves semantically relevant historical turns (like in the 'FAISS Memory' panel) "
            f"and appends them to the system context."
        )
    else:
        # General synthesis response
        snippet = top_doc['text'][:220]
        base_ans = (
            f"Synthesizing an answer using Chunk `{top_doc['chunk_id']}`:\n\n"
            f"**Retrieved Insight**: {snippet}...\n\n"
            f"This RAG source document was evaluated with an RRF score of **{top_doc['rrf_score']:.6f}** for your query. "
            f"If relevant conversation memory was found, it was blended into the context to ensure continuity."
        )

    # Simulated tailoring based on personal context or custom system prompt
    customization_applied = []
    if personal_context.strip():
        customization_applied.append(f"Tailored response for context: '{personal_context.strip()}'")
    if system_context.strip():
        customization_applied.append(f"Applied instructions: '{system_context.strip()}'")
        
    if customization_applied:
        base_ans += "\n\nCustomizations applied:\n" + "\n".join([f"- {item}" for item in customization_applied])
        
    return base_ans

# Gradio chatbot thumbs up/down callback
def handle_like(x: gr.LikeData):
    msg_idx = x.index[0] if isinstance(x.index, (list, tuple)) else x.index
    interaction_idx = msg_idx // 2
    if interaction_idx < len(state.interaction_ids):
        uuid_str = state.interaction_ids[interaction_idx]
        feedback_value = "thumbs_up" if x.liked else "thumbs_down"
        update_log_feedback(uuid_str, feedback_value)

# Gradio custom CSS styling for professional white-and-golden luxury-minimalist aesthetic
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Force light theme colors on Gradio variables for both light mode and dark mode */
:root, .dark {
    --body-bg: #fcfcf9 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #faf8f2 !important;
    --border-color-primary: #e5dfcf !important;
    --border-color-secondary: #e5dfcf !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #e5dfcf !important;
    --input-border-color-focus: #b59461 !important;
    --block-background-fill: #faf8f2 !important;
    --block-border-color: #e5dfcf !important;
    --block-title-text-color: #1a1a17 !important;
    --block-label-text-color: #6e685a !important;
    --body-text-color: #1a1a17 !important;
    --body-text-color-subdued: #6e685a !important;
    --primary-button-background-fill: linear-gradient(135deg, #c5a880 0%, #b59461 100%) !important;
    --primary-button-background-fill-hover: linear-gradient(135deg, #b59461 0%, #a37d37 100%) !important;
    --primary-button-text-color: #ffffff !important;
    --secondary-button-background-fill: #ffffff !important;
    --secondary-button-text-color: #b59461 !important;
    --secondary-button-border-color: #e5dfcf !important;
    --chatbot-code-background-color: #faf8f2 !important;
}

/* Base styles and overrides */
body, .gradio-container, .dark .gradio-container {
    background-color: #fcfcf9 !important;
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1a1a17 !important;
}

/* Titles */
h1, h2, h3, h4, .title-text {
    font-family: 'Outfit', sans-serif !important;
    color: #1a1a17 !important;
    font-weight: 700 !important;
}

/* Sidebar and Main panels */
.sidebar-panel, .dark .sidebar-panel, .main-panel, .dark .main-panel, .custom-card {
    background: #faf8f2 !important;
    border: 1px solid #e5dfcf !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(181, 148, 97, 0.05) !important;
    padding: 20px !important;
}

/* Force Light Backgrounds & Charcoal Text on Inputs/Textareas in all modes */
input, textarea, select, .gr-input, .gr-textarea,
.svelte-1viwjop, .svelte-16ip5nk, .svelte-1puz88e,
.dark input, .dark textarea, .dark select,
.dark .gr-input, .dark .gr-textarea {
    background-color: #ffffff !important;
    color: #1a1a17 !important;
    border: 1px solid #e5dfcf !important;
    border-radius: 8px !important;
    padding: 10px !important;
    transition: all 0.2s ease !important;
}

input:focus, textarea:focus, .dark input:focus, .dark textarea:focus {
    border-color: #b59461 !important;
    box-shadow: 0 0 0 2px rgba(181, 148, 97, 0.15) !important;
}

/* Force light background on all Gradio blocks/containers in both light/dark modes */
div[class*="gr-block"], div[class*="gr-box"], div[class*="gr-form"],
.block, .form, .fieldset, .gr-panel, .gr-card,
.dark div[class*="gr-block"], .dark div[class*="gr-box"], .dark div[class*="gr-form"],
.dark .block, .dark .form, .dark .fieldset, .dark .gr-panel, .dark .gr-card {
    background-color: #faf8f2 !important;
    border-color: #e5dfcf !important;
    color: #1a1a17 !important;
}

/* Primary Button */
button.primary-btn, .dark button.primary-btn {
    background: linear-gradient(135deg, #c5a880 0%, #b59461 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.2s ease-in-out !important;
    cursor: pointer !important;
    padding: 12px 24px !important;
}

button.primary-btn:hover, .dark button.primary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(181, 148, 97, 0.25) !important;
}

/* Secondary Button */
button.secondary-btn, .dark button.secondary-btn {
    background: #ffffff !important;
    color: #b59461 !important;
    border: 1px solid #e5dfcf !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease-in-out !important;
    padding: 10px 20px !important;
}

button.secondary-btn:hover, .dark button.secondary-btn:hover {
    background: #faf8f2 !important;
    border-color: #b59461 !important;
}

/* Accordions */
.gr-accordion, .dark .gr-accordion {
    border: 1px solid #e5dfcf !important;
    background: #ffffff !important;
    border-radius: 8px !important;
    margin-top: 10px !important;
}

.gr-accordion-header, .dark .gr-accordion-header {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: #b59461 !important;
}

/* Ingestion Status Block */
.status-log, .dark .status-log {
    background: #ffffff !important;
    border-left: 4px solid #b59461 !important;
    border-top: 1px solid #e5dfcf !important;
    border-right: 1px solid #e5dfcf !important;
    border-bottom: 1px solid #e5dfcf !important;
    padding: 15px !important;
    border-radius: 6px !important;
    font-family: 'Space Grotesk', monospace !important;
    color: #1a1a17 !important;
}

.status-log *, .dark .status-log * {
    border: none !important;
    box-shadow: none !important;
}

/* Chatbot container & message styling */
.chatbot-container, .dark .chatbot-container {
    background: #ffffff !important;
    border: 1px solid #e5dfcf !important;
    border-radius: 12px !important;
    min-height: 480px !important;
}

.chatbot-container .message, .dark .chatbot-container .message {
    border-radius: 8px !important;
    padding: 12px !important;
}

/* Force bubble colors and text colors in both light and dark mode */
.chatbot-container .message.user, .dark .chatbot-container .message.user,
.chatbot-container .user, .dark .chatbot-container .user {
    background-color: #fcfaf6 !important;
    color: #1a1a17 !important;
    border: 1px solid #ebdcb9 !important;
}

.chatbot-container .message.bot, .dark .chatbot-container .message.bot,
.chatbot-container .bot, .dark .chatbot-container .bot {
    background-color: #ffffff !important;
    color: #1a1a17 !important;
    border: 1px solid #e5dfcf !important;
}

.chatbot-container, .dark .chatbot-container,
.chatbot-container div, .dark .chatbot-container div {
    background-color: #ffffff !important;
    color: #1a1a17 !important;
}

/* Scrollbar customization */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #fcfcf9;
}
::-webkit-scrollbar-thumb {
    background: rgba(181, 148, 97, 0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(181, 148, 97, 0.4);
}

/* Sidebar Navigation Buttons */
button.sidebar-nav-btn, .dark button.sidebar-nav-btn {
    background: #ffffff !important;
    color: #1a1a17 !important;
    border: 1px solid #e5dfcf !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: all 0.2s ease-in-out !important;
    padding: 10px 15px !important;
    width: 100% !important;
    margin-bottom: 8px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    cursor: pointer !important;
}

button.sidebar-nav-btn:hover, .dark button.sidebar-nav-btn:hover {
    background: #faf8f2 !important;
    border-color: #b59461 !important;
    color: #b59461 !important;
}

/* Hide all tab navigation buttons in the main panel except the first one (Chatbot Portal) */
#right-tabs-container .tab-nav button:not(:first-child),
#right-tabs-container div[role="tablist"] button:not(:first-child) {
    display: none !important;
}

/* Re-enable all tab navigation buttons for nested tabs (e.g. inside Session Logs) */
#right-tabs-container .tabitem .tab-nav button,
#right-tabs-container .tabitem div[role="tablist"] button {
    display: inline-block !important;
}
"""

# Build the Gradio Blocks UI
with gr.Blocks(title="Hybrid RAG Workspace", css=custom_css) as demo:
    
    with gr.Row():
        gr.HTML(
            f"""
            <div style="padding: 15px 0; border-bottom: 1px solid #e5dfcf; margin-bottom: 25px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 style="font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 800; background: linear-gradient(to right, #b59461, #cca43b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
                        Hybrid RAG Workspace
                    </h1>
                    <p style="font-family: 'Space Grotesk', sans-serif; color: #6e685a; margin: 5px 0 0 0;">
                        Active Vector Memory and Lexical Document Retrospective Engine
                    </p>
                </div>
                <div style="background: rgba(181, 148, 97, 0.1); border: 1px solid rgba(181, 148, 97, 0.2); padding: 8px 16px; border-radius: 20px; display: flex; align-items: center; gap: 8px;">
                    <span style="width: 8px; height: 8px; background-color: #b59461; border-radius: 50%; display: inline-block; animation: pulse 2s infinite;"></span>
                    <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; color: #b59461; font-weight: 500;">FAISS Memory: Online</span>
                </div>
            </div>
            <style>
            @keyframes pulse {{
                0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(181, 148, 97, 0.7); }}
                70% {{ transform: scale(1); box-shadow: 0 0 0 6px rgba(181, 148, 97, 0); }}
                100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(181, 148, 97, 0); }}
            }}
            </style>
            """
        )
        
    with gr.Row():
        # 1. Left Sidebar - Houses all functions, options, accordions, and tools (Jira Board Style)
        with gr.Column(scale=1, elem_classes="sidebar-panel"):
            gr.Markdown("### Custom Context")
            
            saved_personal, saved_system = load_saved_contexts()
            
            # Personal Context TextBox
            personal_context_input = gr.Textbox(
                label="Personal Context",
                placeholder="Tell us about yourself to tailor responses...",
                value=saved_personal,
                lines=2
            )
            
            # System Context TextBox
            system_context_input = gr.Textbox(
                label="System Context",
                placeholder="Custom system instructions...",
                value=saved_system,
                lines=3
            )

            # Auto-save immediately when user clicks out of the input boxes
            personal_context_input.blur(fn=save_contexts, inputs=[personal_context_input, system_context_input])
            system_context_input.blur(fn=save_contexts, inputs=[personal_context_input, system_context_input])

            gr.Markdown("### Document Source")

            # PDF Upload (supports multiple files)
            pdf_uploader = gr.File(
                label="PDF Document Source(s)",
                file_types=[".pdf"],
                file_count="multiple",
                interactive=True
            )

            gr.Markdown(
                "<div style='font-size: 0.85rem; color: #6e685a; margin-top: -8px; margin-bottom: 12px; line-height: 1.3;'>"
                "Tip: Hold Ctrl (or Cmd) to select/upload multiple PDFs at once. "
                "New uploads are automatically added incrementally without losing previous files."
                "</div>"
            )

            # Ingestion Log Output
            initial_status = (
                "### Ingestion Status\n"
                "- **System Status**: Ready (Waiting for files)\n"
                "- **Loaded Files**: None\n"
                "- **Total Pages**: 0\n"
                "- **Total Words**: 0\n"
                "- **Processed Sections**: 0\n"
            )
            status_display = gr.Markdown(
                value=initial_status,
                elem_classes="status-log"
            )

            # Status Text
            status_text = gr.Textbox(
                label="Log Details",
                value="No documents uploaded. System initialized with default knowledge base.",
                interactive=False
            )

            # Reset Memory Button
            reset_btn = gr.Button(
                "Reset Memory & Index",
                elem_classes="secondary-btn"
            )

            gr.Markdown("### Workspace Views")
            btn_show_chunks = gr.Button("📄 Retrieved Chunks", elem_classes="sidebar-nav-btn")
            btn_show_memory = gr.Button("🧠 FAISS Memory", elem_classes="sidebar-nav-btn")
            btn_show_logs = gr.Button("📊 Session Logs", elem_classes="sidebar-nav-btn")
            btn_show_analyzer = gr.Button("🔍 Feedback Analyzer", elem_classes="sidebar-nav-btn")

        # 2. Main Panel - Pure focus on the chat portal
        with gr.Column(scale=2, elem_classes="main-panel"):
            with gr.Tabs(elem_id="right-tabs-container") as right_tabs:
                with gr.Tab("💬 Chatbot Portal", id="chat_tab"):
                    # Chatbot
                    chatbot = gr.Chatbot(
                        label="Q&A Portal",
                        elem_classes="chatbot-container"
                    )

                    # Query input row
                    with gr.Row():
                        query_input = gr.Textbox(
                            placeholder="Ask a question about the document or RAG concepts...",
                            show_label=False,
                            scale=4
                        )
                        submit_btn = gr.Button(
                            "Search & Generate",
                            elem_classes="primary-btn",
                            scale=1
                        )
                
                with gr.Tab("📄 Retrieved Chunks", id="chunks_tab"):
                    retrieved_docs = gr.Markdown(
                        value="### Retrieved RAG Chunks\n\n*No source documents retrieved yet.*",
                        elem_classes="custom-card"
                    )
                
                with gr.Tab("🧠 FAISS Memory Chunks", id="memory_tab"):
                    retrieved_memory_display = gr.Markdown(
                        value="### Retrieved FAISS Memory Chunks\n\n*No conversation history retrieved from memory.*",
                        elem_classes="custom-card"
                    )
                
                with gr.Tab("📊 Session Logs", id="logs_tab"):
                    gr.Markdown("### Session RAG Interaction Logs")
                    with gr.Tabs():
                        with gr.Tab("Structured Table"):
                            logs_table_display = gr.Markdown(value="### Saved Logs\n\n*No logs generated in this session yet.*", elem_classes="custom-card")
                        with gr.Tab("Raw JSON Logs"):
                            logs_json_display = gr.Code(value="[]", language="json", interactive=False)
                    refresh_logs_btn = gr.Button("Refresh Logs", elem_classes="secondary-btn")
                
                with gr.Tab("🔍 Feedback Analyzer", id="analyzer_tab"):
                    gr.Markdown("### Negative Feedback Analyzer (Current Session)")
                    gr.Markdown(
                        "How it works: After giving Down ratings in the chat, click Run Analysis. "
                        "The system will analyze only the thumbs-down interactions from this session."
                    )
                    with gr.Row():
                        run_analysis_btn = gr.Button(
                            "Run Negative Feedback Analysis",
                            elem_classes="primary-btn"
                        )
                        export_report_btn = gr.Button(
                            "Export Report (Markdown)",
                            elem_classes="secondary-btn"
                        )
                    
                    gr.Markdown("#### Document-wise Negative Feedback Summary")
                    analysis_summary = gr.Markdown("Click Run Analysis to generate.", elem_classes="custom-card")
                    
                    gr.Markdown("#### Per-Interaction Root Cause Analysis")
                    analysis_detail = gr.Markdown("Results will appear here after analysis.", elem_classes="custom-card")


    # ── Event Bindings ───────────────────────────────────────────────────────
    pdf_uploader.change(
        fn=process_pdf,
        inputs=[pdf_uploader],
        outputs=[status_display, status_text]
    )
    
    reset_btn.click(
        fn=reset_memory_callback,
        inputs=[],
        outputs=[status_display, status_text, chatbot, retrieved_docs, retrieved_memory_display]
    ).then(
        fn=refresh_logs_callback,
        inputs=[],
        outputs=[logs_table_display, logs_json_display]
    )
    
    query_input.submit(
        fn=generate_response,
        inputs=[query_input, chatbot, personal_context_input, system_context_input],
        outputs=[chatbot, query_input, retrieved_docs, retrieved_memory_display]
    ).then(
        fn=refresh_logs_callback,
        inputs=[],
        outputs=[logs_table_display, logs_json_display]
    )
    
    submit_btn.click(
        fn=generate_response,
        inputs=[query_input, chatbot, personal_context_input, system_context_input],
        outputs=[chatbot, query_input, retrieved_docs, retrieved_memory_display]
    ).then(
        fn=refresh_logs_callback,
        inputs=[],
        outputs=[logs_table_display, logs_json_display]
    )

    # Bind vote feedback callback
    chatbot.like(fn=handle_like, inputs=[], outputs=[]).then(
        fn=refresh_logs_callback,
        inputs=[],
        outputs=[logs_table_display, logs_json_display]
    )

    # Refresh logs on button click
    refresh_logs_btn.click(
        fn=refresh_logs_callback,
        inputs=[],
        outputs=[logs_table_display, logs_json_display]
    )

    # Feedback Analysis Bindings
    run_analysis_btn.click(
        fn=run_feedback_analysis,
        inputs=[],
        outputs=[analysis_summary, analysis_detail]
    )

    export_report_btn.click(
        fn=export_negative_report,
        inputs=[],
        outputs=[analysis_detail]
    )


    # Sidebar Navigation Bindings to switch Tabs programmatically
    btn_show_chunks.click(fn=lambda: gr.update(selected="chunks_tab"), inputs=[], outputs=[right_tabs])
    btn_show_memory.click(fn=lambda: gr.update(selected="memory_tab"), inputs=[], outputs=[right_tabs])
    btn_show_logs.click(fn=lambda: gr.update(selected="logs_tab"), inputs=[], outputs=[right_tabs])
    btn_show_analyzer.click(fn=lambda: gr.update(selected="analyzer_tab"), inputs=[], outputs=[right_tabs])

    # Load initial logs on page load
    demo.load(
        fn=refresh_logs_callback,
        inputs=[],
        outputs=[logs_table_display, logs_json_display]
    )

if __name__ == "__main__":
    if "SPACE_ID" in os.environ:
        # On Hugging Face Spaces, let the platform handle host and port binding
        print("Running on Hugging Face Spaces...")
        demo.launch()
    else:
        # Bind to 0.0.0.0 and respect the PORT environment variable for local/Render deployment
        port = int(os.environ.get("PORT", 8080))
        print(f"Running locally on port {port}...")
        demo.launch(server_name="0.0.0.0", server_port=port, share=False)
