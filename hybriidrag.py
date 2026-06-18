# -*- coding: utf-8 -*-
"""HybriidRag.ipynb

Exposes FaissMemoryStore and HybridRagEngine classes, and necessary utility functions.
"""

import os
import json
import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default RAG document chunks
DEFAULT_CHUNKS = [
    {
        "text": "Hybrid Retrieval-Augmented Generation (RAG) combines dense semantic retrieval (vector-based, like FAISS or Milvus) with sparse keyword retrieval (lexical-based, like BM25 or TF-IDF). The results of both search types are typically combined using Reciprocal Rank Fusion (RRF) to generate a single re-ranked list of relevant documents.",
        "source": "System Default Knowledge Base",
        "page": 1
    },
    {
        "text": "Reciprocal Rank Fusion (RRF) is a method to combine multiple retrieved document lists. The RRF score for a document d is calculated as: RRF_Score(d) = sum( 1 / (k + r_i(d)) ) for each retriever i, where r_i(d) is the rank of document d in list i, and k is a constant parameter (typically set to 60). RRF does not require calibrating scores between different retrievers.",
        "source": "System Default Knowledge Base",
        "page": 1
    },
    {
        "text": "Conversational memory in RAG systems can be managed using vector databases. Instead of feeding the entire raw chat history into the LLM context, prior exchanges are embedded and indexed. During a new turn, the query is used to retrieve semantically relevant historical context from the vector database (FAISS memory), keeping the context window concise and focused.",
        "source": "System Default Knowledge Base",
        "page": 1
    },
    {
        "text": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.",
        "source": "System Default Knowledge Base",
        "page": 1
    }
]

# FAISS Memory Store
class FaissMemoryStore:
    def __init__(self, embed_model, dim=None,
                 index_path="memory_index.faiss",
                 store_path="memory_store.json"):
        self.embed_model = embed_model
        self.index_path = index_path
        self.store_path = store_path

        if dim is None:
            test_vec = self.embed_model.encode(["test"], convert_to_numpy=True)
            dim = test_vec.shape[1]

        self.dim = dim
        self.memories = []

        if os.path.exists(self.index_path) and os.path.exists(self.store_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
            except Exception as e:
                print(f"Error loading FAISS index/store: {e}. Starting fresh.")
                self.index = faiss.IndexFlatL2(self.dim)
                self.memories = []
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.memories = []

    def save(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save FAISS index/store: {e}")

    def add_memory(self, user_query, assistant_answer):
        memory_text = f"User: {user_query}\nAssistant: {assistant_answer}"

        embedding = self.embed_model.encode([memory_text], convert_to_numpy=True)
        embedding = np.array(embedding).astype("float32")

        self.index.add(embedding)

        memory_item = {
            "id": len(self.memories),
            "user": user_query,
            "assistant": assistant_answer,
            "memory_text": memory_text
        }

        self.memories.append(memory_item)
        self.save()

    def search_memory(self, query, top_k=2):
        if len(self.memories) == 0 or self.index.ntotal == 0:
            return []

        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.memories):
                item = self.memories[idx].copy()
                item["distance"] = float(dist)
                results.append(item)

        return results

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.memories = []
        self.save()


# Hybrid RAG Engine
class HybridRagEngine:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.processed_files = set()
        self.chunks = []
        self.doc_metadata = {}
        self.dense_index = None
        self.bm25 = None
        self.chunk_embeddings = None
        self._rebuild_indices()

    def reset(self):
        self.processed_files = set()
        self.chunks = []
        self.doc_metadata = {}
        self.chunk_embeddings = None
        self._rebuild_indices()

    def ingest_file(self, file_path):
        basename = os.path.basename(file_path)
        if basename in self.processed_files:
            print(f"File {basename} already processed, skipping.")
            return False

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        new_chunks = []
        num_pages = 0
        char_count = 0
        word_count = 0
        extracted = False

        # 1. Try PyMuPDF (fitz)
        try:
            import fitz
            with fitz.open(file_path) as doc:
                num_pages = len(doc)
                for i, page in enumerate(doc):
                    page_number = i + 1
                    text = page.get_text()
                    cleaned_text = text.strip() if text else ""
                    if not cleaned_text:
                        continue
                    char_count += len(cleaned_text)
                    word_count += len(cleaned_text.split())
                    page_chunks = splitter.split_text(cleaned_text)
                    for chunk_text in page_chunks:
                        new_chunks.append({
                            "text": chunk_text,
                            "source": basename,
                            "page": page_number
                        })
            extracted = True
        except Exception:
            pass

        # 2. Try pypdf
        if not extracted:
            try:
                import pypdf
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    num_pages = len(reader.pages)
                    for i, page in enumerate(reader.pages):
                        page_number = i + 1
                        text = page.extract_text()
                        cleaned_text = text.strip() if text else ""
                        if not cleaned_text:
                            continue
                        char_count += len(cleaned_text)
                        word_count += len(cleaned_text.split())
                        page_chunks = splitter.split_text(cleaned_text)
                        for chunk_text in page_chunks:
                            new_chunks.append({
                                "text": chunk_text,
                                "source": basename,
                                "page": page_number
                            })
                extracted = True
            except Exception:
                pass

        # 3. Try pdfplumber
        if not extracted:
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    page_number = i + 1
                    text = page.extract_text()
                    cleaned_text = text.strip() if text else ""
                    if not cleaned_text:
                        continue
                    char_count += len(cleaned_text)
                    word_count += len(cleaned_text.split())
                    page_chunks = splitter.split_text(cleaned_text)
                    for chunk_text in page_chunks:
                        new_chunks.append({
                            "text": chunk_text,
                            "source": basename,
                            "page": page_number
                        })

        if not new_chunks:
            raise ValueError("No text could be extracted or chunked from the PDF. Is it a scanned PDF with no OCR?")

        # Encode ONLY the new chunks!
        new_texts = [c["text"] for c in new_chunks]
        new_embeddings = self.embed_model.encode(new_texts, convert_to_numpy=True).astype("float32")

        if self.chunk_embeddings is None or len(self.chunks) == 0:
            self.chunk_embeddings = new_embeddings
        else:
            self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embeddings])

        self.chunks.extend(new_chunks)
        self.processed_files.add(basename)
        self.doc_metadata[basename] = {
            "pages": num_pages,
            "chars": char_count,
            "words": word_count
        }
        self._rebuild_indices()
        return True

    def _rebuild_indices(self):
        active_chunks = self.chunks if self.chunks else DEFAULT_CHUNKS
        if not active_chunks:
            self.dense_index = None
            self.bm25 = None
            return

        if not self.chunks:
            # Using DEFAULT_CHUNKS
            texts = [c["text"] for c in DEFAULT_CHUNKS]
            embeddings = self.embed_model.encode(texts, convert_to_numpy=True).astype("float32")
        else:
            if self.chunk_embeddings is None:
                texts = [c["text"] for c in self.chunks]
                self.chunk_embeddings = self.embed_model.encode(texts, convert_to_numpy=True).astype("float32")
            embeddings = self.chunk_embeddings

        dim = embeddings.shape[1]
        self.dense_index = faiss.IndexFlatL2(dim)
        self.dense_index.add(embeddings)

        # Rebuild BM25 index
        texts = [c["text"] for c in active_chunks]
        tokenized_texts = []
        for text in texts:
            toks = text.lower().split()
            if not toks:
                toks = [""]
            tokenized_texts.append(toks)
        self.bm25 = BM25Okapi(tokenized_texts)

    def retrieve_hybrid(self, query, top_k=8, k_rrf=60):
        active_chunks = self.chunks if self.chunks else DEFAULT_CHUNKS
        if not active_chunks or self.dense_index is None or self.bm25 is None:
            return [], []

        # Dense retrieval
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.dense_index.search(query_embedding, min(top_k, len(active_chunks)))
        dense_indices = [int(idx) for idx in indices[0] if idx != -1]

        # Sparse retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k].tolist()

        dense_rank_map = {doc_id: rank + 1 for rank, doc_id in enumerate(dense_indices)}
        bm25_rank_map = {doc_id: rank + 1 for rank, doc_id in enumerate(bm25_indices)}

        all_candidates = set(dense_indices).union(set(bm25_indices))
        rrf_scores = {}

        for doc_id in all_candidates:
            score = 0.0
            if doc_id in dense_rank_map:
                score += 1 / (k_rrf + dense_rank_map[doc_id])
            if doc_id in bm25_rank_map:
                score += 1 / (k_rrf + bm25_rank_map[doc_id])
            rrf_scores[doc_id] = score

        final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        top_chunks = []
        for idx, score in final_ranked[:top_k]:
            chunk = active_chunks[idx].copy()
            chunk["chunk_id"] = int(idx)
            chunk["rrf_score"] = float(score)
            top_chunks.append(chunk)

        return top_chunks, final_ranked[:top_k]


# Standalone shared retrieve logic (backward compatible with list-of-strings chunks and model parameters)
def retrieve_hybrid(query, chunks, model, chunk_embeddings, bm25, top_k=8, k_rrf=60):
    normalized_chunks = []
    for i, c in enumerate(chunks):
        if isinstance(c, dict):
            normalized_chunks.append(c)
        else:
            normalized_chunks.append({"text": str(c), "source": "System Default Knowledge Base", "page": 1})

    query_embedding = model.encode([query], convert_to_numpy=True)
    if hasattr(chunk_embeddings, "search"): # FAISS Index object
        query_embedding_f32 = query_embedding.astype("float32")
        distances, indices = chunk_embeddings.search(query_embedding_f32, min(top_k, len(normalized_chunks)))
        dense_indices = [int(idx) for idx in indices[0] if idx != -1]
    else: # Raw numpy array
        from sklearn.metrics.pairwise import cosine_similarity
        dense_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        dense_indices = np.argsort(dense_scores)[::-1][:top_k].tolist()

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k].tolist()

    dense_rank_map = {doc_id: rank + 1 for rank, doc_id in enumerate(dense_indices)}
    bm25_rank_map = {doc_id: rank + 1 for rank, doc_id in enumerate(bm25_indices)}

    all_candidates = set(dense_indices).union(set(bm25_indices))
    rrf_scores = {}
    for doc_id in all_candidates:
        score = 0.0
        if doc_id in dense_rank_map:
            score += 1 / (k_rrf + dense_rank_map[doc_id])
        if doc_id in bm25_rank_map:
            score += 1 / (k_rrf + bm25_rank_map[doc_id])
        rrf_scores[doc_id] = score

    final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    top_chunks = []
    for idx, score in final_ranked[:top_k]:
        chunk = normalized_chunks[idx].copy()
        chunk["chunk_id"] = int(idx)
        chunk["rrf_score"] = float(score)
        top_chunks.append(chunk)

    return top_chunks, final_ranked[:top_k]


# Answer functions
def answer_with_rag(query, client, llm_model, retriever_model, chunk_embeddings, bm25, chunks, top_k=8):
    top_chunks, ranked = retrieve_hybrid(
        query=query,
        chunks=chunks,
        model=retriever_model,
        chunk_embeddings=chunk_embeddings,
        bm25=bm25,
        top_k=top_k
    )

    context = "\n\n".join(
        [
            f"Context {i+1} | Source: {chunk['source']} (Page {chunk['page']}) | Chunk ID: {chunk['chunk_id']} | RRF Score: {chunk['rrf_score']:.6f}\n{chunk['text']}"
            for i, chunk in enumerate(top_chunks)
        ]
    )

    prompt = f"""
You are a helpful assistant answering questions only from the provided context.

Rules:
- Answer using only the context below.
- If the answer is not clearly in the context, say: "The answer is not available in the provided document."
- Do not use outside knowledge.
- Keep the answer clear and accurate.
- If multiple contexts are retrieved, prefer the most directly relevant one.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You answer only from retrieved document context."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content, top_chunks, ranked


def format_chat_history(chat_history, max_turns=3):
    if not chat_history:
        return ""
    recent = chat_history[-max_turns * 2:]
    formatted = []
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {turn['content']}")
    return "\n".join(formatted)


def answer_with_rag_and_memory(query, client, llm_model, retriever_model,
                                chunk_embeddings, bm25, chunks,
                                chat_history, memory_store, top_k=8):

    top_chunks, ranked = retrieve_hybrid(
        query=query,
        chunks=chunks,
        model=retriever_model,
        chunk_embeddings=chunk_embeddings,
        bm25=bm25,
        top_k=top_k
    )

    document_context = "\n\n".join([
        f"Doc Context {i+1} | Source: {chunk['source']} (Page {chunk['page']}) | Chunk ID: {chunk['chunk_id']} | RRF Score: {chunk['rrf_score']:.6f}\n{chunk['text']}"
        for i, chunk in enumerate(top_chunks)
    ])

    memory_results = memory_store.search_memory(query, top_k=2)

    if memory_results:
        memory_context = "\n\n".join([
            f"Past Memory {i+1} (Distance: {r['distance']:.4f}):\n{r['memory_text']}"
            for i, r in enumerate(memory_results)
        ])
    else:
        memory_context = ""

    recent_history = format_chat_history(chat_history, max_turns=3)

    memory_section = f"""
Relevant Past Conversation (from memory search):
{memory_context}
""" if memory_context else ""

    history_section = f"""
Recent Conversation History:
{recent_history}
""" if recent_history else ""

    prompt = f"""
You are a helpful assistant answering questions only from the provided document context.

Rules:
- Answer using only the document context below.
- If the answer is not in the document context, say: "The answer is not available in the provided document."
- Do not use outside knowledge.
- Use past memory and conversation history only to understand follow-up questions.
- Never answer from memory alone. Always ground in document context.
{memory_section}{history_section}
Document Context:
{document_context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You answer only from retrieved document context. Use memory and history only to understand the question better."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})
    memory_store.add_memory(query, answer)

    return answer, top_chunks, ranked, chat_history


if __name__ == "__main__":
    # Wrap Colab and interactive CLI logic here
    try:
        from google.colab import files
        uploaded = files.upload()
        pdf_name = list(uploaded.keys())[0]
        print("Uploaded file:", pdf_name)

        with pdfplumber.open(pdf_name) as pdf:
            print("Number of pages =", len(pdf.pages))
    except (ImportError, Exception):
        print("Not running in Google Colab environment. Interactive execution skipped.")