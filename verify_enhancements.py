# -*- coding: utf-8 -*-
"""
verify_enhancements.py

Automated test runner and verification suite for Quick Hopper Hybrid RAG app enhancements.
"""

import os
import json
import tempfile
import shutil
import unittest
import uuid
from unittest.mock import patch, MagicMock
import numpy as np

# Import actual modules
import hybriidrag
from hybriidrag import HybridRagEngine, FaissMemoryStore, DEFAULT_CHUNKS
import app

class TestHybridRAGActual(unittest.TestCase):
    def setUp(self):
        # We use the real SentenceTransformer model initialized in app
        self.model = app.model
        self.engine = HybridRagEngine(self.model)
        
        # Setup temporary directories for logs and testing files
        self.test_dir = tempfile.mkdtemp()
        self.temp_logs_path = os.path.join(self.test_dir, "test_rag_logs.jsonl")
        
        # Override the LOG_FILE_PATH in app to point to our test logs
        self.original_log_path = app.LOG_FILE_PATH
        app.LOG_FILE_PATH = self.temp_logs_path

    def tearDown(self):
        # Restore app settings and clean up
        app.LOG_FILE_PATH = self.original_log_path
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # R1: Ingestion
    @patch("pdfplumber.open")
    def test_r1_extraction_and_chunking_multiple_documents(self, mock_open):
        # Mock two documents
        mock_pdf1 = MagicMock()
        mock_pdf1.__enter__.return_value = mock_pdf1
        mock_page1_1 = MagicMock()
        mock_page1_1.extract_text.return_value = "Sentence retrieval is semantic. FAISS is standard."
        mock_page1_2 = MagicMock()
        mock_page1_2.extract_text.return_value = "Chunk overlap helps context continuity."
        mock_pdf1.pages = [mock_page1_1, mock_page1_2]
        
        mock_pdf2 = MagicMock()
        mock_pdf2.__enter__.return_value = mock_pdf2
        mock_page2_1 = MagicMock()
        mock_page2_1.extract_text.return_value = "Keyword matching is lexical. BM25 is okapi."
        mock_pdf2.pages = [mock_page2_1]

        def side_effect(filename, *args, **kwargs):
            if "doc1.pdf" in filename:
                return mock_pdf1
            elif "doc2.pdf" in filename:
                return mock_pdf2
            return MagicMock()
            
        mock_open.side_effect = side_effect

        # 1. Ingest doc1.pdf
        res1 = self.engine.ingest_file("doc1.pdf")
        self.assertTrue(res1)
        self.assertIn("doc1.pdf", self.engine.processed_files)
        self.assertEqual(self.engine.doc_metadata["doc1.pdf"]["pages"], 2)
        
        # Verify chunks are dictionaries and have correct metadata
        self.assertTrue(len(self.engine.chunks) > 0)
        for chunk in self.engine.chunks:
            self.assertEqual(chunk["source"], "doc1.pdf")
            self.assertIn(chunk["page"], [1, 2])
            self.assertIn("text", chunk)

        # 2. Ingest doc2.pdf (incremental)
        chunks_count_before = len(self.engine.chunks)
        res2 = self.engine.ingest_file("doc2.pdf")
        self.assertTrue(res2)
        self.assertIn("doc2.pdf", self.engine.processed_files)
        self.assertEqual(self.engine.doc_metadata["doc2.pdf"]["pages"], 1)
        
        chunks_count_after = len(self.engine.chunks)
        self.assertTrue(chunks_count_after > chunks_count_before)
        
        # Verify doc2 chunks
        doc2_chunks = [c for c in self.engine.chunks if c["source"] == "doc2.pdf"]
        self.assertEqual(len(doc2_chunks), 1)
        self.assertEqual(doc2_chunks[0]["page"], 1)
        self.assertEqual(doc2_chunks[0]["text"], "Keyword matching is lexical. BM25 is okapi.")

        # 3. Duplicate upload skipping
        res_dup = self.engine.ingest_file("doc1.pdf")
        self.assertFalse(res_dup)
        self.assertEqual(len(self.engine.chunks), chunks_count_after)

    # R2: Attribution
    @patch("pdfplumber.open")
    def test_r2_direct_retrieval_and_source_attribution(self, mock_open):
        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value = mock_pdf
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "The project name is Quick Hopper RAG system."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "It uses reciprocal rank fusion for hybrid indexing."
        mock_pdf.pages = [mock_page1, mock_page2]
        
        mock_open.return_value = mock_pdf
        
        self.engine.ingest_file("quick_hopper.pdf")
        
        # Search query
        top_chunks, ranked = self.engine.retrieve_hybrid("Quick Hopper RAG")
        self.assertTrue(len(top_chunks) > 0)
        
        # Verify fields and sources mapping
        first_match = top_chunks[0]
        self.assertEqual(first_match["source"], "quick_hopper.pdf")
        self.assertIn("page", first_match)
        self.assertIn("text", first_match)
        self.assertIn("rrf_score", first_match)

    def test_r2_default_chunks_format(self):
        # Clear engine (so it falls back to defaults)
        self.engine.reset()
        
        # Verify DEFAULT_CHUNKS are structured as dictionaries with text, source, page
        for chunk in DEFAULT_CHUNKS:
            self.assertIn("text", chunk)
            self.assertEqual(chunk["source"], "System Default Knowledge Base")
            self.assertEqual(chunk["page"], 1)
            
        # Retrieval with default chunks fallback
        top_chunks, ranked = self.engine.retrieve_hybrid("FAISS")
        self.assertTrue(len(top_chunks) > 0)
        self.assertEqual(top_chunks[0]["source"], "System Default Knowledge Base")
        self.assertEqual(top_chunks[0]["page"], 1)

    # R3 & R4: Feedback logging and retrospective updates
    def test_r3_r4_feedback_logging_and_jsonl_updates(self):
        interaction_id = str(uuid.uuid4())
        query = "How is conversational memory handled?"
        response = "Using FAISS memory store."
        sources = ["docA.pdf (Page 1)", "docB.pdf (Page 2)"]
        
        # 1. Append log entry
        app.append_log_entry(interaction_id, query, response, sources)
        
        # Verify file creation and content
        self.assertTrue(os.path.exists(self.temp_logs_path))
        with open(self.temp_logs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        
        log_data = json.loads(lines[0])
        self.assertEqual(log_data["interaction_id"], interaction_id)
        self.assertEqual(log_data["query"], query)
        self.assertEqual(log_data["response"], response)
        self.assertEqual(log_data["sources"], sources)
        self.assertEqual(log_data["feedback"], "None")
        self.assertIn("timestamp", log_data)
        
        # 2. Retrospective update to thumbs_up
        app.update_log_feedback(interaction_id, "thumbs_up")
        
        with open(self.temp_logs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        log_data_up = json.loads(lines[0])
        self.assertEqual(log_data_up["feedback"], "thumbs_up")
        
        # 3. Consecutive update to thumbs_down
        app.update_log_feedback(interaction_id, "thumbs_down")
        
        with open(self.temp_logs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        log_data_down = json.loads(lines[0])
        self.assertEqual(log_data_down["feedback"], "thumbs_down")

    # Handling of empty/scanned PDFs or mixed invalid uploads
    @patch("pdfplumber.open")
    def test_handling_empty_scanned_pdf(self, mock_open):
        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value = mock_pdf
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "" # No text
        mock_pdf.pages = [mock_page]
        
        mock_open.return_value = mock_pdf
        
        # Verify it raises ValueError
        with self.assertRaises(ValueError):
            self.engine.ingest_file("empty.pdf")

    @patch("pdfplumber.open")
    def test_handling_mixed_invalid_uploads(self, mock_open):
        # Simulate multiple file ingestion in app.py's process_pdf
        mock_pdf_valid = MagicMock()
        mock_pdf_valid.__enter__.return_value = mock_pdf_valid
        mock_page_valid = MagicMock()
        mock_page_valid.extract_text.return_value = "Valid text for RAG."
        mock_pdf_valid.pages = [mock_page_valid]
        
        mock_pdf_empty = MagicMock()
        mock_pdf_empty.__enter__.return_value = mock_pdf_empty
        mock_page_empty = MagicMock()
        mock_page_empty.extract_text.return_value = "" # Invalid empty page
        mock_pdf_empty.pages = [mock_page_empty]
        
        def side_effect(filename, *args, **kwargs):
            if "valid.pdf" in filename:
                return mock_pdf_valid
            elif "empty.pdf" in filename:
                return mock_pdf_empty
            return MagicMock()
            
        mock_open.side_effect = side_effect
        
        # Test app.process_pdf with both files
        app.state.reset()
        
        file1 = MagicMock()
        file1.name = "valid.pdf"
        file2 = MagicMock()
        file2.name = "empty.pdf"
        
        status_md, status_text_val = app.process_pdf([file1, file2])
        
        # Verify valid document was processed, empty raised error and was skipped
        self.assertIn("valid.pdf", app.state.engine.processed_files)
        self.assertNotIn("empty.pdf", app.state.engine.processed_files)
        self.assertEqual(len(app.state.engine.chunks), 1)
        self.assertIn("Errors occurred during ingestion", status_text_val)
        self.assertIn("empty.pdf", status_text_val)

    def test_generate_response_handles_none_history(self):
        # Reset state
        app.state.reset()
        # Call generate_response with history=None
        history, query_out, docs_md, memories_md = app.generate_response(
            query="test query", 
            history=None, 
            personal_context="", 
            system_context=""
        )
        # Verify history was initialized to a list and the query/response was appended
        self.assertIsNotNone(history)
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 2 if app.is_gradio_v5_or_v6 else 1)

def run_verification_suite():
    print("======================================================================")
    print("      STARTING HYBRID RAG E2E TEST VERIFICATION SUITE                 ")
    print("======================================================================")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHybridRAGActual)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("======================================================================")
    print("                           SUMMARY                                    ")
    print("======================================================================")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Passed:           {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed:           {len(result.failures)}")
    print(f"Errors:           {len(result.errors)}")
    print("======================================================================")
    
    if len(result.failures) > 0 or len(result.errors) > 0:
        import sys
        sys.exit(1)
    else:
        print("All tests passed successfully!")

if __name__ == "__main__":
    run_verification_suite()
