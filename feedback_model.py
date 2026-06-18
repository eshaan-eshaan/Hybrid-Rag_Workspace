"""
feedback_model.py — CSV-Trained Local Feedback Analyzer (Pharma Edition)

Trains on pharma Q&A query+feedback pairs from a CSV dataset.
Uses semantic similarity (all-MiniLM-L6-v2) + weak-spot type classification
to analyze why an answer would receive negative feedback, without API calls.

CSV expected columns:
  id, company, product, section, query, intent,
  expected_feedback, feedback, weak_spot_type
"""

import csv
import os
import re
import numpy as np
import faiss
from collections import Counter
from sentence_transformers import SentenceTransformer


# ── Pharma Weak-Spot Type Definitions ────────────────────────────────────────
# Human-readable descriptions and actionable guidance for each weak-spot type.
WEAK_SPOT_LABELS = {
    "pitch_clarity": "Unclear Doctor Pitch Messaging",
    "failure_mode": "Overstated Claims / Failure Risk",
    "evidence_missing_or_overstated": "Evidence Missing or Overstated",
    "safety_gap": "Safety Information Gap",
    "unsafe_comparison": "Unsafe Competitor Comparison",
    "analysis_signal": "Analysis / Meta-Pattern Signal",
    "policy_violation": "Regulatory Policy Violation",
    "none": "No Specific Weak Spot",
}

WEAK_SPOT_ROOT_CAUSES = {
    "pitch_clarity": (
        "The response lacks clear, doctor-facing messaging. "
        "The pitch may be vague, off-brand, or missing the clinical value proposition."
    ),
    "failure_mode": (
        "The response overstates product claims or misses known failure modes. "
        "This could lead to misleading field communications."
    ),
    "evidence_missing_or_overstated": (
        "The response either lacks supporting clinical evidence or overstates "
        "the strength of available evidence (e.g., citing Phase II data as definitive)."
    ),
    "safety_gap": (
        "The response omits critical safety information, contraindications, "
        "or adverse reaction warnings required for compliant communication."
    ),
    "unsafe_comparison": (
        "The response makes direct or implied comparative claims against "
        "competitor products without proper qualifying language or evidence basis."
    ),
    "analysis_signal": (
        "The query tests the system's ability to handle meta-analytical or "
        "cross-referencing patterns. The response may lack analytical depth."
    ),
    "policy_violation": (
        "The response contains language or claims that could violate "
        "regulatory policies (e.g., off-label promotion, unapproved indications)."
    ),
    "none": (
        "No specific weak spot was identified in the training data, "
        "but the response may still be incomplete or need refinement."
    ),
}

WEAK_SPOT_SUGGESTIONS = {
    "pitch_clarity": (
        "Rewrite the doctor-facing pitch to be concise, clinically grounded, "
        "and aligned with the product's approved indications. Include key "
        "differentiators and a clear call-to-action."
    ),
    "failure_mode": (
        "Add disclaimers for known limitations. Ensure claims are grounded in "
        "approved labeling. Include failure mode documentation and risk language."
    ),
    "evidence_missing_or_overstated": (
        "Cite specific clinical trial data with proper context (Phase, N, endpoints). "
        "Avoid superlatives unless supported by head-to-head RCT data."
    ),
    "safety_gap": (
        "Add a dedicated safety section covering: contraindications, common adverse "
        "effects, drug interactions, and black-box warnings if applicable."
    ),
    "unsafe_comparison": (
        "Replace direct competitor comparisons with factual, evidence-based "
        "differentiation. Use language like 'may offer' instead of 'is better than'. "
        "Include proper disclaimers."
    ),
    "analysis_signal": (
        "Strengthen the document's analytical content. Add cross-references, "
        "summary tables, and pattern-based Q&A to support field reps."
    ),
    "policy_violation": (
        "Review the response against regulatory guidelines. Remove any off-label "
        "promotion, unapproved indication claims, or misleading comparative data. "
        "Engage medical-legal review."
    ),
    "none": (
        "Review the document section for completeness, accuracy, and clarity. "
        "Consider adding more detail or examples to improve answer quality."
    ),
}

# Sections that appear in the pharma CSV and their human-readable descriptions
SECTION_LABELS = {
    "doctor_pitch": "Doctor Pitch / Detailing",
    "negative_spots": "Known Weak Spots / Failure Modes",
    "evidence_story": "Evidence & Clinical Story",
    "safety": "Safety & Contraindications",
    "comparison_language": "Competitor Comparison Language",
    "meta_pattern": "Meta-Analysis & Patterns",
}


class FeedbackAnalyzerModel:
    """
    A local, zero-API feedback analyzer trained on a pharma CSV dataset.
    Uses semantic similarity search to find the closest known queries
    and weak-spot classifications to generate root-cause analysis.
    """

    def __init__(self, embed_model=None):
        """
        Args:
            embed_model: A pre-loaded SentenceTransformer model. If None,
                         one will be loaded (reuses the app's model to save RAM).
        """
        self.embed_model = embed_model
        self.trained = False

        # Training data
        self.entries = []          # list of dicts from CSV rows
        self.embeddings = None     # numpy array of embeddings
        self.index = None          # FAISS index
        self.positive_entries = [] # entries with thumbs up
        self.negative_entries = [] # entries with thumbs down

    def train(self, csv_path):
        """
        Load the pharma CSV, embed all queries, and build the FAISS index.

        Args:
            csv_path: Path to the CSV file with columns:
                      id, company, product, section, query, intent,
                      expected_feedback, feedback, weak_spot_type
        """
        if self.embed_model is None:
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.entries = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feedback_raw = row.get("feedback", "").strip()
                is_neg = feedback_raw in ("\U0001f44e", "👎", "thumbs_down", "negative")

                # Also check expected_feedback as backup signal
                expected = row.get("expected_feedback", "").strip()
                if expected in ("\U0001f44e", "👎", "thumbs_down", "negative"):
                    is_neg = True

                entry = {
                    "id": int(row.get("id", 0)),
                    "company": row.get("company", "").strip(),
                    "product": row.get("product", "").strip(),
                    "section": row.get("section", "").strip(),
                    "query": row.get("query", "").strip(),
                    "intent": row.get("intent", "").strip(),
                    "feedback": feedback_raw,
                    "weak_spot_type": row.get("weak_spot_type", "none").strip(),
                    "is_negative": is_neg,
                }
                self.entries.append(entry)

        self.positive_entries = [e for e in self.entries if not e["is_negative"]]
        self.negative_entries = [e for e in self.entries if e["is_negative"]]

        # Embed queries for similarity search
        query_texts = [e["query"] for e in self.entries]
        self.embeddings = self.embed_model.encode(query_texts, normalize_embeddings=True)
        self.embeddings = np.array(self.embeddings, dtype="float32")

        # Build FAISS index (cosine similarity via inner product on normalized vectors)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        self.trained = True

        # Build weak-spot distribution stats
        ws_dist = Counter(e["weak_spot_type"] for e in self.negative_entries)
        return {
            "total": len(self.entries),
            "positive": len(self.positive_entries),
            "negative": len(self.negative_entries),
            "embedding_dim": dim,
            "weak_spot_distribution": dict(ws_dist),
        }

    def _find_similar(self, query_text, top_k=5):
        """Find the top-K most similar entries from the training set."""
        query_vec = self.embed_model.encode([query_text], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")

        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self.entries[idx]
            results.append({
                "entry": entry,
                "similarity": float(score),
            })
        return results

    def analyze(self, query, answer=""):
        """
        Analyze a single query (+ optional answer) and return structured analysis.

        Args:
            query: The user's question
            answer: The chatbot's response (optional — used for heuristic fallback)

        Returns:
            dict with keys: predicted_feedback, confidence, root_cause,
                           error_type, missing_info, suggestion, weak_spot_type,
                           section, company, product, similar_entries
        """
        q_lower = query.lower()
        is_legacy_chemistry = (
            "ncert" in q_lower
            or "chemistry" in q_lower
            or "colligative" in q_lower
            or "molarity" in q_lower
            or "molality" in q_lower
            or "solute" in q_lower
            or "solvent" in q_lower
        )
        if is_legacy_chemistry:
            return {
                "predicted_feedback": "⚠️ Disabled",
                "confidence": 0.0,
                "root_cause": "Query belongs to legacy NCERT Chemistry content. Feedback analysis is disabled for this domain.",
                "error_type": "Legacy Domain",
                "weak_spot_type": "none",
                "missing_info": "N/A",
                "suggestion": "Please ask pharma-related questions or upload pharma documents.",
                "similar_entries": [],
            }

        if not self.trained:
            return {
                "predicted_feedback": "unknown",
                "confidence": 0.0,
                "root_cause": "Model not trained. Call train() first.",
                "error_type": "N/A",
                "missing_info": "N/A",
                "suggestion": "Train the model with a CSV dataset.",
                "weak_spot_type": "none",
                "similar_entries": [],
            }

        similar = self._find_similar(query, top_k=5)

        if not similar:
            return self._fallback_analysis(query, answer)

        top_match = similar[0]
        top_entry = top_match["entry"]
        top_sim = top_match["similarity"]

        # Count how many of the top matches are negative vs positive
        neg_count = sum(1 for s in similar if s["entry"]["is_negative"])
        pos_count = len(similar) - neg_count

        # Weighted negative score
        neg_weight = sum(
            s["similarity"] for s in similar if s["entry"]["is_negative"]
        )
        pos_weight = sum(
            s["similarity"] for s in similar if not s["entry"]["is_negative"]
        )
        total_weight = neg_weight + pos_weight
        neg_ratio = neg_weight / total_weight if total_weight > 0 else 0.5

        # Decision threshold
        if top_sim > 0.85 and top_entry["is_negative"]:
            predicted = "\U0001f44e"
            confidence = min(top_sim, 0.99)
        elif neg_ratio > 0.5:
            predicted = "\U0001f44e"
            confidence = neg_ratio * top_sim
        elif top_sim > 0.85 and not top_entry["is_negative"]:
            predicted = "\U0001f44d"
            confidence = min(top_sim, 0.99)
        else:
            predicted = "\U0001f44d" if pos_count >= neg_count else "\U0001f44e"
            confidence = max(pos_weight, neg_weight) / total_weight if total_weight > 0 else 0.5

        # Build the analysis
        if predicted == "\U0001f44e":
            analysis = self._build_negative_analysis(query, answer, similar)
        else:
            analysis = self._build_positive_analysis(query, answer, similar)

        analysis["predicted_feedback"] = predicted
        analysis["confidence"] = round(confidence, 3)
        analysis["similar_entries"] = [
            {
                "query": s["entry"]["query"],
                "company": s["entry"]["company"],
                "product": s["entry"]["product"],
                "section": s["entry"]["section"],
                "weak_spot": s["entry"]["weak_spot_type"],
                "feedback": "\U0001f44d" if not s["entry"]["is_negative"] else "\U0001f44e",
                "similarity": round(s["similarity"], 3),
            }
            for s in similar[:3]
        ]
        return analysis

    def _build_negative_analysis(self, query, answer, similar):
        """Generate root cause analysis for a predicted-negative answer using weak spots."""
        # Collect weak-spot types from similar negative matches
        neg_similar = [s for s in similar if s["entry"]["is_negative"]]

        if neg_similar:
            # Find the dominant weak-spot type from similar negatives
            ws_types = [s["entry"]["weak_spot_type"] for s in neg_similar
                       if s["entry"]["weak_spot_type"] != "none"]

            if ws_types:
                dominant_ws = Counter(ws_types).most_common(1)[0][0]
                best_neg = neg_similar[0]

                label = WEAK_SPOT_LABELS.get(dominant_ws, dominant_ws)
                root_cause = WEAK_SPOT_ROOT_CAUSES.get(dominant_ws, "Unknown weak spot detected.")
                suggestion = WEAK_SPOT_SUGGESTIONS.get(dominant_ws, "Review and improve the document.")
                section = best_neg["entry"]["section"]
                section_label = SECTION_LABELS.get(section, section)

                return {
                    "root_cause": root_cause,
                    "error_type": label,
                    "weak_spot_type": dominant_ws,
                    "section": section,
                    "section_label": section_label,
                    "company": best_neg["entry"]["company"],
                    "product": best_neg["entry"]["product"],
                    "missing_info": (
                        f"The document section '{section_label}' needs improvement. "
                        f"Similar queries about "
                        f"{'product ' + best_neg['entry']['product'] if best_neg['entry']['product'] else 'this company'} "
                        f"received negative feedback due to '{label}' issues."
                    ),
                    "suggestion": suggestion,
                    "matched_from": best_neg["entry"]["query"],
                }

            # Negatives found but all have weak_spot_type = "none"
            best_neg = neg_similar[0]
            return {
                "root_cause": (
                    "Similar queries received negative feedback, but no specific "
                    "weak-spot pattern was identified in the training data."
                ),
                "error_type": "Unclassified Negative Feedback",
                "weak_spot_type": "none",
                "section": best_neg["entry"]["section"],
                "company": best_neg["entry"]["company"],
                "product": best_neg["entry"]["product"],
                "missing_info": "The document may be incomplete or inaccurate for this type of query.",
                "suggestion": (
                    "Review the document content related to this query. "
                    "Ensure completeness, accuracy, and compliance with guidelines."
                ),
            }

        # No negative similar entries — use heuristic
        return self._heuristic_negative(query, answer)

    def _heuristic_negative(self, query, answer):
        """Heuristic analysis with pharma-domain awareness."""
        q_lower = query.lower()
        a_lower = answer.lower() if answer else ""

        # Pharma-specific heuristic patterns
        if any(w in q_lower for w in ["compar", "versus", "vs", "better than", "competitor"]):
            return {
                "root_cause": WEAK_SPOT_ROOT_CAUSES["unsafe_comparison"],
                "error_type": WEAK_SPOT_LABELS["unsafe_comparison"],
                "weak_spot_type": "unsafe_comparison",
                "missing_info": "Comparative claims need evidence basis and qualifying language.",
                "suggestion": WEAK_SPOT_SUGGESTIONS["unsafe_comparison"],
            }

        if any(w in q_lower for w in ["safety", "adverse", "side effect", "contraindic", "warning"]):
            return {
                "root_cause": WEAK_SPOT_ROOT_CAUSES["safety_gap"],
                "error_type": WEAK_SPOT_LABELS["safety_gap"],
                "weak_spot_type": "safety_gap",
                "missing_info": "Safety and contraindication data may be incomplete.",
                "suggestion": WEAK_SPOT_SUGGESTIONS["safety_gap"],
            }

        if any(w in q_lower for w in ["evidence", "clinical", "trial", "study", "data", "proof"]):
            return {
                "root_cause": WEAK_SPOT_ROOT_CAUSES["evidence_missing_or_overstated"],
                "error_type": WEAK_SPOT_LABELS["evidence_missing_or_overstated"],
                "weak_spot_type": "evidence_missing_or_overstated",
                "missing_info": "Clinical evidence citations may be missing or overrepresented.",
                "suggestion": WEAK_SPOT_SUGGESTIONS["evidence_missing_or_overstated"],
            }

        if any(w in q_lower for w in ["pitch", "doctor", "rep ", "detailing", "message", "position"]):
            return {
                "root_cause": WEAK_SPOT_ROOT_CAUSES["pitch_clarity"],
                "error_type": WEAK_SPOT_LABELS["pitch_clarity"],
                "weak_spot_type": "pitch_clarity",
                "missing_info": "Doctor-facing messaging may lack clarity or clinical grounding.",
                "suggestion": WEAK_SPOT_SUGGESTIONS["pitch_clarity"],
            }

        if any(w in q_lower for w in ["overstate", "claim", "wrong", "fail", "risk", "mislead"]):
            return {
                "root_cause": WEAK_SPOT_ROOT_CAUSES["failure_mode"],
                "error_type": WEAK_SPOT_LABELS["failure_mode"],
                "weak_spot_type": "failure_mode",
                "missing_info": "The document may contain overstated claims or miss known failure modes.",
                "suggestion": WEAK_SPOT_SUGGESTIONS["failure_mode"],
            }

        if any(w in q_lower for w in ["approv", "regulat", "compliance", "off-label", "policy", "deploy"]):
            return {
                "root_cause": WEAK_SPOT_ROOT_CAUSES["policy_violation"],
                "error_type": WEAK_SPOT_LABELS["policy_violation"],
                "weak_spot_type": "policy_violation",
                "missing_info": "Regulatory compliance content may need review.",
                "suggestion": WEAK_SPOT_SUGGESTIONS["policy_violation"],
            }

        # Generic fallback
        if answer and len(answer.strip()) < 30:
            return {
                "root_cause": "The response is very short and likely incomplete.",
                "error_type": "Insufficient Detail",
                "weak_spot_type": "none",
                "missing_info": "A more detailed explanation with supporting data is needed.",
                "suggestion": "Expand the knowledge base with more detailed content for this query type.",
            }

        return {
            "root_cause": "The answer may be vague, inaccurate, or off-topic for this query.",
            "error_type": "Possible Inaccuracy",
            "weak_spot_type": "none",
            "missing_info": "More precise or structured information may be needed.",
            "suggestion": "Review the response for factual correctness and completeness.",
        }

    def _build_positive_analysis(self, query, answer, similar):
        """Analysis for a predicted-positive answer."""
        top = similar[0]["entry"] if similar else {}
        return {
            "root_cause": "The answer appears to align with known correct response patterns.",
            "error_type": "None — Answer Likely Correct",
            "weak_spot_type": top.get("weak_spot_type", "none"),
            "section": top.get("section", ""),
            "company": top.get("company", ""),
            "product": top.get("product", ""),
            "missing_info": "No missing information detected.",
            "suggestion": "The answer aligns with known correct responses in the training data.",
        }

    def _fallback_analysis(self, query, answer):
        """When no similar entries are found at all."""
        return {
            "predicted_feedback": "\u26a0\ufe0f Unknown",
            "confidence": 0.0,
            "root_cause": "No similar entries found in the training data.",
            "error_type": "Out of Domain",
            "weak_spot_type": "none",
            "missing_info": "This query is outside the scope of the training dataset.",
            "suggestion": "Consider adding relevant Q&A pairs to the training CSV to cover this topic.",
            "similar_entries": [],
        }

    def batch_analyze(self, text_block):
        """
        Analyze a block of text containing one or more Q&A pairs or freeform chunks.

        Expected format (flexible):
            Q: <question>
            A: <answer>

        Also accepts freeform text (treated as a single query to analyze).

        Returns:
            list of analysis dicts
        """
        if not self.trained:
            return [{"error": "Model not trained. Call train() first."}]

        pairs = self._parse_qa_pairs(text_block)
        results = []
        for q, a in pairs:
            result = self.analyze(q, a)
            result["input_query"] = q
            result["input_answer"] = a
            results.append(result)
        return results

    def _parse_qa_pairs(self, text):
        """
        Parse text into (query, answer) pairs. Supports multiple formats:
        - "Q: ... A: ..." blocks
        - "Query: ... Answer: ..." blocks
        - Plain text (treated as query with empty answer)
        """
        # Try structured Q/A parsing
        pattern = (
            r'(?:Q(?:uery|uestion)?)\\s*[:]\\s*(.*?)\\s*'
            r'(?:A(?:nswer)?|Response)\\s*[:]\\s*(.*?)'
            r'(?=(?:Q(?:uery|uestion)?)\\s*[:]|\\Z)'
        )
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            return [(q.strip(), a.strip()) for q, a in matches if q.strip()]

        # Fallback: treat each non-empty paragraph as a chunk to analyze
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        if not chunks:
            chunks = [text.strip()]

        results = []
        for chunk in chunks:
            lines = chunk.strip().split("\n")
            if len(lines) >= 2:
                results.append((lines[0].strip(), " ".join(lines[1:]).strip()))
            else:
                # Treat as a query (the model matches on query similarity)
                results.append((chunk.strip(), ""))
        return results

    def get_training_stats(self):
        """Return training data statistics as markdown for display."""
        if not self.trained:
            return "Model not trained yet."

        # Weak-spot distribution
        ws_counts = Counter(e["weak_spot_type"] for e in self.negative_entries)

        # Company distribution
        company_counts = Counter(e["company"] for e in self.entries)

        # Section distribution
        section_counts = Counter(
            e["section"] for e in self.entries
            if e["section"] in SECTION_LABELS
        )

        stats_md = "### \U0001f4ca Training Data Statistics\n\n"
        stats_md += "| Metric | Value |\n|--------|-------|\n"
        stats_md += f"| Total Training Samples | **{len(self.entries)}** |\n"
        stats_md += f"| Positive (\U0001f44d) | **{len(self.positive_entries)}** ({100*len(self.positive_entries)//len(self.entries)}%) |\n"
        stats_md += f"| Negative (\U0001f44e) | **{len(self.negative_entries)}** ({100*len(self.negative_entries)//len(self.entries)}%) |\n"
        stats_md += f"| Companies | **{len(company_counts)}** |\n"

        # Product count (excluding empty)
        products = set(e["product"] for e in self.entries if e["product"])
        stats_md += f"| Products | **{len(products)}** |\n\n"

        # Weak-spot breakdown
        if ws_counts:
            stats_md += "### \U0001f50c Weak Spot Distribution (Negative Feedback)\n\n"
            stats_md += "| Weak Spot Type | Count |\n|----------------|-------|\n"
            for ws, count in ws_counts.most_common():
                label = WEAK_SPOT_LABELS.get(ws, ws)
                stats_md += f"| {label} | **{count}** |\n"
            stats_md += "\n"

        # Company breakdown
        if company_counts:
            stats_md += "### \U0001f3e2 Company Distribution\n\n"
            stats_md += "| Company | Total | \U0001f44e Negative |\n|---------|-------|----------|\n"
            neg_by_company = Counter(e["company"] for e in self.negative_entries)
            for company, total in company_counts.most_common():
                neg = neg_by_company.get(company, 0)
                stats_md += f"| {company} | {total} | {neg} |\n"
            stats_md += "\n"

        # Section breakdown
        if section_counts:
            stats_md += "### \U0001f4d1 Section Distribution\n\n"
            stats_md += "| Section | Count |\n|---------|-------|\n"
            for sec, count in section_counts.most_common():
                label = SECTION_LABELS.get(sec, sec)
                stats_md += f"| {label} | {count} |\n"

        return stats_md

    def format_analysis_markdown(self, result):
        """Format a single analysis result as a markdown card."""
        pred = result.get("predicted_feedback", "?")
        conf = result.get("confidence", 0)
        conf_pct = int(conf * 100)

        if conf_pct >= 80:
            conf_color = "\U0001f7e2"
        elif conf_pct >= 50:
            conf_color = "\U0001f7e1"
        else:
            conf_color = "\U0001f534"

        md = "---\n\n"

        if result.get("input_query"):
            md += f"**\u2753 Query**: {result['input_query']}\n\n"
        if result.get("input_answer"):
            answer_preview = result["input_answer"]
            if len(answer_preview) > 300:
                answer_preview = answer_preview[:297] + "..."
            md += f"**\U0001f4ac Answer**: {answer_preview}\n\n"

        md += f"**Predicted Feedback**: {pred}  |  **Confidence**: {conf_color} {conf_pct}%\n\n"

        # Pharma-specific fields
        ws_type = result.get("weak_spot_type", "none")
        ws_label = WEAK_SPOT_LABELS.get(ws_type, ws_type)
        md += f"**\U0001f50d Weak Spot Type**: {ws_label}\n\n"

        if result.get("company"):
            md += f"**\U0001f3e2 Company**: {result['company']}"
            if result.get("product"):
                md += f"  |  **\U0001f48a Product**: {result['product']}"
            md += "\n\n"

        section = result.get("section", "")
        if section:
            section_label = SECTION_LABELS.get(section, section)
            md += f"**\U0001f4d1 Section**: {section_label}\n\n"

        md += f"**\U0001f4cb Root Cause**: {result.get('root_cause', 'N/A')}\n\n"
        md += f"**\U0001f4a1 Missing Information**: {result.get('missing_info', 'N/A')}\n\n"
        md += f"**\U0001f6e0\ufe0f Suggestion**: {result.get('suggestion', 'N/A')}\n\n"

        similar = result.get("similar_entries", [])
        if similar:
            md += "**\U0001f4ce Similar Training Examples**:\n\n"
            for s in similar:
                sim_pct = int(s["similarity"] * 100)
                ws = WEAK_SPOT_LABELS.get(s.get("weak_spot", "none"), s.get("weak_spot", ""))
                md += (
                    f"- {s['feedback']} *\"{s['query']}\"* "
                    f"({s.get('company', '')}"
                    f"{' / ' + s['product'] if s.get('product') else ''}) "
                    f"— {ws} ({sim_pct}% match)\n"
                )
            md += "\n"

        return md
