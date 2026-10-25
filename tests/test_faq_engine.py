"""
Unit and Integration Tests for NLP FAQ Matching Engine.
Covers preprocessing, TF-IDF vectorization, cosine similarity,
confidence thresholds, and fallback behavior.
"""

import json
import os
import tempfile
import unittest

from chatbot.faq_engine import FAQEngine, DEFAULT_FALLBACK_MESSAGE
from chatbot.preprocessing import TextPreprocessor, preprocess_text


class TestTextPreprocessing(unittest.TestCase):
    """Tests for text normalization and tokenization."""

    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_lowercase_and_whitespace(self):
        raw = "   HOW Can   I Pay  MY   Tuition   FEES?   "
        cleaned = self.preprocessor.preprocess(raw)
        self.assertNotIn("HOW", cleaned)
        self.assertNotIn("  ", cleaned)
        self.assertIn("pay", cleaned)
        self.assertIn("tuition", cleaned)

    def test_punctuation_removal(self):
        raw = "Where is the library??? !!! #campus @books $$"
        cleaned = self.preprocessor.preprocess(raw)
        self.assertNotIn("?", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("#", cleaned)
        self.assertIn("library", cleaned)

    def test_contractions_expansion(self):
        raw = "can't won't i'm"
        expanded = self.preprocessor.expand_contractions(raw)
        self.assertIn("cannot", expanded)
        self.assertIn("will not", expanded)
        self.assertIn("i am", expanded)

    def test_empty_and_none_input(self):
        self.assertEqual(self.preprocessor.preprocess(""), "")
        self.assertEqual(self.preprocessor.preprocess("    "), "")
        self.assertEqual(self.preprocessor.preprocess(None), "")
        self.assertEqual(preprocess_text(""), "")

    def test_negation_preservation(self):
        # Negations like 'not', 'no', 'cannot' should NOT be stripped as stopwords
        raw = "I do not want to apply"
        cleaned = self.preprocessor.preprocess(raw)
        self.assertIn("not", cleaned)


class TestFAQEngine(unittest.TestCase):
    """Tests for FAQEngine retrieval, cosine similarity, and thresholding."""

    @classmethod
    def setUpClass(cls):
        # Ensure faqs.json exists
        cls.faq_file = "faqs.json"
        if not os.path.exists(cls.faq_file):
            raise FileNotFoundError("faqs.json is missing for test suite.")
        cls.engine = FAQEngine(cls.faq_file, similarity_threshold=0.25)

    def test_exact_question_match(self):
        query = "How can I apply for admission?"
        res = self.engine.find_best_match(query)
        self.assertTrue(res["matched"])
        self.assertAlmostEqual(res["confidence"], 1.0, delta=0.05)
        self.assertEqual(res["category"], "Admissions")
        self.assertIn("admissions portal", res["answer"])

    def test_paraphrased_question(self):
        query = "what is the cost of tuition fees?"
        res = self.engine.find_best_match(query)
        self.assertTrue(res["matched"])
        self.assertGreaterEqual(res["confidence"], 0.25)
        self.assertEqual(res["category"], "Tuition Fees")

    def test_different_capitalization(self):
        query = "HOW CAN I PAY MY TUITION FEES?"
        res = self.engine.find_best_match(query)
        self.assertTrue(res["matched"])
        self.assertAlmostEqual(res["confidence"], 1.0, delta=0.05)
        self.assertEqual(res["category"], "Tuition Fees")

    def test_extra_whitespace_and_punctuation(self):
        query = "   Where   is the   emergency health center located on campus??? !!!   "
        res = self.engine.find_best_match(query)
        self.assertTrue(res["matched"])
        self.assertAlmostEqual(res["confidence"], 1.0, delta=0.05)
        self.assertEqual(res["category"], "Contact Information")

    def test_unrelated_question_triggers_fallback(self):
        unrelated_queries = [
            "how do I make a chocolate cake?",
            "what is the distance from Earth to Mars?",
            "who won the world cup in 1998?",
            "recipe for Italian pizza dough"
        ]
        for q in unrelated_queries:
            res = self.engine.find_best_match(q)
            self.assertFalse(res["matched"], f"Query '{q}' should not match any FAQ")
            self.assertEqual(res["answer"], DEFAULT_FALLBACK_MESSAGE)
            self.assertIsNone(res["matched_question"])
            self.assertIsNone(res["category"])

    def test_empty_query(self):
        empty_queries = ["", "   ", "\t\n", None]
        for q in empty_queries:
            res = self.engine.find_best_match(q)
            self.assertFalse(res["matched"])
            self.assertEqual(res["confidence"], 0.0)

    def test_consecutive_queries(self):
        queries = [
            "What are the library hours?",
            "How do I apply for hostel room allocation?",
            "What is the college attendance policy?",
            "When are final semester examinations?"
        ]
        expected_categories = [
            "Library Hours",
            "Hostel Facilities",
            "Attendance Requirements",
            "Examination Dates"
        ]
        for q, expected_cat in zip(queries, expected_categories):
            res = self.engine.find_best_match(q)
            self.assertTrue(res["matched"], f"Query '{q}' should be matched")
            self.assertEqual(res["category"], expected_cat)

    def test_configurable_threshold(self):
        query = "library operating hours"
        # Normal threshold -> match
        lenient_engine = FAQEngine(self.faq_file, similarity_threshold=0.20)
        res_lenient = lenient_engine.find_best_match(query)
        self.assertTrue(res_lenient["matched"])

        # Extremely strict threshold -> fallback
        strict_engine = FAQEngine(self.faq_file, similarity_threshold=0.99)
        res_strict = strict_engine.find_best_match(query)
        self.assertFalse(res_strict["matched"])
        self.assertEqual(res_strict["answer"], DEFAULT_FALLBACK_MESSAGE)

    def test_malformed_faq_file_handling(self):
        # 1. Non-existent file
        with self.assertRaises(FileNotFoundError):
            FAQEngine("non_existent_file.json")

        # 2. Invalid JSON file
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
            tmp.write("{ invalid json")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                FAQEngine(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 3. Empty list FAQ file
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
            tmp.write("[]")
            empty_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                FAQEngine(empty_path)
        finally:
            if os.path.exists(empty_path):
                os.remove(empty_path)


if __name__ == "__main__":
    unittest.main()
