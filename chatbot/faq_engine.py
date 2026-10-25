"""
FAQ Matching Engine using TF-IDF and Cosine Similarity.

Loads FAQs, computes vector embeddings for questions using scikit-learn's
TfidfVectorizer, and calculates cosine similarity to match user queries
against predefined answers.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import TextPreprocessor, preprocess_text

DEFAULT_FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find a reliable answer to that question. "
    "Please try rephrasing it or contact the appropriate college department."
)


class FAQEngine:
    """
    NLP-based FAQ retrieval engine using TF-IDF and Cosine Similarity.
    """

    def __init__(
        self,
        faq_file_path: str = "faqs.json",
        similarity_threshold: float = 0.25,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        """
        Initialize the FAQ engine.

        :param faq_file_path: Path to the JSON file containing FAQ records.
        :param similarity_threshold: Minimum cosine similarity score (0.0 to 1.0)
                                     required to consider a match valid.
        :param preprocessor: TextPreprocessor instance (uses default if None).
        """
        self.faq_file_path = faq_file_path
        self.similarity_threshold = float(similarity_threshold)
        self.preprocessor = preprocessor or TextPreprocessor()

        self.faqs: List[Dict[str, Any]] = []
        self.raw_questions: List[str] = []
        self.cleaned_questions: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        # Load and fit index
        self.load_and_index()

    def load_and_index(self) -> None:
        """
        Loads FAQs from the JSON file, preprocesses questions,
        and fits the TF-IDF vectorizer.
        """
        if not os.path.exists(self.faq_file_path):
            raise FileNotFoundError(f"FAQ file not found at: {self.faq_file_path}")

        with open(self.faq_file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in FAQ file '{self.faq_file_path}': {e}")

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("FAQ file must contain a non-empty list of FAQ objects.")

        self.faqs = data
        self.raw_questions = [item.get("question", "") for item in self.faqs]

        # Preprocess each question using our NLP pipeline
        self.cleaned_questions = [
            self.preprocessor.preprocess(q) for q in self.raw_questions
        ]

        # Use unigrams and bigrams with sublinear TF scaling
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            min_df=1,
        )

        # Fit and transform the corpus
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_questions)

    def find_best_match(self, query: str) -> Dict[str, Any]:
        """
        Processes a user question, computes cosine similarity with stored FAQs,
        and returns the top match or fallback response.

        :param query: Raw question string from the user.
        :return: Dict containing answer, matched_question, category,
                 confidence score, and matched flag.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return {
                "answer": "Please ask a question so I can assist you.",
                "matched_question": None,
                "category": None,
                "confidence": 0.0,
                "matched": False,
            }

        # Check if index is properly populated
        if self.tfidf_matrix is None or self.vectorizer is None or len(self.faqs) == 0:
            return {
                "answer": DEFAULT_FALLBACK_MESSAGE,
                "matched_question": None,
                "category": None,
                "confidence": 0.0,
                "matched": False,
            }

        # Step 1: Preprocess user query
        cleaned_query = self.preprocessor.preprocess(query)

        # If preprocessing left nothing (e.g., pure punctuation), fallback
        if not cleaned_query:
            # Try raw query lowercased
            cleaned_query = query.lower().strip()

        # Step 2: Vectorize user query using the fitted TF-IDF vectorizer
        query_vector = self.vectorizer.transform([cleaned_query])

        # Step 3: Compute cosine similarity between query and all stored FAQs
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Step 4: Identify best match index and top score
        best_index = int(np.argmax(similarity_scores))
        best_score = float(similarity_scores[best_index])

        # Round confidence score for presentation
        confidence_rounded = round(best_score, 4)

        # Step 5: Compare against similarity threshold
        if best_score >= self.similarity_threshold:
            matched_faq = self.faqs[best_index]
            return {
                "answer": matched_faq.get("answer", ""),
                "matched_question": matched_faq.get("question", ""),
                "category": matched_faq.get("category", "General"),
                "confidence": confidence_rounded,
                "matched": True,
                "faq_id": matched_faq.get("id"),
            }
        else:
            return {
                "answer": DEFAULT_FALLBACK_MESSAGE,
                "matched_question": None,
                "category": None,
                "confidence": confidence_rounded,
                "matched": False,
            }

    def get_categories(self) -> List[str]:
        """Returns a sorted list of unique FAQ categories."""
        categories = set(item.get("category", "General") for item in self.faqs)
        return sorted(list(categories))

    def get_faqs_summary(self) -> List[Dict[str, Any]]:
        """
        Returns a simplified list of FAQs (id, category, question)
        useful for populating suggested questions and topic lists.
        """
        return [
            {
                "id": item.get("id"),
                "category": item.get("category", "General"),
                "question": item.get("question", ""),
            }
            for item in self.faqs
        ]

    def get_suggested_questions(self, limit: int = 6) -> List[Dict[str, str]]:
        """
        Returns a curated selection of suggested questions from distinct categories.
        """
        suggestions = []
        seen_categories = set()

        # First pick one question per unique category up to limit
        for item in self.faqs:
            cat = item.get("category", "General")
            if cat not in seen_categories:
                seen_categories.add(cat)
                suggestions.append({
                    "question": item.get("question"),
                    "category": cat
                })
            if len(suggestions) >= limit:
                break

        # Fill up if fewer categories than limit
        if len(suggestions) < limit:
            for item in self.faqs:
                if len(suggestions) >= limit:
                    break
                q = item.get("question")
                if not any(s["question"] == q for s in suggestions):
                    suggestions.append({
                        "question": q,
                        "category": item.get("category", "General")
                    })

        return suggestions
