"""
Chatbot package initialization.
Provides the FAQEngine and NLP preprocessing utilities.
"""

from .preprocessing import TextPreprocessor, preprocess_text
from .faq_engine import FAQEngine
from .gemini_service import GeminiService

__all__ = ["TextPreprocessor", "preprocess_text", "FAQEngine", "GeminiService"]
