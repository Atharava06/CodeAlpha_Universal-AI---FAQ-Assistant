"""
Text Preprocessing Module for FAQ Chatbot.

Provides robust NLP cleaning and normalization:
- Lowercasing and whitespace normalization
- Contraction expansion
- Punctuation removal
- NLTK tokenization with fallback to regex
- Stopwords filtering with negation/query preservation
- Lemmatization with WordNet (with fallback)
- Safe offline-ready error handling
"""

import re
import string
from typing import List, Optional, Set

# Attempt to import NLTK and optional components
_NLTK_AVAILABLE = False
_STOPWORDS_SET: Set[str] = set()
_LEMMATIZER = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


def ensure_nltk_resources() -> bool:
    """
    Safely initialize required NLTK corpora.
    Returns True if NLTK resources are available, False otherwise.
    """
    global _STOPWORDS_SET, _LEMMATIZER
    if not _NLTK_AVAILABLE:
        return False

    # Check and initialize stopwords
    try:
        from nltk.corpus import stopwords
        _STOPWORDS_SET = set(stopwords.words("english"))
    except (LookupError, AttributeError):
        try:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            _STOPWORDS_SET = set(stopwords.words("english"))
        except Exception:
            _STOPWORDS_SET = FALLBACK_STOPWORDS

    # Check and initialize lemmatizer
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        lem.lemmatize("students")  # quick verify
        _LEMMATIZER = lem
    except (LookupError, AttributeError):
        try:
            nltk.download("wordnet", quiet=True)
            from nltk.stem import WordNetLemmatizer
            _LEMMATIZER = WordNetLemmatizer()
        except Exception:
            _LEMMATIZER = None

    # Check tokenizer resource
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except (LookupError, AttributeError):
        try:
            nltk.data.find("tokenizers/punkt")
        except (LookupError, AttributeError):
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass

    return True


# Fallback stop words when NLTK stopwords are unavailable
FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}

# Essential negations to ALWAYS preserve so intent is not flipped
PRESERVED_WORDS = {
    "not", "no", "never", "nor", "neither", "cannot", "without"
}

# Contraction mapping
CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bdon't\b": "do not",
    r"\bdn't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bi'm\b": "i am",
    r"\byou're\b": "you are",
    r"\bthey're\b": "they are",
    r"\bwe're\b": "we are",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bwhat's\b": "what is",
    r"\bwhere's\b": "where is",
    r"\bwho's\b": "who is",
    r"\bhow's\b": "how is",
    r"\bthere's\b": "there is",
    r"\blet's\b": "let us",
    r"\bi've\b": "i have",
    r"\byou've\b": "you have",
    r"\bwe've\b": "we have",
    r"\bi'll\b": "i will",
    r"\byou'll\b": "you will",
    r"\bhe'll\b": "he will",
    r"\bshe'll\b": "she will",
    r"\bthey'll\b": "they will",
}


class TextPreprocessor:
    """
    Configurable text preprocessor for FAQ queries and documents.
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        preserve_negations: bool = True,
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.preserve_negations = preserve_negations

        # Initialize NLTK resources
        ensure_nltk_resources()

        # Build active stopwords set
        base_stopwords = _STOPWORDS_SET if _STOPWORDS_SET else FALLBACK_STOPWORDS
        if self.preserve_negations:
            self.stopwords = base_stopwords - PRESERVED_WORDS
        else:
            self.stopwords = set(base_stopwords)

    def expand_contractions(self, text: str) -> str:
        """Expands common English contractions."""
        lower = text.lower()
        for pattern, replacement in CONTRACTIONS.items():
            lower = re.sub(pattern, replacement, lower)
        return lower

    def clean_characters(self, text: str) -> str:
        """Removes punctuation and keeps alphanumeric tokens and spaces."""
        # Replace punctuation characters with whitespace
        cleaned = re.sub(r"[^\w\s]", " ", text)
        # Collapse multiple whitespace characters
        return re.sub(r"\s+", " ", cleaned).strip()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into words using NLTK word_tokenize with regex fallback.
        """
        if not text:
            return []

        if _NLTK_AVAILABLE:
            try:
                from nltk.tokenize import word_tokenize
                return word_tokenize(text)
            except Exception:
                pass

        # Robust regex fallback
        return re.findall(r"\b\w+\b", text)

    def lemmatize_token(self, token: str) -> str:
        """Lemmatizes a single token if lemmatizer is available."""
        global _LEMMATIZER
        if not self.lemmatize:
            return token

        if _LEMMATIZER is None and _NLTK_AVAILABLE:
            try:
                from nltk.stem import WordNetLemmatizer
                _LEMMATIZER = WordNetLemmatizer()
            except Exception:
                _LEMMATIZER = None

        if _LEMMATIZER is not None:
            try:
                # Lemmatize both as noun and verb for best normalization
                lemmatized_noun = _LEMMATIZER.lemmatize(token, pos="n")
                return _LEMMATIZER.lemmatize(lemmatized_noun, pos="v")
            except Exception:
                return token

        return token

    def preprocess(self, text: Optional[str]) -> str:
        """
        Full preprocessing pipeline:
        1. Handle empty / null inputs
        2. Lowercase and expand contractions
        3. Remove punctuation
        4. Tokenize
        5. Filter stopwords (preserving key words)
        6. Lemmatize
        7. Rejoin into clean string
        """
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        text = text.strip()
        if not text:
            return ""

        # Step 1: Contractions & lowercasing
        text = self.expand_contractions(text)

        # Step 2: Clean special characters
        text = self.clean_characters(text)
        if not text:
            return ""

        # Step 3: Tokenize
        tokens = self.tokenize(text)

        # Step 4: Stopwords filter & lemmatization
        processed_tokens: List[str] = []
        for token in tokens:
            token = token.lower()
            if not token:
                continue

            if self.remove_stopwords and token in self.stopwords:
                continue

            token = self.lemmatize_token(token)
            processed_tokens.append(token)

        # If all tokens were filtered out (e.g. "what is"), return original cleaned tokens
        if not processed_tokens and tokens:
            processed_tokens = tokens

        return " ".join(processed_tokens)


# Singleton preprocessor for application-wide convenience
_DEFAULT_PREPROCESSOR = TextPreprocessor()


def preprocess_text(text: Optional[str]) -> str:
    """
    Convenience function to preprocess a text string using default settings.
    """
    return _DEFAULT_PREPROCESSOR.preprocess(text)
