
from __future__ import annotations

import html
import re

_HTML_TAG = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_MULTI_SPACE = re.compile(r"\s+")

# Small, deliberately conservative stopword list -- big enough to remove
# noise words, small enough to avoid needing an external download.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "to", "from", "in", "on", "as", "so", "than", "too", "very",
    "just", "my", "your", "its", "our", "their", "have", "has", "had",
    "do", "does", "did", "will", "would", "can", "could", "should",
}

# Negation words are intentionally excluded from STOPWORDS removal even
# though many generic stopword lists include them.
NEGATIONS = {"not", "no", "never", "n't", "cannot", "cant", "dont", "wont"}


def clean_text(text: str) -> str:
    """Lowercase, decode HTML entities, strip tags/punctuation, remove stopwords (keeping negations)."""
    text = html.unescape(text)  # &#34; -> ", &amp; -> &, etc. -- real Amazon review text has these
    text = text.lower()
    text = _HTML_TAG.sub(" ", text)
    text = text.replace("n't", " not")  # normalize contractions before stripping punctuation
    text = _NON_ALPHA.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()

    tokens = [
        tok for tok in text.split()
        if tok in NEGATIONS or tok not in STOPWORDS
    ]
    return " ".join(tokens)


def clean_batch(texts: list) -> list:
    return [clean_text(t) for t in texts]