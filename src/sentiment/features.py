
from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))


def build_count_vectorizer(max_features: int = 5000) -> CountVectorizer:
    return CountVectorizer(max_features=max_features, ngram_range=(1, 2))