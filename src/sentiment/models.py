
from __future__ import annotations

from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


class MajorityBaseline:
    """Always predicts whichever class was most frequent in training data."""

    def __init__(self):
        self.majority_label = None

    def fit(self, X, y):
        self.majority_label = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return [self.majority_label] * n


MODEL_REGISTRY = {
    "baseline": lambda: MajorityBaseline(),
    "naive_bayes": lambda: MultinomialNB(),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
}


def get_model(name: str):
    try:
        return MODEL_REGISTRY[name]()
    except KeyError as exc:
        valid = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{name}'. Valid options: {valid}") from exc