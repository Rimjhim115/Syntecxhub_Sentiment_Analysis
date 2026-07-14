
from __future__ import annotations

_LABEL_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

_pipeline_cache = None


def _get_pipeline():
    global _pipeline_cache
    if _pipeline_cache is None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "RoBERTa comparison requires the optional dependencies. "
                "Install with: pip install -r requirements-transformers.txt"
            ) from exc
        _pipeline_cache = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            framework="pt",  # force PyTorch, skip TensorFlow entirely
            )
    return _pipeline_cache


def predict_with_roberta(text: str) -> dict:
    """Runs one prediction through the pretrained model. Slow on CPU by design --
    that latency is itself part of the accuracy-vs-speed tradeoff this project
    is meant to illustrate, not a bug to hide."""
    import time

    clf = _get_pipeline()
    start = time.perf_counter()
    result = clf(text)[0]
    elapsed = time.perf_counter() - start

    label = _LABEL_MAP.get(result["label"], result["label"].lower())
    return {
        "text": text,
        "sentiment": label,
        "confidence": float(result["score"]),
        "time_seconds": elapsed,
    }