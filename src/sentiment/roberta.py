
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
    result = clf(text, truncation=True, max_length=512)[0]
    elapsed = time.perf_counter() - start

    label = _LABEL_MAP.get(result["label"], result["label"].lower())
    return {
        "text": text,
        "sentiment": label,
        "confidence": float(result["score"]),
        "time_seconds": elapsed,
    }


def evaluate_roberta_on_test(
    csv_path: str | None = None,
    sample_size: int | None = None,
    test_size: float = 0.25,
    seed: int = 42,
    eval_limit: int = 300,
):

    import random
    import time

    from sentiment.data import train_test_texts
    from sentiment.evaluate import evaluate

    _, X_test_raw, _, y_test = train_test_texts(
        csv_path, test_size=test_size, seed=seed, sample_size=sample_size
    )

    if eval_limit and len(X_test_raw) > eval_limit:
        rng = random.Random(seed)
        indices = rng.sample(range(len(X_test_raw)), eval_limit)
        X_test_raw = [X_test_raw[i] for i in indices]
        y_test = [y_test[i] for i in indices]

    clf = _get_pipeline()
    start = time.perf_counter()
    raw_results = clf(X_test_raw, batch_size=16, truncation=True, max_length=512)
    elapsed = time.perf_counter() - start

    y_pred = [_LABEL_MAP.get(r["label"], r["label"].lower()) for r in raw_results]

    result = evaluate("roberta", y_test, y_pred, X_test_raw)
    avg_ms_per_example = (elapsed / len(X_test_raw)) * 1000 if X_test_raw else 0.0

    return result, avg_ms_per_example, len(X_test_raw)