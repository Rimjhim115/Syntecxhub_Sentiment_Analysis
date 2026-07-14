
from __future__ import annotations

from pathlib import Path
from typing import List

import joblib

from sentiment.data import train_test_texts
from sentiment.evaluate import EvalResult, evaluate
from sentiment.features import build_tfidf_vectorizer
from sentiment.models import get_model
from sentiment.preprocess import clean_batch

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"

MODELS_TO_COMPARE = ["baseline", "naive_bayes", "logistic_regression"]


def run_training(csv_path: str | None = None, save_best: bool = True, sample_size: int | None = None) -> List[EvalResult]:
    X_train_raw, X_test_raw, y_train, y_test = train_test_texts(csv_path, sample_size=sample_size)
    X_train_clean = clean_batch(X_train_raw)
    X_test_clean = clean_batch(X_test_raw)

    vectorizer = build_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(X_train_clean)
    X_test = vectorizer.transform(X_test_clean)

    results: List[EvalResult] = []
    trained_models = {}

    for name in MODELS_TO_COMPARE:
        model = get_model(name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        result = evaluate(name, y_test, list(preds), X_test_raw)
        results.append(result)
        trained_models[name] = model

    if save_best:
        # Pick the best *real* model (skip the baseline on purpose --
        # saving "always guess positive" as the production model would
        # defeat the point of having a baseline at all).
        real_results = [r for r in results if r.model_name != "baseline"]
        best = max(real_results, key=lambda r: r.f1_macro)
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        joblib.dump(trained_models[best.model_name], MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print(f"\nSaved best model ('{best.model_name}') to {MODEL_PATH}")

    return results


def load_saved_model():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python main.py train` first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_text(text: str) -> dict:
    model, vectorizer = load_saved_model()
    cleaned = clean_batch([text])
    X = vectorizer.transform(cleaned)
    prediction = model.predict(X)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(max(proba))

    return {"text": text, "sentiment": prediction, "confidence": confidence}