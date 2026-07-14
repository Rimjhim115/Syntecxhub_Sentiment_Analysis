
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABELS = ["negative", "neutral", "positive"]


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    f1_macro: float
    confusion: list = field(default_factory=list)
    misclassified: List[dict] = field(default_factory=list)


def evaluate(model_name: str, y_true: list, y_pred: list, texts: list, max_examples: int = 5) -> EvalResult:
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    misclassified = []
    for text, true_label, pred_label in zip(texts, y_true, y_pred):
        if true_label != pred_label:
            misclassified.append({"text": text, "true": true_label, "predicted": pred_label})
        if len(misclassified) >= max_examples:
            break

    return EvalResult(
        model_name=model_name,
        accuracy=accuracy,
        f1_macro=f1,
        confusion=cm,
        misclassified=misclassified,
    )


def print_comparison_table(results: List[EvalResult]) -> None:
    headers = ["Model", "Accuracy", "F1 (macro)"]
    widths = [22, 12, 12]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for r in results:
        row = [r.model_name, f"{r.accuracy:.3f}", f"{r.f1_macro:.3f}"]
        print("".join(c.ljust(w) for c, w in zip(row, widths)))


def print_confusion_matrix(result: EvalResult) -> None:
    print(f"\nConfusion matrix — {result.model_name} (rows=actual, cols=predicted)")
    header = "".ljust(10) + "".join(label.ljust(10) for label in LABELS)
    print(header)
    for label, row in zip(LABELS, result.confusion):
        print(label.ljust(10) + "".join(str(v).ljust(10) for v in row))


def print_misclassified(result: EvalResult) -> None:
    if not result.misclassified:
        print(f"\n{result.model_name}: no misclassified examples in the sample shown.")
        return
    print(f"\nSample misclassifications — {result.model_name}")
    for ex in result.misclassified:
        print(f"  actual={ex['true']:<9} predicted={ex['predicted']:<9} text=\"{ex['text'][:80]}\"")