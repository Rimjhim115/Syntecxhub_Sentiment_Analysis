
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

POSITIVE_WORDS = {
    "great", "love", "loved", "amazing", "excellent", "perfect", "best",
    "awesome", "fantastic", "happy", "recommend", "wonderful", "good",
    "nice", "sturdy", "durable", "reliable", "easy", "fast", "works",
}

NEGATIVE_WORDS = {
    "terrible", "horrible", "awful", "broke", "broken", "disappointed",
    "disappointing", "waste", "worst", "bad", "poor", "cheap", "stopped",
    "defective", "damaged", "useless", "junk", "avoid", "refund",
    "cracked", "died", "leak", "drain",
}


@dataclass
class MismatchReport:
    total_reviews: int
    mismatch_count: int
    mismatch_rate: float
    examples: List[dict] = field(default_factory=list)


def lexicon_score(text: str) -> int:
    words = text.lower().split()
    pos = sum(1 for w in words if w.strip(".,!?;:\"'") in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,!?;:\"'") in NEGATIVE_WORDS)
    return pos - neg


def find_rating_text_mismatches(
    df: pd.DataFrame,
    min_word_diff: int = 2,
    max_examples: int = 10,
) -> MismatchReport:
    
    examples = []
    mismatch_count = 0

    for _, row in df.iterrows():
        score = lexicon_score(row["text"])
        label = row["sentiment"]

        is_mismatch = (
            (label == "negative" and score >= min_word_diff)
            or (label == "positive" and score <= -min_word_diff)
        )

        if is_mismatch:
            mismatch_count += 1
            if len(examples) < max_examples:
                examples.append({
                    "text": row["text"],
                    "label": label,
                    "lexicon_score": score,
                })

    total = len(df)
    rate = mismatch_count / total if total else 0.0

    return MismatchReport(
        total_reviews=total,
        mismatch_count=mismatch_count,
        mismatch_rate=rate,
        examples=examples,
    )


def print_mismatch_report(report: MismatchReport) -> None:
    print(f"Checked {report.total_reviews} reviews")
    print(
        f"Found {report.mismatch_count} where the text's word choice strongly "
        f"disagrees with the star-rating label ({report.mismatch_rate:.1%})"
    )
    if report.examples:
        print("\nExamples:")
        for ex in report.examples:
            print(f"  label={ex['label']:<9} lexicon_score={ex['lexicon_score']:+d}  text=\"{ex['text'][:90]}\"")
    print(
        "\nNote: this uses a simple word-counting heuristic (no negation handling), "
        "so it's meant to surface interesting candidates for manual review, not "
        "to be treated as ground truth on its own."
    )