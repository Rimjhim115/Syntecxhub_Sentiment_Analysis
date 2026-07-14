
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

SAMPLE_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "sample_reviews.csv"

# Maps alternate column names (as found in different dataset sources) to
# the canonical 'text' / 'score' names this project uses internally.
_TEXT_COLUMN_ALIASES = ["Text", "text", "reviewText", "review_text", "Review"]
_SCORE_COLUMN_ALIASES = ["Score", "score", "overall", "rating", "Rating"]


def score_to_sentiment(score: int) -> str:
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def _find_column(df: pd.DataFrame, aliases: list) -> str:
    for name in aliases:
        if name in df.columns:
            return name
    raise ValueError(
        f"Could not find any of these columns in the data: {aliases}. "
        f"Actual columns found: {df.columns.tolist()}"
    )


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        # Amazon review dumps are JSON-lines, not a single JSON array.
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def load_reviews(
    csv_path: str | Path | None = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load reviews and attach a 'sentiment' column derived from the star rating.

    Pass a path to a full dataset (.csv or .json-lines) for real training,
    or leave it blank to use the small bundled sample. For large files,
    pass `sample_size` to randomly subsample rows -- there's no real
    benefit to training on 500K+ rows for this project's purposes, and
    it slows every run down for no meaningful accuracy gain.
    """
    path = Path(csv_path) if csv_path else SAMPLE_DATA_PATH
    df = _read_any(path)

    text_col = _find_column(df, _TEXT_COLUMN_ALIASES)
    score_col = _find_column(df, _SCORE_COLUMN_ALIASES)

    df = df.dropna(subset=[text_col, score_col]).copy()

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)

    df["sentiment"] = df[score_col].astype(float).astype(int).apply(score_to_sentiment)
    return df[[text_col, "sentiment"]].rename(columns={text_col: "text"})


def train_test_texts(
    csv_path: str | Path | None = None,
    test_size: float = 0.25,
    seed: int = 42,
    sample_size: Optional[int] = None,
):
    from sklearn.model_selection import train_test_split

    df = load_reviews(csv_path, sample_size=sample_size, seed=seed)
    return train_test_split(
        df["text"].tolist(),
        df["sentiment"].tolist(),
        test_size=test_size,
        random_state=seed,
        stratify=df["sentiment"] if df["sentiment"].nunique() > 1 else None,
    )