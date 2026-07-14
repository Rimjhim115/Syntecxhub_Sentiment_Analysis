import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment.data import load_reviews, score_to_sentiment


def test_score_to_sentiment_boundaries():
    assert score_to_sentiment(1) == "negative"
    assert score_to_sentiment(2) == "negative"
    assert score_to_sentiment(3) == "neutral"
    assert score_to_sentiment(4) == "positive"
    assert score_to_sentiment(5) == "positive"


def test_load_reviews_uses_bundled_sample_by_default():
    df = load_reviews()
    assert "text" in df.columns
    assert "sentiment" in df.columns
    assert len(df) > 0
    assert set(df["sentiment"].unique()) <= {"negative", "neutral", "positive"}


def test_load_reviews_has_all_three_classes_in_sample():
    df = load_reviews()
    assert set(df["sentiment"].unique()) == {"negative", "neutral", "positive"}