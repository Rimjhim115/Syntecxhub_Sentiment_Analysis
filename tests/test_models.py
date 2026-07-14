import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment.models import MajorityBaseline, get_model


def test_majority_baseline_predicts_most_common_label():
    baseline = MajorityBaseline()
    baseline.fit(None, ["positive", "positive", "negative", "positive"])
    preds = baseline.predict([0, 0])  # length 2 dummy input
    assert preds == ["positive", "positive"]


def test_get_model_returns_correct_types():
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    assert isinstance(get_model("naive_bayes"), MultinomialNB)
    assert isinstance(get_model("logistic_regression"), LogisticRegression)
    assert isinstance(get_model("baseline"), MajorityBaseline)


def test_get_model_rejects_unknown_name():
    import pytest
    with pytest.raises(ValueError):
        get_model("not_a_real_model")