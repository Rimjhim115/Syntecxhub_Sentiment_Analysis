import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from sentiment.analysis import find_rating_text_mismatches, lexicon_score


def test_lexicon_score_positive_text():
    assert lexicon_score("This is amazing and I love it") > 0


def test_lexicon_score_negative_text():
    assert lexicon_score("Terrible, broke immediately, complete waste") < 0


def test_lexicon_score_neutral_text_near_zero():
    assert lexicon_score("It arrived on Tuesday in a box") == 0


def test_finds_positive_label_with_negative_text():
    df = pd.DataFrame({
        "text": ["The only reason this loses a star is it doesnt fit, disappointed"],
        "sentiment": ["positive"],
    })
    report = find_rating_text_mismatches(df, min_word_diff=1)
    assert report.mismatch_count == 1


def test_no_mismatch_when_text_and_label_agree():
    df = pd.DataFrame({
        "text": ["This is amazing, I love it, best purchase ever"],
        "sentiment": ["positive"],
    })
    report = find_rating_text_mismatches(df, min_word_diff=1)
    assert report.mismatch_count == 0