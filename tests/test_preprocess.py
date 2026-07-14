import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment.preprocess import clean_text


def test_lowercases_and_strips_punctuation():
    assert clean_text("This Was GREAT!!!") == "great"


def test_keeps_negation_words():
    cleaned = clean_text("This was not good")
    assert "not" in cleaned.split()


def test_removes_html_tags():
    cleaned = clean_text("<br>Great product<br/>")
    assert "br" not in cleaned


def test_handles_contraction_negation():
    cleaned = clean_text("I don't like this")
    assert "not" in cleaned.split()

def test_decodes_html_entities():
    cleaned = clean_text('This was a &#34;great&#34; product &amp; I loved it')
    assert "34" not in cleaned  # the raw entity code shouldn't leak into tokens
    assert "amp" not in cleaned  # &amp; should decode to "&", not survive as the word "amp"