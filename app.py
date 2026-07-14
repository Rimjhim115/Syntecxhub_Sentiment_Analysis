
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from sentiment.pipeline import predict_text

st.set_page_config(page_title="Sentiment Analysis Tool", page_icon="💬")

st.title("💬 Sentiment Analysis Tool")
st.caption("Classical ML sentiment classification, trained on Amazon reviews.")

text = st.text_area(
    "Enter a review:",
    placeholder="e.g. This case fits perfectly and the material feels premium",
    height=100,
)

if st.button("Analyze sentiment", type="primary"):
    if not text.strip():
        st.warning("Type something first.")
    else:
        try:
            result = predict_text(text)
        except FileNotFoundError:
            st.error(
                "No trained model found. Run `python main.py train` first, "
                "then restart this app."
            )
        else:
            sentiment = result["sentiment"]
            confidence = result["confidence"]

            color = {"positive": "green", "neutral": "orange", "negative": "red"}[sentiment]
            st.markdown(f"### Sentiment: :{color}[{sentiment.upper()}]")
            if confidence is not None:
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")

st.divider()

st.subheader("Model comparison")
st.caption("Evaluated on the same held-out test split, 30K sampled reviews.")
st.table(
    {
        "Model": ["Baseline", "Naive Bayes", "Logistic Regression (used above)", "RoBERTa (pretrained)"],
        "Accuracy": ["0.766", "0.798", "0.759", "0.767"],
        "F1 (macro)": ["0.289", "0.433", "0.616", "0.562"],
        "Speed": ["instant", "instant", "instant", "~371 ms"],
    }
)
st.caption(
    "Logistic Regression was picked as the production model for its F1 score, "
    "not its accuracy -- it's the only model that reliably distinguishes all "
    "three classes instead of defaulting to 'positive'."
)