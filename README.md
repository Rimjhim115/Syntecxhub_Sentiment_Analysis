[![CI](https://github.com/Rimjhim115/Syntexhub_Sentiment_Analysis/actions/workflows/ci.yml/badge.svg
An end-to-end NLP sentiment analysis platform built during my AI Internship to classify Amazon product reviews into Negative, Neutral, and Positive sentiments using classical Machine Learning techniques.
# Features:
Trained on "Amazon Cell Phones & Accessories reviews
- HTML entity decoding
- Lowercasing
- Punctuation removal
- Tokenization
- TF-IDF feature extraction
- Multiple sentiment classification models
- Majority Class Baseline
- Multinomial Naive Bayes
- Logistic Regression
- Pretrained RoBERTa
-  Performance evaluation using:
- Accuracy
- Precision
- Recall
- Macro F1 Score
- Confusion Matrices
- Interactive Streamlit web application
- Command Line Interface (CLI)
- Automated unit tests with GitHub Actions CI

# 📂 Dataset

- ~194,000 customer reviews
- Ratings converted into three sentiment classes

| Rating | Sentiment |

| ⭐ 1–2 | Negative |
| ⭐ 3 | Neutral |
| ⭐ 4–5 | Positive |

# Project Pipeline


Amazon Reviews
       │
       ▼
Text Preprocessing
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Model Training
(Baseline • Naive Bayes • Logistic Regression)
       │
       ▼
Model Evaluation
(Accuracy • Precision • Recall • Macro F1)
       │
       ▼
RoBERTa Benchmark
       │
       ▼
Confusion Matrix & Error Analysis
       │
       ▼
Streamlit Web Application

#  Key Findings

### Logistic Regression vs Pretrained RoBERTa

A pretrained RoBERTa model was evaluated on the exact same held-out test split.

Interestingly, Logistic Regression achieved a higher Macro F1 score on this dataset while providing significantly faster inference.

### Dataset Quality Analysis

Beyond model evaluation, the project investigates the dataset itself.

Results:

- 30,000 reviews analyzed
- 539 potential label mismatches detected
- Approximately **1.8%** mismatch rate

This analysis provides additional insight into dataset quality while acknowledging the limitations of rule-based sentiment methods.

#  Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Streamlit
- Matplotlib
- Seaborn
- Hugging Face Transformers
- PyTest
- GitHub Actions


