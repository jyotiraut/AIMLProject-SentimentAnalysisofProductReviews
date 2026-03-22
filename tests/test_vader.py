
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Constants matching vader_sentiment.py ─────────────────────────────────────
POS_THRESH =  0.05
NEG_THRESH = -0.05

analyser = SentimentIntensityAnalyzer()


def classify(compound: float) -> str:
    """Mirror the classify_vader logic from vader_sentiment.py."""
    if compound >= POS_THRESH:  return "Positive"
    if compound <= NEG_THRESH:  return "Negative"
    return "Neutral"


# ── Score range tests ─────────────────────────────────────────────────────────

class TestVADERScoreRange:
    def _scores(self, text):
        return analyser.polarity_scores(text)

    def test_compound_in_range(self):
        s = self._scores("This product is amazing!")
        assert -1.0 <= s["compound"] <= 1.0

    def test_pos_in_range(self):
        s = self._scores("Excellent quality!")
        assert 0.0 <= s["pos"] <= 1.0

    def test_neu_in_range(self):
        s = self._scores("This is a product.")
        assert 0.0 <= s["neu"] <= 1.0

    def test_neg_in_range(self):
        s = self._scores("Terrible and awful!")
        assert 0.0 <= s["neg"] <= 1.0

    def test_scores_sum_to_one(self):
        s = self._scores("Good product with fast delivery.")
        total = round(s["pos"] + s["neu"] + s["neg"], 4)
        assert abs(total - 1.0) < 0.01

    def test_empty_text_compound_zero(self):
        s = self._scores("")
        assert s["compound"] == 0.0


# ── Classification threshold tests ───────────────────────────────────────────

class TestClassification:
    def test_clearly_positive(self):
        assert classify(0.8)  == "Positive"

    def test_barely_positive(self):
        assert classify(0.05) == "Positive"

    def test_clearly_negative(self):
        assert classify(-0.8) == "Negative"

    def test_barely_negative(self):
        assert classify(-0.05) == "Negative"

    def test_neutral_zero(self):
        assert classify(0.0) == "Neutral"

    def test_neutral_just_below_pos(self):
        assert classify(0.04) == "Neutral"

    def test_neutral_just_above_neg(self):
        assert classify(-0.04) == "Neutral"

    def test_boundary_pos_threshold(self):
        # Exactly at threshold should be Positive
        assert classify(POS_THRESH) == "Positive"

    def test_boundary_neg_threshold(self):
        # Exactly at threshold should be Negative
        assert classify(NEG_THRESH) == "Negative"


# ── Sentiment direction tests (sanity checks on real text) ────────────────────

class TestSentimentDirection:
    def test_positive_review_positive_compound(self):
        text = "This is an absolutely wonderful product. I love it!"
        s    = analyser.polarity_scores(text)
        assert s["compound"] > 0

    def test_negative_review_negative_compound(self):
        text = "Terrible product. Complete waste of money. Very disappointed."
        s    = analyser.polarity_scores(text)
        assert s["compound"] < 0

    def test_neutral_review_near_zero(self):
        text = "The product arrived."
        s    = analyser.polarity_scores(text)
        assert -0.3 <= s["compound"] <= 0.3

    def test_positive_review_classifies_positive(self):
        text  = "Excellent quality! Highly recommend this product!"
        score = analyser.polarity_scores(text)["compound"]
        assert classify(score) == "Positive"

    def test_negative_review_classifies_negative(self):
        text  = "Broken on arrival. Extremely poor quality. Terrible!"
        score = analyser.polarity_scores(text)["compound"]
        assert classify(score) == "Negative"

    def test_stronger_sentiment_higher_abs_compound(self):
        mild   = analyser.polarity_scores("Good product.")["compound"]
        strong = analyser.polarity_scores("Absolutely amazing! Best product ever!")["compound"]
        assert strong > mild


# ── Accuracy logic test ───────────────────────────────────────────────────────

class TestAccuracyComputation:
    def test_perfect_accuracy(self):
        y_true = ["Positive", "Negative", "Neutral"]
        y_pred = ["Positive", "Negative", "Neutral"]
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        acc     = correct / len(y_true)
        assert acc == 1.0

    def test_zero_accuracy(self):
        y_true = ["Positive", "Positive", "Positive"]
        y_pred = ["Negative", "Negative", "Negative"]
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        acc     = correct / len(y_true)
        assert acc == 0.0

    def test_partial_accuracy(self):
        y_true = ["Positive", "Positive", "Negative", "Negative"]
        y_pred = ["Positive", "Negative", "Negative", "Positive"]
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        acc     = correct / len(y_true)
        assert acc == 0.5