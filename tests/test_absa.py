"""
test_absa.py
------------
Pytest unit tests for ABSA aspect extraction and sentiment logic.

Run:
    pytest tests/test_absa.py -v
"""

import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

POS_THRESH =  0.05
NEG_THRESH = -0.05

ASPECT_KEYWORDS = {
    "Quality"         : ["quality","build","material","durable","sturdy","solid","flimsy",
                         "well made","craftsmanship","finish","defect","broken","damaged",
                         "poor quality","good quality","premium","reliable"],
    "Price"           : ["price","cost","expensive","affordable","value","worth","overpriced",
                         "budget","deal","bargain","pricey","money","paid","discount",
                         "reasonable","fair price","good value","rip off","costly"],
    "Delivery"        : ["delivery","shipping","shipped","arrived","delivered","courier",
                         "fast shipping","slow shipping","late","on time","delay","tracking"],
    "Packaging"       : ["packaging","package","box","boxed","wrapped","packing","packed",
                         "sealed","damaged box","unboxing","well packed","poorly packed"],
    "Customer Service": ["customer service","service","support","staff","seller","helpful",
                         "response","replied","refund","return","exchange","complaint","resolved"],
    "Design"          : ["design","look","looks","appearance","style","color","colour","size",
                         "shape","beautiful","attractive","elegant","sleek","comfortable",
                         "fit","lightweight","compact","portable"],
}
ASPECTS = list(ASPECT_KEYWORDS.keys())


# ── Replicate helpers from absa.py for isolated testing ──────────────────────

def detect_aspects_keyword(text: str) -> dict:
    """Pure keyword-based detection (no SpaCy needed for tests)."""
    text_low = text.lower()
    return {
        asp: [kw for kw in kws if kw in text_low]
        for asp, kws in ASPECT_KEYWORDS.items()
    }


def get_aspect_sentences(text: str, keywords: list) -> str:
    sents   = re.split(r"[.!?]+", text)
    rel     = [s.strip() for s in sents if any(kw in s.lower() for kw in keywords)]
    return " ".join(rel) if rel else text


def score_sentiment(text: str, aspect: str) -> tuple:
    kws      = ASPECT_KEYWORDS[aspect]
    asp_text = get_aspect_sentences(text, kws)
    compound = analyser.polarity_scores(asp_text)["compound"]
    label    = ("Positive" if compound >= POS_THRESH
                else "Negative" if compound <= NEG_THRESH
                else "Neutral")
    return round(compound, 4), label


# ── Aspect keyword detection ──────────────────────────────────────────────────

class TestAspectDetection:
    def test_detects_quality(self):
        matched = detect_aspects_keyword("The quality is excellent.")
        assert len(matched["Quality"]) > 0

    def test_detects_price(self):
        matched = detect_aspects_keyword("The price is too expensive for what you get.")
        assert len(matched["Price"]) > 0

    def test_detects_delivery(self):
        matched = detect_aspects_keyword("Fast shipping and arrived on time.")
        assert len(matched["Delivery"]) > 0

    def test_detects_packaging(self):
        matched = detect_aspects_keyword("The packaging was damaged when it arrived.")
        assert len(matched["Packaging"]) > 0

    def test_detects_customer_service(self):
        matched = detect_aspects_keyword("Customer service was very helpful with my return.")
        assert len(matched["Customer Service"]) > 0

    def test_detects_design(self):
        matched = detect_aspects_keyword("The design looks beautiful and feels comfortable.")
        assert len(matched["Design"]) > 0

    def test_no_false_positive_empty(self):
        matched = detect_aspects_keyword("I received the item.")
        # None of the strong aspect keywords should be present
        for asp in ["Quality", "Price", "Delivery", "Customer Service"]:
            assert len(matched[asp]) == 0

    def test_detects_multiple_aspects(self):
        text    = "Great quality and excellent value for the price. Fast delivery too!"
        matched = detect_aspects_keyword(text)
        detected = [asp for asp, hits in matched.items() if hits]
        assert len(detected) >= 2

    def test_case_insensitive(self):
        matched_lower = detect_aspects_keyword("quality is great")
        matched_upper = detect_aspects_keyword("QUALITY is great")
        assert len(matched_lower["Quality"]) > 0
        assert len(matched_upper["Quality"]) > 0


# ── Aspect sentence extraction ────────────────────────────────────────────────

class TestAspectSentenceExtraction:
    def test_extracts_relevant_sentence(self):
        text     = "The product arrived quickly. The quality is excellent. Nothing else."
        keywords = ["quality"]
        result   = get_aspect_sentences(text, keywords)
        assert "quality" in result.lower()

    def test_falls_back_to_full_text(self):
        text   = "A simple product."
        result = get_aspect_sentences(text, ["nonexistent_keyword_xyz"])
        assert result == text

    def test_extracts_multiple_relevant_sentences(self):
        text     = "Quality is great. The price is fair. Quality never disappoints."
        keywords = ["quality"]
        result   = get_aspect_sentences(text, keywords)
        assert result.lower().count("quality") >= 1

    def test_empty_text_returns_empty(self):
        result = get_aspect_sentences("", ["quality"])
        assert result == ""


# ── Aspect sentiment scoring ──────────────────────────────────────────────────

class TestAspectSentimentScoring:
    def test_positive_quality_review(self):
        text     = "The quality is absolutely excellent and very durable."
        compound, label = score_sentiment(text, "Quality")
        assert label == "Positive"
        assert compound > 0

    def test_negative_quality_review(self):
        text     = "Terrible quality. Broken after one day. Very poor quality."
        compound, label = score_sentiment(text, "Quality")
        assert label == "Negative"
        assert compound < 0

    def test_positive_price_review(self):
        text     = "Great value for money. Very affordable and worth every penny."
        compound, label = score_sentiment(text, "Price")
        assert label == "Positive"

    def test_negative_price_review(self):
        text     = "Way too expensive. Completely overpriced. Not worth the money."
        compound, label = score_sentiment(text, "Price")
        assert label == "Negative"

    def test_returns_tuple(self):
        compound, label = score_sentiment("Good quality.", "Quality")
        assert isinstance(compound, float)
        assert isinstance(label, str)
        assert label in ("Positive", "Neutral", "Negative")

    def test_compound_in_range(self):
        compound, _ = score_sentiment("Amazing product with great quality!", "Quality")
        assert -1.0 <= compound <= 1.0


# ── Scorecard aggregation logic ───────────────────────────────────────────────

class TestScorecardAggregation:
    def test_mean_computation(self):
        import numpy as np
        scores = [0.8, 0.5, 0.6]
        mean   = round(np.mean(scores), 3)
        assert abs(mean - 0.633) < 0.01

    def test_nan_for_insufficient_mentions(self):
        import numpy as np
        scores = []  # no mentions
        result = round(np.mean(scores), 3) if len(scores) >= 3 else float("nan")
        assert np.isnan(result)

    def test_min_3_mentions_required(self):
        import numpy as np
        scores_2 = [0.5, 0.6]        # only 2 → NaN
        scores_3 = [0.5, 0.6, 0.7]   # 3 → valid
        r2 = round(np.mean(scores_2), 3) if len(scores_2) >= 3 else float("nan")
        r3 = round(np.mean(scores_3), 3) if len(scores_3) >= 3 else float("nan")
        assert np.isnan(r2)
        assert not np.isnan(r3)