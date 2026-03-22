"""
test_preprocessing.py
---------------------
Unit tests for src/preprocessing.py
Run with: pytest tests/test_preprocessing.py -v
"""
import sys
sys.path.append('..')

import pytest
import pandas as pd
from src.preprocessing import (
    clean_text,
    assign_sentiment_label,
    tokenise_and_lemmatise,
    stratified_sample,
    split_data,
)


# ── clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_removes_html():
    assert "<br>" not in clean_text("Hello <br> world")

def test_clean_text_removes_url():
    assert "http" not in clean_text("Visit http://example.com today")

def test_clean_text_removes_special_chars():
    result = clean_text("Great product!!! 100% worth it #amazon")
    assert "!" not in result and "%" not in result and "#" not in result

def test_clean_text_lowercases():
    assert clean_text("GREAT PRODUCT") == "great product"

def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_clean_text_collapses_whitespace():
    result = clean_text("too   many   spaces")
    assert "  " not in result


# ── assign_sentiment_label ────────────────────────────────────────────────────

def test_label_positive():
    assert assign_sentiment_label(5) == "Positive"
    assert assign_sentiment_label(4) == "Positive"

def test_label_neutral():
    assert assign_sentiment_label(3) == "Neutral"

def test_label_negative():
    assert assign_sentiment_label(2) == "Negative"
    assert assign_sentiment_label(1) == "Negative"


# ── tokenise_and_lemmatise ────────────────────────────────────────────────────

def test_tokenise_returns_tuple():
    tokens, joined = tokenise_and_lemmatise("the product was amazing")
    assert isinstance(tokens, list)
    assert isinstance(joined, str)

def test_tokenise_removes_stopwords():
    tokens, _ = tokenise_and_lemmatise("the product was amazing")
    assert "the" not in tokens
    assert "was" not in tokens

def test_tokenise_lemmatises():
    tokens, _ = tokenise_and_lemmatise("running products are amazing")
    # 'running' should be lemmatised to 'running' or 'run'
    assert any(t in tokens for t in ["run", "running"])

def test_tokenise_filters_short_tokens():
    tokens, _ = tokenise_and_lemmatise("a it be the good")
    assert all(len(t) > 2 for t in tokens)


# ── stratified_sample ─────────────────────────────────────────────────────────

def test_stratified_sample_size():
    df = pd.DataFrame({
        'text': ['review'] * 200,
        'overall': [5, 4, 3, 2, 1] * 40,
        'sentiment_label': ['Positive'] * 120 + ['Neutral'] * 40 + ['Negative'] * 40,
    })
    sampled = stratified_sample(df, n=100, random_state=42)
    assert len(sampled) == 100

def test_stratified_sample_class_balance():
    df = pd.DataFrame({
        'text': ['review'] * 300,
        'overall': [5] * 150 + [3] * 75 + [1] * 75,
        'sentiment_label': ['Positive'] * 150 + ['Neutral'] * 75 + ['Negative'] * 75,
    })
    sampled = stratified_sample(df, n=120, random_state=42)
    counts  = sampled['sentiment_label'].value_counts(normalize=True)
    assert abs(counts.get('Positive', 0) - 0.5) < 0.05


# ── split_data ────────────────────────────────────────────────────────────────

def test_split_sizes():
    df = pd.DataFrame({
        'text': ['r'] * 1000,
        'overall': [5] * 500 + [3] * 250 + [1] * 250,
        'sentiment_label': ['Positive'] * 500 + ['Neutral'] * 250 + ['Negative'] * 250,
        'processed_text': ['review'] * 1000,
        'review_length': [5] * 1000,
    })
    train, val, test = split_data(df, train_ratio=0.70, val_ratio=0.15)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert abs(len(train) / len(df) - 0.70) < 0.02
    assert abs(len(val)   / len(df) - 0.15) < 0.02
    assert abs(len(test)  / len(df) - 0.15) < 0.02

def test_split_no_overlap():
    df = pd.DataFrame({
        'text': [f'r{i}' for i in range(500)],
        'overall': [5] * 300 + [3] * 100 + [1] * 100,
        'sentiment_label': ['Positive'] * 300 + ['Neutral'] * 100 + ['Negative'] * 100,
        'processed_text': ['review'] * 500,
        'review_length': [5] * 500,
    })
    train, val, test = split_data(df)
    train_texts = set(train['text'])
    val_texts   = set(val['text'])
    test_texts  = set(test['text'])
    assert len(train_texts & val_texts)  == 0
    assert len(train_texts & test_texts) == 0
    assert len(val_texts   & test_texts) == 0