

import os
import re
import logging
import pandas as pd
import numpy as np
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from sklearn.model_selection import train_test_split

# ── Download NLTK resources 
for resource in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

# ── Logging ─
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants
SAMPLE_SIZE  = 50_000
RANDOM_STATE = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15

RAW_DATA_PATH = "data/raw/amazon_reviews.csv"
PROCESSED_DIR = "data/processed/"

# ── Column names matching dataset 
COL_TEXT     = "reviewText"
COL_RATING   = "rating"         
COL_SUMMARY  = "summary"
COL_TIME     = "reviewTime"
COL_PRODUCT  = "itemName"        
COL_CATEGORY = "category"
COL_PRICE    = "price"
COL_BRAND    = "brand"
COL_VERIFIED = "verified"

STOPWORDS  = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


#load

def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Amazon reviews CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame.
    """
    log.info("Loading raw data from: %s", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("Raw shape: %d rows x %d cols", *df.shape)
    log.info("Columns found: %s", df.columns.tolist())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATE & NORMALISE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalise all columns for this specific dataset.

    Handles:
        - rating as int (may come as float 5.0)
        - price as float (comes as "$1.63 " string)
        - verified as bool (comes as "TRUE"/"FALSE" string)
        - category whitespace stripping
        - dropping rows with missing reviewText or rating

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned, type-corrected DataFrame.

    Raises:
        ValueError: If reviewText or rating columns are missing.
    """
    # ── Check required columns ────────────────────────────────────────────────
    for required in [COL_TEXT, COL_RATING]:
        if required not in df.columns:
            raise ValueError(
                f"Required column '{required}' not found.\n"
                f"Available columns: {df.columns.tolist()}"
            )

    # ── Category fallback ─────────────────────────────────────────────────────
    if COL_CATEGORY not in df.columns:
        log.warning("'%s' column missing — setting to 'General'", COL_CATEGORY)
        df[COL_CATEGORY] = "General"

    # ── Drop rows missing text or rating ──────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=[COL_TEXT, COL_RATING]).reset_index(drop=True)
    log.info("Dropped %d rows with missing text/rating", before - len(df))

    # ── Convert rating to int (handles "5", "5.0", 5.0) ──────────────────────
    df[COL_RATING] = pd.to_numeric(df[COL_RATING], errors="coerce")
    df = df.dropna(subset=[COL_RATING]).reset_index(drop=True)
    df[COL_RATING] = df[COL_RATING].astype(int)

    # ── Clamp ratings to valid 1–5 range ─────────────────────────────────────
    invalid = ~df[COL_RATING].between(1, 5)
    if invalid.sum() > 0:
        log.warning("Dropping %d rows with ratings outside 1–5", invalid.sum())
        df = df[~invalid].reset_index(drop=True)

    # ── Clean price: "$1.63 " → 1.63 ─────────────────────────────────────────
    if COL_PRICE in df.columns:
        df[COL_PRICE] = (
            df[COL_PRICE]
            .astype(str)
            .str.replace(r"[\$,\s]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )
        log.info("Price column cleaned — %d non-null values", df[COL_PRICE].notna().sum())

    # ── Normalise verified: "TRUE"/"FALSE" → bool ────────────────────────────
    if COL_VERIFIED in df.columns:
        df[COL_VERIFIED] = (
            df[COL_VERIFIED]
            .astype(str).str.upper()
            .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
        )

    # ── Normalise category whitespace ─────────────────────────────────────────
    df[COL_CATEGORY] = df[COL_CATEGORY].astype(str).str.strip()

    log.info("Categories found: %s",
             df[COL_CATEGORY].value_counts().head(10).to_dict())
    log.info("After validation: %d rows remain", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. SENTIMENT LABELS
# ─────────────────────────────────────────────────────────────────────────────

def assign_sentiment_label(rating: int) -> str:
    """
    Map 1–5 star rating to 3-class sentiment label.

        >= 4  ->  Positive
        == 3  ->  Neutral
        <= 2  ->  Negative

    Args:
        rating: Integer star rating.

    Returns:
        Sentiment label string.
    """
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"


def add_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'sentiment_label' column derived from the rating column."""
    df["sentiment_label"] = df[COL_RATING].apply(assign_sentiment_label)
    log.info("Sentiment distribution:\n%s",
             df["sentiment_label"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRATIFIED SAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def stratified_sample(
    df: pd.DataFrame,
    n: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Draw a stratified sample of n rows preserving class proportions.

    If the dataset has fewer than n rows, returns the full dataset.

    Args:
        df:           Input DataFrame (must have 'sentiment_label').
        n:            Target sample size.
        random_state: Reproducibility seed.

    Returns:
        Sampled DataFrame with reset index.
    """
    if len(df) <= n:
        log.warning("Dataset has only %d rows — using full dataset", len(df))
        return df.reset_index(drop=True)

    df_sample, _ = train_test_split(
        df,
        train_size=n,
        stratify=df["sentiment_label"],
        random_state=random_state,
    )
    df_sample = df_sample.reset_index(drop=True)
    log.info("Sampled %d rows (stratified by sentiment)", len(df_sample))
    log.info("Sample distribution:\n%s",
             df_sample["sentiment_label"].value_counts().to_string())
    return df_sample


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean a single review string.

    Steps:
        1. Lowercase
        2. Remove HTML tags  (<br>, <p>, etc.)
        3. Remove URLs
        4. Remove special characters and digits
        5. Collapse extra whitespace

    Args:
        text: Raw review string.

    Returns:
        Cleaned lowercase string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)            # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"[^a-z\s]", " ", text)           # special chars & digits
    text = re.sub(r"\s+", " ", text).strip()         # extra whitespace
    return text


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text to reviewText column and add 'cleaned_text'.
    Drops any reviews that become empty after cleaning.
    """
    log.info("Cleaning %d review texts...", len(df))
    df["cleaned_text"] = df[COL_TEXT].apply(clean_text)

    before = len(df)
    df = df[df["cleaned_text"].str.len() > 10].reset_index(drop=True)
    log.info("Dropped %d reviews that became empty after cleaning",
             before - len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. TOKENISE, STOPWORD REMOVAL, LEMMATISE
# ─────────────────────────────────────────────────────────────────────────────

def tokenise_and_lemmatise(text: str):
    """
    Tokenise a cleaned string, remove stopwords, and lemmatise.

    Args:
        text: Cleaned review string.

    Returns:
        Tuple of (token_list, joined_string).
    """
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]
    return tokens, " ".join(tokens)


def apply_tokenisation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply tokenise_and_lemmatise to cleaned_text.

    Adds columns:
        tokens          — list of lemmatised tokens
        processed_text  — space-joined token string (model input)
        review_length   — number of tokens
    """
    log.info("Tokenising and lemmatising %d reviews...", len(df))
    results = df["cleaned_text"].apply(tokenise_and_lemmatise)
    df["tokens"]         = results.apply(lambda x: x[0])
    df["processed_text"] = results.apply(lambda x: x[1])
    df["review_length"]  = df["tokens"].apply(len)
    log.info(
        "Tokenisation done. Avg length: %.1f tokens | Median: %.0f tokens",
        df["review_length"].mean(),
        df["review_length"].median(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. PARSE DATES
# ─────────────────────────────────────────────────────────────────────────────

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse reviewTime to datetime and add 'year_month' Period column
    for trend analysis in the Streamlit dashboard.
    """
    if COL_TIME in df.columns:
        df["review_date"] = pd.to_datetime(df[COL_TIME], errors="coerce")
        df["year_month"]  = df["review_date"].dt.to_period("M")
        log.info("Parsed review dates. Date range: %s to %s",
                 df["review_date"].min(), df["review_date"].max())
    else:
        log.warning("'%s' column not found — skipping date parsing", COL_TIME)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. TRAIN / VAL / TEST SPLIT (70 / 15 / 15)
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
    random_state: int  = RANDOM_STATE,
):
    """
    Stratified 3-way split preserving class proportions.

    Args:
        df:           Processed DataFrame (must have 'sentiment_label').
        train_ratio:  Fraction for training   (default 0.70)
        val_ratio:    Fraction for validation  (default 0.15)
        random_state: Reproducibility seed.

    Returns:
        Tuple of (df_train, df_val, df_test).
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 4)

    # Step 1: carve out test set
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["sentiment_label"],
        random_state=random_state,
    )

    # Step 2: split remainder into train and val
    val_adjusted = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_adjusted,
        stratify=df_train_val["sentiment_label"],
        random_state=random_state,
    )

    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    log.info("Split sizes  →  Train: %d | Val: %d | Test: %d",
             len(df_train), len(df_val), len(df_test))

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        dist = (
            split["sentiment_label"]
            .value_counts(normalize=True)
            .mul(100).round(1)
            .to_dict()
        )
        log.info("%s class balance: %s", name, dist)

    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_splits(
    df_full:  pd.DataFrame,
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    out_dir:  str = PROCESSED_DIR,
) -> None:
    """
    Save full processed dataset and three split CSVs to out_dir.

    Files written:
        full_processed.csv  — all 50k rows
        train.csv           — 70% training split
        val.csv             — 15% validation split
        test.csv            — 15% test split

    Args:
        df_full:  Complete processed DataFrame.
        df_train: Training split.
        df_val:   Validation split.
        df_test:  Test split.
        out_dir:  Output directory path.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Columns to save
    base_cols = [
        COL_TEXT, "cleaned_text", "processed_text",
        "review_length", COL_RATING, "sentiment_label",
    ]
    optional = [
        COL_SUMMARY, COL_TIME, COL_PRODUCT, COL_CATEGORY,
        COL_BRAND, COL_PRICE, COL_VERIFIED, "year_month",
    ]
    save_cols = base_cols + [c for c in optional if c in df_full.columns]

    # Serialise token lists as strings so they survive CSV round-trip
    # Reload with: import ast; df['tokens'] = df['tokens_str'].apply(ast.literal_eval)
    for df in [df_full, df_train, df_val, df_test]:
        if "tokens" in df.columns:
            df["tokens_str"] = df["tokens"].apply(str)

    files = {
        "full_processed.csv": df_full,
        "train.csv":          df_train,
        "val.csv":            df_val,
        "test.csv":           df_test,
    }

    for fname, df in files.items():
        cols  = [c for c in save_cols + ["tokens_str"] if c in df.columns]
        fpath = os.path.join(out_dir, fname)
        df[cols].to_csv(fpath, index=False)
        size_kb = os.path.getsize(fpath) / 1024
        log.info("Saved  %-25s  (%7.0f KB,  %d rows)", fname, size_kb, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: TOP N-GRAMS  (used by eda.ipynb and utils.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_top_ngrams(token_series: pd.Series, n: int = 1, top_k: int = 20):
    """
    Compute the most frequent n-grams across a Series of token lists.

    Args:
        token_series: Pandas Series where each element is a list of tokens.
        n:            n-gram size (1 = unigram, 2 = bigram).
        top_k:        Number of top n-grams to return.

    Returns:
        List of (ngram_string, count) tuples.
    """
    all_ngrams = [
        " ".join(gram)
        for tokens in token_series
        if isinstance(tokens, list)
        for gram in (list(ngrams(tokens, n)) if n > 1 else [(t,) for t in tokens])
    ]
    return Counter(all_ngrams).most_common(top_k)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(raw_path: str = RAW_DATA_PATH) -> dict:
    """
    Run the complete preprocessing pipeline end-to-end.

    Args:
        raw_path: Path to raw Amazon reviews CSV.

    Returns:
        Dict with keys: 'full', 'train', 'val', 'test'
        each holding the corresponding DataFrame.
    """
    log.info("=" * 55)
    log.info("  STARTING PREPROCESSING PIPELINE")
    log.info("=" * 55)

    df = load_raw(raw_path)
    df = validate_columns(df)
    df = add_sentiment_labels(df)
    df = stratified_sample(df, n=SAMPLE_SIZE)
    df = apply_cleaning(df)
    df = apply_tokenisation(df)
    df = parse_dates(df)

    df_train, df_val, df_test = split_data(df)
    save_splits(df, df_train, df_val, df_test)

    log.info("=" * 55)
    log.info("  PIPELINE COMPLETE")
    log.info("  Total: %d  |  Train: %d  |  Val: %d  |  Test: %d",
             len(df), len(df_train), len(df_val), len(df_test))
    log.info("  Saved to: %s", PROCESSED_DIR)
    log.info("=" * 55)

    return {"full": df, "train": df_train, "val": df_val, "test": df_test}


if __name__ == "__main__":
    run_pipeline()