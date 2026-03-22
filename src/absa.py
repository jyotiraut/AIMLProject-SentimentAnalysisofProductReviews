import os, re, logging, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy

warnings.filterwarnings("ignore")
nltk.download("vader_lexicon", quiet=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths (FIXED: absolute paths) ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "full_processed.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

SCORECARD_PATH = os.path.join(OUTPUT_DIR, "absa_scorecard.csv")
DETAIL_PATH    = os.path.join(OUTPUT_DIR, "absa_review_details.csv")

# ── Thresholds ─────────────────────────────────────────────────────────────────
POS_THRESH =  0.05
NEG_THRESH = -0.05

# ── Aspects ────────────────────────────────────────────────────────────────────
ASPECT_KEYWORDS = {
    "Quality": ["quality","build","material","durable","sturdy","broken","damaged"],
    "Price": ["price","cost","expensive","cheap","value","worth"],
    "Delivery": ["delivery","shipping","arrived","late","delay"],
    "Packaging": ["packaging","box","wrapped","damaged box"],
    "Customer Service": ["service","support","refund","return"],
    "Design": ["design","look","style","color","size","fit"],
}
ASPECTS = list(ASPECT_KEYWORDS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    log.info(f"Loading data from {PROCESSED_PATH}")
    if not os.path.exists(PROCESSED_PATH):
        log.error("❌ Processed file not found!")
        return None

    df = pd.read_csv(PROCESSED_PATH)

    text_col = "reviewText" if "reviewText" in df.columns else "cleaned_text"
    df["absa_text"] = df[text_col].fillna("").astype(str)

    if "category" not in df.columns:
        df["category"] = "General"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SPACY
# ─────────────────────────────────────────────────────────────────────────────
def load_spacy():
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except:
        import subprocess, sys
        subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm", disable=["ner"])


# ─────────────────────────────────────────────────────────────────────────────
# ABSA LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def get_aspect_sentences(text, keywords):
    sentences = re.split(r"[.!?]+", text)
    rel = [s for s in sentences if any(k in s.lower() for k in keywords)]
    return " ".join(rel) if rel else text


def score_sentiment(text, aspect, analyser):
    txt = get_aspect_sentences(text, ASPECT_KEYWORDS[aspect])
    score = analyser.polarity_scores(txt)["compound"]

    label = "Positive" if score >= POS_THRESH else "Negative" if score <= NEG_THRESH else "Neutral"
    return score, label


def run_absa(df, analyser):
    log.info("Running ABSA...")

    for asp in ASPECTS:
        df[f"aspect_{asp}"] = np.nan
        df[f"aspect_{asp}_label"] = None

    for i, row in df.iterrows():
        text = row["absa_text"].lower()

        for asp, kws in ASPECT_KEYWORDS.items():
            if any(k in text for k in kws):
                score, label = score_sentiment(text, asp, analyser)
                df.at[i, f"aspect_{asp}"] = score
                df.at[i, f"aspect_{asp}_label"] = label

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SCORECARD (FIXED)
# ─────────────────────────────────────────────────────────────────────────────
def build_scorecard(df):
    log.info("Building scorecard...")

    scorecard = {}

    for cat in df["category"].unique():
        df_cat = df[df["category"] == cat]
        row = {}

        for asp in ASPECTS:
            col = f"aspect_{asp}"
            vals = df_cat[col].dropna()

            # 🔥 FIX: allow even 1 value
            row[asp] = round(vals.mean(), 3) if len(vals) >= 1 else np.nan

        scorecard[cat] = row

    sc = pd.DataFrame(scorecard).T
    print("\n=== SCORECARD ===\n", sc)

    return sc


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS (FIXED)
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(df, scorecard):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        if scorecard is not None and not scorecard.empty:
            scorecard.to_csv(SCORECARD_PATH)
            log.info(f"✅ Scorecard saved: {SCORECARD_PATH}")
        else:
            log.error("❌ Scorecard empty! Saving fallback file.")
            pd.DataFrame({"error": ["No data"]}).to_csv(SCORECARD_PATH)
    except Exception as e:
        log.error(f"❌ Error saving scorecard: {e}")

    df.to_csv(DETAIL_PATH, index=False)
    log.info(f"✅ Detail file saved: {DETAIL_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_absa_pipeline():
    log.info("===== ABSA START =====")

    df = load_data()
    if df is None:
        return

    nlp = load_spacy()
    analyser = SentimentIntensityAnalyzer()

    df = run_absa(df, analyser)
    scorecard = build_scorecard(df)

    save_outputs(df, scorecard)

    log.info("===== ABSA DONE =====")


if __name__ == "__main__":
    run_absa_pipeline()