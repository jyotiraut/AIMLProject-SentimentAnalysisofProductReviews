
import os
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no pop-up windows
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

warnings.filterwarnings("ignore")
nltk.download("vader_lexicon", quiet=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Absolute paths (work regardless of where you run the script from) ─────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data",    "processed", "full_processed.csv")
RESULTS_PATH   = os.path.join(BASE_DIR, "outputs", "vader_results.csv")
PLOTS_DIR      = os.path.join(BASE_DIR, "outputs", "eda_plots")

POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05

COLORS = {
    "Positive": "#2A8AC8",
    "Neutral" : "#E8A020",
    "Negative": "#E24B4A",
}

plt.rcParams.update({
    "figure.dpi"       : 130,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.size"        : 11,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_processed(path: str = PROCESSED_PATH) -> pd.DataFrame:
    """Load full_processed.csv produced by preprocessing.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find: {path}\n"
            "Run  python src/preprocessing.py  first."
        )
    log.info("Loading: %s", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded %d rows x %d cols", *df.shape)

    for col in ["sentiment_label"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing — run preprocessing.py first.")

    # ── Decide which text column to score ─────────────────────────────────────
    # IMPORTANT: Use reviewText (original) NOT cleaned_text.
    # Cleaning removes punctuation (!?...) which VADER uses for sentiment.
    # "Not good!!!" scored on cleaned_text → "not good" (weaker signal)
    # "Not good!!!" scored on reviewText   → VADER sees the !!! (stronger negative)
    if "reviewText" in df.columns:
        df["vader_input"] = df["reviewText"].fillna("").astype(str)
        log.info("Using reviewText for VADER (original text, better accuracy)")
    elif "cleaned_text" in df.columns:
        df["vader_input"] = df["cleaned_text"].fillna("").astype(str)
        log.warning("reviewText not found — falling back to cleaned_text")
    else:
        raise ValueError("Neither reviewText nor cleaned_text found.")

    if "category" not in df.columns:
        df["category"] = "General"

    print(f"\n  Loaded {len(df):,} reviews")
    print(f"  Ground truth distribution:")
    print(f"  {df['sentiment_label'].value_counts().to_dict()}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. APPLY VADER
# ─────────────────────────────────────────────────────────────────────────────

def apply_vader(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run VADER on every review using vader_input column (reviewText).
    Adds: vader_compound, vader_pos, vader_neu, vader_neg
    """
    analyser = SentimentIntensityAnalyzer()
    log.info("Running VADER on %d reviews ...", len(df))
    print("  Scoring with VADER on original reviewText ...")

    scores = df["vader_input"].apply(
        lambda text: analyser.polarity_scores(text) if text.strip()
                     else {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
    )

    df["vader_compound"] = scores.apply(lambda s: round(s["compound"], 4))
    df["vader_pos"]      = scores.apply(lambda s: round(s["pos"],      4))
    df["vader_neu"]      = scores.apply(lambda s: round(s["neu"],      4))
    df["vader_neg"]      = scores.apply(lambda s: round(s["neg"],      4))

    print("  VADER scoring complete.")
    print(f"  Compound stats:\n  {df['vader_compound'].describe().round(4).to_dict()}\n")

    # Show how many reviews fall into each bucket
    neg = (df["vader_compound"] <= -0.05).sum()
    pos = (df["vader_compound"] >= 0.05).sum()
    neu = len(df) - neg - pos
    print(f"  Pre-classification counts:")
    print(f"    Positive (>=0.05)  : {pos:,}")
    print(f"    Neutral  (-0.05 to 0.05): {neu:,}")
    print(f"    Negative (<=-0.05) : {neg:,}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASSIFY
# ─────────────────────────────────────────────────────────────────────────────

def classify_vader(compound: float) -> str:
    """compound >= 0.05 → Positive | <= -0.05 → Negative | else → Neutral"""
    if compound >= POSITIVE_THRESHOLD:  return "Positive"
    if compound <= NEGATIVE_THRESHOLD:  return "Negative"
    return "Neutral"


def apply_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Apply classify_vader to vader_compound → adds vader_label column."""
    df["vader_label"] = df["vader_compound"].apply(classify_vader)
    print("  VADER predicted distribution:")
    print(f"  {df['vader_label'].value_counts().to_dict()}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vader(df: pd.DataFrame) -> dict:
    """Accuracy, Macro F1, Precision, Recall, Confusion Matrix vs ground truth."""
    y_true  = df["sentiment_label"]
    y_pred  = df["vader_label"]
    labels  = ["Positive", "Neutral", "Negative"]

    accuracy  = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro", labels=labels)
    precision = precision_score(y_true, y_pred, average="macro",
                                labels=labels, zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro",
                             labels=labels, zero_division=0)
    report    = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=labels)

    print("=" * 50)
    print("  VADER EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy        : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Macro Precision : {precision:.4f}")
    print(f"  Macro Recall    : {recall:.4f}")
    print(f"\n{report}")
    print("=" * 50)

    return {
        "accuracy": accuracy, "macro_f1": macro_f1,
        "precision": precision, "recall": recall,
        "report": report, "confusion_matrix": cm, "labels": labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS  (saved as PNG, no pop-ups)
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fpath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def plot_compound_distribution(df: pd.DataFrame) -> None:
    """KDE by true label + histogram by predicted label."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, color in COLORS.items():
        subset = df[df["sentiment_label"] == label]["vader_compound"]
        subset.plot.kde(ax=axes[0], label=label, color=color, linewidth=2)
    axes[0].axvline(POSITIVE_THRESHOLD, color="gray", linestyle="--", linewidth=1,
                    label=f"Threshold (+{POSITIVE_THRESHOLD})")
    axes[0].axvline(NEGATIVE_THRESHOLD, color="gray", linestyle=":", linewidth=1,
                    label=f"Threshold ({NEGATIVE_THRESHOLD})")
    axes[0].set_title("Compound Score Distribution by True Label",
                      fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("VADER Compound Score")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    for label, color in COLORS.items():
        subset = df[df["vader_label"] == label]["vader_compound"]
        axes[1].hist(subset, bins=60, alpha=0.55, label=f"Pred: {label}",
                     color=color, edgecolor="none")
    axes[1].set_title("Compound Score Histogram (Predicted Label)",
                      fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("VADER Compound Score")
    axes[1].set_ylabel("Count")
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[1].legend()
    plt.tight_layout()
    _save(fig, "vader_01_compound_distribution.png")


def plot_confusion_matrix(metrics: dict) -> None:
    """Confusion matrix — counts and row %."""
    cm     = metrics["confusion_matrix"]
    labels = metrics["labels"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in [
        (axes[0], cm,     "d",   "Confusion Matrix (counts)"),
        (axes[1], cm_pct, ".1f", "Confusion Matrix (row %)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, ax=ax, cbar=False)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    plt.tight_layout()
    _save(fig, "vader_02_confusion_matrix.png")


def plot_scores_per_category(df: pd.DataFrame) -> None:
    """Avg compound + pos/neu/neg scores per category."""
    top_cats  = df["category"].value_counts().head(8).index.tolist()
    df_cat    = df[df["category"].isin(top_cats)]
    cat_means = (
        df_cat.groupby("category")[
            ["vader_compound", "vader_pos", "vader_neg", "vader_neu"]]
        .mean().round(3).reindex(top_cats)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bars = axes[0].barh(
        cat_means.index, cat_means["vader_compound"],
        color=[COLORS["Positive"] if v >= POSITIVE_THRESHOLD
               else COLORS["Negative"] if v <= NEGATIVE_THRESHOLD
               else COLORS["Neutral"]
               for v in cat_means["vader_compound"]],
        edgecolor="none",
    )
    axes[0].axvline(0, color="gray", linewidth=0.8)
    axes[0].set_title("Avg Compound Score per Category",
                      fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Mean Compound Score")
    for bar in bars:
        w = bar.get_width()
        axes[0].text(w + 0.005 if w >= 0 else w - 0.005,
                     bar.get_y() + bar.get_height()/2,
                     f"{w:.3f}", va="center",
                     ha="left" if w >= 0 else "right", fontsize=9)

    x, width = np.arange(len(top_cats)), 0.25
    for i, (col, color, name) in enumerate(zip(
        ["vader_pos", "vader_neu", "vader_neg"],
        [COLORS["Positive"], COLORS["Neutral"], COLORS["Negative"]],
        ["Positive score", "Neutral score", "Negative score"],
    )):
        axes[1].bar(x + i*width, cat_means[col], width,
                    label=name, color=color, edgecolor="none", alpha=0.85)
    axes[1].set_title("Avg Pos / Neu / Neg Scores per Category",
                      fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([c[:18] for c in top_cats],
                             rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Mean Score")
    axes[1].legend()
    plt.tight_layout()
    _save(fig, "vader_03_scores_per_category.png")


def plot_label_distribution_per_category(df: pd.DataFrame) -> None:
    """Stacked bar: predicted label % per category."""
    top_cats = df["category"].value_counts().head(8).index.tolist()
    df_cat   = df[df["category"].isin(top_cats)]

    cat_label = df_cat.groupby(["category","vader_label"]).size().unstack(fill_value=0)
    for col in ["Positive","Neutral","Negative"]:
        if col not in cat_label.columns: cat_label[col] = 0
    cat_pct = (cat_label[["Positive","Neutral","Negative"]]
               .div(cat_label.sum(axis=1), axis=0).mul(100).round(1).reindex(top_cats))

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom  = np.zeros(len(cat_pct))
    for label, color in COLORS.items():
        vals = cat_pct[label].values
        bars = ax.bar(cat_pct.index, vals, bottom=bottom,
                      label=label, color=color, edgecolor="none", alpha=0.85)
        for bar, val, bot in zip(bars, vals, bottom):
            if val > 5:
                ax.text(bar.get_x()+bar.get_width()/2, bot+val/2,
                        f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals
    ax.set_title("VADER Predicted Sentiment per Category (%)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Percentage of Reviews")
    ax.set_xlabel("Product Category")
    ax.set_ylim(0, 105)
    ax.legend(title="Predicted label")
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    _save(fig, "vader_04_label_dist_per_category.png")


def plot_accuracy_summary(metrics: dict) -> None:
    """Horizontal bar: Accuracy, Macro F1, Precision, Recall."""
    names = ["Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]
    vals  = [metrics["accuracy"], metrics["macro_f1"],
             metrics["precision"], metrics["recall"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, vals,
                   color=["#2A8AC8","#1D9E75","#E8A020","#E24B4A"],
                   edgecolor="none", height=0.5)
    ax.set_xlim(0, 1.1)
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("VADER Performance Summary",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Score")
    for bar, val in zip(bars, vals):
        ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}  ({val*100:.1f}%)", va="center", fontsize=10)
    plt.tight_layout()
    _save(fig, "vader_05_performance_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame, path: str = RESULTS_PATH) -> None:
    """Save DataFrame with VADER columns to vader_results.csv."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Drop the temp scoring column before saving
    df = df.drop(columns=["vader_input"], errors="ignore")

    save_cols = [
        "reviewText", "cleaned_text", "processed_text",
        "sentiment_label", "rating", "category", "brand", "itemName",
        "reviewTime",
        "vader_compound", "vader_pos", "vader_neu", "vader_neg", "vader_label",
    ]
    cols = [c for c in save_cols if c in df.columns]
    df[cols].to_csv(path, index=False)

    size_kb = os.path.getsize(path) / 1024
    print(f"\n  VADER results saved → {path}  ({size_kb:,.0f} KB,  {len(df):,} rows)")
    print(f"  Columns saved: {cols}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_vader_pipeline() -> dict:
    """Run full VADER pipeline end-to-end."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 55)
    print("  VADER PIPELINE STARTING")
    print(f"  Looking for data at : {PROCESSED_PATH}")
    print(f"  Will save results to: {RESULTS_PATH}")
    print(f"  Will save plots to  : {PLOTS_DIR}")
    print("=" * 55)

    df      = load_processed()
    df      = apply_vader(df)
    df      = apply_classification(df)
    metrics = evaluate_vader(df)

    print("\n  Generating plots ...")
    plot_compound_distribution(df)
    plot_confusion_matrix(metrics)
    plot_scores_per_category(df)
    plot_label_distribution_per_category(df)
    plot_accuracy_summary(metrics)

    save_results(df)

    print()
    print("=" * 55)
    print("  VADER PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Accuracy : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"  Results  : {RESULTS_PATH}")
    print(f"  Plots    : {PLOTS_DIR}")
    print("=" * 55)

    return {"df": df, "metrics": metrics}


if __name__ == "__main__":
    run_vader_pipeline()