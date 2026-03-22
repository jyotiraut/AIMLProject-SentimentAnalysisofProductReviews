import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("started")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH      = "data/processed/train.csv"
VAL_PATH        = "data/processed/val.csv"
TEST_PATH       = "data/processed/test.csv"
VADER_PATH      = "outputs/vader_results.csv"
MODEL_SAVE_DIR  = "models/bert_checkpoint/"
PLOTS_DIR       = "outputs/eda_plots/"
RESULTS_PATH    = "outputs/bert_results.csv"

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"   # swap to "bert-base-uncased" if you have GPU
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE     = 2                           # EarlyStopping patience

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {0: "Negative", 1: "Neutral",  2: "Positive"}
LABELS   = ["Positive", "Neutral", "Negative"]

plt.rcParams.update({
    "figure.dpi"       : 130,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.size"        : 11,
})
COLORS = {"Positive": "#2A8AC8", "Neutral": "#E8A020", "Negative": "#E24B4A"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_splits():
    """
    Load train, val, test CSVs produced by preprocessing.py.

    Returns:
        Tuple of (df_train, df_val, df_test).
    """
    log.info("Loading data splits...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_val   = pd.read_csv(VAL_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    for df, name in [(df_train, "Train"), (df_val, "Val"), (df_test, "Test")]:
        df["processed_text"] = df["processed_text"].fillna("").astype(str)
        df["label_id"]       = df["sentiment_label"].map(LABEL2ID)
        log.info("%s: %d rows — %s", name,
                 len(df), df["sentiment_label"].value_counts().to_dict())

    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# 2. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenised review encodings and labels.

    Args:
        texts:     List of review strings.
        labels:    List of integer label IDs.
        tokenizer: HuggingFace tokenizer.
        max_len:   Maximum token length.
    """

    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ─────────────────────────────────────────────────────────────────────────────
# 3. METRICS FUNCTION FOR TRAINER
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """
    Compute accuracy and macro F1 during training.
    Called by HuggingFace Trainer after each eval step.

    Args:
        eval_pred: EvalPrediction object with logits and label_ids.

    Returns:
        Dict with accuracy and macro_f1.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy" : accuracy_score(labels, preds),
        "macro_f1" : f1_score(labels, preds, average="macro", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. FINE-TUNE BERT
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune_bert(df_train, df_val):
    """
    Fine-tune distilbert-base-uncased for 3-class sentiment.

    Uses:
        - HuggingFace Trainer API
        - AdamW optimiser (built into Trainer)
        - Linear warmup schedule
        - Weight decay regularisation
        - EarlyStopping on eval_loss (patience=PATIENCE)
        - Best model saved automatically to MODEL_SAVE_DIR

    Args:
        df_train: Training DataFrame.
        df_val:   Validation DataFrame.

    Returns:
        Tuple of (trainer, model, tokenizer).
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device.upper())
    log.info("Loading tokenizer and model: %s", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    log.info("Tokenising train (%d) and val (%d) sets...",
             len(df_train), len(df_val))
    train_dataset = ReviewDataset(
        df_train["processed_text"], df_train["label_id"], tokenizer
    )
    val_dataset = ReviewDataset(
        df_val["processed_text"], df_val["label_id"], tokenizer
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = MODEL_SAVE_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LR,
        weight_decay                = WEIGHT_DECAY,
        warmup_ratio                = WARMUP_RATIO,  # linear warmup
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,          # saves best checkpoint
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        logging_dir                 = os.path.join(MODEL_SAVE_DIR, "logs"),
        logging_steps               = 50,
        save_total_limit            = 2,             # keep only 2 checkpoints
        fp16                        = torch.cuda.is_available(),  # half precision on GPU
        report_to                   = "none",        # disable wandb/mlflow
    )

    # ── Trainer with EarlyStopping ────────────────────────────────────────────
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    log.info("=" * 55)
    log.info("  STARTING BERT FINE-TUNING")
    log.info("  Model      : %s", MODEL_NAME)
    log.info("  Epochs     : %d (early stop patience=%d)", EPOCHS, PATIENCE)
    log.info("  Batch size : %d", BATCH_SIZE)
    log.info("  LR         : %s  |  Weight decay: %s", LR, WEIGHT_DECAY)
    log.info("  Warmup     : %s of total steps", WARMUP_RATIO)
    log.info("=" * 55)

    trainer.train()

    # ── Save best model and tokenizer ─────────────────────────────────────────
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    log.info("Best model saved to: %s", MODEL_SAVE_DIR)

    return trainer, model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATE BERT ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bert(model, tokenizer, df_test):
    """
    Run the fine-tuned model on the test set and compute full metrics.

    Args:
        model:     Fine-tuned BERT model.
        tokenizer: Corresponding tokenizer.
        df_test:   Test DataFrame.

    Returns:
        Dict with accuracy, macro_f1, precision, recall,
              confusion_matrix, report, predictions.
    """
    log.info("Evaluating BERT on test set (%d rows)...", len(df_test))

    test_dataset = ReviewDataset(
        df_test["processed_text"], df_test["label_id"], tokenizer
    )

    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    preds_output = trainer.predict(test_dataset)

    y_pred_ids = np.argmax(preds_output.predictions, axis=1)
    y_true_ids = df_test["label_id"].values

    y_pred = [ID2LABEL[i] for i in y_pred_ids]
    y_true = [ID2LABEL[i] for i in y_true_ids]

    accuracy  = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=LABELS)
    report    = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)

    log.info("=" * 55)
    log.info("  BERT TEST SET RESULTS")
    log.info("=" * 55)
    log.info("  Accuracy        : %.4f  (%.1f%%)", accuracy, accuracy * 100)
    log.info("  Macro F1        : %.4f", macro_f1)
    log.info("  Macro Precision : %.4f", precision)
    log.info("  Macro Recall    : %.4f", recall)
    log.info("\nClassification Report:\n%s", report)

    df_test = df_test.copy()
    df_test["bert_label"] = y_pred
    df_test.to_csv(RESULTS_PATH, index=False)
    log.info("BERT predictions saved to: %s", RESULTS_PATH)

    return {
        "accuracy"        : accuracy,
        "macro_f1"        : macro_f1,
        "precision"       : precision,
        "recall"          : recall,
        "confusion_matrix": cm,
        "report"          : report,
        "predictions"     : y_pred,
        "df_test"         : df_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. TF-IDF + LOGISTIC REGRESSION BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_tfidf_baseline(df_train, df_val, df_test):
    """
    Train a TF-IDF + Logistic Regression baseline for comparison.

    Args:
        df_train: Training DataFrame.
        df_val:   Validation DataFrame (used only for reporting).
        df_test:  Test DataFrame.

    Returns:
        Dict with accuracy, macro_f1, precision, recall, confusion_matrix.
    """
    log.info("Training TF-IDF + Logistic Regression baseline...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )),
    ])

    pipeline.fit(df_train["processed_text"], df_train["sentiment_label"])

    y_pred = pipeline.predict(df_test["processed_text"])
    y_true = df_test["sentiment_label"]

    accuracy  = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=LABELS)

    log.info("TF-IDF Baseline — Accuracy: %.4f | Macro F1: %.4f",
             accuracy, macro_f1)

    return {
        "accuracy"        : accuracy,
        "macro_f1"        : macro_f1,
        "precision"       : precision,
        "recall"          : recall,
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. LOAD VADER METRICS FROM SAVED RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def load_vader_metrics(vader_path=VADER_PATH, test_path=TEST_PATH):
    """
    Re-compute VADER metrics on the test split for a fair comparison.
    Falls back to a default dict if vader_results.csv is not found.

    Returns:
        Dict with accuracy, macro_f1, precision, recall.
    """
    try:
        df_vader = pd.read_csv(vader_path)
        df_test  = pd.read_csv(test_path)

        # Match on cleaned_text to get test-split rows only
        df_merged = df_test.merge(
            df_vader[["cleaned_text", "vader_label"]],
            on="cleaned_text", how="left"
        ).dropna(subset=["vader_label"])

        if len(df_merged) == 0:
            raise ValueError("No matching rows found after merge.")

        y_true = df_merged["sentiment_label"]
        y_pred = df_merged["vader_label"]

        return {
            "accuracy" : accuracy_score(y_true, y_pred),
            "macro_f1" : f1_score(y_true, y_pred, average="macro",
                                  labels=LABELS, zero_division=0),
            "precision": precision_score(y_true, y_pred, average="macro",
                                         labels=LABELS, zero_division=0),
            "recall"   : recall_score(y_true, y_pred, average="macro",
                                      labels=LABELS, zero_division=0),
        }
    except Exception as e:
        log.warning("Could not load VADER results (%s) — using placeholder.", e)
        return {"accuracy": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, title, filename, out_dir=PLOTS_DIR):
    """Plot counts + row-% confusion matrix side by side."""
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, t in [
        (axes[0], cm,     "d",   f"{title} (counts)"),
        (axes[1], cm_pct, ".1f", f"{title} (row %)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS,
                    linewidths=0.5, ax=ax, cbar=False)
        ax.set_title(t, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    fpath = os.path.join(out_dir, filename)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Saved: %s", fpath)


def plot_model_comparison(bert_m, vader_m, tfidf_m, out_dir=PLOTS_DIR):
    """
    Plot side-by-side bar chart comparing BERT, VADER, and TF-IDF
    across Accuracy, Macro F1, Precision, and Recall.
    """
    metrics  = ["Accuracy", "Macro F1", "Precision", "Recall"]
    models   = ["BERT (DistilBERT)", "VADER", "TF-IDF + LR"]
    m_colors = ["#2A8AC8", "#E8A020", "#1D9E75"]

    bert_vals  = [bert_m["accuracy"],  bert_m["macro_f1"],
                  bert_m["precision"], bert_m["recall"]]
    vader_vals = [vader_m["accuracy"], vader_m["macro_f1"],
                  vader_m["precision"], vader_m["recall"]]
    tfidf_vals = [tfidf_m["accuracy"], tfidf_m["macro_f1"],
                  tfidf_m["precision"], tfidf_m["recall"]]

    x     = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (vals, name, color) in enumerate(
        zip([bert_vals, vader_vals, tfidf_vals], models, m_colors)
    ):
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=color, edgecolor="none", alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: BERT vs VADER vs TF-IDF + LR",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend()
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    fpath = os.path.join(out_dir, "bert_03_model_comparison.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Saved: %s", fpath)


def plot_training_history(trainer, out_dir=PLOTS_DIR):
    """Plot train loss and eval loss curves across epochs."""
    history = trainer.state.log_history

    train_loss = [(e["epoch"], e["loss"])
                  for e in history if "loss" in e and "eval_loss" not in e]
    eval_data  = [(e["epoch"], e["eval_loss"], e.get("eval_accuracy", None),
                   e.get("eval_macro_f1", None))
                  for e in history if "eval_loss" in e]

    if not train_loss or not eval_data:
        log.warning("No training history available to plot.")
        return

    t_epochs, t_loss         = zip(*train_loss)
    e_epochs, e_loss, e_acc, e_f1 = zip(*eval_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(t_epochs, t_loss, label="Train loss",
                 color="#2A8AC8", linewidth=2)
    axes[0].plot(e_epochs, e_loss, label="Val loss",
                 color="#E24B4A", linewidth=2, marker="o")
    axes[0].set_title("Training & Validation Loss",
                      fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Val accuracy and F1
    if any(v is not None for v in e_acc):
        axes[1].plot(e_epochs, e_acc, label="Val Accuracy",
                     color="#2A8AC8", linewidth=2, marker="o")
    if any(v is not None for v in e_f1):
        axes[1].plot(e_epochs, e_f1, label="Val Macro F1",
                     color="#1D9E75", linewidth=2, marker="s")
    axes[1].set_title("Validation Metrics per Epoch",
                      fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    fpath = os.path.join(out_dir, "bert_02_training_history.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Saved: %s", fpath)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_bert_pipeline():
    """
    Run the full BERT fine-tuning and evaluation pipeline.

    Returns:
        Dict with keys: bert_metrics, tfidf_metrics, vader_metrics, trainer.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    log.info("=" * 55)
    log.info("  BERT FINE-TUNING PIPELINE")
    log.info("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    df_train, df_val, df_test = load_splits()

    # ── TF-IDF baseline (fast — run first) ───────────────────────────────────
    tfidf_metrics = train_tfidf_baseline(df_train, df_val, df_test)

    # ── BERT fine-tuning ──────────────────────────────────────────────────────
    trainer, model, tokenizer = fine_tune_bert(df_train, df_val)

    # ── Training history plot ─────────────────────────────────────────────────
    plot_training_history(trainer)

    # ── BERT test evaluation ──────────────────────────────────────────────────
    bert_metrics = evaluate_bert(model, tokenizer, df_test)

    # ── BERT confusion matrix ─────────────────────────────────────────────────
    plot_confusion_matrix(
        bert_metrics["confusion_matrix"],
        "BERT Confusion Matrix",
        "bert_01_confusion_matrix.png",
    )

    # ── VADER metrics on test split ───────────────────────────────────────────
    vader_metrics = load_vader_metrics()

    # ── 3-way comparison ──────────────────────────────────────────────────────
    plot_model_comparison(bert_metrics, vader_metrics, tfidf_metrics)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("  FINAL COMPARISON  (Test Set)")
    log.info("=" * 55)
    log.info("  %-20s  Acc: %.3f  F1: %.3f",
             "BERT (DistilBERT)", bert_metrics["accuracy"], bert_metrics["macro_f1"])
    log.info("  %-20s  Acc: %.3f  F1: %.3f",
             "TF-IDF + LR", tfidf_metrics["accuracy"], tfidf_metrics["macro_f1"])
    log.info("  %-20s  Acc: %.3f  F1: %.3f",
             "VADER", vader_metrics["accuracy"], vader_metrics["macro_f1"])
    log.info("  Model saved to: %s", MODEL_SAVE_DIR)
    log.info("=" * 55)

    return {
        "bert_metrics" : bert_metrics,
        "tfidf_metrics": tfidf_metrics,
        "vader_metrics": vader_metrics,
        "trainer"      : trainer,
    }


if __name__ == "__main__":
    run_bert_pipeline()