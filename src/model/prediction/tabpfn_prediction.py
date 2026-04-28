"""
TabPFN Prediction Model for HMDA Mortgage Loan Approval
========================================================

TabPFN (Tabular Prior-Data Fitted Network) is a pre-trained transformer for
small-to-medium tabular data. It performs *in-context learning*: training data
is fed into the model at inference time rather than via gradient updates.

Because TabPFN v2 is designed for ~10,000 training rows and our HMDA training
set has 156,379 rows, we use a **stratified subsample of 10,000 rows** so the
model stays within its pretraining regime. The full 39,095-row test set is
still used for evaluation, so the comparison with other models remains fair on
the test side.

Outputs saved to `reports/results/prediction/` (all prefixed with `tabpfn_`):
    - tabpfn_metrics.json                 : accuracy, precision, recall, F1, ROC-AUC
    - tabpfn_classification_report.csv    : per-class precision/recall/F1
    - tabpfn_confusion_matrix.csv         : raw confusion matrix
    - tabpfn_confusion_matrix.png         : heatmap
    - tabpfn_roc_curve.png                : ROC curve
    - tabpfn_precision_recall_curve.png   : Precision–Recall curve
    - tabpfn_feature_importance_top20.png : top-20 permutation importances
    - tabpfn_feature_importance_all.csv   : full importance table (permutation)
    - tabpfn_test_predictions.csv         : row-level predictions with probabilities

----------------------------------------------------------------------
ONE-TIME SETUP (required by TabPFN v2)
----------------------------------------------------------------------
TabPFN downloads its pretrained weights from Hugging Face under a free license
that requires a one-time acceptance.

1. Register / log in at https://ux.priorlabs.ai
2. Accept the license on the "Licenses" tab
3. Copy your API key from https://ux.priorlabs.ai/account
4. Export it before running this script:
       export TABPFN_TOKEN="<your-api-key>"
   Or set it inside Python:
       import os; os.environ["TABPFN_TOKEN"] = "<your-api-key>"

----------------------------------------------------------------------
INSTALLATION
----------------------------------------------------------------------
    pip install tabpfn

----------------------------------------------------------------------
USAGE
----------------------------------------------------------------------
    export TABPFN_TOKEN="<your-api-key>"
    python3 src/model/prediction/tabpfn_prediction.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "split" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "split" / "test.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results" / "prediction"

TARGET_COL = "target"
RANDOM_STATE = 42

# TabPFN v2 is pretrained for up to ~10K samples. We stratified-subsample to
# stay within its design regime.
TRAIN_SUBSAMPLE_SIZE = 1_000

# Permutation importance is *very* expensive with TabPFN (each permutation
# requires a full re-prediction). Use a small test sample.
PERM_SAMPLE_SIZE = 300
PERM_N_REPEATS = 1

# Predicting on a large test set in one shot can exhaust GPU/MPS memory.
# Splitting predictions into small batches keeps memory bounded. Lower this
# if you still hit OOM (e.g., 1000 or 500); raise it for faster runs.
PREDICT_BATCH_SIZE = 100

# Permutation importance is informative but slow on CPU (1-2 hours typical).
# Set to False to skip it — main metrics and curves will still be produced.
COMPUTE_FEATURE_IMPORTANCE = True


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data():
    print("[1/6] Loading data ...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"       train shape = {train.shape} | test shape = {test.shape}")

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL].astype(int)
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL].astype(int)
    return X_train, y_train, X_test, y_test


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """TabPFN expects numeric input. Label-encode categorical columns using
    codes from the union of train+test, and remember the column indices so
    we can pass `categorical_features_indices` to TabPFN."""
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    print(f"[2/6] Encoding {len(cat_cols)} categorical columns: {cat_cols}")

    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in cat_cols:
        combined = pd.concat(
            [X_train[col].astype(str), X_test[col].astype(str)],
            axis=0,
        ).reset_index(drop=True)
        codes, _ = pd.factorize(combined)
        X_train[col] = codes[: len(X_train)]
        X_test[col] = codes[len(X_train):]

    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
    return X_train, X_test, cat_indices


def stratified_subsample(X: pd.DataFrame, y: pd.Series, size: int):
    """Stratified subsample preserving the class ratio."""
    if len(X) <= size:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Batched prediction helpers (avoid GPU/MPS out-of-memory)
# --------------------------------------------------------------------------- #
def predict_proba_batched(clf, X, batch_size: int = PREDICT_BATCH_SIZE,
                          verbose: bool = True):
    """Run clf.predict_proba in batches and concatenate results.

    Prints per-batch progress so long CPU runs don't look frozen.
    """
    import time, sys
    preds = []
    n = len(X)
    n_batches = (n + batch_size - 1) // batch_size
    t_start = time.time()
    for i, start in enumerate(range(0, n, batch_size), 1):
        end = min(start + batch_size, n)
        t0 = time.time()
        preds.append(clf.predict_proba(X[start:end]))
        if verbose:
            elapsed_total = time.time() - t_start
            elapsed_batch = time.time() - t0
            eta = (elapsed_total / i) * (n_batches - i)
            print(f"       batch {i}/{n_batches} "
                  f"[{start}:{end}] took {elapsed_batch:.1f}s | "
                  f"elapsed {elapsed_total/60:.1f}m | "
                  f"ETA {eta/60:.1f}m",
                  flush=True)
            sys.stdout.flush()
    return np.vstack(preds)


class BatchedClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible thin wrapper that batches predict_proba calls.

    Used so that scikit-learn's permutation_importance — which internally
    calls predict_proba on the (possibly large) sample — never asks TabPFN
    for more than `batch_size` rows at once.

    Inherits from ClassifierMixin so sklearn's `is_classifier` returns True
    and scoring functions (e.g., roc_auc) route through predict_proba.
    """

    # Belt-and-suspenders: works on older sklearn versions that key off this
    _estimator_type = "classifier"

    def __init__(self, base_clf=None, batch_size: int = PREDICT_BATCH_SIZE):
        self.base_clf = base_clf
        self.batch_size = batch_size
        if base_clf is not None and hasattr(base_clf, "classes_"):
            self.classes_ = base_clf.classes_

    # No-op fit so sklearn's check_is_fitted is satisfied
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return predict_proba_batched(self.base_clf, X, self.batch_size,
                                     verbose=False)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def train_model(X_train_np, y_train_np, cat_indices):
    print(f"[3/6] Fitting TabPFN on {len(X_train_np):,} subsampled rows ...")
    clf = TabPFNClassifier(
        # On Apple Silicon Macs, MPS has a hard ~12 GiB limit and TabPFN's
        # in-context learning loads the full training set per prediction,
        # which can blow that budget. CPU uses system RAM (16-32 GiB on
        # most Macs) and runs reliably. Change to "auto" if you have a
        # CUDA GPU with sufficient VRAM.
        device="cpu",
        n_estimators=4,                  # internal ensemble (lower = faster)
        categorical_features_indices=cat_indices,
        random_state=RANDOM_STATE,
        # Required to allow >1,000 training samples on CPU. Our 10K subsample
        # is still within TabPFN v2's pretraining regime; this flag just
        # bypasses the CPU-speed safety check.
        ignore_pretraining_limits=True,
    )
    clf.fit(X_train_np, y_train_np)
    print("       Fit complete (TabPFN does not iteratively train — "
          "it stores the data for in-context inference).")
    return clf


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(clf, X_test_np, y_test, results_dir: Path):
    print(f"[4/6] Predicting on full test set "
          f"({len(X_test_np):,} rows, batch_size={PREDICT_BATCH_SIZE}) ...")
    y_pred_proba = predict_proba_batched(clf, X_test_np)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "average_precision": average_precision_score(y_test, y_pred_proba),
        "n_test": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
        "train_subsample_size": int(TRAIN_SUBSAMPLE_SIZE),
    }
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"       {k:>20s} = {v:.4f}")
        else:
            print(f"       {k:>20s} = {v}")

    # Save metrics
    (results_dir / "tabpfn_metrics.json").write_text(json.dumps(metrics, indent=2))

    report_dict = classification_report(
        y_test, y_pred, output_dict=True,
        target_names=["Denied (0)", "Approved (1)"],
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        results_dir / "tabpfn_classification_report.csv", index=True
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Denied", "Actual Approved"],
        columns=["Pred Denied", "Pred Approved"],
    )
    cm_df.to_csv(results_dir / "tabpfn_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
        xticklabels=["Denied", "Approved"],
        yticklabels=["Denied", "Approved"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("TabPFN Confusion Matrix")
    fig.tight_layout()
    fig.savefig(results_dir / "tabpfn_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"TabPFN (AUC = {roc_auc:.4f})", color="#ff7f0e")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — TabPFN")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "tabpfn_roc_curve.png", dpi=150)
    plt.close(fig)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"TabPFN (AP = {ap:.4f})", color="#8c564b")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve — TabPFN")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "tabpfn_precision_recall_curve.png", dpi=150)
    plt.close(fig)

    # Test predictions
    preds_df = pd.DataFrame({
        "y_true": y_test.reset_index(drop=True),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    })
    preds_df.to_csv(results_dir / "tabpfn_test_predictions.csv", index=False)

    return metrics


# --------------------------------------------------------------------------- #
# Feature importance via permutation
# --------------------------------------------------------------------------- #
def plot_feature_importance(
    clf, X_test_df: pd.DataFrame, X_test_np, y_test,
    results_dir: Path, top_n: int = 20,
):
    """
    TabPFN has no built-in importance score, so we use permutation importance.
    Each permutation triggers a full re-prediction, which is expensive — so we
    use a small subsample of the test set.
    """
    print(f"[5/6] Computing permutation importance "
          f"(sample={PERM_SAMPLE_SIZE}, repeats={PERM_N_REPEATS}) "
          "— this is the slow step.")

    sample_size = min(PERM_SAMPLE_SIZE, len(X_test_np))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_test_np), size=sample_size, replace=False)
    X_sample = X_test_np[idx]
    y_sample = y_test.values[idx] if hasattr(y_test, "values") else y_test[idx]

    # Wrap the classifier so each predict_proba call inside permutation_importance
    # is automatically batched (avoids MPS/CUDA OOM during repeated permutations).
    batched_clf = BatchedClassifier(clf, batch_size=PREDICT_BATCH_SIZE)

    result = permutation_importance(
        batched_clf, X_sample, y_sample,
        scoring="roc_auc",
        n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,                # TabPFN already uses internal parallelism
    )

    imp_df = (
        pd.DataFrame({
            "feature": X_test_df.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(results_dir / "tabpfn_feature_importance_all.csv", index=False)

    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["importance_mean"],
            xerr=top["importance_std"], color="#ff9896", ecolor="grey")
    ax.set_xlabel("Drop in ROC-AUC when feature is permuted")
    ax.set_title(f"TabPFN Top {top_n} Feature Importances (Permutation)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "tabpfn_feature_importance_top20.png", dpi=150)
    plt.close(fig)
    return imp_df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test, cat_indices = encode_categoricals(X_train, X_test)

    # Stratified subsample to stay within TabPFN's pretraining regime
    X_train_sub, y_train_sub = stratified_subsample(
        X_train, y_train, TRAIN_SUBSAMPLE_SIZE
    )
    print(f"       Subsampled training set: {len(X_train_sub):,} rows "
          f"(positive rate = {y_train_sub.mean():.4f})")

    # TabPFN expects numpy arrays
    X_train_np = X_train_sub.values.astype(np.float32)
    y_train_np = y_train_sub.values.astype(int)
    X_test_np = X_test.values.astype(np.float32)

    clf = train_model(X_train_np, y_train_np, cat_indices)
    metrics = evaluate(clf, X_test_np, y_test, RESULTS_DIR)

    imp_df = None
    if COMPUTE_FEATURE_IMPORTANCE:
        imp_df = plot_feature_importance(
            clf, X_test, X_test_np, y_test, RESULTS_DIR, top_n=20
        )
    else:
        print("[5/6] Skipping permutation importance "
              "(COMPUTE_FEATURE_IMPORTANCE = False).")

    print("\n" + "=" * 60)
    print("TabPFN Prediction — Summary")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:>20s} : {v:.4f}" if isinstance(v, float) else f"{k:>20s} : {v}")
    if imp_df is not None:
        print("\nTop 5 features by permutation importance:")
        print(imp_df.head(5).to_string(index=False))
    print(f"\nAll outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
