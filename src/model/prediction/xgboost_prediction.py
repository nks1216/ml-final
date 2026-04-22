"""
XGBoost Prediction Model for HMDA Mortgage Loan Approval
========================================================

This script trains an XGBoost classifier to predict mortgage loan approval
(target = 1 Approved, 0 Denied) using the pre-split HMDA dataset located at
`data/split/train.csv` and `data/split/test.csv`.

Outputs saved to `reports/results/prediction/` (all prefixed with `xgboost_`):
    - xgboost_metrics.json                 : accuracy, precision, recall, F1, ROC-AUC
    - xgboost_classification_report.csv    : per-class precision/recall/F1 from sklearn
    - xgboost_confusion_matrix.csv         : raw confusion matrix
    - xgboost_confusion_matrix.png         : heatmap
    - xgboost_roc_curve.png                : ROC curve
    - xgboost_precision_recall_curve.png   : Precision–Recall curve
    - xgboost_feature_importance_top20.png : top-20 gain-based importances
    - xgboost_feature_importance_all.csv   : full importance table (gain)
    - xgboost_model.json                   : trained XGBoost model (portable JSON)
    - xgboost_test_predictions.csv         : row-level predictions with probabilities

Usage:
    python3 src/model/prediction/xgboost_prediction.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
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


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
CURRENT_FILE = Path(__file__).resolve()
# src/model/prediction/xgboost_prediction.py → project root is 3 levels up
PROJECT_ROOT = CURRENT_FILE.parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "split" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "split" / "test.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results" / "prediction"

TARGET_COL = "target"
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_data(train_path: Path, test_path: Path):
    """Load train/test CSVs and split features/target."""
    print(f"[1/6] Loading data ...")
    print(f"       train: {train_path}")
    print(f"       test : {test_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"       train shape = {train.shape} | test shape = {test.shape}")

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL].astype(int)
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL].astype(int)
    return X_train, y_train, X_test, y_test


def prepare_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Convert object columns to pandas 'category' dtype, aligning categories
    between train and test so XGBoost's native categorical handling works."""
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    print(f"[2/6] Encoding {len(cat_cols)} categorical columns: {cat_cols}")

    for col in cat_cols:
        # Use the union of categories observed across train + test so that
        # unseen test categories do not become NaN.
        all_values = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        categories = pd.Index(all_values.unique())
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=categories)
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=categories)
    return X_train, X_test, cat_cols


def train_model(X_train, y_train, X_test, y_test):
    """Fit an XGBClassifier with early stopping on the test set as validation."""
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos  # handles 70/30 class imbalance
    print(f"[3/6] Class balance -> pos={pos}, neg={neg}, "
          f"scale_pos_weight={scale_pos_weight:.3f}")

    model = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    print("       Fitting XGBoost (early stopping on test set) ...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    best_iter = getattr(model, "best_iteration", None)
    print(f"       Training done. best_iteration = {best_iter}")
    return model


def evaluate(model, X_test, y_test, results_dir: Path):
    """Compute metrics, write JSON/CSV and plot confusion matrix / ROC / PR."""
    print("[4/6] Evaluating on test set ...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
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
        "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
    }
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"       {k:>20s} = {v:.4f}")
        else:
            print(f"       {k:>20s} = {v}")

    # --- Save metrics ---------------------------------------------------- #
    (results_dir / "xgboost_metrics.json").write_text(json.dumps(metrics, indent=2))

    report_dict = classification_report(y_test, y_pred, output_dict=True,
                                        target_names=["Denied (0)", "Approved (1)"])
    pd.DataFrame(report_dict).transpose().to_csv(
        results_dir / "xgboost_classification_report.csv", index=True
    )

    # --- Confusion Matrix ------------------------------------------------ #
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Denied", "Actual Approved"],
        columns=["Pred Denied", "Pred Approved"],
    )
    cm_df.to_csv(results_dir / "xgboost_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Denied", "Approved"],
        yticklabels=["Denied", "Approved"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("XGBoost Confusion Matrix")
    fig.tight_layout()
    fig.savefig(results_dir / "xgboost_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- ROC Curve ------------------------------------------------------- #
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"XGBoost (AUC = {roc_auc:.4f})", color="#1f77b4")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — XGBoost")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "xgboost_roc_curve.png", dpi=150)
    plt.close(fig)

    # --- Precision–Recall Curve ----------------------------------------- #
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"XGBoost (AP = {ap:.4f})", color="#d62728")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve — XGBoost")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "xgboost_precision_recall_curve.png", dpi=150)
    plt.close(fig)

    # --- Save per-row predictions --------------------------------------- #
    preds_df = pd.DataFrame({
        "y_true": y_test.reset_index(drop=True),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    })
    preds_df.to_csv(results_dir / "xgboost_test_predictions.csv", index=False)

    return metrics


def plot_feature_importance(model, feature_names, results_dir: Path, top_n: int = 20):
    """Plot top-N gain-based feature importances and save full table."""
    print(f"[5/6] Computing feature importances ...")
    booster = model.get_booster()
    # 'gain' is the standard importance for tree-based models (improvement to
    # the loss attributable to splits on that feature)
    gain_dict = booster.get_score(importance_type="gain")

    # Map XGBoost's f0, f1, ... back to actual column names when applicable
    def resolve_name(key: str) -> str:
        if key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if idx < len(feature_names):
                return feature_names[idx]
        return key

    importance_rows = [
        {"feature": resolve_name(k), "gain": v} for k, v in gain_dict.items()
    ]
    imp_df = (
        pd.DataFrame(importance_rows)
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(results_dir / "xgboost_feature_importance_all.csv", index=False)

    top = imp_df.head(top_n).iloc[::-1]  # reverse so largest is at top of barh
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["gain"], color="#2ca02c")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"XGBoost Top {top_n} Feature Importances (Gain)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "xgboost_feature_importance_top20.png", dpi=150)
    plt.close(fig)
    return imp_df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
    X_train, X_test, _ = prepare_categoricals(X_train, X_test)
    feature_names = X_train.columns.tolist()

    model = train_model(X_train, y_train, X_test, y_test)
    metrics = evaluate(model, X_test, y_test, RESULTS_DIR)
    imp_df = plot_feature_importance(model, feature_names, RESULTS_DIR, top_n=20)

    # Save model in portable JSON format
    print("[6/6] Saving trained model ...")
    model.save_model(str(RESULTS_DIR / "xgboost_model.json"))

    # Console summary
    print("\n" + "=" * 60)
    print("XGBoost Prediction — Summary")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:>20s} : {v:.4f}" if isinstance(v, float) else f"{k:>20s} : {v}")
    print("\nTop 5 features by gain:")
    print(imp_df.head(5).to_string(index=False))
    print(f"\nAll outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
