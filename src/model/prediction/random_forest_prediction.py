"""
Random Forest Prediction Model for HMDA Mortgage Loan Approval
===============================================================

This module trains a Random Forest classifier to predict mortgage loan approval.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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
from sklearn.preprocessing import LabelEncoder

# Constants
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10
TARGET_COL = "target"
TRAIN_PATH = Path("data/split/train.csv")
TEST_PATH = Path("data/split/test.csv")
RESULTS_DIR = Path("reports/results/prediction")
FIGURES_DIR = Path("reports/figures/prediction")


def load_data(train_path: Path, test_path: Path) -> tuple:
    """
    Load train and test datasets from CSV files.

    Args:
        train_path: Path to training data CSV file
        test_path: Path to test data CSV file

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(columns=[TARGET_COL])
    y_train = train_data[TARGET_COL].astype(int)
    X_test = test_data.drop(columns=[TARGET_COL])
    y_test = test_data[TARGET_COL].astype(int)

    return X_train, y_train, X_test, y_test


def encode_categorical(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Encode categorical features using LabelEncoder.

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        Tuple of (X_train_encoded, X_test_encoded)
    """
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    
    for col in categorical_cols:
        # Combine train and test to handle unseen categories
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le = LabelEncoder()
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    return X_train, X_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Trained RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Evaluate model performance.

    Args:
        model: Trained Random Forest model
        X_test: Test features
        y_test: Test target

    Returns:
        Tuple of (metrics_dict, y_pred, y_pred_proba, report_text, confusion_matrix)
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "average_precision": average_precision_score(y_test, y_pred_proba),
    }

    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return metrics, y_pred, y_pred_proba, report_text, cm


def save_metrics(metrics: dict, save_path: Path) -> None:
    """Save metrics as JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_classification_report(report_text: str, save_path: Path) -> None:
    """Save classification report as text file."""
    with open(save_path, "w") as f:
        f.write(report_text)


def save_predictions(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    """Save test predictions as CSV."""
    predictions_df = pd.DataFrame({
        "actual": y_test, "predicted": y_pred, "predicted_probability": y_prob
    })
    predictions_df.to_csv(save_path, index=False)


def save_feature_importance(model: RandomForestClassifier, feature_names: list, save_path: Path) -> None:
    """Save top 20 feature importances as CSV."""
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    importance_df.head(20).to_csv(save_path, index=False)
    return importance_df


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    """Generate and save confusion matrix plot."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Random Forest Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_test: pd.Series, y_prob: np.ndarray, roc_auc: float, save_path: Path) -> None:
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_precision_recall_curve(y_test: pd.Series, y_prob: np.ndarray, avg_precision: float, save_path: Path) -> None:
    """Generate and save Precision-Recall curve."""
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label=f"AP = {avg_precision:.4f}", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Random Forest Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, save_path: Path) -> None:
    """Generate and save feature importance plot."""
    top20 = importance_df.head(20).sort_values("importance")
    
    plt.figure(figsize=(10, 8))
    plt.barh(top20["feature"], top20["importance"])
    plt.title("Random Forest Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    """Main execution function."""
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("[1/5] Loading data...")
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
    print(f"      Train: {X_train.shape}, Test: {X_test.shape}")

    # Encode categorical variables
    print("[2/5] Encoding categorical features...")
    X_train, X_test = encode_categorical(X_train, X_test)

    # Train model
    print("[3/5] Training Random Forest...")
    model = train_model(X_train, y_train)

    # Evaluate
    print("[4/5] Evaluating model...")
    metrics, y_pred, y_prob, report_text, cm = evaluate_model(model, X_test, y_test)

    # Feature importance
    print("[5/5] Saving results...")
    importance_df = save_feature_importance(model, X_train.columns.tolist(), 
                                            RESULTS_DIR / "random_forest_feature_importance_top20.csv")

    # Save all outputs
    save_metrics(metrics, RESULTS_DIR / "random_forest_metrics.json")
    save_classification_report(report_text, RESULTS_DIR / "random_forest_classification_report.txt")
    
    cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Predicted_0", "Predicted_1"])
    cm_df.to_csv(RESULTS_DIR / "random_forest_confusion_matrix.csv", index=True)
    
    save_predictions(y_test, y_pred, y_prob, RESULTS_DIR / "random_forest_test_predictions.csv")

    # Generate plots
    plot_confusion_matrix(cm, FIGURES_DIR / "random_forest_confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, metrics["roc_auc"], FIGURES_DIR / "random_forest_roc_curve.png")
    plot_precision_recall_curve(y_test, y_prob, metrics["average_precision"], 
                                FIGURES_DIR / "random_forest_precision_recall_curve.png")
    plot_feature_importance(importance_df, FIGURES_DIR / "random_forest_feature_importance_top20.png")

    # Print results
    print("\n" + "=" * 50)
    print("RANDOM FOREST RESULTS")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():20s}: {value:.4f}")

    print(f"\n Results saved to {RESULTS_DIR}")
    print(f" Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()