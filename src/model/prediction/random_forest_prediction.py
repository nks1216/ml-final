"""
Random Forest Prediction Model for HMDA Mortgage Loan Approval
===============================================================

Trains a Random Forest classifier and evaluates its performance.
Uses OrdinalEncoder for categorical features (appropriate for tree-based models)
and wraps all preprocessing in a scikit-learn Pipeline.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_COL = "target"
TRAIN_PATH = Path("data/split/train.csv")
TEST_PATH = Path("data/split/test.csv")
RESULTS_DIR = Path("reports/results/prediction")
FIGURES_DIR = Path("reports/figures/prediction")

# ---------------------------------------------------------------------------
# Helper functions (plotting and saving)
# ---------------------------------------------------------------------------
def save_confusion_matrix_plot(cm, save_path: Path) -> None:
    """Save confusion matrix figure."""
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


def save_roc_curve(y_test, y_prob, roc_auc, save_path: Path) -> None:
    """Save ROC curve figure."""
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


def save_precision_recall_curve(y_test, y_prob, avg_precision, save_path: Path) -> None:
    """Save Precision-Recall curve figure."""
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


def save_feature_importance_plot(importance_df: pd.DataFrame, save_path: Path) -> None:
    """Save horizontal bar chart of top 20 feature importances."""
    top20 = importance_df.head(20).sort_values("importance")
    plt.figure(figsize=(10, 8))
    plt.barh(top20["feature"], top20["importance"])
    plt.title("Random Forest Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main() -> None:
    """Load data, train Random Forest, evaluate, and save all outputs."""

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load data
    # ----------------------------
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ----------------------------
    # Identify column types
    # ----------------------------
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumber of numeric features:", len(numeric_features))
    print("Number of categorical features:", len(categorical_features))

    # ----------------------------
    # Preprocessing pipeline
    # ----------------------------
    # OrdinalEncoder is chosen over OneHotEncoder because Random Forest
    # is a tree-based model that handles integer-coded categories without
    # assuming any ordinal relationship between them. This keeps the
    # feature space compact and makes importance scores directly
    # interpretable per original feature.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical_features,
            ),
        ]
    )

    # ----------------------------
    # Full model pipeline
    # ----------------------------
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ----------------------------
    # Train
    # ----------------------------
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # ----------------------------
    # Predict
    # ----------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ----------------------------
    # Metrics
    # ----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Random Forest Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # ----------------------------
    # Save metrics and reports
    # ----------------------------
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "average_precision": float(avg_precision),
    }

    with open(RESULTS_DIR / "random_forest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(RESULTS_DIR / "random_forest_classification_report.txt", "w") as f:
        f.write(report)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    )
    cm_df.to_csv(RESULTS_DIR / "random_forest_confusion_matrix.csv", index=True)

    predictions_df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred,
        "predicted_probability": y_prob,
    })
    predictions_df.to_csv(RESULTS_DIR / "random_forest_test_predictions.csv", index=False)

    # ----------------------------
    # Feature importances
    # ----------------------------
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    importance_df.head(20).to_csv(
        RESULTS_DIR / "random_forest_feature_importance_top20.csv", index=False
    )

    # ----------------------------
    # Plots
    # ----------------------------
    save_confusion_matrix_plot(cm, FIGURES_DIR / "random_forest_confusion_matrix.png")
    save_roc_curve(y_test, y_prob, roc_auc, FIGURES_DIR / "random_forest_roc_curve.png")
    save_precision_recall_curve(
        y_test, y_prob, avg_precision,
        FIGURES_DIR / "random_forest_precision_recall_curve.png",
    )
    save_feature_importance_plot(
        importance_df,
        FIGURES_DIR / "random_forest_feature_importance_top20.png",
    )

    print("\nSaved Random Forest results and figures successfully.")


if __name__ == "__main__":
    main()