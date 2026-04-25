"""
Random Forest Prediction Model for HMDA Mortgage Loan Approval
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

# ----------------------------
# Load data
# ----------------------------
train = pd.read_csv("data/split/train.csv")
test = pd.read_csv("data/split/test.csv")

X_train = train.drop(columns=["target"])
y_train = train["target"]

X_test = test.drop(columns=["target"])
y_test = test["target"]

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ----------------------------
# Encode categorical variables
# ----------------------------
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

print("\nNumber of categorical features:", len(categorical_features))

for col in categorical_features:
    # Combine train and test to handle unseen categories
    combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
    le = LabelEncoder()
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# ----------------------------
# Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# ----------------------------
# Train
# ----------------------------
print("\nTraining Random Forest model...")
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
# Save results
# ----------------------------
results_dir = "reports/results/prediction"
figures_dir = "reports/figures/prediction"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Save metrics JSON
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "average_precision": float(avg_precision),
}

with open(f"{results_dir}/random_forest_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save classification report
with open(f"{results_dir}/random_forest_classification_report.txt", "w") as f:
    f.write(report)

# Save confusion matrix CSV
cm_df = pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
)
cm_df.to_csv(f"{results_dir}/random_forest_confusion_matrix.csv", index=True)

# Save feature importances (top 20)
feature_names = X_train.columns.tolist()
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
})
importance_df = importance_df.sort_values("importance", ascending=False)
importance_df.head(20).to_csv(f"{results_dir}/random_forest_feature_importance_top20.csv", index=False)

# ----------------------------
# Save test predictions
# ----------------------------
predictions_df = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "predicted_probability": y_prob,
})
predictions_df.to_csv(f"{results_dir}/random_forest_test_predictions.csv", index=False)
print(f"Saved test predictions to {results_dir}/random_forest_test_predictions.csv")

# ----------------------------
# Confusion matrix plot
# ----------------------------
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
plt.savefig(f"{figures_dir}/random_forest_confusion_matrix.png", dpi=300)
plt.close()

# ----------------------------
# ROC curve
# ----------------------------
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
plt.savefig(f"{figures_dir}/random_forest_roc_curve.png", dpi=300)
plt.close()

# ----------------------------
# Precision-Recall curve
# ----------------------------
precisions, recalls, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions, label=f"AP = {avg_precision:.4f}", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest Precision-Recall Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{figures_dir}/random_forest_precision_recall_curve.png", dpi=300)
plt.close()

# ----------------------------
# Feature importance plot (top 20)
# ----------------------------
top20 = importance_df.head(20).sort_values("importance")

plt.figure(figsize=(10, 8))
plt.barh(top20["feature"], top20["importance"])
plt.title("Random Forest Top 20 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{figures_dir}/random_forest_feature_importance_top20.png", dpi=300)
plt.close()

print("\nSaved random forest results and figures successfully.")