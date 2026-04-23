import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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
# Identify column types
# ----------------------------
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

print("\nNumber of numeric features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))

# ----------------------------
# Preprocessing
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ----------------------------
# Model pipeline
# ----------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            ),
        ),
    ]
)

# ----------------------------
# Train
# ----------------------------
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

print("\n=== Logistic Regression Results ===")
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

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "average_precision": float(avg_precision),
}

with open(f"{results_dir}/logistic_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(f"{results_dir}/logistic_classification_report.txt", "w") as f:
    f.write(report)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
)
cm_df.to_csv(f"{results_dir}/logistic_confusion_matrix.csv", index=True)

predictions_df = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "predicted_probability": y_prob,
})
predictions_df.to_csv(f"{results_dir}/logistic_test_predictions.csv", index=False)

# ----------------------------
# Confusion matrix plot
# ----------------------------
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Logistic Regression Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["Actual 0", "Actual 1"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{figures_dir}/logistic_confusion_matrix.png", dpi=300)
plt.close()

# ----------------------------
# ROC curve
# ----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{figures_dir}/logistic_roc_curve.png", dpi=300)
plt.close()

# ----------------------------
# Precision-Recall curve
# ----------------------------
precisions, recalls, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions, label=f"AP = {avg_precision:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Logistic Regression Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{figures_dir}/logistic_precision_recall_curve.png", dpi=300)
plt.close()

# ----------------------------
# Top 20 coefficients
# ----------------------------
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["classifier"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
})
coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

coef_df.head(20).to_csv(f"{results_dir}/logistic_top20_coefficients.csv", index=False)

top20 = coef_df.head(20).sort_values("coefficient")

plt.figure(figsize=(10, 8))
plt.barh(top20["feature"], top20["coefficient"])
plt.title("Top 20 Logistic Regression Coefficients")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig(f"{figures_dir}/logistic_top20_coefficients.png", dpi=300)
plt.close()

print("\nSaved logistic regression results and figures successfully.")