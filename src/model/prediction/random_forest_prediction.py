import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv("data/split/train.csv")
test_data = pd.read_csv("data/split/test.csv")

X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]
X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]

string_columns = X_train.select_dtypes(include=['object']).columns

if len(string_columns) > 0:
    print(f"   Found {len(string_columns)} text columns that need conversion")
    
    for col in string_columns:
        print(f"\n   Processing column: '{col}'")
        
        combined = pd.concat([X_train[col], X_test[col]]).astype(str)
        unique_values = combined.unique()
        print(f"     Found {len(unique_values)} unique values")
        
        le = LabelEncoder()
        le.fit(combined)
        
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
        print(f"     ✓ Converted successfully")
else:
    print("   No text columns found - good to go!")

print("\n3. Checking for numeric columns stored as text...")
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        try:
            X_train[col] = pd.to_numeric(X_train[col])
            X_test[col] = pd.to_numeric(X_test[col])
            print(f"   Converted '{col}' to numbers")
        except:
            pass

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"ROC-AUC:            {roc_auc:.4f}")
print(f"Average Precision:  {avg_precision:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("reports/figures/prediction/rf_confusion_matrix.png", dpi=300)
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.savefig("reports/figures/prediction/rf_roc_curve.png", dpi=300)
plt.show()

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve, 'g-', label=f'PR Curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.legend()
plt.savefig("reports/figures/prediction/rf_precision_recall_curve.png", dpi=300)
plt.show()

importances = model.feature_importances_
indices = np.argsort(importances)[-20:]

plt.figure(figsize=(10, 8))
plt.barh(range(20), importances[indices])
plt.yticks(range(20), [X_train.columns[i] for i in indices])
plt.xlabel('Importance')
plt.title('Random Forest Top 20 Feature Importances')
plt.tight_layout()
plt.savefig("reports/figures/prediction/rf_feature_importances.png", dpi=300)
plt.show()

print("\nTop 5 Most Important Features:")
for i in range(5):
    idx = indices[-1-i]
    print(f"   {i+1}. {X_train.columns[idx]}: {importances[idx]:.4f}")

results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "average_precision": avg_precision
}

results_df = pd.DataFrame([results])
results_df.to_csv("reports/results/prediction/rf_model_metrics.csv", index=False)