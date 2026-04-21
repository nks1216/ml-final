import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# 1. Set Matplotlib backend to 'Agg' for non-interactive environments (WSL/Servers)
import matplotlib
matplotlib.use('Agg')

tqdm.pandas()

def train_catboost_model(data_path):
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return None

    print(f"Step 1: Loading dataset...")
    df = pd.read_csv(data_path, low_memory=False)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # FIX: Pandas4Warning - Explicitly include 'object' and 'string'
    cat_features = X.select_dtypes(include=['object', 'string']).columns.tolist()
    print(f"Detected {len(cat_features)} categorical features.")
    
    print("Step 2: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100 
    )
    
    print("Step 3: Training...")
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), early_stopping_rounds=50)
    
    print("\nStep 4: Evaluating...")
    y_pred = model.predict(X_test)
    print(f"\nFinal Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # --- Step 5: Visualizing and Saving Plot ---
    print("Step 5: Saving feature importance plot...")
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    # FIX: FutureWarning - Assign 'Feature' to hue and set legend=False
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='magma', hue='Feature', legend=False)
    
    plt.title('Top 20 Features Influencing Mortgage Decisions')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.tight_layout()

    # --- Directory Logic for Saving ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(base_dir, '../../reports/figures/')
    
    # Create directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    
    save_path = os.path.join(report_dir, 'feature_importance.png')
    plt.savefig(save_path)
    print(f"Plot successfully saved to: {save_path}")
    
    return model

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(base_dir, '../../data/clean/hmda_cleaned.csv')
    train_catboost_model(data_file_path)