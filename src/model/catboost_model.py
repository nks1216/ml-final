import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib

# Use 'Agg' for headless environments (no display needed)
matplotlib.use('Agg')

def train_catboost_model():
    # --- 1. Robust Path Setup ---
    # Get the absolute path of the current script (ml-final/src/model/catboost_model.py)
    current_script_path = os.path.abspath(__file__)
    # Go up two levels to get to the project root (ml-final/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    
    # Define absolute paths
    data_path = os.path.join(project_root, 'data/clean/hmda_cleaned.csv')
    results_dir = os.path.join(project_root, 'reports/results/catboost/')
    figures_dir = os.path.join(project_root, 'reports/figures/catboost/')

    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"--- Checkpoint 1: Paths Initialized ---")
    print(f"Data Path: {data_path}")
    print(f"Figures will be saved to: {figures_dir}")

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}. Please check the file location.")
        return

    # --- 2. Data Loading & Preprocessing ---
    print(f"\n--- Checkpoint 2: Loading Data ---")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Define target and features
    target_col = 'target'
    if target_col not in df.columns:
        print(f"ERROR: '{target_col}' column not found in dataset!")
        return

    # Handle numeric columns (convert 'Unknown' or strings to NaN)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify categorical features
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    # Fill NaN in categorical with 'Unknown' so CatBoost doesn't complain
    X[cat_features] = X[cat_features].astype(str).replace('nan', 'Unknown')

    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"Categorical features detected: {len(cat_features)}")

    # --- 3. Data Splitting ---
    print("\n--- Checkpoint 3: Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. Model Training ---
    print("\n--- Checkpoint 4: Starting Training ---")
    # Check if multiclass
    num_classes = y.nunique()
    loss_fn = 'MultiClass' if num_classes > 2 else 'Logloss'
    
    model = CatBoostClassifier(
        iterations=500, # Reduced for faster initial test
        learning_rate=0.05,
        depth=6,
        loss_function=loss_fn,
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100,
        train_dir=results_dir,
        allow_writing_files=True
    )

    model.fit(
        X_train, y_train, 
        cat_features=cat_features, 
        eval_set=(X_test, y_test), 
        early_stopping_rounds=50
    )

    # --- 5. Evaluation ---
    print("\n--- Checkpoint 5: Evaluating ---")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- 6. Saving Feature Importance Plot ---
    print("\n--- Checkpoint 6: Saving Figure ---")
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False).head(20)

    plt.figure(figsize=(12, 10))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=importance_df, 
        palette='magma', 
        hue='Feature', 
        legend=False
    )
    plt.title(f'Top 20 Features for {target_col} Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.tight_layout()

    save_path = os.path.join(figures_dir, 'feature_importance.png')
    plt.savefig(save_path)
    plt.close()

    if os.path.exists(save_path):
        print(f"SUCCESS! Figure saved at: {save_path}")
    else:
        print("ERROR: Plot was not saved successfully.")

if __name__ == "__main__":
    train_catboost_model()