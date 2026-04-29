"""
Fairness Through Blindness Comparative Analysis
===============================================

Compares Post-hoc Audit (full model) vs Fairness Through Blindness 
(model without race/gender features) to demonstrate why proxy variables 
make "blindness" ineffective.

Outputs:
    - Figures: reports/figures/fairness/blindness_*.png
    - Results: reports/results/fairness/blindness_comparison_*.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
from pathlib import Path
from fairlearn.metrics import (
    MetricFrame, selection_rate, true_positive_rate, false_positive_rate,
    demographic_parity_difference, equalized_odds_difference
)
from sklearn.metrics import accuracy_score

# --------------------------------------------------------------------------- #
# 1. Configuration & Paths
# --------------------------------------------------------------------------- #
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "split" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "split" / "test.csv"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "fairness"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results" / "fairness"

TARGET_COL = "target"
RANDOM_STATE = 42

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2. Load & Preprocess (Same as audit)
# --------------------------------------------------------------------------- #
def load_and_preprocess():
    print("[1/6] Loading and preprocessing data...")
    test_df = pd.read_csv(TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH)

    # Age mapping
    age_labels = {0: "<25", 1: "25-34", 2: "35-44", 3: "45-54", 4: "55-64", 5: "65-74", 6: ">74"}
    test_df['audit_age'] = pd.to_numeric(test_df['applicant_age'], errors='coerce').fillna(-1).astype(int).map(age_labels)

    # Race mapping
    def map_race(r):
        if r == "White": return "White"
        if r == "Black or African American": return "Black"
        if r == "Asian": return "Asian"
        if r in ["American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander", "Joint"]:
            return "Other"
        return np.nan
    test_df['audit_race'] = test_df['derived_race'].map(map_race)

    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL, 'audit_age', 'audit_race'])
    
    # Categorical encoding
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        all_vals = pd.concat([X_train[col], X_test[col]]).astype(str)
        categories = pd.Index(all_vals.unique())
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=categories)

    return X_train, X_test, train_df[TARGET_COL].astype(int), test_df[TARGET_COL].astype(int), test_df

# --------------------------------------------------------------------------- #
# 3. Train Blind Model (Remove race/gender)
# --------------------------------------------------------------------------- #
def train_blind_model(X_train, y_train):
    print("[2/6] Training Blind Model (without race/gender)...")
    
    X_train_blind = X_train.drop(['derived_race', 'derived_sex'], axis=1, errors='ignore')
    
    # Convert object columns to category (match XGBoost requirements)
    cat_cols = X_train_blind.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X_train_blind[col] = X_train_blind[col].astype('category')
    
    blind_model = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=RANDOM_STATE,
        n_estimators=100
    )
    blind_model.fit(X_train_blind, y_train)
    
    model_path = RESULTS_DIR / "xgboost_blind_model.pkl"
    joblib.dump(blind_model, model_path)
    print(f"✓ Blind model saved to {model_path}")
    
    return blind_model

# --------------------------------------------------------------------------- #
# 4. Get Predictions from Both Models
# --------------------------------------------------------------------------- #
def get_predictions(full_model, blind_model, X_test):
    print("[3/6] Getting predictions from both models...")
    
    X_test_blind = X_test.drop(['derived_race', 'derived_sex'], axis=1, errors='ignore')
    
    # Convert object columns to category (match training)
    cat_cols = X_test_blind.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X_test_blind[col] = X_test_blind[col].astype('category')
    
    full_pred = full_model.predict(X_test)
    blind_pred = blind_model.predict(X_test_blind)
    
    return full_pred, blind_pred

# --------------------------------------------------------------------------- #
# 5. Compare Fairness Across Race & Age
# --------------------------------------------------------------------------- #
def compare_fairness(full_pred, blind_pred, y_test, full_test_df):
    print("[4/6] Comparing fairness metrics (Race & Age)...")
    
    comparison_results = {}
    
    for attribute, col_name in [('Race', 'audit_race'), ('Age', 'audit_age')]:
        mask = full_test_df[col_name].notna()
        y_t = y_test[mask]
        full_p = full_pred[mask]
        blind_p = blind_pred[mask]
        sens = full_test_df.loc[mask, col_name]
        
        # Order groups
        orders = {
            'Race': ['White', 'Black', 'Asian', 'Other'],
            'Age': ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
        }
        
        # Full model metrics
        mf_full = MetricFrame(
            metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate, 'TPR': true_positive_rate, 'FPR': false_positive_rate},
            y_true=y_t, y_pred=full_p, sensitive_features=sens
        )
        full_results = mf_full.by_group.reindex(orders[attribute])
        full_results.columns = [f'Full_{col}' for col in full_results.columns]
        
        # Blind model metrics
        mf_blind = MetricFrame(
            metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate, 'TPR': true_positive_rate, 'FPR': false_positive_rate},
            y_true=y_t, y_pred=blind_p, sensitive_features=sens
        )
        blind_results = mf_blind.by_group.reindex(orders[attribute])
        blind_results.columns = [f'Blind_{col}' for col in blind_results.columns]
        
        # Combine
        comparison_df = pd.concat([full_results, blind_results], axis=1)
        comparison_df.to_csv(RESULTS_DIR / f"blindness_comparison_{attribute.lower()}.csv")
        
        comparison_results[attribute] = comparison_df
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Selection Rate Comparison
        comparison_df[['Full_selection_rate', 'Blind_selection_rate']].plot(
            kind='bar', ax=axes[0], rot=45, color=['steelblue', 'coral']
        )
        axes[0].set_title(f'{attribute}: Selection Rate (Full vs Blind Model)', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Selection Rate')
        axes[0].legend(['Full Model', 'Blind Model'])
        
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.3f', fontsize=9)
        
        # TPR Comparison
        comparison_df[['Full_TPR', 'Blind_TPR']].plot(
            kind='bar', ax=axes[1], rot=45, color=['steelblue', 'coral']
        )
        axes[1].set_title(f'{attribute}: TPR (Full vs Blind Model)', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend(['Full Model', 'Blind Model'])
        
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%.3f', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"blindness_vs_audit_{attribute.lower()}.png", dpi=150)
        plt.close()
        
        print(f"✓ {attribute} comparison saved")
    
    return comparison_results

# --------------------------------------------------------------------------- #
# 6. Main Execution
# --------------------------------------------------------------------------- #
def main():
    # Load data
    X_train, X_test, y_train, y_test, full_test_df = load_and_preprocess()
    
    # Load pre-trained FULL model
    MODEL_PATH = PROJECT_ROOT / "reports" / "results" / "prediction" / "xgboost_model.json"
    print(f"[2/6] Loading pre-trained XGBoost model from: {MODEL_PATH}")
    
    full_model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=RANDOM_STATE)
    if MODEL_PATH.exists():
        full_model.load_model(str(MODEL_PATH))
        print("✓ Full model loaded")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    # Train blind model
    blind_model = train_blind_model(X_train, y_train)
    
    # Get predictions
    full_pred, blind_pred = get_predictions(full_model, blind_model, X_test)
    
    # Compare fairness
    comparison_results = compare_fairness(full_pred, blind_pred, y_test, full_test_df)
    
    print("\n[6/6] ✓ Fairness Through Blindness analysis complete!")
    print("Results saved to reports/results/fairness/ and reports/figures/fairness/")

if __name__ == "__main__":
    main()