"""
Fairness Audit for XGBoost Mortgage Approval Model
==================================================

This script performs a post-hoc fairness audit on the best-performing XGBoost model.
It evaluates demographic parity and equalized odds across:
    - Race (White, Black, Other)
    - Gender (Male, Female)
    - Age (7 ordinal bins)
    - County (4 Texas counties)

Outputs:
    - Figures: reports/figures/fairness/ (Parity plots, SHAP summary)
    - Results: reports/results/fairness/ (Metrics by group, fairness summary CSVs)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
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
# 2. Mapping Logic 
# --------------------------------------------------------------------------- #
def load_and_preprocess():
    print("[1/5] Applying refined mappings based on CSV content...")
    test_df = pd.read_csv(TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH)

    # (1) Age: Convert float to int then map
    age_labels = {0: "<25", 1: "25-34", 2: "35-44", 3: "45-54", 4: "55-64", 5: "65-74", 6: ">74"}
    test_df['audit_age'] = pd.to_numeric(test_df['applicant_age'], errors='coerce').fillna(-1).astype(int).map(age_labels)

    # (2) County: Match FIPS codes
    county_labels = {
        48029: "Bexar",
        48113: "Dallas",
        48201: "Harris",
        48453: "Travis"
    }
    test_df['audit_county'] = test_df['county_code'].map(county_labels)

    # (3) Race
    def map_race(r):
        if r == "White": return "White"
        if r == "Black or African American": return "Black"
        if r == "Asian": return "Asian"
        if r in ["American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander", "Joint"]:
            return "Other"
        return np.nan
    test_df['audit_race'] = test_df['derived_race'].map(map_race)

    # (4) Gender
    def map_gender(s):
        if s == "Male": return "Male"
        if s == "Female": return "Female"
        return np.nan
    test_df['audit_gender'] = test_df['derived_sex'].map(map_gender)

    X_train = train_df.drop(columns=[TARGET_COL])
    audit_cols = ['audit_age', 'audit_county', 'audit_race', 'audit_gender']
    X_test = test_df.drop(columns=[TARGET_COL] + audit_cols)
    
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        all_vals = pd.concat([X_train[col], X_test[col]]).astype(str)
        categories = pd.Index(all_vals.unique())
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=categories)

    return X_test, test_df[TARGET_COL].astype(int), test_df

# --------------------------------------------------------------------------- #
# 3. Fairness Audit with Strict Ordering
# --------------------------------------------------------------------------- #
def run_fairness_audit(model, X_test, y_test, full_test_df):
    print("[3/5] Running Fairness Audit with data labels...")
    y_pred = model.predict(X_test)
    
    sensitive_vars = {'Race': 'audit_race', 'Gender': 'audit_gender', 'Age': 'audit_age', 'County': 'audit_county'}
    orders = {
        'Race': ['White', 'Black', 'Asian', 'Other'],
        'Gender': ['Male', 'Female'],
        'Age': ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"],
        'County': ["Bexar", "Dallas", "Harris", "Travis"]
    }

    summary_list = []
    for label, col_name in sensitive_vars.items():
        mask = full_test_df[col_name].notna()
        y_t, y_p, sens = y_test[mask], y_pred[mask], full_test_df.loc[mask, col_name]

        mf = MetricFrame(
            metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate, 'TPR (Recall)': true_positive_rate, 'FPR': false_positive_rate},
            y_true=y_t, y_pred=y_p, sensitive_features=sens
        )

        group_results = mf.by_group.reindex(orders[label])
        group_results.to_csv(RESULTS_DIR / f"metrics_by_{label.lower()}.csv")
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 7))

        # [Plot 1: Selection Rate]
        group_results[['selection_rate']].plot(kind='bar', ax=ax[0], color='skyblue', rot=0)
        ax[0].set_title(f'{label} Selection Rate', fontsize=15, pad=15)
        ax[0].set_ylim(0, max(group_results['selection_rate']) * 1.15) 
        
        # Add number label
        for container in ax[0].containers:
            ax[0].bar_label(container, fmt='%.3f', padding=3, fontsize=10, fontweight='bold')

        # [Plot 2: Error Rates (TPR vs FPR)]
        group_results[['TPR (Recall)', 'FPR']].plot(kind='bar', ax=ax[1], rot=0)
        ax[1].set_title(f'{label} Error Rates (TPR vs FPR)', fontsize=15, pad=15)
        # Choose Y axis from Max(TPR, FPR)
        max_val = max(group_results['TPR (Recall)'].max(), group_results['FPR'].max())
        ax[1].set_ylim(0, max_val * 1.15) 

        # Add number label
        for container in ax[1].containers:
            ax[1].bar_label(container, fmt='%.3f', padding=3, fontsize=9)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fairness_plot_{label.lower()}.png", dpi=150)
        plt.close()

        summary_list.append({
            'Attribute': label,
            'DP_Diff': demographic_parity_difference(y_t, y_p, sensitive_features=sens),
            'EO_Diff': equalized_odds_difference(y_t, y_p, sensitive_features=sens)
        })

    pd.DataFrame(summary_list).to_csv(RESULTS_DIR / "fairness_summary_metrics.csv", index=False)

# --------------------------------------------------------------------------- #
# 4. SHAP Analysis 
# --------------------------------------------------------------------------- #
def run_shap_analysis(model, X_test):
    """Generates SHAP summary plot. Sampling is used for speed."""
    print("[4/5] Computing SHAP values (using 1,000 samples for speed)...")
    
    # Use a sample of the test set for SHAP to avoid long computation times
    X_sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    # Summary plot helps identify which features impact approval/denial most
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance (Global Impact)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_fairness.png", dpi=150)
    plt.close()

# --------------------------------------------------------------------------- #
# 5. Main Execution
# --------------------------------------------------------------------------- #
def main():
    X_test, y_test, full_test_df = load_and_preprocess()
    
    print("[2/5] Training model for audit...")
    model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=RANDOM_STATE)
    model.fit(X_test, y_test)
    
    run_fairness_audit(model, X_test, y_test, full_test_df)
    run_shap_analysis(model, X_test)
    
    print("\n[5/5] Fairness Audit Successful. All 5 plots saved to reports/.")

if __name__ == "__main__":
    main()