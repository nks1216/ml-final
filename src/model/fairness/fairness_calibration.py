"""
Calibration Fairness Analysis
=============================

Evaluates whether the model's predicted probabilities are well-calibrated
across demographic groups (Race, Age, Gender, County). 

A well-calibrated model means: "When I predict 70% approval, 70% actually 
get approved" - this should hold for ALL demographic groups.

Outputs:
    - Calibration curves by race/age/gender/county
    - Expected Calibration Error (ECE) metrics
    - Calibration plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# --------------------------------------------------------------------------- #
# Configuration
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
# Load Data & Preprocess
# --------------------------------------------------------------------------- #
def load_and_preprocess():
    print("[1/4] Loading and preprocessing data...")
    test_df = pd.read_csv(TEST_PATH)
    
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

    # Gender mapping
    def map_gender(s):
        if s == "Male": return "Male"
        if s == "Female": return "Female"
        return np.nan
    test_df['audit_gender'] = test_df['derived_sex'].map(map_gender)

    # County mapping
    county_labels = {
        48029: "Bexar",
        48113: "Dallas",
        48201: "Harris",
        48453: "Travis"
    }
    test_df['audit_county'] = test_df['county_code'].map(county_labels)

    X_test = test_df.drop(columns=[TARGET_COL, 'audit_age', 'audit_race', 'audit_gender', 'audit_county'])
    y_test = test_df[TARGET_COL].astype(int)
    
    # Convert object columns to category (match XGBoost requirements)
    cat_cols = X_test.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X_test[col] = X_test[col].astype('category')
    
    return X_test, y_test, test_df

# --------------------------------------------------------------------------- #
# Get Predicted Probabilities
# --------------------------------------------------------------------------- #
def get_predictions(X_test):
    print("[2/4] Loading model and getting predicted probabilities...")
    
    MODEL_PATH = PROJECT_ROOT / "reports" / "results" / "prediction" / "xgboost_model.json"
    model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=RANDOM_STATE)
    
    if MODEL_PATH.exists():
        model.load_model(str(MODEL_PATH))
        print("✓ Model loaded")
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
        return None
    
    # Get predicted probabilities (not just class predictions)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of approval (class 1)
    
    return y_pred_proba

# --------------------------------------------------------------------------- #
# Calculate Calibration Metrics
# --------------------------------------------------------------------------- #
def calculate_ece(y_true, y_pred_proba, n_bins=10):
    """
    Expected Calibration Error (ECE)
    Measures average difference between predicted probability and actual frequency
    Lower is better (0 = perfectly calibrated)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ece = 0
    bin_accs = []
    bin_confs = []
    
    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred_proba[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(y_true)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
    
    return ece, bin_centers, bin_accs, bin_confs

# --------------------------------------------------------------------------- #
# Analyze Calibration by Group (Race, Age, Gender, County)
# --------------------------------------------------------------------------- #
def analyze_calibration(y_test, y_pred_proba, test_df):
    print("[3/4] Analyzing calibration by Race, Age, Gender, County...")
    
    calibration_results = []
    
    for group_label, col_name in [('Race', 'audit_race'), ('Age', 'audit_age'), 
                                   ('Gender', 'audit_gender'), ('County', 'audit_county')]:
        print(f"\n  Analyzing {group_label}...")
        
        groups = test_df[col_name].dropna().unique()
        groups = sorted(groups) if group_label == 'Age' else sorted(groups)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for group in groups:
            mask = test_df[col_name] == group
            y_group = y_test[mask]
            y_pred_group = y_pred_proba[mask]
            
            if len(y_group) < 10:  # Skip groups with too few samples
                continue
            
            # Calculate ECE
            ece, bin_centers, bin_accs, bin_confs = calculate_ece(y_group.values, y_pred_group)
            
            # Brier Score (squared error)
            brier = brier_score_loss(y_group, y_pred_group)
            
            # Actual approval rate
            actual_approval = y_group.mean()
            mean_predicted_proba = y_pred_group.mean()
            
            calibration_results.append({
                'Group': group_label,
                'Subgroup': group,
                'Sample_Size': len(y_group),
                'Actual_Approval_Rate': actual_approval,
                'Mean_Predicted_Probability': mean_predicted_proba,
                'Calibration_Gap': abs(actual_approval - mean_predicted_proba),
                'ECE': ece,
                'Brier_Score': brier
            })
            
            # Plot 1: Calibration Curve
            prob_true, prob_pred = calibration_curve(y_group, y_pred_group, n_bins=5, strategy='uniform')
            axes[0].plot(prob_pred, prob_true, marker='o', label=f'{group} (ECE: {ece:.3f})', linewidth=2)
            
            # Plot 2: Predicted vs Actual
            axes[1].scatter(mean_predicted_proba, actual_approval, s=200, alpha=0.6, label=group)
        
        # Perfect calibration line
        axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        axes[0].set_xlabel('Mean Predicted Probability', fontsize=12)
        axes[0].set_ylabel('Actual Approval Rate', fontsize=12)
        axes[0].set_title(f'{group_label}: Calibration Curve', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        
        # Perfect calibration line for scatter
        axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        axes[1].set_xlabel('Mean Predicted Probability', fontsize=12)
        axes[1].set_ylabel('Actual Approval Rate', fontsize=12)
        axes[1].set_title(f'{group_label}: Predicted vs Actual', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"calibration_{group_label.lower()}.png", dpi=150)
        plt.close()
        
        print(f"  ✓ {group_label} calibration analysis saved")
    
    return pd.DataFrame(calibration_results)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    X_test, y_test, test_df = load_and_preprocess()
    y_pred_proba = get_predictions(X_test)
    
    if y_pred_proba is None:
        return
    
    # Analyze calibration
    calibration_df = analyze_calibration(y_test, y_pred_proba, test_df)
    
    # Save results
    calibration_df.to_csv(RESULTS_DIR / "calibration_fairness_metrics.csv", index=False)
    
    print("\n[4/4] ✓ Calibration Fairness analysis complete!")
    print("\nKey Findings:")
    print(calibration_df.to_string())
    print(f"\nResults saved to {RESULTS_DIR / 'calibration_fairness_metrics.csv'}")
    print(f"Plots saved to {FIGURES_DIR}/calibration_*.png")

if __name__ == "__main__":
    main()
    