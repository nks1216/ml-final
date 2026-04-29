"""
Intersectional Fairness Analysis
==================================

Analyzes fairness across intersections of multiple demographic groups.
For example: Black Women, Asian Men, Young White applicants, etc.

This reveals compound discrimination that single-dimension analysis misses.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    selection_rate, true_positive_rate, false_positive_rate,
    demographic_parity_difference, equalized_odds_difference
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "split" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "split" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "reports" / "results" / "prediction" / "xgboost_model.json"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "fairness"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results" / "fairness"

TARGET_COL = "target"
RANDOM_STATE = 42

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Load & Preprocess Data
# --------------------------------------------------------------------------- #
def load_and_preprocess():
    """Load test data and create intersection labels."""
    print("[1/4] Loading data and creating intersections...")
    
    test_df = pd.read_csv(TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    
    # Map demographics
    age_labels = {0: "<25", 1: "25-34", 2: "35-44", 3: "45-54", 4: "55-64", 5: "65-74", 6: ">74"}
    test_df['audit_age'] = pd.to_numeric(test_df['applicant_age'], errors='coerce').fillna(-1).astype(int).map(age_labels)
    
    def map_race(r):
        if r == "White": return "White"
        if r == "Black or African American": return "Black"
        if r == "Asian": return "Asian"
        return np.nan
    test_df['audit_race'] = test_df['derived_race'].map(map_race)
    
    def map_gender(s):
        if s == "Male": return "Male"
        if s == "Female": return "Female"
        return np.nan
    test_df['audit_gender'] = test_df['derived_sex'].map(map_gender)
    
    # Create intersections
    test_df['race_gender'] = test_df['audit_race'] + " " + test_df['audit_gender']
    test_df['race_age'] = test_df['audit_race'] + " " + test_df['audit_age']
    
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL, 'audit_age', 'audit_race', 'audit_gender', 'race_gender', 'race_age'])
    
    # Handle categorical columns
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        all_vals = pd.concat([X_train[col], X_test[col]]).astype(str)
        categories = pd.Index(all_vals.unique())
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=categories)
    
    return X_test, test_df[TARGET_COL].astype(int), test_df

# --------------------------------------------------------------------------- #
# Load Pre-trained Model
# --------------------------------------------------------------------------- #
def load_model():
    """Load the pre-trained XGBoost model."""
    print("[2/4] Loading pre-trained XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model

# --------------------------------------------------------------------------- #
# Intersectional Fairness Analysis
# --------------------------------------------------------------------------- #
def run_intersectional_analysis(model, X_test, y_test, full_test_df):
    """Analyze fairness across intersections."""
    print("[3/4] Computing intersectional fairness metrics...")
    
    y_pred = model.predict(X_test)
    
    # Analyze two key intersections
    intersections = {
        'Race × Gender': 'race_gender',
        'Race × Age': 'race_age'
    }
    
    for intersection_name, col in intersections.items():
        print(f"  Analyzing {intersection_name}...")
        
        # Get valid rows
        mask = full_test_df[col].notna()
        y_t = y_test[mask]
        y_p = y_pred[mask]
        sens = full_test_df.loc[mask, col]
        
        # Calculate metrics per intersection
        results_list = []
        for group in sorted(sens.unique()):
            group_mask = sens == group
            group_y_true = y_t[group_mask]
            group_y_pred = y_p[group_mask]
            
            if len(group_y_true) < 10:  # Skip small groups
                continue
            
            results_list.append({
                'Intersection': group,
                'Count': len(group_y_true),
                'Accuracy': accuracy_score(group_y_true, group_y_pred),
                'Selection_Rate': selection_rate(group_y_true, group_y_pred),
                'TPR': true_positive_rate(group_y_true, group_y_pred),
                'FPR': false_positive_rate(group_y_true, group_y_pred),
                'Approved': group_y_pred.sum(),
                'Denied': (1 - group_y_pred).sum()
            })
        
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('Selection_Rate')
        
        # Save results
        filename = f"intersectional_{intersection_name.lower().replace(' × ', '_')}_metrics.csv"
        results_df.to_csv(RESULTS_DIR / filename, index=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Selection Rate
        results_df.set_index('Intersection')['Selection_Rate'].plot(
            kind='barh', ax=axes[0, 0], color='skyblue'
        )
        axes[0, 0].set_title(f'{intersection_name}: Selection Rate by Group', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Selection Rate')
        for i, v in enumerate(results_df['Selection_Rate']):
            axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
        
        # Plot 2: TPR vs FPR
        ax2 = axes[0, 1]
        x = np.arange(len(results_df))
        width = 0.35
        ax2.bar(x - width/2, results_df['TPR'], width, label='TPR (Recall)', color='steelblue')
        ax2.bar(x + width/2, results_df['FPR'], width, label='FPR', color='orange')
        ax2.set_xlabel('Group')
        ax2.set_ylabel('Rate')
        ax2.set_title(f'{intersection_name}: Error Rates (TPR vs FPR)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['Intersection'], rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        
        # Plot 3: Approved vs Denied Counts
        ax3 = axes[1, 0]
        results_df_sorted = results_df.sort_values('Approved')
        ax3.barh(results_df_sorted['Intersection'], results_df_sorted['Approved'], label='Approved', color='green', alpha=0.7)
        ax3.barh(results_df_sorted['Intersection'], results_df_sorted['Denied'], 
                left=results_df_sorted['Approved'], label='Denied', color='red', alpha=0.7)
        ax3.set_title(f'{intersection_name}: Approval Counts', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Count')
        ax3.legend()
        
        # Plot 4: Accuracy
        results_df_sorted2 = results_df.sort_values('Accuracy')
        axes[1, 1].barh(results_df_sorted2['Intersection'], results_df_sorted2['Accuracy'], color='purple', alpha=0.7)
        axes[1, 1].set_title(f'{intersection_name}: Model Accuracy by Group', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Accuracy')
        for i, v in enumerate(results_df_sorted2['Accuracy']):
            axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
        axes[1, 1].set_xlim(0, 1.1)
        
        plt.tight_layout()
        figname = f"intersectional_{intersection_name.lower().replace(' × ', '_')}.png"
        plt.savefig(FIGURES_DIR / figname, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {figname}")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    X_test, y_test, full_test_df = load_and_preprocess()
    model = load_model()
    run_intersectional_analysis(model, X_test, y_test, full_test_df)
    print("\n[4/4] ✓ Intersectional fairness analysis complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()