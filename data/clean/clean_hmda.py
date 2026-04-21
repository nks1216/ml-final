import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Initialize tqdm for pandas
tqdm.pandas()

def clean_hmda_data(input_path):
    # 1. Load the dataset
    if not os.path.exists(input_path):
        print(f"Error: Raw data not found at {input_path}")
        return None

    print(f"Step 1: Loading raw data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Initial shape: {df.shape}")

    # 2. Refine the target variable (action_taken)
    # Focus only on definitive outcomes: 1 (Approved) and 3 (Denied)
    df = df[df['action_taken'].isin([1, 3])].copy()
    
    # Create binary target with progress bar
    print("Step 2: Creating target variable...")
    df['target'] = df['action_taken'].progress_apply(lambda x: 1 if x == 1 else 0)
    
    # 3. Define columns to be dropped
    drop_cols = [
        'action_taken',
        'denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4',
        'purchaser_type', 'hoepa_status', 'interest_rate', 'rate_spread', 
        'total_loan_costs', 'total_points_and_fees', 'origination_charges',
        'discount_points', 'lender_credits', 'prepayment_penalty_term', 'intro_rate_period',
        'activity_year', 'lei', 'state_code', 'census_tract', 
        'applicant_ethnicity-1', 'applicant_ethnicity-2', 'applicant_ethnicity-3', 'applicant_ethnicity-4', 'applicant_ethnicity-5',
        'co-applicant_ethnicity-1', 'co-applicant_ethnicity-2', 'co-applicant_ethnicity-3', 'co-applicant_ethnicity-4', 'co-applicant_ethnicity-5',
        'applicant_race-1', 'applicant_race-2', 'applicant_race-3', 'applicant_race-4', 'applicant_race-5',
        'co-applicant_race-1', 'co-applicant_race-2', 'co-applicant_race-3', 'co-applicant_race-4', 'co-applicant_race-5',
        'applicant_sex', 'co-applicant_sex', 'applicant_age_above_62', 'co-applicant_age_above_62'
    ]
    
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 4. Handle Numerical Data
    numeric_cols = ['loan_amount', 'property_value', 'income', 'loan_to_value_ratio', 'debt_to_income_ratio']
    
    print("Step 3: Processing numeric columns...")
    for col in tqdm(numeric_cols, desc="Cleaning numeric fields"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # 5. Handle Categorical Data
    print("Step 4: Handling missing categorical values...")
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    print(f"Cleaned shape: {df.shape}")
    return df

if __name__ == "__main__":
    # --- Path Configuration ---
    # Get the directory where this script (clean_hmda.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Raw data is in ../raw/ folder relative to this script
    raw_data_path = os.path.join(current_dir, '../raw/hmda_raw_2023_TX_big4.csv')
    
    # Output will be saved in the same folder as this script
    output_data_path = os.path.join(current_dir, 'hmda_cleaned.csv')

    # Execute cleaning
    cleaned_df = clean_hmda_data(raw_data_path)
    
    if cleaned_df is not None:
        print(f"Step 5: Saving cleaned data to {output_data_path}...")
        cleaned_df.to_csv(output_data_path, index=False)
        print("Success!")