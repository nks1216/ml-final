import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Initialize tqdm for pandas integration
tqdm.pandas()

def clean_hmda_data(input_path):
    """
    Final cleaning pipeline for HMDA data.
    Incorporates ordinal mapping for age, range conversion for DTI/Units,
    and rigorous leakage prevention for academic fairness analysis.
    """
    # 1. Load the dataset
    if not os.path.exists(input_path):
        print(f"Error: Raw data not found at {input_path}")
        return None

    print(f"Step 1: Loading raw data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Initial shape: {df.shape}")

    # 2. Define Target Variable (action_taken)
    # Focus only on Approved (1) and Denied (3)
    df = df[df['action_taken'].isin([1, 3])].copy()
    print("Step 2: Creating binary target variable...")
    df['target'] = df['action_taken'].progress_apply(lambda x: 1 if x == 1 else 0)
    
    # 3. Handle Data Leakage and Redundant Metadata
    drop_cols = [
        'action_taken', 'denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4',
        'purchaser_type', 'hoepa_status', 'interest_rate', 'rate_spread', 
        'total_loan_costs', 'total_points_and_fees', 'origination_charges',
        'discount_points', 'lender_credits', 'prepayment_penalty_term', 'intro_rate_period',
        'activity_year', 'lei', 'state_code', 'census_tract',
        
        # --- Critical: Removing ALL internal decision markers (Leakage) ---
        'aus-1', 'aus-2', 'aus-3', 'aus-4', 'aus-5', 
        'initially_payable_to_institution',
        
        # --- Metadata Removal ---
        'applicant_ethnicity_observed', 'co-applicant_ethnicity_observed', 
        'applicant_race_observed', 'co-applicant_race_observed', 
        'applicant_sex_observed', 'co-applicant_sex_observed',
        
        # --- Demographic Redundancy (Dropping raw inputs, keeping 'derived' versions) ---
        'applicant_ethnicity-1', 'applicant_ethnicity-2', 'applicant_ethnicity-3', 'applicant_ethnicity-4', 'applicant_ethnicity-5',
        'co-applicant_ethnicity-1', 'co-applicant_ethnicity-2', 'co-applicant_ethnicity-3', 'co-applicant_ethnicity-4', 'co-applicant_ethnicity-5',
        'applicant_race-1', 'applicant_race-2', 'applicant_race-3', 'applicant_race-4', 'applicant_race-5',
        'co-applicant_race-1', 'co-applicant_race-2', 'co-applicant_race-3', 'co-applicant_race-4', 'co-applicant_race-5',
        'applicant_sex', 'co-applicant_sex', 'applicant_age_above_62', 'co-applicant_age_above_62' # <-- 쉼표 추가됨
    ]
    
    # Ensure drop list uniqueness and drop columns only if they exist in the current DataFrame
    df = df.drop(columns=[col for col in list(set(drop_cols)) if col in df.columns])

    # 4. Processing Ordinal and Range-based Features
    print("Step 3: Processing Numeric and Ordinal transformations...")

    # A. Ordinal Mapping for Age Groups
    age_map = {'<25': 0, '25-34': 1, '35-44': 2, '45-54': 3, '55-64': 4, '65-74': 5, '>74': 6}
    for age_col in ['applicant_age', 'co-applicant_age']:
        if age_col in df.columns:
            df[age_col] = df[age_col].map(age_map)

    # B. Midpoint Conversion for Ranges (DTI, Units)
    def convert_range_to_value(val):
        if pd.isna(val) or val in ['Unknown', 'Exempt', '8888', '9999']:
            return np.nan
        val = str(val).replace('%', '').replace('>', '').replace('<', '').strip()
        if '-' in val:
            try:
                low, high = map(float, val.split('-'))
                return (low + high) / 2
            except: return np.nan
        try: return float(val)
        except: return np.nan

    if 'debt_to_income_ratio' in df.columns:
        df['debt_to_income_ratio'] = df['debt_to_income_ratio'].apply(convert_range_to_value)
    if 'total_units' in df.columns:
        df['total_units'] = df['total_units'].apply(convert_range_to_value)

    # 5. Final Numeric Imputation
    numeric_cols = [
        'loan_amount', 'property_value', 'income', 'loan_to_value_ratio', 
        'debt_to_income_ratio', 'loan_term', 'total_units', 
        'applicant_age', 'co-applicant_age', 'tract_population',
        'tract_minority_population_percent', 'ffiec_msa_md_median_family_income',
        'tract_to_msa_income_percentage', 'tract_owner_occupied_units',
        'tract_one_to_four_family_homes', 'tract_median_age_of_housing_units'
    ]
    
    for col in tqdm(numeric_cols, desc="Finalizing numeric fields"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # 6. Categorical Handling
    print("Step 4: Handling categorical variables...")
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    print(f"Final Cleaned shape: {df.shape}")
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(current_dir, '../raw/hmda_raw_2023_TX_big4.csv')
    output_data_path = os.path.join(current_dir, 'hmda_cleaned.csv')

    cleaned_df = clean_hmda_data(raw_data_path)
    if cleaned_df is not None:
        print(f"Step 5: Saving to {output_data_path}...")
        cleaned_df.to_csv(output_data_path, index=False)
        print("Success!")