import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_hmda_data():
    """
    Splits the cleaned HMDA dataset into training and testing sets.
    Uses stratified sampling to maintain the balance of the target variable.
    """
    
    # --- 1. Path Resolution ---
    # ML-FINAL/src/data/split_data.py
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    
    input_path = os.path.join(project_root, 'data', 'clean', 'hmda_cleaned.csv')
    split_dir = os.path.join(project_root, 'data', 'split')
    
    # Create the output directory
    os.makedirs(split_dir, exist_ok=True)

    # --- 2. Load Cleaned Data ---
    if not os.path.exists(input_path):
        print(f"Error: Cleaned data not found at {input_path}")
        return

    print(f"Loading cleaned data from: {input_path}")
    df = pd.read_csv(input_path)

    # --- 3. Stratified Split (80/20) ---
    print("Splitting data into Train (80%) and Test (20%) sets...")
    
    # random_state=42 ensures reproducibility
    # stratify=df['target'] maintains the ratio of approved/denied cases
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['target']
    )

    # --- 4. Save Split Data ---
    # Recommended filenames: train.csv and test.csv
    train_output = os.path.join(split_dir, 'train.csv')
    test_output = os.path.join(split_dir, 'test.csv')

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    # --- 5. Verification ---
    print("-" * 30)
    print(f"SUCCESS: Data split completed.")
    print(f"Saved Train set to: {train_output} ({len(train_df)} rows)")
    print(f"Saved Test set to : {test_output} ({len(test_df)} rows)")
    print("-" * 30)
    
    # Print distribution to verify stratification
    print("\nTarget Distribution (Ratio):")
    print("Train Set:\n", train_df['target'].value_counts(normalize=True))
    print("Test Set :\n", test_df['target'].value_counts(normalize=True))

if __name__ == "__main__":
    split_hmda_data()