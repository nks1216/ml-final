import pandas as pd
import requests
import io
import os

def download_and_save_hmda(year, state, county=None):
    """
    Downloads HMDA data for specific counties and saves it as a CSV file.
    """
    # Create 'data' directory if it doesn't exist to prevent FileNotFoundError
    os.makedirs('data', exist_ok=True)
    
    base_url = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    
    if county:
        query_params = f"?years={year}&states={state}&counties={county}"
        # If multiple counties are passed, name the file 'big4' for clarity
        counties_label = "big4" if "," in county else county
        file_name = f"hmda_raw_{year}_{state}_{counties_label}.csv"
    else:
        query_params = f"?years={year}&states={state}"
        file_name = f"hmda_raw_{year}_{state}.csv"
        
    full_url = base_url + query_params
    
    print(f"Fetching data from: {full_url}")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(full_url, headers=headers)
    
    if response.status_code == 200:
        if response.text.startswith("<!doctype html>") or "<html>" in response.text[:100]:
            print("Received HTML instead of CSV. Request might be too large.")
            return None
            
        df = pd.read_csv(io.StringIO(response.text), low_memory=False)
        
        # Save to CSV (Relative path: data/...)
        save_path = os.path.join("data", file_name)
        df.to_csv(save_path, index=False)
        
        print(f"Success! Data saved to: {save_path}")
        print(f"Total rows retrieved: {len(df)}")
        return df
    else:
        print(f"HTTP Error: {response.status_code}")
        return None

# --- Main Execution ---
# Define the Big 4 Counties FIPS codes:
# Travis (48453), Harris (48201), Dallas (48113), Bexar (48029)
big4_fips = "48453,48201,48113,48029"

# Call the function with the Big 4 counties
raw_df = download_and_save_hmda(2023, "TX", county=big4_fips)

if raw_df is not None:
    print("\n--- Value Counts by County ---")
    # 48453: Travis, 48201: Harris, 48113: Dallas, 48029: Bexar
    print(raw_df['county_code'].value_counts())