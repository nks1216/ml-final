import pandas as pd
import requests
import io
import os

def download_and_save_hmda(year, state, county=None):
    """
    Downloads HMDA data and saves it as a CSV file for reproducibility.
    """
    base_url = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    
    if county:
        query_params = f"?years={year}&states={state}&counties={county}"
        file_name = f"hmda_raw_{year}_{state}_{county}.csv"
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
            
        # Convert response to DataFrame
        df = pd.read_csv(io.StringIO(response.text), low_memory=False)
        
        # Save to CSV using a relative path
        # index=False prevents pandas from adding an unneccessary ID column
        df.to_csv(f"data/{file_name}", index=False)
        
        print(f"Success! Data saved to: {file_name}")
        print(f"Total rows retrieved: {len(df)}")
        return df
    else:
        print(f"HTTP Error: {response.status_code}")
        return None

# Execute the function
# We use Travis County (48453) for the initial project setup
raw_df = download_and_save_hmda(2023, "TX", county="48453")

if raw_df is not None:
    print("\n--- Preview of Saved Data ---")
    print(raw_df.head())