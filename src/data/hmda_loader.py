import pandas as pd
import requests
import io
import os

def download_and_save_hmda(year, state, county=None):
    """
    Downloads HMDA data from the FFIEC API and saves it to the raw data directory.
    """
    
    # Define directory paths based on the script location
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    
    # Create the target directory
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # API endpoint configuration
    base_url = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    
    # Set query parameters and filename based on input
    if county:
        query_params = f"?years={year}&states={state}&counties={county}"
        counties_label = "big4" if "," in county else county
        file_name = f"hmda_raw_{year}_{state}_{counties_label}.csv"
    else:
        query_params = f"?years={year}&states={state}"
        file_name = f"hmda_raw_{year}_{state}.csv"
        
    full_url = base_url + query_params
    
    # Send request to the FFIEC API
    print(f"Fetching data from: {full_url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    
    # Validate and save the retrieved data
    if response.status_code == 200:
        if response.text.startswith("<!doctype html>") or "<html>" in response.text[:100]:
            print("Error: Received HTML. The dataset might be too large for the API.")
            return None
            
        df = pd.read_csv(io.StringIO(response.text), low_memory=False)
        save_path = os.path.join(raw_data_dir, file_name)
        df.to_csv(save_path, index=False)
        
        print(f"Success! Data saved to: {save_path}")
        print(f"Total rows: {len(df)}")
        return df
    else:
        print(f"HTTP Error: {response.status_code}")
        return None

if __name__ == "__main__":
    # Travis (48453), Harris (48201), Dallas (48113), Bexar (48029)
    big4_fips = "48453,48201,48113,48029"
    
    # Download 2023 data for Big 4 Texas counties
    raw_df = download_and_save_hmda(2023, "TX", county=big4_fips)

    if raw_df is not None:
        print("\nDistribution by County:")
        if 'county_code' in raw_df.columns:
            print(raw_df['county_code'].value_counts())