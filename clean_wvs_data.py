import pandas as pd
import numpy as np
import re

# Configuration
INPUT_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_Cross-National_Wave_7_csv_v6_0.xlsx'
OUTPUT_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_Cross-National_Wave_7_cleaned.csv'

def clean_data():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        # Load the dataset
        df = pd.read_excel(INPUT_FILE)
        print(f"Original shape: {df.shape}")
        
        # 1. Filter Columns: Q1 to Q290
        print("Filtering columns Q1 to Q290...")
        
        # Generate the list of target columns
        # We look for columns that match the pattern Q<number> where number is between 1 and 290
        # We also need to handle potential suffixes or variations if they exist, but strict Q1-Q290 is the request.
        
        all_cols = df.columns.tolist()
        target_cols = []
        
        for col in all_cols:
            # Check if column is exactly Q<number>
            match = re.match(r'^Q(\d+)$', str(col))
            if match:
                num = int(match.group(1))
                if 1 <= num <= 290:
                    target_cols.append(col)
        
        # Sort columns by their number to ensure order (Q1, Q2, ... Q10, not Q1, Q10, Q100...)
        target_cols.sort(key=lambda x: int(x[1:]))
        
        if not target_cols:
            print("WARNING: No columns matching Q1-Q290 found!")
            print("First 20 columns in dataset:", all_cols[:20])
            return
            
        print(f"Found {len(target_cols)} columns matching criteria.")
        df_filtered = df[target_cols].copy()
        
        # 2. Handle Missing Values (Negative values -> NaN)
        print("Handling missing values (replacing negative values with NaN)...")
        # WVS uses -1, -2, -4, -5 etc. for missing/inapplicable
        # We assume all numeric columns with negative values are missing codes
        
        # Get numeric columns
        numeric_cols = df_filtered.select_dtypes(include=['number']).columns
        
        # Count missing before
        n_negatives = (df_filtered[numeric_cols] < 0).sum().sum()
        print(f"Found {n_negatives} negative values to replace.")
        
        # Replace
        for col in numeric_cols:
            df_filtered.loc[df_filtered[col] < 0, col] = np.nan
            
        # 3. Drop Empty Columns
        print("Dropping completely empty columns...")
        df_filtered.dropna(axis=1, how='all', inplace=True)
        
        # 4. Deduplication
        print("Removing duplicates...")
        initial_rows = len(df_filtered)
        df_filtered.drop_duplicates(inplace=True)
        if len(df_filtered) < initial_rows:
            print(f"Removed {initial_rows - len(df_filtered)} duplicate rows.")
        
        # 5. Save
        print(f"Saving cleaned data to {OUTPUT_FILE}...")
        df_filtered.to_csv(OUTPUT_FILE, index=False)
        print("Done.")
        print(f"Final shape: {df_filtered.shape}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clean_data()
