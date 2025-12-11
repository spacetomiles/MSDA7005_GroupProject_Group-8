import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os

# --- Configuration ---
WAVE6_FILE = 'WV6_Data_csv_v20201117.csv'
WAVE5_FILE = 'WV5_Data_csv_v20180912.csv'

# Output files
WAVE6_FEATURE_PLOT = 'feature_importance_wave6.png'
WAVE5_FEATURE_PLOT = 'feature_importance_wave5.png'
WAVE6_TOP_FEATURES_CSV = 'wave6_top_features.csv'
WAVE5_TOP_FEATURES_CSV = 'wave5_top_features.csv'



def parse_args_and_run(file_path, wave_name, target_col, output_csv=None, predictor_prefix='V', max_predictor_num=300):
    analyze_wave(file_path, wave_name, target_col, output_csv, predictor_prefix, max_predictor_num)

def analyze_wave(file_path, wave_name, target_col, output_csv=None, predictor_prefix='V', max_predictor_num=300):
    print(f"\n========================================================")
    print(f"   ANALYZING {wave_name}")
    print(f"========================================================")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return

    print(f"Loading data from {file_path}...")
    try:
        # WVS data is semicolon separated
        df = pd.read_csv(file_path, sep=';', low_memory=False)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Original shape: {df.shape}")

    # 1. Filter Columns
    # Keep target and predictors V1..Vn
    # Also keep country/wave vars if needed, but for now just predictors
    
    # Identify predictors: V followed by digits, within range
    predictors = []
    for c in df.columns:
        if c == target_col:
            continue
        if c.startswith(predictor_prefix) and c[len(predictor_prefix):].isdigit():
            num = int(c[len(predictor_prefix):])
            if 1 <= num <= max_predictor_num:
                predictors.append(c)
    
    print(f"Identified {len(predictors)} potential predictors.")
    
    cols_to_keep = predictors + [target_col]
    # Filter df to exist columns only
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep]
    
    print(f"Filtered DataFrame shape: {df.shape}")

    # 2. Handle Negative Values (Missing)
    print("Replacing negative values with NaN...")
    num_neg = (df.select_dtypes(include=np.number) < 0).sum().sum()
    print(f"Found {num_neg} negative values.")
    # Efficiently replace negative values
    for col in df.select_dtypes(include=np.number).columns:
         df.loc[df[col] < 0, col] = np.nan

    # 3. Drop completely empty rows/cols
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')

    # 4. Imputation
    print("Imputing missing values...")
    # Drop rows where target is missing
    if target_col not in df.columns:
        print(f"Error: Target column {target_col} not found in filtered data!")
        return

    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    
    # Simple Median Imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # --- SAVE CLEANED & IMPUTED DATA ---
    if output_csv:
        print(f"Saving cleaned and imputed data to {output_csv}...")
        # Combine X and y back together
        df_imputed = X_imputed.copy()
        df_imputed[target_col] = y.values
        df_imputed.to_csv(output_csv, index=False)
        print("Done.")

    # 5. AdaBoost
    print("Running AdaBoost...")
    base_estimator = DecisionTreeClassifier(max_depth=1)
    ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
    ada.fit(X_imputed, y)
    
    importances = ada.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    # Top 10
    print(f"Top 10 Features for {wave_name}:")
    print(feature_imp_df.head(10).to_string(index=False))
    
    # Save Top Features
    csv_name = WAVE6_TOP_FEATURES_CSV if wave_name == 'Wave 6' else WAVE5_TOP_FEATURES_CSV
    feature_imp_df.head(20).to_csv(csv_name, index=False)
    print(f"Saved top features to {csv_name}")

    # Plot
    plt.figure(figsize=(10, 6))
    top_plot = feature_imp_df.head(20).sort_values(by='Importance', ascending=True)
    plt.barh(top_plot['Feature'], top_plot['Importance'], color='skyblue')
    plt.title(f'Top 20 Predictors ({wave_name})')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    plot_name = WAVE6_FEATURE_PLOT if wave_name == 'Wave 6' else WAVE5_FEATURE_PLOT
    plt.savefig(plot_name)
    print(f"Saved plot to {plot_name}")
    plt.close()

def main():
    # Wave 6 Analysis
    # Target: V23 (Satisfaction with your life)
    analyze_wave(WAVE6_FILE, 'Wave 6', 'V23', 
                 output_csv='WVS_Wave6_cleaned_imputed.csv', 
                 predictor_prefix='V', max_predictor_num=229)

    # Wave 5 Analysis
    # Target: V22 (Satisfaction with your life)
    analyze_wave(WAVE5_FILE, 'Wave 5', 'V22', 
                 output_csv='WVS_Wave5_cleaned_imputed.csv',
                 predictor_prefix='V', max_predictor_num=234)

if __name__ == "__main__":
    main()
