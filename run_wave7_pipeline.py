import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel
import os

# --- Configuration ---
RAW_FILE = 'WVS_Cross-National_Wave_7_csv_v6_0.xlsx'
CLEANED_FILE = 'WVS_Cross-National_Wave_7_cleaned.csv'
FEATURE_PLOT_ADA = 'feature_importance_adaboost.svg'
COEF_PLOT_ORDINAL = 'ordinal_coefficients.svg'
REGRESSION_TABLE_CSV = 'regression_coefficients.csv'

# Selected Features for Final Model (Data Driven)
SELECTED_FEATURES = [
    'Q50', 'Q48', 'Q46', 'Q164', 'Q112', 'Q110', 'Q120', 'Q55', 'Q47'
]

def main():
    print("========================================================")
    print("   WVS WAVE 7 ANALYSIS PIPELINE (CLEANED)")
    print("========================================================")

    # ---------------------------------------------------------
    # 1. Data Cleaning
    # ---------------------------------------------------------
    print("\n[Step 1] Data Cleaning...")
    if os.path.exists(CLEANED_FILE):
        print(f"Cleaned file found at {CLEANED_FILE}. Loading...")
        df = pd.read_csv(CLEANED_FILE)
    else:
        print(f"Loading raw data from {RAW_FILE} (This may take a moment)...")
        try:
            df = pd.read_excel(RAW_FILE)
        except FileNotFoundError:
            print("Error: Raw file not found! Please ensure it is in the directory.")
            return

        print(f"Original shape: {df.shape}")
        
        # Filter Columns Q1-Q290
        cols_to_keep = [c for c in df.columns if c.startswith('Q') and c[1:].isdigit() and 1 <= int(c[1:]) <= 290]
        df = df[cols_to_keep]
        print(f"Filtered to {len(df.columns)} columns (Q1-Q290).")
        
        # Handle Negative Values
        print("Replacing negative values with NaN...")
        num_neg = (df.select_dtypes(include=np.number) < 0).sum().sum()
        print(f"Found {num_neg} negative values.")
        df[df < 0] = np.nan
        
        # Deduplicate
        print("Removing duplicates...")
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_rows - len(df)} duplicate rows.")
        
        # Save
        df.to_csv(CLEANED_FILE, index=False)
        print(f"Saved cleaned data to {CLEANED_FILE}")

    # ---------------------------------------------------------
    # 2. Imputation
    # ---------------------------------------------------------
    print("\n[Step 2] Imputation...")
    # Drop rows where target Q49 is missing
    df = df.dropna(subset=['Q49']).reset_index(drop=True)
    y = df['Q49'].astype(int)
    X = df.drop(columns=['Q49'])
    
    print("Imputing missing predictors with Median...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # ---------------------------------------------------------
    # 3. Feature Selection (AdaBoost)
    # ---------------------------------------------------------
    print("\n[Step 3] Feature Selection (AdaBoost)...")
    print("Training AdaBoostClassifier to rank features...")
    base_estimator = DecisionTreeClassifier(max_depth=1)
    ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
    ada.fit(X_imputed, y)
    
    importances = ada.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    print("Top 10 Features:")
    print(feature_imp_df.head(10).to_string(index=False))
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    top_plot = feature_imp_df.head(20).sort_values(by='Importance', ascending=True)
    plt.barh(top_plot['Feature'], top_plot['Importance'], color='skyblue')
    plt.title('Top 20 Predictors (AdaBoost)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(FEATURE_PLOT_ADA)
    print(f"Saved AdaBoost feature importance plot to {FEATURE_PLOT_ADA}")

    # ---------------------------------------------------------
    # 4. Standardize & Prepare for OLR
    # ---------------------------------------------------------
    print("\n[Step 4] Preparing for Ordinal Logistic Regression...")
    # Use selected features (AdaBoost Top Features)
    X_final = X_imputed[SELECTED_FEATURES].copy()
    
    # Recoding (consistency with analysis findings)
    # Q46 High is Poor health in W7, so reverse. (1-4)
    # Q47 High is Unhappy in W7, so reverse. (1-4)
    if 'Q46' in X_final.columns:
        X_final['Q46'] = 5 - X_final['Q46'] # 1->4, 4->1
    if 'Q47' in X_final.columns:
        X_final['Q47'] = 5 - X_final['Q47'] # 1->4, 4->1
        
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_final), columns=X_final.columns)

    # ---------------------------------------------------------
    # 5. Ordinal Logistic Regression
    # ---------------------------------------------------------
    print("\n[Step 5] Ordinal Logistic Regression...")
    try:
        mod = OrderedModel(y, X_scaled, distr='logit')
        res = mod.fit(method='bfgs', disp=False)
        
        print(res.summary())
        
        # Save Regression Table
        params = res.params
        conf = res.conf_int()
        results_df = pd.DataFrame({
            'Feature': params.index,
            'Coefficient': params.values,
            'P-value': res.pvalues.values,
            'Odds Ratio': np.exp(params.values),
            'CI Lower': np.exp(conf.iloc[:, 0].values),
            'CI Upper': np.exp(conf.iloc[:, 1].values)
        })
        # Filter cutpoints
        results_df = results_df[~results_df['Feature'].str.contains('/')]
        results_df.to_csv(REGRESSION_TABLE_CSV, index=False)
        print(f"Saved regression coefficients table to {REGRESSION_TABLE_CSV}")

        # Plot Coefficients
        plt.figure(figsize=(10, 6))
        
        # Filter out cutpoints for plot
        predictor_mask = [x for x in params.index if '/' not in x]
        plot_params = params.loc[predictor_mask].sort_values(key=abs)
        
        bars = plt.barh(plot_params.index, plot_params.values, color='steelblue')
        plt.axvline(0, color='grey', linewidth=0.8)
        plt.title('Ordinal Logistic Regression Coefficients')
        plt.xlabel('Standardized Coefficient')
        plt.tight_layout()
        plt.savefig(COEF_PLOT_ORDINAL)
        print(f"Saved Ordinal Regression Coefficient Plot to {COEF_PLOT_ORDINAL}")
        
    except Exception as e:
        print(f"Regression failed: {e}")
    
    print("\nPipeline Complete.")

if __name__ == "__main__":
    main()
