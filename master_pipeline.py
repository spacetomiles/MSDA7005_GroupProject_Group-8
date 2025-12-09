import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from statsmodels.miscmodels.ordinal_model import OrderedModel
import os

# --- Configuration ---
RAW_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_Cross-National_Wave_7_csv_v6_0.xlsx'
CLEANED_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_Cross-National_Wave_7_cleaned.csv'
IMPUTED_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_imputed.csv'
ROC_PLOT_BINARY = '/Users/a11/Downloads/HKU/7005/7005 group project/roc_curve_ordinal_binary.svg'
ROC_PLOT_MULTI = '/Users/a11/Downloads/HKU/7005/7005 group project/roc_curve_ordinal_multiclass.svg'
FEATURE_PLOT_ADA = '/Users/a11/Downloads/HKU/7005/7005 group project/feature_importance_adaboost.svg'
FEATURE_PLOT_ORDINAL = '/Users/a11/Downloads/HKU/7005/7005 group project/feature_importance_ordinal.svg'
FEATURE_PLOT_RF = '/Users/a11/Downloads/HKU/7005/7005 group project/feature_importance_rf.svg'
COEF_PLOT_ORDINAL = '/Users/a11/Downloads/HKU/7005/7005 group project/ordinal_coefficients.svg'
ROC_PLOT_RF_MULTI = '/Users/a11/Downloads/HKU/7005/7005 group project/roc_curve_rf_multiclass.svg'
REGRESSION_TABLE_CSV = '/Users/a11/Downloads/HKU/7005/7005 group project/regression_coefficients.csv'

# Selected Features for Final Model
SELECTED_FEATURES = [
    'Q50', 'Q48', 'Q46', 'Q164', 'Q112', 'Q110', 'Q120', 'Q55', 'Q47'
]

def main():
    print("========================================================")
    print("   WVS ANALYSIS MASTER PIPELINE")
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
            print("Error: Raw file not found!")
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
    # 4. Recoding & Preparation for Final Model
    # ---------------------------------------------------------
    print("\n[Step 4] Recoding Variables...")
    X_final = X_imputed[SELECTED_FEATURES].copy()
    
    # Recode Q46 (Happiness): 1=Very Happy -> 4=Very Happy
    print("Recoding Q46 (Happiness): Reversing scale so Higher = Happier")
    X_final['Q46'] = 5 - X_final['Q46']
    
    # Recode Q47 (Health): 1=Very Good -> 5=Very Good
    print("Recoding Q47 (Health): Reversing scale so Higher = Healthier")
    X_final['Q47'] = 6 - X_final['Q47']
    
    # Standardize
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
        
        print("\n--- Odds Ratios (Interpretation) ---")
        params = res.params
        conf = res.conf_int()
        # --- Save Regression Table ---
        results_df = pd.DataFrame({
            'Feature': params.index,
            'Coefficient': params.values,
            'Std Error': res.bse.values,
            'z-score': res.tvalues.values,
            'P-value': res.pvalues.values,
            'Odds Ratio': np.exp(params.values),
            'CI Lower (2.5%)': np.exp(conf.iloc[:, 0].values),
            'CI Upper (97.5%)': np.exp(conf.iloc[:, 1].values)
        })
        # Filter only predictors (exclude threshold parameters like 1/2, 2/3)
        predictors_df = results_df[results_df['Feature'].isin(SELECTED_FEATURES)].copy()
        predictors_df.to_csv(REGRESSION_TABLE_CSV, index=False)
        print(f"Saved regression coefficients table to {REGRESSION_TABLE_CSV}")

        # --- 5b. Generate Ordinal Regression Coefficient Bar Chart ---
        print("Generating Ordinal Regression Coefficient Bar Chart...")
        
        # Extract coefficients
        params = res.params
        
        # Filter out cutpoints (only keep predictors)
        predictor_mask = [x for x in params.index if '/' not in x]
        params = params.loc[predictor_mask]
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({'coef': params})
        plot_df['abs_coef'] = plot_df['coef'].abs()
        plot_df = plot_df.sort_values(by='abs_coef', ascending=True) # Ascending for barh (bottom to top)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Create horizontal bar chart
        bars = plt.barh(plot_df.index, plot_df['coef'], color='steelblue')
        
        # Add vertical line at 0
        plt.axvline(x=0, color='black', linewidth=0.8)
        
        # Add grid
        plt.grid(True, axis='x', linestyle=':', alpha=0.6)
        
        # Set x-axis limits with padding to prevent label overlap
        x_min, x_max = plot_df['coef'].min(), plot_df['coef'].max()
        padding = 0.2  # Add 0.2 padding on both sides
        plt.xlim(x_min - padding, x_max + padding)
        
        # Add value annotations
        for bar in bars:
            width = bar.get_width()
            # Position text slightly outside the bar
            offset = 0.02 if width >= 0 else -0.02
            ha = 'left' if width >= 0 else 'right'
            plt.text(width + offset, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f}', 
                     va='center', ha=ha, fontsize=10, fontweight='bold')
        
        plt.title('Predictors Ranked by |Standardized Coefficient| (Ordinal Logistic Regression)')
        plt.xlabel('Standardized Coefficient (Log-odds)')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(COEF_PLOT_ORDINAL)
        print(f"Saved Ordinal Regression Coefficient Plot to {COEF_PLOT_ORDINAL}")

        # Calculate McFadden's Pseudo R-squared
        if hasattr(res, 'llnull'):
            mcfadden_r2 = 1 - (res.llf / res.llnull)
            print(f"\nMcFadden's Pseudo R-squared: {mcfadden_r2:.4f}")
            
    except Exception as e:
        print(f"Regression failed: {e}")

    # ---------------------------------------------------------
    # 6. Random Forest Model
    # ---------------------------------------------------------
    print("\n[Step 6] Random Forest Model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    rf_importances = rf.feature_importances_
    rf_imp_df = pd.DataFrame({'Feature': X_scaled.columns, 'Importance': rf_importances})
    rf_imp_df = rf_imp_df.sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(rf_imp_df['Feature'], rf_imp_df['Importance'], color='mediumpurple')
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(FEATURE_PLOT_RF)
    print(f"Saved Random Forest feature importance plot to {FEATURE_PLOT_RF}")

    # --- Cross Validation ---
    print("Running 5-Fold Cross-Validation on Random Forest...")
    cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # --- RF Multi-class ROC (Micro-average) ---
    print("Generating Random Forest Multi-class ROC (Micro-average)...")
    y_prob_rf = rf.predict_proba(X_scaled)
    y_bin = label_binarize(y, classes=sorted(y.unique()))
    n_classes = y_bin.shape[1]
    
    # Compute micro-average ROC curve and ROC area
    fpr_rf, tpr_rf, _ = roc_curve(y_bin.ravel(), y_prob_rf.ravel())
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Micro-average ROC (AUC = {roc_auc_rf:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Random Forest ROC (Micro-average)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(ROC_PLOT_RF_MULTI)
    print(f"Saved Random Forest Micro-average ROC to {ROC_PLOT_RF_MULTI}")
    print(f"RF Multi-class AUC: {roc_auc_rf:.4f}")

    # --- Save RF Report and Table ---
    print("Generating Random Forest Report...")
    
    # 1. Feature Importance CSV
    rf_importance_df = pd.DataFrame({
        'Feature': SELECTED_FEATURES,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    rf_csv_path = '/Users/a11/Downloads/HKU/7005/7005 group project/rf_feature_importance.csv'
    rf_importance_df.to_csv(rf_csv_path, index=False)
    print(f"Saved Random Forest Feature Importance Table to {rf_csv_path}")

    # 2. Model Summary Report (Markdown)
    report_path = '/Users/a11/Downloads/HKU/7005/7005 group project/rf_model_report.md'
    with open(report_path, 'w') as f:
        f.write("# Random Forest Model Report\n\n")
        f.write("## 1. Model Performance\n")
        f.write(f"- **Model Type**: Random Forest Classifier (100 trees)\n")
        f.write(f"- **Target Variable**: Q49 (Life Satisfaction, Scale 1-10)\n")
        f.write(f"- **Cross-Validation Accuracy (5-fold)**: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)\n")
        f.write(f"- **Micro-average AUC**: {roc_auc_rf:.4f}\n\n")
        f.write("## 2. Feature Importance Ranking\n")
        f.write("| Rank | Feature | Importance Score | Description |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        # Dictionary for descriptions (simplified for report)
        descriptions = {
            'Q50': 'Satisfaction with financial situation',
            'Q48': 'Control over your life',
            'Q46': 'Happiness (Recoded)',
            'Q164': 'Importance of God',
            'Q112': 'Confidence: The Government',
            'Q110': 'Confidence: The Press',
            'Q120': 'Confidence: The UN',
            'Q55': 'Freedom of choice and control',
            'Q47': 'Health (Recoded)'
        }
        
        for i, (index, row) in enumerate(rf_importance_df.iterrows()):
            desc = descriptions.get(row['Feature'], '')
            f.write(f"| {i+1} | **{row['Feature']}** | {row['Importance']:.4f} | {desc} |\n")
            
    print(f"Saved Random Forest Model Report to {report_path}")

    # ---------------------------------------------------------
    # 7. ROC Curves (Ordinal Regression)
    # ---------------------------------------------------------
    print("\n[Step 7] Generating ROC Curves...")
    y_prob = res.predict(X_scaled)
    
    # Binary ROC (High Satisfaction > 5)
    y_binary = (y > 5).astype(int)
    # Sum probs for classes > 5 (indices 5-9)
    # Assuming classes are 1-10 sorted
    high_sat_indices = [i for i, c in enumerate(sorted(y.unique())) if c > 5]
    y_prob_binary = y_prob.iloc[:, high_sat_indices].sum(axis=1)
    
    fpr, tpr, _ = roc_curve(y_binary, y_prob_binary)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Binary ROC: High Life Satisfaction (>5)')
    plt.legend(loc="lower right")
    plt.savefig(ROC_PLOT_BINARY)
    print(f"Binary ROC AUC: {roc_auc:.4f}")
    print(f"Saved Binary ROC plot to {ROC_PLOT_BINARY}")
    
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
