import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Configuration
IMPUTED_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_imputed_median.csv'
OUTPUT_PLOT = '/Users/a11/Downloads/HKU/7005/7005 group project/feature_importance_plot.png'
TOP_N = 20

def visualize_importance():
    print(f"Loading data from {IMPUTED_FILE}...")
    try:
        df = pd.read_csv(IMPUTED_FILE)
    except FileNotFoundError:
        print("Error: Imputed file not found. Please run the modeling pipeline first.")
        return

    y = df['Q49']
    X = df.drop(columns=['Q49'])
    
    print("Training AdaBoostClassifier...")
    base_estimator = DecisionTreeClassifier(max_depth=1)
    ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
    ada.fit(X, y)
    
    # Get importances
    importances = ada.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    # Print List
    print(f"\n--- Top {TOP_N} Feature Importance Ranking ---")
    print(feature_imp_df.head(TOP_N).to_string(index=False))
    
    # Plot
    print(f"\nGenerating plot for top {TOP_N} features...")
    plt.figure(figsize=(12, 8))
    
    # Plot top N
    top_data = feature_imp_df.head(TOP_N).sort_values(by='Importance', ascending=True) # Ascending for horizontal bar chart
    
    plt.barh(top_data['Feature'], top_data['Importance'], color='skyblue')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature')
    plt.title('Top Predictors of Life Satisfaction (AdaBoost)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    visualize_importance()
