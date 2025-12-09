# MSDA7005 Group Project (Group 8)

**Project Title:** What Drives Life Satisfaction? Insights from the World Values Survey Using Theory-Driven and Data-Driven Approaches

This repository contains the source code and analysis scripts for our group project investigating the determinants of life satisfaction using World Values Survey (WVS) data from Waves 5, 6, and 7.

Dataset and WVS Variable codebooks can be found at:https://www.worldvaluessurvey.org/WVSContents.jsp


## Repository Contents

*   **`master_pipeline.py`**: The main execution pipeline for WVS Wave 7. It performs data cleaning, imputation, AdaBoost feature selection, Ordinal Logistic Regression, and Random Forest modeling.
*   **`analyze_wave5_6.py`**: The analysis script for WVS Waves 5 and 6, performing consistent cleaning, imputation, and AdaBoost feature extraction.
*   **`visualize_feature_importance.py`**: Script to generate feature importance visualizations.
*   **`generate_report_plots.py`**: Script used to generate the specific high-quality plots used in the final report.
*   **`clean_wvs_data.py`**: Standalone module for cleaning WVS Wave 7 data.
*   **`adaboost.ipynb`**: Jupyter Notebook containing exploratory AdaBoost analysis.

## Prerequisites

To run these scripts, you need **Python 3.x** and the following libraries installed:

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn
*   statsmodels
*   openpyxl

You can install the dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data Setup

**Note:** The raw World Values Survey datasets are **not** included in this repository due to size and redistribution limits. You must download them from the [official WVS website](https://www.worldvaluessurvey.org/) and place them in the project root directory.

Required Files:
1.  **Wave 5**: `WV5_Data_csv_v20180912.csv`
2.  **Wave 6**: `WV6_Data_csv_v20201117.csv`
3.  **Wave 7**: `WVS_Cross-National_Wave_7_csv_v6_0.xlsx`

## Execution Guide

### 1. Analyze Wave 7 (Main Model)
To run the full pipeline for Wave 7, including data cleaning, imputation, and model training (AdaBoost, Ordinal Regression, Random Forest):

```bash
python master_pipeline.py
```
*   **Outputs**:
    *   `WVS_Cross-National_Wave_7_cleaned.csv`: Cleaned dataset.
    *   `feature_importance_plot.png`: Feature importance graph.
    *   `rf_model_report.md`: Summary of the Random Forest model.
    *   Various ROC curve and coefficient plots.

### 2. Analyze Waves 5 & 6 (Longitudinal Checks)
To run the comparative analysis for Waves 5 and 6:

```bash
python analyze_wave5_6.py
```
*   **Outputs**:
    *   `WVS_Wave5_cleaned_imputed.csv`: Cleaned/Imputed data for Wave 5.
    *   `WVS_Wave6_cleaned_imputed.csv`: Cleaned/Imputed data for Wave 6.
    *   `feature_importance_wave5.png` & `feature_importance_wave6.png`: Comparison plots.

### 3. Generate Report Plots
To reproduce the specific styled plots used in the final report:

```bash
python generate_report_plots.py
```
