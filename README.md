# MSDA7005 Group Project (Group 8)

**Project Title:** What Drives Life Satisfaction? Insights from the World Values Survey Using Theory-Driven and Data-Driven Approaches

This repository contains the source code and analysis scripts for our group project investigating the determinants of life satisfaction using World Values Survey (WVS) data from Waves 5, 6, and 7.

## Repository Contents

*   **`run_multi_wave_analysis.py`**: **[NEW]** The core script for the Data-Driven analysis. It runs the Ordered Logistic Regression on AdaBoost-selected features for Waves 5, 6, and 7 and generates the ranking plots (`Figure_4...`, `Figure_5...`, etc.).
*   **`master_pipeline.py`**: The main execution pipeline for WVS Wave 7. It performs data cleaning, imputation, AdaBoost feature selection, Ordinal Logistic Regression, and Random Forest modeling.
*   **`analyze_wave5_6.py`**: The analysis script for WVS Waves 5 and 6, performing consistent cleaning, imputation, and AdaBoost feature extraction.
*   **`generate_report_plots.py`**: Script used to generate the specific high-quality plots used in the final report.
*   **`clean_wvs_data.py`**: Standalone module for cleaning WVS Wave 7 data.
*   **`adaboost.ipynb`**: Jupyter Notebook containing exploratory AdaBoost analysis.
*   **`Theory-driven_SDT_analysis.ipynb`**: Jupyter Notebook containing the initial theory-driven analysis (SDT Framework).

## Prerequisites

To run these scripts, you need **Python 3.x** and the following libraries installed:

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn
*   statsmodels
*   openpyxl

You can install the dependencies using the provided `package requirements.txt`:

```bash
pip install -r "package requirements.txt"
```

## Data Setup

**Note:** The raw World Values Survey datasets are **not** included in this repository due to size and redistribution limits. You must download them from the [official WVS website](https://www.worldvaluessurvey.org/) and place them in the project root directory.

Required Files:
1.  **Wave 5**: `WV5_Data_csv_v20180912.csv`
2.  **Wave 6**: `WV6_Data_csv_v20201117.csv`
3.  **Wave 7**: `WVS_Cross-National_Wave_7_csv_v6_0.xlsx` (or `WVS_imputed_median.csv` if creating from intermediate steps)

## Execution Guide

### 1. Multi-Wave Analysis (Data-Driven Results)
To replicate the Ordered Logistic Regression results and generate the key figures (Top 10 Predictors for Waves 5, 6, 7):

```bash
python run_multi_wave_analysis.py
```
*   **Outputs**:
    *   `Figure_4_Wave_5.png`: Ranking of predictors for Wave 5.
    *   `Figure_5_Wave_6.png`: Ranking of predictors for Wave 6.
    *   `Figure_6_Wave_7.png`: Ranking of predictors for Wave 7.

### 2. Analyze Wave 7 (Main Pipeline)
To run the full pipeline for Wave 7, including data cleaning, imputation, and model training (AdaBoost, Ordinal Regression, Random Forest):

```bash
python master_pipeline.py
```
*   **Outputs**:
    *   `WVS_Cross-National_Wave_7_cleaned.csv`: Cleaned dataset.
    *   `feature_importance_plot.png`: Feature importance graph.
    *   `rf_model_report.md`: Summary of the Random Forest model.

### 3. Analyze Waves 5 & 6 (Longitudinal Checks)
To run the comparative analysis for Waves 5 and 6:

```bash
python analyze_wave5_6.py
```
*   **Outputs**:
    *   `WVS_Wave5_cleaned_imputed.csv`: Cleaned/Imputed data for Wave 5.
    *   `WVS_Wave6_cleaned_imputed.csv`: Cleaned/Imputed data for Wave 6.
    *   `feature_importance_wave5.png` & `feature_importance_wave6.png`: Comparison plots.

### Report Content
*   **`Data_Driven_Analysis.md`**: Contains the draft text for the "Data-Driven Analysis" section of the final report, including Table 2 and discussion of the figures.
