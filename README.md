# MSDA7005 Group Project (Group 8)

**Project Title:** What Drives Life Satisfaction? Insights from the World Values Survey Using Theory-Driven and Data-Driven Approaches

This repository contains the analysis scripts for our project. We utilize World Values Survey (WVS) data (Waves 5, 6, and 7) to identify predictors of life satisfaction using two approaches:
1.  **Data-Driven**: Feature selection via AdaBoost followed by Ordered Logistic Regression.
2.  **Theory-Driven**: Self-Determination Theory (SDT) variables analysed via Ordered Logistic Regression.

## Repository Contents

*   **`run_wave7_pipeline.py`**: Performs cleaning, imputation, and AdaBoost feature selection for **Wave 7**.
*   **`analyze_wave5_6.py`**: Performs cleaning and AdaBoost feature selection for **Waves 5 & 6**.
*   **`run_multi_wave_analysis.py`**: Takes the *top features* identified and performed the final Ordered Logistic Regression models with standardized coefficient plots (`Figure_4...`, `Figure_5...`, etc.) used in the final report.

## Prerequisites

*   Python 3.x
*   pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, openpyxl

## Comparisons

*   **Data-Driven Analysis**: See `run_multi_wave_analysis.py` execution output and generated figures.
*   **Theory-Driven Analysis**: Variables based on SDT (Competence, Autonomy, Relatedness) are analyzed in `Theory-driven_SDT_analysis.ipynb`.

