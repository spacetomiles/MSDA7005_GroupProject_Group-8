# MSDA7005 Group Project (Group 8)

**Project Title:** What Drives Life Satisfaction? Insights from the World Values Survey Using Theory-Driven and Data-Driven Approaches

This repository contains the analysis scripts for our project. We utilize World Values Survey (WVS) data (Waves 5, 6, and 7) to identify predictors of life satisfaction using two approaches:
1.  **Data-Driven**: Feature selection via AdaBoost followed by Ordered Logistic Regression.
2.  **Theory-Driven**: Self-Determination Theory (SDT) variables analysed via Ordered Logistic Regression.

## Repository Contents

*   **`run_multi_wave_analysis.py`**: Performs the Data-Driven analysis (AdaBoost features) for Waves 5, 6, and 7. Generates the coefficient ranking figures used in the report.
*   **`run_wave7_pipeline.py`**: A complete pipeline for Wave 7 specifically. Handles raw data cleaning, imputation, AdaBoost feature selection, and Ordered Logistic Regression.
*   **`analyze_wave5_6.py`**: Analysis script for Waves 5 and 6, performing consistent cleaning and AdaBoost feature extraction.

## Prerequisites

*   Python 3.x
*   pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, openpyxl

Install dependencies:
```bash
pip install -r requirements.txt
```

## Comparisons

*   **Data-Driven Analysis**: See `run_multi_wave_analysis.py` execution output and generated figures.
*   **Theory-Driven Analysis**: Variables based on SDT (Competence, Autonomy, Relatedness) are analyzed in `Theory-driven_SDT_analysis.ipynb`.

## How to Run

1.  **Generate Multi-Wave Data-Driven Results:**
    ```bash
    python run_multi_wave_analysis.py
    ```
    This generates the visual comparisons for all three waves as seen in the report.

2.  **Run Wave 7 Full Processing:**
    ```bash
    python run_wave7_pipeline.py
    ```
