import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Data Preparation ---

# Wave 7 Data (extracted from analysis)
wave7_data = {
    'Feature': ['Q50', 'Q48', 'Q46', 'Q164', 'Q110', 'Q112', 'Q120', 'Q159', 'Q55', 'Q47'],
    'Importance': [0.445, 0.294, 0.173, 0.020, 0.015, 0.015, 0.012, 0.011, 0.009, 0.006],
    'Description': [
        'Financial Satisfaction', 'Freedom of Choice', 'State of Health', 'Importance of God',
        'Income Equality', 'Corruption in Gov\'t', 'Confidence: Gov\'t', 'Science & Tech Impact',
        'Job Satisfaction', 'Feeling of Happiness'
    ]
}

# Wave 6 Data (from wave6_top_features.csv)
wave6_data = {
    'Feature': ['V59', 'V10', 'V55', 'V11', 'V56', 'V141', 'V160', 'V193', 'V152', 'V154'],
    'Importance': [0.378, 0.294, 0.200, 0.036, 0.028, 0.022, 0.015, 0.014, 0.013, 0.009], # Approx values from summary
    'Description': [
        'Financial Satisfaction', 'Feeling of Happiness', 'Freedom of Choice', 'State of Health',
        'Importance of God', 'Democracy: Tax Rich', 'Confidence: Courts', 'Justifiable: Homosexuality',
        'Confidence: Press', 'Confidence: Police'
    ]
}

# Wave 5 Data (from wave5_top_features.csv)
wave5_data = {
    'Feature': ['V68', 'V10', 'V46', 'V3', 'V11', 'V192', 'V202', 'V118', 'V222', 'V170'],
    'Importance': [0.420, 0.288, 0.171, 0.040, 0.025, 0.017, 0.015, 0.013, 0.011, 0.009], # Approx values from summary
    'Description': [
        'Financial Satisfaction', 'Feeling of Happiness', 'Freedom of Choice', 'Leisure Time',
        'State of Health', 'Justifiable: Homosexuality', 'Justifiable: Fare Avoidance',
        'Democracy: Tax Rich', 'World Citizen', 'Secure in Neighborhood'
    ]
}

# --- Plotting Function ---

def create_plot(data_dict, wave_name, filename, color_palette="viridis"):
    df = pd.DataFrame(data_dict)
    # Sort by importance
    df = df.sort_values('Importance', ascending=True)
    
    # Create label with code
    df['Label'] = df['Description'] + ' (' + df['Feature'] + ')'

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create horizontal bar plot
    bars = plt.barh(df['Label'], df['Importance'], color=sns.color_palette(color_palette, len(df)))
    
    # Add values to end of bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', 
                 va='center', ha='left', fontsize=10, color='black')

    plt.title(f'Top 10 Predictors of Life Satisfaction - {wave_name}', fontsize=14, loc='left', pad=20)
    plt.xlabel('Feature Importance (AdaBoost)', fontsize=12)
    plt.xlim(0, max(df['Importance']) * 1.15) # Add space for labels
    
    # Clean up spines
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

# --- Execution ---

def main():
    output_dir = '/Users/a11/.gemini/antigravity/brain/c382b794-6317-44bc-8aa7-4b61d214383c'
    
    create_plot(wave7_data, "Wave 7 (2017-2022)", os.path.join(output_dir, "wvs_wave7_importance.png"), "mako")
    create_plot(wave6_data, "Wave 6 (2010-2014)", os.path.join(output_dir, "wvs_wave6_importance.png"), "rocket")
    create_plot(wave5_data, "Wave 5 (2005-2009)", os.path.join(output_dir, "wvs_wave5_importance.png"), "viridis")

if __name__ == "__main__":
    main()
