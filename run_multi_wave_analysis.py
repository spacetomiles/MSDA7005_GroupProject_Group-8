import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import StandardScaler
import os

# --- Configuration ---
# File Paths
FILES = {
    'Wave_5': 'WV5_Data_csv_v20180912.csv',
    'Wave_6': 'WV6_Data_csv_v20201117.csv',
    'Wave_7': 'WVS_imputed_median.csv'
}

# Variable Mappings & Labels
# Extracted from user screenshots + standard WVS knowledge
CONFIG = {
    'Wave_5': {
        'Target': 'V22',
        'Features': ['V68', 'V10', 'V46', 'V3', 'V11', 'V192', 'V202', 'V118', 'V222', 'V170'],
        'Labels': {
            'V68': 'Financial Satisfaction',
            'V10': 'Feeling of Happiness', # Scale REVERSE? Usually 1=Very Happy, 4=Not at all. High is bad.
            'V46': 'Freedom of Choice', # 1-10. High is good.
            'V3': 'Leisure Time', # 1=Very important... 4=Not important? Or 1=Important. CHECK. Usually 1=Very Important.
            'V11': 'State of Health', # 1=Very good... 4=Poor. High is bad. REVERSE.
            'V192': 'Justifiable: Homosexuality', # 1-10. High is more justifiable? CHECK.
            'V202': 'Justifiable: Fare Avoidance',
            'V118': 'Democracy: Tax Rich', 
            'V222': 'World Citizen',
            'V170': 'Secure in Neighborhood',
            'V22': 'Life Satisfaction'
        },
        'Reverse': ['V10', 'V11'] # Happiness, Health (1-4 scales where 1 is best)
    },
    'Wave_6': {
        'Target': 'V23',
        'Features': ['V59', 'V10', 'V55', 'V11', 'V56', 'V141', 'V160', 'V193', 'V152', 'V154'],
        'Labels': {
            'V59': 'Financial Satisfaction',
            'V10': 'Feeling of Happiness',
            'V55': 'Freedom of Choice',
            'V11': 'State of Health',
            'V56': 'Importance of God', # 1=Very imp... 10=Not? Or 10? WVS 6 V56 is 10 point? No, usually 1-10 importance.
            'V141': 'Democracy: Tax Rich',
            'V160': 'Confidence: Courts',
            'V193': 'Justifiable: Homosexuality',
            'V152': 'Confidence: Press',
            'V154': 'Confidence: Police',
            'V23': 'Life Satisfaction'
        },
        'Reverse': ['V10', 'V11']
    },
    'Wave_7': {
        'Target': 'Q49',
        'Features': ['Q50', 'Q48', 'Q46', 'Q164', 'Q110', 'Q112', 'Q120', 'Q159', 'Q55', 'Q47'],
        'Labels': {
            'Q50': 'Financial Satisfaction',
            'Q48': 'Freedom of Choice',
            'Q46': 'State of Health', # 1=Very good... 4=Poor. Reverse.
            'Q164': 'Importance of God', # 1=Very imp... 10=Not imp. Scale Varies. Usually 10 is very important in newer waves? Check. 
            # Actually Q164 W7: "How important is God in your life" 10 mean very important, 1 mean not at all.
            # Q46 W7: 1=Very good, 4=Very poor. REVERSE.
            # Q47 W7: Happiness. 1=Very happy, 4=Not at all. REVERSE.
            'Q110': 'Income Equality',
            'Q112': 'Corruption in Gov\'t',
            'Q120': 'Confidence: Gov\'t',
            'Q159': 'Science/Tech Impact',
            'Q55': 'Job Satisfaction',
            'Q47': 'Feeling of Happiness',
            'Q49': 'Life Satisfaction'
        },
        'Reverse': ['Q46', 'Q47']
    }
}

OUTPUT_DIR = 'multi_wave_analysis_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess(wave_name, config):
    filepath = FILES[wave_name]
    print(f"[{wave_name}] Loading data from {filepath}...")
    
    # Read CSV
    # Wave 5/6 are Semicolon separated often, Wave 7 is comma.
    try:
        # Try finding the file or fallback to clean versions
        if not os.path.exists(filepath):
            # Fallbacks for specific clean files known in environment
            if 'Wave 5' in wave_name: filepath = 'WVS_Wave5_cleaned_imputed.csv'
            elif 'Wave 6' in wave_name: filepath = 'WVS_Wave6_cleaned_imputed.csv'
            elif 'Wave 7' in wave_name: filepath = 'WVS_imputed_median.csv'
        
        # Determine separator
        sep = ';' if 'WV5' in filepath or 'WV6' in filepath else ','
        
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Filter Features + Target
    feats = config['Features']
    target = config['Target']
    needed_cols = feats + [target]
    
    # Check if cols exist
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns in {wave_name}: {missing}")
        # Try to continue? Or return.
        return None

    df_subset = df[needed_cols].copy()
    
    # Handle Numeric
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    
    # Handle Negatives (Missing)
    df_subset[df_subset < 0] = np.nan
    
    # Impute (Median)
    df_subset = df_subset.fillna(df_subset.median())
    
    # Reverse Codes
    for col in config['Reverse']:
        if col in df_subset.columns:
            # Reversing 1..4 to 4..1
            # New = Max + Min - Old
            mx = df_subset[col].max()
            mn = df_subset[col].min()
            df_subset[col] = mx + mn - df_subset[col]
            
    return df_subset

def run_analysis_for_wave(wave_name):
    cfg = CONFIG[wave_name]
    df = load_and_preprocess(wave_name, cfg)
    if df is None: return

    print(f"[{wave_name}] Running Ordered Logit...")
    X = df[cfg['Features']]
    y = df[cfg['Target']].astype(int)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Model
    model = OrderedModel(y, X_scaled, distr='logit')
    try:
        res = model.fit(method='bfgs', disp=False)
    except:
        print(f"[{wave_name}] BFGS failed, trying different solver...")
        res = model.fit(method='lbfgs', disp=False)

    # Plot
    plot_coefficients(res, wave_name, cfg)

def plot_coefficients(model_res, wave_name, config):
    print(f"[{wave_name}] Generating Plot...")
    params = model_res.params
    feats = config['Features']
    
    # Extract only feature coefs (ignore threshold cuts 1/2 etc)
    coefs = params[feats]
    pvalues = model_res.pvalues[feats]
    
    # Create DF
    plot_df = pd.DataFrame({
        'Feature': feats,
        'Coef': coefs.values,
        'P-value': pvalues.values,
        'Label': [config['Labels'].get(f, f) for f in feats]
    })
    
    # Sort by Abs Coef
    plot_df['AbsCoef'] = plot_df['Coef'].abs()
    plot_df = plot_df.sort_values(by='AbsCoef', ascending=True)

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(plot_df['Label'], plot_df['Coef'], color=np.where(plot_df['Coef']>0, 'skyblue', 'salmon'), edgecolor='black')
    
    plt.title(f'Top 10 Predictors of Life Satisfaction - {wave_name.replace("_", " ")}')
    plt.xlabel('Standardized Coefficient')
    plt.axvline(0, color='grey', linewidth=0.8)

    # Calculate max absolute coef for symmetrical padding
    max_abs = plot_df['Coef'].abs().max()
    padding = max_abs * 0.3  # 30% padding for labels
    plt.xlim(-max_abs - padding, max_abs + padding)

    # Annotate with Stars AND Values
    for bar, p, val in zip(bars, plot_df['P-value'], plot_df['Coef']):
        width = bar.get_width()
        
        # Stars
        stars = ''
        if p < 0.001: stars = '***'
        elif p < 0.01: stars = '**'
        elif p < 0.05: stars = '*'
        
        # Value Text
        val_text = f'{val:.3f}'
        
        # Combine
        full_text = f'{val_text} {stars}'
        
        # Positioning
        x_pos = width
        align = 'left' if width > 0 else 'right'
        offset = padding * 0.1 if width > 0 else -padding * 0.1
        
        plt.text(x_pos + offset, bar.get_y() + bar.get_height()/2, full_text, 
                 va='center', ha=align, fontsize=10, fontweight='bold')
    
    # Adjust layout to fit text
    plt.tight_layout()
    
    # Determine Figure Number based on Wave
    fig_num = '4' if 'Wave_5' in wave_name else '5' if 'Wave_6' in wave_name else '6'
    filename = f'Figure_{fig_num}_{wave_name}.png'
    save_path = filename # Save to current directory
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # bbox_inches='tight' helps with y-axis labels too
    print(f"[{wave_name}] Saved plot to {save_path}")
    plt.close()

def main():
    for wave in ['Wave_5', 'Wave_6', 'Wave_7']:
        run_analysis_for_wave(wave)
        print("-" * 30)

if __name__ == "__main__":
    main()
