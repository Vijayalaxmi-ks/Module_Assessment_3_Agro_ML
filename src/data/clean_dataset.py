import os
import io
import pandas as pd
import numpy as np

def clean_crop_data():
    # 1. Resolve paths dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    raw_path = os.path.join(project_root, "data", "raw", "crop_intervention_raw.csv")
    processed_dir = os.path.join(project_root, "data", "processed")
    doc_dir = os.path.join(project_root, "docs", "data_cards")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(doc_dir, exist_ok=True)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}. Run make_dataset.py first.")

    df = pd.read_csv(raw_path)
    print(f" Loaded Raw Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    cleaning_logs = []

    # 2. Text Standardization
    df['crop_type'] = df['crop_type'].str.capitalize()
    df['growth_stage'] = df['growth_stage'].str.capitalize()
    cleaning_logs.append("Standardized casing for 'crop_type' and 'growth_stage'.")

    # 3. Outlier / Sensor Glitch Handling
    invalid_temp_count = ((df['avg_temp_c'] < -10) | (df['avg_temp_c'] > 60)).sum()
    df.loc[(df['avg_temp_c'] < -10) | (df['avg_temp_c'] > 60), 'avg_temp_c'] = np.nan
    
    invalid_hum_count = ((df['avg_humidity_pct'] < 0) | (df['avg_humidity_pct'] > 100)).sum()
    df.loc[(df['avg_humidity_pct'] < 0) | (df['avg_humidity_pct'] > 100), 'avg_humidity_pct'] = np.nan
    
    cleaning_logs.append(f"Flagged {invalid_temp_count} invalid temperature glitches (-999 C) as NaN.")
    cleaning_logs.append(f"Flagged {invalid_hum_count} invalid humidity glitches (>100%) as NaN.")

    # 4. Impute Missing Values (Grouped by Crop Type & Growth Stage)
    numerical_cols = ['avg_temp_c', 'avg_humidity_pct', 'soil_moisture_pct', 'soil_ph', 'ndvi_index']
    
    for col in numerical_cols:
        missing_before = df[col].isnull().sum()
        df[col] = df.groupby(['crop_type', 'growth_stage'])[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())
        cleaning_logs.append(f"Imputed {missing_before} missing/glitched values in '{col}' using group medians.")

    # 5. Save Cleaned Dataset
    output_path = os.path.join(processed_dir, "crop_intervention_cleaned.csv")
    df.to_csv(output_path, index=False)
    print(f" Cleaned dataset saved to: {output_path}")

    # 6. Capture DataFrame Info as String
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # 7. Generate Data Cleaning Documentation
    log_path = os.path.join(doc_dir, "data_cleaning_log.md")
    with open(log_path, "w") as f:
        f.write("# Data Cleaning & Quality Assurance Log\n\n")
        f.write(f"**Total Records Processed:** {len(df)}\n\n")
        f.write("## Actions Taken:\n")
        for log in cleaning_logs:
            f.write(f"- {log}\n")
        f.write("\n## Cleaned Data Schema Summary:\n")
        f.write("```text\n")
        f.write(info_str)
        f.write("\n```\n")
        
    print(f" Data cleaning documentation saved to: {log_path}")

if __name__ == "__main__":
    clean_crop_data()