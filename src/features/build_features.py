import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def build_features():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    input_path = os.path.join(project_root, "data", "processed", "crop_intervention_cleaned.csv")
    output_dir = os.path.join(project_root, "data", "processed")
    doc_dir = os.path.join(project_root, "docs", "data_cards")
    
    df = pd.read_csv(input_path)
    print(f" Loaded Cleaned Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Feature Engineering: Domain Domain Metrics
    # Vapor Pressure Deficit (VPD approximation in kPa)
    svp = 0.61078 * np.exp((17.27 * df['avg_temp_c']) / (df['avg_temp_c'] + 237.3))
    avp = svp * (df['avg_humidity_pct'] / 100.0)
    df['vpd_kpa'] = np.round(svp - avp, 3)
    
    # NPK Total Load
    df['total_npk_ppm'] = df['nitrogen_ppm'] + df['phosphorus_ppm'] + df['potassium_ppm']
    
    # Soil pH Deviation from Neutral (6.5 - 7.5)
    df['ph_deviation'] = np.abs(df['soil_ph'] - 7.0)

    # 2. One-Hot Encoding Categoricals
    categorical_cols = ['region', 'crop_type', 'growth_stage']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Drop Identifier columns not useful for training
    features_df = df_encoded.drop(columns=['field_id'])

    # 3. Train-Test Split (80/20 Stratified)
    X = features_df.drop(columns=['intervention_required'])
    y = features_df['intervention_required']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save Processed Datasets
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f" Features engineered and split into Train ({X_train.shape[0]}) / Test ({X_test.shape[0]}) sets.")

    # 4. Generate Feature Documentation
    doc_path = os.path.join(doc_dir, "feature_engineering.md")
    with open(doc_path, "w") as f:
        f.write("# Feature Engineering & Data Pipeline Architecture\n\n")
        f.write("## Engineered Domain Features\n")
        f.write("- **`vpd_kpa`**: Vapor Pressure Deficit derived from ambient temperature and relative humidity.\n")
        f.write("- **`total_npk_ppm`**: Aggregate soil macronutrient load (N + P + K).\n")
        f.write("- **`ph_deviation`**: Absolute deviation of soil pH from ideal neutral state (7.0).\n\n")
        f.write("## Dataset Dimensions\n")
        f.write(f"- Train Set: {X_train.shape[0]} samples, {X_train.shape[1]} features\n")
        f.write(f"- Test Set: {X_test.shape[0]} samples, {X_test.shape[1]} features\n")
        f.write(f"- Target Class Distribution (Positive/Intervention Needed): {y.mean():.2%}\n")

    print(f" Feature documentation written to: {doc_path}")

if __name__ == "__main__":
    build_features()