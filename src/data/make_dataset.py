import pandas as pd
import numpy as np
import os

def generate_raw_crop_data(n_samples=10000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Base Categorical Variables
    field_ids = [f"PLOT_{np.random.randint(100, 250):03d}" for _ in range(n_samples)]
    crop_types = np.random.choice(['Wheat', 'Cotton', 'Sugarcane', 'Soybean', 'Tomato'], size=n_samples, p=[0.25, 0.25, 0.2, 0.15, 0.15])
    growth_stages = np.random.choice(['Vegetative', 'Flowering', 'Fruiting', 'Maturation'], size=n_samples, p=[0.35, 0.3, 0.2, 0.15])
    regions = np.random.choice(['Zone_A_North', 'Zone_B_West', 'Zone_C_South', 'Zone_D_East'], size=n_samples)
    
    # 2. Continuous Telemetry Data (Real Relationships)
    temperature = np.random.normal(loc=28.5, scale=5.2, size=n_samples)
    humidity = np.random.normal(loc=65.0, scale=12.0, size=n_samples)
    soil_moisture = np.random.normal(loc=45.0, scale=15.0, size=n_samples)
    
    # Soil Chemistry
    soil_ph = np.random.normal(loc=6.8, scale=0.8, size=n_samples)
    nitrogen_level = np.random.normal(loc=120.0, scale=30.0, size=n_samples)
    phosphorus_level = np.random.normal(loc=45.0, scale=15.0, size=n_samples)
    potassium_level = np.random.normal(loc=180.0, scale=40.0, size=n_samples)
    
    # NDVI (Normalized Difference Vegetation Index: -0.1 to 0.9)
    # Dependent on soil moisture, nitrogen, and crop health
    ndvi = 0.2 + (0.003 * soil_moisture) + (0.001 * nitrogen_level) + np.random.normal(0, 0.08, size=n_samples)
    ndvi = np.clip(ndvi, -0.1, 0.95)
    
    # Pest Pressure Index (0 to 100)
    # High temp + high humidity drive pest pressure
    pest_index = (0.8 * temperature) + (0.6 * humidity) - (0.2 * soil_moisture) + np.random.normal(0, 10, size=n_samples)
    pest_index = np.clip(pest_index, 0, 100)

    # 3. Target Variable: Intervention Required (Non-Linear Rule with Noise)
    # Intervention required if severe pest pressure, extreme soil moisture loss, or severe nutrient drop
    stress_score = (
        (pest_index > 65).astype(int) * 3.5 +
        (ndvi < 0.35).astype(int) * 2.5 +
        (soil_moisture < 25).astype(int) * 2.0 +
        ((soil_ph < 5.5) | (soil_ph > 8.0)).astype(int) * 1.5 +
        np.random.normal(0, 1.2, size=n_samples)
    )
    
    # Binary Target (Asymmetric classes ~ 22% positive interventions)
    intervention_required = (stress_score > 3.8).astype(int)
    
    # 4. Inject Real-World Messiness (Data Anomalies & Noise)
    df = pd.DataFrame({
        'field_id': field_ids,
        'region': regions,
        'crop_type': crop_types,
        'growth_stage': growth_stages,
        'avg_temp_c': np.round(temperature, 2),
        'avg_humidity_pct': np.round(humidity, 2),
        'soil_moisture_pct': np.round(soil_moisture, 2),
        'soil_ph': np.round(soil_ph, 2),
        'nitrogen_ppm': np.round(nitrogen_level, 2),
        'phosphorus_ppm': np.round(phosphorus_level, 2),
        'potassium_ppm': np.round(potassium_level, 2),
        'ndvi_index': np.round(ndvi, 3),
        'pest_pressure_score': np.round(pest_index, 1),
        'intervention_required': intervention_required
    })
    
    # Messiness A: Missing Values (Sensor dropouts)
    missing_cols = ['avg_humidity_pct', 'soil_moisture_pct', 'soil_ph', 'ndvi_index']
    for col in missing_cols:
        mask = np.random.rand(len(df)) < 0.05  # 5% missing
        df.loc[mask, col] = np.nan
        
    # Messiness B: Sensor Glitches / Outliers (e.g. Temp = -999, Humidity = 300%)
    df.loc[np.random.choice(n_samples, 25, replace=False), 'avg_temp_c'] = -999.0
    df.loc[np.random.choice(n_samples, 20, replace=False), 'avg_humidity_pct'] = 250.0
    
    # Messiness C: Inconsistent Text (Data entry typos)
    df.loc[np.random.choice(n_samples, 40, replace=False), 'crop_type'] = 'wheat'  # lowercase
    df.loc[np.random.choice(n_samples, 30, replace=False), 'growth_stage'] = 'FLOWERING'  # uppercase
    
    return df

if __name__ == "__main__":
    raw_df = generate_raw_crop_data(n_samples=10000)
    
    # Get absolute path to project root (2 levels up from src/data/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    output_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, "crop_intervention_raw.csv")
    raw_df.to_csv(filepath, index=False)
    print(f" Raw dataset successfully created with {len(raw_df)} records at:\n -> {filepath}")