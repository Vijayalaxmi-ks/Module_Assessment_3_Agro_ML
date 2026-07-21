import os
import joblib
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def project_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    model_path = os.path.join(project_root, "models", "best_crop_intervention_model.joblib")
    features_path = os.path.join(project_root, "data", "processed", "X_train.csv")
    return model_path, features_path

def test_artifacts_exist(project_paths):
    model_path, features_path = project_paths
    assert os.path.exists(model_path), f"Model file missing at {model_path}"
    assert os.path.exists(features_path), f"Features CSV missing at {features_path}"

def test_model_inference_output(project_paths):
    model_path, features_path = project_paths
    
    # Load model and feature columns
    model = joblib.load(model_path)
    feature_columns = pd.read_csv(features_path, nrows=1).columns.tolist()

    # Create dummy sample row matching feature columns
    sample_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
    
    # Populate key numerical features
    sample_df['avg_temp_c'] = 28.5
    sample_df['avg_humidity_pct'] = 65.0
    sample_df['soil_moisture_pct'] = 45.0
    sample_df['soil_ph'] = 6.8
    sample_df['nitrogen_ppm'] = 120.0
    sample_df['phosphorus_ppm'] = 45.0
    sample_df['potassium_ppm'] = 180.0
    sample_df['ndvi_index'] = 0.65
    sample_df['pest_pressure_score'] = 20.0
    sample_df['vpd_kpa'] = 1.35
    sample_df['total_npk_ppm'] = 345.0
    sample_df['ph_deviation'] = 0.2

    # Predict probability
    proba = model.predict_proba(sample_df)[0][1]

    # Verification checks
    assert isinstance(proba, (float, np.float64))
    assert 0.0 <= proba <= 1.0