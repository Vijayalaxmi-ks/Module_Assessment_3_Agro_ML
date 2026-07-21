# 🌾 AgroGenAI: Early Crop Intervention & Decision Support System

**AgroGenAI** is a machine-learning-powered precision agriculture decision support system designed to predict crop stress and recommend early agricultural interventions within a critical 7-day window. By integrating soil chemistry ($N, P, K, pH$), micro-climate parameters (Temperature, Relative Humidity), satellite vegetation indices ($\text{NDVI}$), and biological stress metrics (Pest Pressure Score), the system computes real-time risk scores and actionable field advice.

---

## 📁 Repository Directory Structure

```text
📂 Module_Assessment_3_Agro_ML/
├── 📂 app/
│   └── 📜 app.py                      # Interactive Streamlit Web Application
├── 📂 data/
│   ├── 📂 external/                   # External datasets & supplementary metadata
│   ├── 📂 processed/                  # Cleaned & feature-engineered datasets
│   │   ├── 📊 crop_intervention_dataset_engineered.csv
│   │   ├── 📊 X_test.csv
│   │   ├── 📊 X_train.csv
│   │   ├── 📊 y_test.csv
│   │   └── 📊 y_train.csv
│   └── 📂 raw/                        # Raw telemetry sensor inputs
│       └── 📊 crop_intervention_dataset.csv
├── 📂 docs/                           # Project documentation & engineering reports
├── 📂 models/
│   └── 📦 best_crop_intervention_model.joblib # Serialized Random Forest Model
├── 📂 notebooks/                      # Exploratory Data Analysis & Modeling Notebooks
├── 📂 src/                            # Core Source Code Modules
│   ├── 📂 data/                       # Data processing scripts
│   │   ├── 📜 __init__.py
│   │   ├── 📜 clean_dataset.py
│   │   └── 📜 make_dataset.py
│   ├── 📂 features/                   # Feature engineering pipeline scripts
│   │   ├── 📜 __init__.py
│   │   └── 📜 build_features.py
│   ├── 📂 models/                     # Model training & hyperparameter tuning
│   │   ├── 📜 __init__.py
│   │   └── 📜 train_model.py
│   └── 📂 utils/                      # Helper & logging functions
│       └── 📜 __init__.py
├── 📂 tests/                          # Automated Pytest Suite
│   ├── 📜 __init__.py
│   ├── 📜 test_features.py            # Unit tests for domain feature formulas (VPD, NPK)
│   └── 📜 test_model.py               # Integration tests for model artifact loading & inference
├── 📄 .gitignore                      # Git exclusion rules
├── 📜 README.md                       # High-level project summary & documentation
└── 📜 requirements.txt                # Python dependency configuration

```

## 🛠️ Technology Stack

* **Programming Language:** Python 3.10+
* **Machine Learning & Pipeline:** `scikit-learn`, `joblib`, `pandas`, `numpy`
* **Visualization & Frontend UI:** `streamlit`, `plotly`
* **Testing & Quality Assurance:** `pytest`
* **Version Control & Cloud Hosting:** Git, Streamlit Community Cloud

---

## 🚀 Quickstart & Setup Instructions

### 1. Clone & Navigate to Repository
```bash
git clone (https://github.com/Vijayalaxmi-ks/Module_Assessment_3_Agro_ML.git)
cd Module_Assessment_3_Agro_ML

1. Activate Virtual Environment - 
.\.venv\Scripts\Activate.ps1

2. Install Required Dependencies - 
pip install -r requirements.txt

3. Launch the Interactive Dashboard -
streamlit run app/app.py

4. Run Automated Test Suite - 
pytest -v -W ignore::DeprecationWarning
```

## 📊 Core Features & Key Innovations

⚡ 1-Click Field Diagnostic Presets: Instant loading of real-world scenarios including Optimal Field, Heat & Drought Stress, Severe Pest Outbreak, and Nitrogen Deficiency.

🌡️ Real-Time Transpiration Driver ($\text{VPD}$): Calculates atmospheric Vapor Pressure Deficit dynamically from ambient temperature and humidity telemetry.

🎨 Theme-Adaptive Visual Analytics: Custom-styled Plotly gauge visualizers and CSS metric blocks engineered for seamless readability across Light and Dark UI modes.

🧪 Automated Testing Pipeline: Integrated pytest verification covering critical domain formulas (VPD, NPK total, pH divergence) and serialized model artifact inference.

🎯 Actionable Decision Support: Direct mapping of risk probabilities ($P \ge 0.50$) to practical field interventions such as fertigation schedules and biopesticide deployment.