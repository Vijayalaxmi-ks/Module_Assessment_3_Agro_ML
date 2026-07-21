# 🌾 AgroGenAI: Early Crop Intervention & Decision Support System
## **Engineering & Technical Implementation Report**

---

## **Executive Summary**

**AgroGenAI** is an end-to-end, machine-learning-powered decision support system designed to predict crop stress and recommend early agricultural interventions within a critical 7-day window. By fusing soil chemistry parameters ($N, P, K, pH$), micro-climate telemetry (Temperature, Relative Humidity), satellite-derived crop canopy density ($\text{NDVI}$), and biological stress factors (Pest Pressure Score), the system calculates real-time risk probabilities and provides actionable field guidance for agronomists and farmers.

---

## **1. System Architecture & Tech Stack**

### **1.1 End-to-End System Data Flow**

| Stage | Subsystem | Inputs / Process | Primary Output |
| :--- | :--- | :--- | :--- |
| **Stage 1** | **Telemetry Data Ingestion** | Soil NPK & pH, Ambient Temp & Humidity, NDVI Index, Pest Score | Raw Sensor Feature Vector |
| **Stage 2** | **Feature Engineering Engine** | Real-time Vapor Pressure Deficit ($\text{VPD}$) calculation, Total NPK aggregation, pH deviation, Categorical One-Hot Encoding | Processed Model Feature Matrix |
| **Stage 3** | **ML Inference Engine** | Trained Random Forest Classifier (`joblib` pipeline) evaluating feature matrix | Probability Score ($P \ge 0.50$) |
| **Stage 4** | **Decision & Advisory Dashboard** | Streamlit UI, Plotly gauge visualizers, Dynamic field advice generator | Real-time Risk Score & Intervention Guidance |

---

### **1.2 Technology Stack**
* **Language:** Python 3.10+
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `joblib`
* **Visualization & UI:** `streamlit`, `plotly`
* **Development & Deployment:** Git, Streamlit Community Cloud

---

## **2. Dataset Architecture & Feature Engineering**

### **2.1 Raw Telemetry Parameters**

| Parameter | Symbol / Variable | Unit | Range / Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Crop Type** | `crop_type` | Categorical | Cotton, Soybean, Sugarcane, Tomato, Wheat | Target cultivation species |
| **Growth Stage** | `growth_stage` | Categorical | Vegetative, Flowering, Fruiting, Maturation | Morphological crop development phase |
| **Region** | `region` | Categorical | Zone_A_North, Zone_B_West, Zone_C_South, Zone_D_East | Geographical agro-climatic zone |
| **Temperature** | `avg_temp_c` | °C | 10.0 – 45.0 | Ambient field temperature |
| **Humidity** | `avg_humidity_pct` | % | 20.0 – 100.0 | Ambient relative humidity |
| **Soil Moisture** | `soil_moisture_pct`| % | 5.0 – 80.0 | Volumetric soil water content |
| **Soil pH** | `soil_ph` | pH scale | 4.0 – 9.5 | Soil acidity / alkalinity balance |
| **Nitrogen** | `nitrogen_ppm` | ppm | 10.0 – 300.0 | Available soil Nitrogen concentration |
| **Phosphorus** | `phosphorus_ppm` | ppm | 5.0 – 100.0 | Available soil Phosphorus concentration |
| **Potassium** | `potassium_ppm` | ppm | 20.0 – 400.0 | Available soil Potassium concentration |
| **NDVI Index** | `ndvi_index` | Index | -0.10 – 0.95 | Normalized Difference Vegetation Index |
| **Pest Pressure** | `pest_pressure_score`| Score | 0.0 – 100.0 | Quantified pest infestation risk level |

---

### **2.2 Engineered Domain Features**

#### **1. Vapor Pressure Deficit ($\text{VPD}$)**
VPD measures the drying power of the atmosphere and directly drives plant transpiration:

$$\text{SVP} = 0.61078 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right)$$

$$\text{AVP} = \text{SVP} \times \left(\frac{RH}{100}\right)$$

$$\text{VPD} = \text{SVP} - \text{AVP}$$

*where $T$ is temperature in °C and $RH$ is relative humidity in %.*

#### **2. Total Macro-Nutrients ($\text{Total NPK}$)**

$$\text{Total NPK} = \text{Nitrogen}_{\text{ppm}} + \text{Phosphorus}_{\text{ppm}} + \text{Potassium}_{\text{ppm}}$$

#### **3. Soil pH Divergence ($\Delta \text{pH}$)**

$$\Delta \text{pH} = |\text{Soil pH} - 7.0|$$

---

## **3. Model Training & Evaluation Pipeline**

### **3.1 Data Preparation & Preprocessing**
1. **Handling Categorical Encoding:** One-Hot Encoding applied to categorical variables (`crop_type`, `growth_stage`, `region`).
2. **Train / Test Partitioning:** 80/20 train-test split with stratified sampling to preserve target class balance ($y = 1$: Intervention Required, $y = 0$: Optimal Field Conditions).

### **3.2 Model Performance Comparison**

| Classifier Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **94.2%** | **93.8%** | **95.1%** | **0.944** | **0.982** |
| **XGBoost Classifier** | 93.6% | 93.1% | 94.5% | 0.938 | 0.978 |
| **Gradient Boosting** | 92.1% | 91.5% | 93.0% | 0.922 | 0.969 |
| **Logistic Regression** | 81.4% | 80.2% | 82.7% | 0.814 | 0.887 |

> **Selected Model:** **Random Forest Classifier** was selected as the core inference engine due to its superior ROC-AUC score and strong generalization across non-linear soil-climate interactions.

---

## **4. Dashboard Architecture & User Experience**

The Streamlit web application (`app/app.py`) provides an intuitive dashboard tailored for both technical researchers and non-technical agricultural extension workers.

### **Key UI Features**
1. **1-Click Diagnostic Presets:**
   * 🟢 **Optimal Healthy Field:** Resets telemetry to baseline healthy conditions.
   * 🟡 **High Heat & Drought Stress:** Simulates high temperature ($39.5^\circ\text{C}$), low humidity ($25\%$), and low soil moisture ($12\%$).
   * 🔴 **Severe Pest Outbreak:** Simulates elevated pest pressure ($88/100$) and compromised vegetation ($NDVI = 0.41$).
   * 🧪 **Nitrogen Deficiency:** Simulates severely depleted soil Nitrogen ($22\text{ ppm}$).
2. **Theme-Adaptive Visual Analytics:**
   * Fully transparent Plotly gauge charts and high-contrast CSS metrics that dynamically render cleanly in both Light and Dark application themes.
3. **Actionable Agricultural Guidance:**
   * Automatically translates risk probability scores into practical field steps (e.g., drip irrigation schedules, bio-pesticide deployment, fertigation).

---

## **5. Deployment Guide**

### **5.1 Local Execution Steps**

1. Navigate to the project root directory: `cd AgroGenAI`
2. Activate your virtual environment: `.\.venv\Scripts\Activate.ps1`
3. Install required packages: `pip install -r requirements.txt`
4. Launch the application: `streamlit run app/app.py`

### **5.2 Cloud Deployment (Streamlit Community Cloud)**

1. Commit and push all project files (`app/app.py`, `models/`, `data/processed/X_train.csv`, `requirements.txt`) to a GitHub repository.
2. Sign in to Streamlit Community Cloud (https://share.streamlit.io).
3. Click "New App", connect your GitHub repository, and select the repository branch.
4. Set the Main file path to `app/app.py`.
5. Select "Deploy!" to launch the live application instance.

---

## **6. Future Roadmap**

* **📡 IoT Telemetry Integration:** Direct integration with ESP32 soil moisture sensors and MQTT weather stations for real-time live data streaming.
* **🖼️ Multi-Modal Crop Disease Vision:** Fusing tabular sensor data with deep learning image classification (CNN / Generative AI) for leaf disease identification.
* **🌐 Local Language Localization:** Multilingual support (Hindi, Marathi, Kannada) to enhance accessibility for regional farming communities.

---
*AgroGenAI Project Documentation — Version 2.0*