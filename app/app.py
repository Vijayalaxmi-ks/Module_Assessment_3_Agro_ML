import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgroGenAI | Smart Crop Intervention System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light & Dark Theme Polish
st.markdown("""
    <style>
    /* Global Card Wrapper */
    .telemetry-card {
        background-color: var(--background-secondary-color, #ffffff);
        border: 1px solid var(--border-color, #e0e0e0);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }
    
    /* Alert Status Banners */
    .status-alert {
        padding: 16px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .status-danger {
        background-color: #fde8e8;
        color: #9b1c1c;
        border: 1px solid #f8b4b4;
    }
    .status-success {
        background-color: #def7ec;
        color: #03543f;
        border: 1px solid #84e1bc;
    }
    
    /* Custom Metric Styling */
    .metric-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-color, #4a5568);
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: var(--text-color, #1a202c);
    }
    .metric-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }
    .badge-normal { background-color: #e6fffa; color: #234e52; }
    .badge-warning { background-color: #feefc3; color: #744210; }
    .badge-danger { background-color: #fed7d7; color: #742a2a; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Model & Metadata Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_features():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    model_path = os.path.join(project_root, "models", "best_crop_intervention_model.joblib")
    train_features_path = os.path.join(project_root, "data", "processed", "X_train.csv")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model artifact not found at `{model_path}`. Run `train_model.py` first.")
        st.stop()
        
    model = joblib.load(model_path)
    feature_columns = pd.read_csv(train_features_path, nrows=1).columns.tolist()
    return model, feature_columns

model, feature_columns = load_model_and_features()

# -----------------------------------------------------------------------------
# 3. Header & Quick Preset Selector
# -----------------------------------------------------------------------------
st.title("🌾 AgroGenAI: Early Crop Intervention & Risk Analysis")
st.markdown("##### *AI-Powered Precision Agriculture Decision Support System*")

# Scenario Presets for Non-Technical Users
st.markdown("#### ⚡ Quick Field Presets (1-Click Diagnostics)")
preset_cols = st.columns(4)

scenario = None
if preset_cols[0].button("🟢 Optimal Healthy Field", use_container_width=True):
    scenario = "optimal"
if preset_cols[1].button("🟡 High Heat & Drought Stress", use_container_width=True):
    scenario = "drought"
if preset_cols[2].button("🔴 Severe Pest Outbreak", use_container_width=True):
    scenario = "pest"
if preset_cols[3].button("🧪 Nitrogen Deficiency", use_container_width=True):
    scenario = "nitrogen"

# Default Preset Values
defaults = {
    "crop": "Cotton", "stage": "Flowering", "region": "Zone_B_West",
    "temp": 28.5, "humidity": 65.0, "moisture": 45.0,
    "ph": 6.8, "n": 120.0, "p": 45.0, "k": 180.0,
    "ndvi": 0.65, "pest": 20.0
}

if scenario == "optimal":
    defaults.update({"temp": 26.0, "humidity": 70.0, "moisture": 55.0, "n": 150.0, "ndvi": 0.82, "pest": 10.0})
elif scenario == "drought":
    defaults.update({"temp": 39.5, "humidity": 25.0, "moisture": 12.0, "ndvi": 0.32, "pest": 40.0})
elif scenario == "pest":
    defaults.update({"pest": 88.0, "ndvi": 0.41, "moisture": 35.0})
elif scenario == "nitrogen":
    defaults.update({"n": 22.0, "ndvi": 0.38})

st.divider()

# -----------------------------------------------------------------------------
# 4. Sidebar: Telemetry & Input Panel
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Field Telemetry Controls")

crop_type = st.sidebar.selectbox("Crop Type", ["Cotton", "Soybean", "Sugarcane", "Tomato", "Wheat"], index=["Cotton", "Soybean", "Sugarcane", "Tomato", "Wheat"].index(defaults["crop"]))
growth_stage = st.sidebar.selectbox("Growth Stage", ["Vegetative", "Flowering", "Fruiting", "Maturation"], index=["Vegetative", "Flowering", "Fruiting", "Maturation"].index(defaults["stage"]))
region = st.sidebar.selectbox("Region", ["Zone_A_North", "Zone_B_West", "Zone_C_South", "Zone_D_East"], index=["Zone_A_North", "Zone_B_West", "Zone_C_South", "Zone_D_East"].index(defaults["region"]))

st.sidebar.subheader("🌤️ Weather & Soil Moisture")
temp = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, float(defaults["temp"]), 0.5)
humidity = st.sidebar.slider("Humidity (%)", 20.0, 100.0, float(defaults["humidity"]), 1.0)
soil_moisture = st.sidebar.slider("Soil Moisture (%)", 5.0, 80.0, float(defaults["moisture"]), 1.0)

st.sidebar.subheader("🧪 Soil Nutrients (NPK & pH)")
soil_ph = st.sidebar.slider("Soil pH", 4.0, 9.5, float(defaults["ph"]), 0.1)
nitrogen = st.sidebar.number_input("Nitrogen (ppm)", 10.0, 300.0, float(defaults["n"]), 5.0)
phosphorus = st.sidebar.number_input("Phosphorus (ppm)", 5.0, 100.0, float(defaults["p"]), 2.0)
potassium = st.sidebar.number_input("Potassium (ppm)", 20.0, 400.0, float(defaults["k"]), 5.0)

st.sidebar.subheader("📡 Satellite & Health Indices")
ndvi = st.sidebar.slider("NDVI (Crop Canopy Density)", -0.10, 0.95, float(defaults["ndvi"]), 0.01)
pest_pressure = st.sidebar.slider("Pest Pressure Score (0-100)", 0.0, 100.0, float(defaults["pest"]), 1.0)

# -----------------------------------------------------------------------------
# 5. Feature Engineering & Prediction Engine
# -----------------------------------------------------------------------------
def construct_input_data():
    svp = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    avp = svp * (humidity / 100.0)
    vpd_kpa = np.round(svp - avp, 3)
    total_npk = nitrogen + phosphorus + potassium
    ph_dev = np.abs(soil_ph - 7.0)

    data = {
        'avg_temp_c': temp,
        'avg_humidity_pct': humidity,
        'soil_moisture_pct': soil_moisture,
        'soil_ph': soil_ph,
        'nitrogen_ppm': nitrogen,
        'phosphorus_ppm': phosphorus,
        'potassium_ppm': potassium,
        'ndvi_index': ndvi,
        'pest_pressure_score': pest_pressure,
        'vpd_kpa': vpd_kpa,
        'total_npk_ppm': total_npk,
        'ph_deviation': ph_dev
    }

    input_df = pd.DataFrame([data])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    if f"crop_type_{crop_type}" in input_df.columns:
        input_df[f"crop_type_{crop_type}"] = 1
    if f"growth_stage_{growth_stage}" in input_df.columns:
        input_df[f"growth_stage_{growth_stage}"] = 1
    if f"region_{region}" in input_df.columns:
        input_df[f"region_{region}"] = 1

    return input_df[feature_columns], vpd_kpa, total_npk, ph_dev

input_df, vpd_kpa, total_npk, ph_dev = construct_input_data()

# Calculate Prediction Probability
proba = model.predict_proba(input_df)[0][1]
requires_intervention = proba >= 0.50

# -----------------------------------------------------------------------------
# 6. Main Visual Analytics Dashboard
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📈 Real-Time Risk Score Indicator")
    
    # Styled Semi-Arc Gauge Chart
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={'suffix': "%", 'font': {'size': 44, 'family': 'sans-serif'}},
        title={'text': f"Risk Index ({crop_type} - {growth_stage})", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#888888"},
            'bar': {'color': "#e53e3e" if requires_intervention else "#38a169", 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': "rgba(56, 161, 105, 0.18)"},
                {'range': [40, 70], 'color': "rgba(221, 107, 32, 0.18)"},
                {'range': [70, 100], 'color': "rgba(229, 62, 62, 0.18)"}
            ],
            'threshold': {
                'line': {'color': "#e53e3e", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    gauge_fig.update_layout(
        height=260, 
        margin=dict(l=20, r=20, t=30, b=10), 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Action Recommendation Card
    if requires_intervention:
        st.markdown(f"""
        <div class="status-alert status-danger">
            🚨 IMMEDIATE FIELD INTERVENTION RECOMMENDED
            <br><span style="font-size:14px; font-weight:normal;">Calculated intervention probability ({proba:.1%}) exceeds safety threshold (50.0%).</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-alert status-success">
            ✅ FIELD HEALTH OPTIMAL — NO ACTION REQUIRED
            <br><span style="font-size:14px; font-weight:normal;">Calculated intervention probability ({proba:.1%}) is within healthy boundaries.</span>
        </div>
        """, unsafe_allow_html=True)

    # Actionable Advisory Steps
    st.subheader("💡 Actionable Field Guidance")
    if requires_intervention:
        st.markdown("""
        * 🌊 **Irrigation Priority:** Moisture reserves are low relative to temperature ($VPD > 2.0\text{ kPa}$). Initiate drip irrigation.
        * 🐛 **Pest Management:** High pest pressure detected. Apply targeted organic or chemical controls.
        * 🧪 **Nutrient Boost:** Consider nitrogen fertigation if canopy density ($NDVI < 0.50$) is struggling.
        """)
    else:
        st.markdown("""
        * 🟢 Soil moisture and climate conditions are in equilibrium.
        * 🟢 Maintain standard monitoring and irrigation routine.
        """)

with right_col:
    st.subheader("📊 Primary Telemetry Metrics")
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        badge_cls = "badge-danger" if vpd_kpa > 2.2 else "badge-normal"
        badge_txt = "High Stress" if vpd_kpa > 2.2 else "Optimal"
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="metric-title">VPD Transpiration Stress</div>
            <div class="metric-value">{vpd_kpa} <span style="font-size:14px;">kPa</span></div>
            <div class="metric-badge {badge_cls}">{badge_txt}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="telemetry-card">
            <div class="metric-title">Total NPK Nutrients</div>
            <div class="metric-value">{total_npk:.0f} <span style="font-size:14px;">ppm</span></div>
            <div class="metric-badge badge-normal">Balanced Ratio</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        badge_cls2 = "badge-danger" if ndvi < 0.45 else "badge-normal"
        badge_txt2 = "Low Canopy Density" if ndvi < 0.45 else "Healthy Canopy"
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="metric-title">NDVI Vegetation Health</div>
            <div class="metric-value">{ndvi:.2f}</div>
            <div class="metric-badge {badge_cls2}">{badge_txt2}</div>
        </div>
        """, unsafe_allow_html=True)

        badge_cls3 = "badge-danger" if pest_pressure > 50 else "badge-normal"
        badge_txt3 = "Action Needed" if pest_pressure > 50 else "Controlled"
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="metric-title">Pest Pressure Score</div>
            <div class="metric-value">{pest_pressure:.0f} <span style="font-size:14px;">/ 100</span></div>
            <div class="metric-badge {badge_cls3}">{badge_txt3}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # Feature Input Breakdown Bar Chart
    st.subheader("🔍 Key Risk Drivers")
    features_impact = pd.DataFrame({
        'Parameter': ['Pest Pressure', 'Soil Moisture', 'NDVI Index', 'Temperature', 'Nitrogen'],
        'Value': [pest_pressure, soil_moisture, ndvi * 100, temp, nitrogen]
    })
    
    bar_fig = px.bar(
        features_impact, 
        x='Value', 
        y='Parameter', 
        orientation='h',
        color='Value',
        color_continuous_scale="Viridis"
    )
    bar_fig.update_layout(
        height=220, 
        margin=dict(l=10, r=10, t=10, b=10), 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(bar_fig, use_container_width=True)

st.divider()

st.caption("🤖 AgroGenAI Version 2.0 — Developed for Crop Health Intervention and Decision Support.")

import streamlit as st
import streamlit.components.v1 as components

# ==========================================
# INTERACTIVE CROP-SPECIFIC STRESS SIMULATOR
# ==========================================
st.markdown("---")
st.header("🎮 Interactive Crop-Specific Stress Simulator")
st.caption("Simulate real-time environmental impact on specific crops grown on Bhumi.")

# 1. Fetch Crop Type from main controls or provide selectbox fallback
# (Reuses the main crop selection variable if defined earlier in your script)
if 'crop_type' in locals():
    selected_crop = crop_type
elif 'crop_choice' in locals():
    selected_crop = crop_choice
else:
    selected_crop = st.selectbox(
        "Select Crop Type for Simulation", 
        ["Cotton", "Soybean", "Sugarcane", "Tomato", "Wheat"]
    )

col1, col2 = st.columns(2)
with col1:
    temp_sim = st.slider("Simulated Temperature (°C)", 15, 48, 38)
    rain_days = st.slider("Days Without Rain", 0, 30, 14)
with col2:
    moisture_sim = st.slider("Soil Moisture Level (%)", 5, 80, 15)

# 2. Stress Index Calculation
stress_index = max(0, min(100, int((temp_sim * 1.3) + (rain_days * 2.2) - (moisture_sim * 0.9))))

st.subheader(f"Simulated Stress Index for {selected_crop}: {stress_index} / 100")

# 3. Dynamic Interactive HTML5 Canvas Code
canvas_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ margin: 0; background-color: #0e1117; display: flex; justify-content: center; align-items: center; font-family: sans-serif; }}
        canvas {{ background: linear-gradient(to bottom, #1a1c24, #0e1117); border-radius: 12px; border: 1px solid #2d313e; }}
    </style>
</head>
<body>
    <canvas id="cropCanvas" width="600" height="300"></canvas>
    <script>
        const canvas = document.getElementById('cropCanvas');
        const ctx = canvas.getContext('2d');
        const crop = "{selected_crop}";
        const stress = {stress_index};

        // Calculate dynamic properties based on stress slider values
        let leafColor, stemColor, plantHeight, droopAngle, statusText;

        if (stress < 35) {{
            leafColor = '#4CAF50';  // Healthy Green
            stemColor = '#2E7D32';
            plantHeight = 160;
            droopAngle = 0;
            statusText = "Lush & Healthy Growth";
        }} else if (stress < 70) {{
            leafColor = '#D4AC0D';  // Yellowing / Stressed
            stemColor = '#8BC34A';
            plantHeight = 120;
            droopAngle = 0.5;
            statusText = "Moderate Stress (Wilting & Growth Stunted)";
        }} else {{
            leafColor = '#795548';  // Dying Brown
            stemColor = '#4E342E';
            plantHeight = 75;       // Shrinks under severe stress
            droopAngle = 1.2;       // Strong downward bend
            statusText = "Critical Drought Wilting & Failure Risk";
        }}

        let wind = 0;

        function drawScene() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw Bhumi / Soil
            ctx.fillStyle = '#3E2723';
            ctx.fillRect(0, canvas.height - 30, canvas.width, 30);
            ctx.fillStyle = '#4E342E';
            ctx.fillRect(0, canvas.height - 30, canvas.width, 4);

            const cx = canvas.width / 2;
            const cy = canvas.height - 30;
            const sway = stress < 70 ? Math.sin(wind) * 4 : 0;
            wind += 0.05;

            // Render specific crop visuals dynamically
            if (crop === "Wheat") {{
                drawWheat(cx, cy, sway);
            }} else if (crop === "Sugarcane") {{
                drawSugarcane(cx, cy, sway);
            }} else if (crop === "Tomato") {{
                drawTomato(cx, cy, sway);
            }} else if (crop === "Cotton") {{
                drawCotton(cx, cy, sway);
            }} else {{
                drawSoybean(cx, cy, sway);
            }}

            // Overlay Text
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '14px sans-serif';
            ctx.fillText("🌾 " + crop + " Status: " + statusText, 20, 30);

            requestAnimationFrame(drawScene);
        }}

        // COTTON DRAWING
        function drawCotton(cx, cy, sway) {{
            let topY = cy - plantHeight;
            let bendX = cx + (droopAngle * 25) + sway;

            // Main Stem
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.quadraticCurveTo(cx + sway, cy - (plantHeight * 0.5), bendX, topY);
            ctx.strokeStyle = stemColor;
            ctx.lineWidth = 8;
            ctx.stroke();

            // Dynamic Cotton Bolls / Leaves attached to stem
            const nodes = [0.4, 0.7, 1.0];
            nodes.forEach((t, i) => {{
                let nx = cx + (bendX - cx) * t;
                let ny = cy - (plantHeight * t);
                let side = i % 2 === 0 ? 1 : -1;

                // Leaves
                ctx.beginPath();
                ctx.ellipse(nx + (side * 20), ny + (droopAngle * 15), 14, 8, (side * 0.5) + droopAngle, 0, Math.PI * 2);
                ctx.fillStyle = leafColor;
                ctx.fill();

                // Cotton Puff (shrinks & turns dark when dying)
                ctx.beginPath();
                let puffSize = stress > 69 ? 6 : 11;
                ctx.arc(nx + (side * 25), ny + (droopAngle * 20), puffSize, 0, Math.PI * 2);
                ctx.fillStyle = stress > 69 ? '#B0BEC5' : '#FFFFFF';
                ctx.fill();
            }});
        }}

        // SOYBEAN DRAWING
        function drawSoybean(cx, cy, sway) {{
            let topY = cy - plantHeight;
            let bendX = cx + (droopAngle * 20) + sway;

            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.quadraticCurveTo(cx + sway, cy - (plantHeight * 0.5), bendX, topY);
            ctx.strokeStyle = stemColor;
            ctx.lineWidth = 7;
            ctx.stroke();

            // Trifoliate Leaf Clusters
            [0.3, 0.6, 0.9].forEach((t) => {{
                let nx = cx + (bendX - cx) * t;
                let ny = cy - (plantHeight * t);

                ctx.beginPath();
                ctx.arc(nx - 20, ny + (droopAngle * 15), 10, 0, Math.PI * 2);
                ctx.arc(nx + 20, ny + (droopAngle * 15), 10, 0, Math.PI * 2);
                ctx.arc(nx, ny - 10 + (droopAngle * 15), 10, 0, Math.PI * 2);
                ctx.fillStyle = leafColor;
                ctx.fill();
            }});
        }}

        // SUGARCANE DRAWING
        function drawSugarcane(cx, cy, sway) {{
            let topY = cy - (plantHeight * 1.3);
            let bendX = cx + (droopAngle * 35) + sway;

            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(bendX, topY);
            ctx.strokeStyle = stemColor;
            ctx.lineWidth = 12;
            ctx.stroke();

            // Long Arching Leaves
            [0.3, 0.5, 0.7, 0.9].forEach((t) => {{
                let ly = cy - (plantHeight * 1.3 * t);
                let lx = cx + (bendX - cx) * t;

                ctx.beginPath();
                ctx.moveTo(lx, ly);
                ctx.quadraticCurveTo(lx + 50, ly + (droopAngle * 40), lx + 70, ly + 20 + (droopAngle * 50));
                ctx.moveTo(lx, ly);
                ctx.quadraticCurveTo(lx - 50, ly + (droopAngle * 40), lx - 70, ly + 20 + (droopAngle * 50));
                ctx.strokeStyle = leafColor;
                ctx.lineWidth = 4;
                ctx.stroke();
            }});
        }}

        // TOMATO DRAWING
        function drawTomato(cx, cy, sway) {{
            let topY = cy - plantHeight;
            let bendX = cx + (droopAngle * 20) + sway;

            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.quadraticCurveTo(cx + sway, cy - (plantHeight * 0.5), bendX, topY);
            ctx.strokeStyle = stemColor;
            ctx.lineWidth = 8;
            ctx.stroke();

            // Branches and Tomato Fruit
            [0.4, 0.7, 0.95].forEach((t, i) => {{
                let nx = cx + (bendX - cx) * t;
                let ny = cy - (plantHeight * t);
                let side = i % 2 === 0 ? 1 : -1;

                // Leaves
                ctx.beginPath();
                ctx.arc(nx + (side * 22), ny + (droopAngle * 12), 12, 0, Math.PI * 2);
                ctx.fillStyle = leafColor;
                ctx.fill();

                // Tomatoes (Wrinkle/Shrivel if high stress)
                ctx.beginPath();
                let r = stress > 69 ? 6 : 10;
                ctx.arc(nx + (side * 15), ny + 15 + (droopAngle * 20), r, 0, Math.PI * 2);
                ctx.fillStyle = stress > 69 ? '#5D4037' : '#E53935';
                ctx.fill();
            }});
        }}

        // WHEAT DRAWING
        function drawWheat(cx, cy, sway) {{
            for (let i = -2; i <= 2; i++) {{
                let x = cx + (i * 15);
                let topY = cy - plantHeight + (Math.abs(i) * 10);
                let bendX = x + (i * 8) + (droopAngle * 20) + sway;

                ctx.beginPath();
                ctx.moveTo(x, cy);
                ctx.quadraticCurveTo(x + sway, cy - (plantHeight * 0.5), bendX, topY);
                ctx.strokeStyle = stemColor;
                ctx.lineWidth = 3;
                ctx.stroke();

                // Grain Head Spike
                ctx.beginPath();
                ctx.ellipse(bendX, topY, 6, 14, (i * 0.2) + droopAngle, 0, Math.PI * 2);
                ctx.fillStyle = stress > 69 ? '#5D4037' : (stress > 34 ? '#D4AC0D' : '#F4D03F');
                ctx.fill();
            }}
        }}

        drawScene();
    </script>
</body>
</html>
"""

components.html(canvas_html, height=320)

# 4. Status Description
if stress_index < 35:
    st.success(f"🟢 **Status: Optimal Growth for {selected_crop}**")
    st.markdown(f"""
    * **Plant State:** Stomata open, active photosynthesis.
    * **Visual Impact:** Deep green canopy, healthy root uptake on Bhumi.
    * **Action Needed:** None. Normal irrigation schedule.
    """)
elif stress_index < 70:
    st.warning(f"🟡 **Status: Moderate Drought Stress for {selected_crop}**")
    st.markdown(f"""
    * **Plant State:** Stomata closing to conserve moisture.
    * **Visual Impact:** Growth stunted, yellowing leaves (NDVI drop).
    * **Action Needed:** Initiate micro-irrigation within 48 hours.
    """)
else:
    st.error(f"🔴 **Status: Critical Wilting Risk for {selected_crop}**")
    st.markdown(f"""
    * **Plant State:** Loss of turgor pressure, high yield damage.
    * **Visual Impact:** Severe plant shrinkage, browning, and drooping on Bhumi.
    * **Action Needed:** Immediate emergency watering and soil protection.
    """)