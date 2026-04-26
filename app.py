import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Mining AI Predictive Maintenance", layout="wide")
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.header-box {
    background: linear-gradient(90deg, #003366, #0059b3);
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.subheader-box {
    background: linear-gradient(90deg, #0059b3, #3399ff);
    padding: 10px;
    border-radius: 8px;
    color: white;
    font-size: 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():
    model = pickle.load(open("failure_model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    try:
        rul_model = pickle.load(open("rul_model.pkl", "rb"))
    except:
        rul_model = None
    return model, encoders, rul_model

model, encoders, rul_model = load_models()

# ================================
# FUNCTIONS
# ================================
def encode_data(df):
    for col in ["Asset_Type","Mine_Site","Shift"]:
        mapping = dict(zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_)))
        df[col] = df[col].map(mapping).fillna(-1)
    return df

def add_missing_features(df):
    df["Temp_7D_Avg"] = df["Temperature"]
    df["Vibration_7D_Avg"] = df["Vibration"]
    df["Temp_Trend"] = 0
    df["Vibration_Trend"] = 0
    df["Prev_Temperature"] = df["Temperature"]
    df["Prev_Vibration"] = df["Vibration"]
    return df

def align_features(df):
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    return df[model_features]

def get_failure_mode(row):
    if row["Temperature"] > 85:
        return "Overheating"
    elif row["Vibration"] > 4:
        return "Mechanical Failure"
    elif row["Pressure"] > 70:
        return "Hydraulic Failure"
    elif row["Dust_Index"] > 1000:
        return "Environmental"
    else:
        return "Normal"

def risk_label(x):
    if x > 0.7:
        return "HIGH"
    elif x > 0.4:
        return "MEDIUM"
    else:
        return "LOW"

# ================================
# UI
# ================================

# ================================
# LOGO HEADER (UPDATED)
# ================================
col1, col2, col3 = st.columns([1.2,2,1.2])

with col1:
    st.markdown("""
    <div style='
        background:#e6f2ff;
        padding:5px 10px;
        border-radius:10px;
        display:inline-block;
    '>
    """, unsafe_allow_html=True)

    st.image("pulse_logo.png", width=160)   # ✅ FIXED

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="header-box">Mining Predictive Maintenance AI System</div>', unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='display:flex; justify-content:flex-end;'>
        <div style='
            background:#e6f2ff;
            padding:5px 10px;
            border-radius:10px;
            display:inline-block;
        '>
    """, unsafe_allow_html=True)

    st.image("digitide_logo.jpg", width=140)   # ✅ FIXED

    st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("<div style='margin-top:-15px;'></div>", unsafe_allow_html=True)

# ================================
# SINGLE PREDICTION
# ================================
st.markdown('<div class="subheader-box">Real-Time Equipment Health & Failure Assessment</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    asset_type = st.selectbox("Asset Type", ["Excavator","Crusher","Conveyor","Pump","Drill"])
    mine_site = st.selectbox("Mine Site", ["Gold","Phosphate","Aluminium"])
    shift = st.selectbox("Shift", ["Day","Night"])
    asset_age = st.slider("Asset Age",1,20,5)
    operator_exp = st.slider("Operator Experience",1,20,5)

with col2:
    temperature = st.slider("Temperature",30,120,70)
    vibration = st.slider("Vibration",0.5,10.0,2.0)
    pressure = st.slider("Pressure",10,100,40)
    load = st.slider("Load %",0,100,70)
    runtime = st.slider("Runtime Hours",0,300,100)

with col3:
    ambient_temp = st.slider("Ambient Temp",20,60,40)
    dust_index = st.slider("Dust Index",100,1500,500)
    humidity = st.slider("Humidity",0,100,30)
    fatigue = st.slider("Fatigue",0.0,1.0,0.5)
    workforce = st.slider("Workforce",0.0,1.0,0.8)

st.subheader("Operational Conditions")

def switch_card(title, desc, key):
    col1, col2 = st.columns([4,2])

    with col1:
        st.markdown(f"**{title}**")
        st.caption(desc)

    with col2:
        value = st.toggle("", key=key)
        st.markdown(
            f"<div style='text-align:center; font-weight:bold; color:#0e6efd;'>"
            f"{'Yes' if value else 'No'}"
            f"</div>",
            unsafe_allow_html=True
        )

    return 1 if value else 0


col4, col5 = st.columns(2)

with col4:
    maintenance = switch_card("Maintenance Done", "Equipment maintenance completed", "maintenance")
    overdue = switch_card("Overdue Maintenance", "Maintenance delay beyond schedule", "overdue")
    sandstorm = switch_card("Sandstorm", "Extreme dusty weather condition", "sandstorm")
    extreme_heat = switch_card("Extreme Heat", "High temperature environmental stress", "extreme_heat")

with col5:
    heavy_rain = switch_card("Heavy Rain", "High rainfall impacting operations", "heavy_rain")
    operator_error = switch_card("Operator Error", "Manual operational mistake occurred", "operator_error")
    safety = switch_card("Safety Incident", "Safety-related event recorded", "safety")


# ================================
# PREDICT
# ================================

st.markdown("""
<style>
div.stButton > button {
    background-color: #003366;
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 25px;
    border-radius: 8px;
    border: none;
}
div.stButton > button:hover {
    background-color: #0059b3;
}
</style>
""", unsafe_allow_html=True)

st.markdown("---")

# ================================
# RESULTS + BATCH SIDE BY SIDE
# ================================
col_left, col_right = st.columns([1,1])

# ---------------- LEFT (RESULTS) ----------------
with col_left:

    if st.button("Run Failure Risk Assessment"):

        df = pd.DataFrame([{
            "Asset_Type": asset_type,
            "Mine_Site": mine_site,
            "Asset_Age": asset_age,
            "Operator_Experience": operator_exp,
            "Health_Index": 0.7,
            "Temperature": temperature,
            "Vibration": vibration,
            "Pressure": pressure,
            "Load_Percentage": load,
            "Runtime_Hours": runtime,
            "Ambient_Temperature": ambient_temp,
            "Dust_Index": dust_index,
            "Humidity": humidity,
            "Shift": shift,
            "Maintenance_Flag": maintenance,
            "Overdue_Maintenance": overdue,
            "Sandstorm_Flag": sandstorm,
            "Extreme_Heat_Flag": extreme_heat,
            "Heavy_Rain_Flag": heavy_rain,
            "Seismic_Flag": 0,
            "Operator_Error_Flag": operator_error,
            "Fatigue_Level": fatigue,
            "Workforce_Availability": workforce,
            "Safety_Incident_Flag": safety
        }])

        df = encode_data(df)
        df = add_missing_features(df)
        df_model = align_features(df)

        prob = model.predict_proba(df_model)[0][1]
        risk = risk_label(prob)
        failure_mode = get_failure_mode(df.iloc[0])
        rul = int(rul_model.predict(df_model)[0]) if rul_model else "N/A"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Failure Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#003366"},
                'steps': [
                    {'range': [0, 40], 'color': '#d6eaf8'},
                    {'range': [40, 70], 'color': '#aed6f1'},
                    {'range': [70, 100], 'color': '#5dade2'}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### Risk: **{risk}**")
        st.markdown(f"### Failure Mode: **{failure_mode}**")
        st.markdown(f"### Remaining Useful Life: **{rul} days**")


# ---------------- RIGHT (BATCH) ----------------
with col_right:

    st.markdown("### Multi-Asset Risk Evaluation")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df_input = df.copy()

        df = encode_data(df)
        df = add_missing_features(df)
        df_model = align_features(df)

        probs = model.predict_proba(df_model)[:,1]

        df_input["Breakdown_Flag"] = (probs > 0.5).astype(int)
        df_input["Risk"] = [risk_label(x) for x in probs]
        df_input["Failure_Prob"] = probs

        if rul_model:
            df_input["RUL"] = rul_model.predict(df_model)

        df_input["Failure_Mode"] = df_input.apply(get_failure_mode, axis=1)

        st.dataframe(df_input, use_container_width=True)

        st.download_button("Download CSV", df_input.to_csv(index=False), "output.csv")

     
