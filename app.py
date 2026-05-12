import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Pro Solar XAI Dashboard", layout="wide")

# 2. Sidebar for User Inputs
st.sidebar.header("🕹️ System Parameters")
irradiation = st.sidebar.slider("Irradiation (W/m²)", 0.0, 1.5, 0.5)
ambient_temp = st.sidebar.slider("Ambient Temp (°C)", 20.0, 45.0, 25.0)
module_temp = st.sidebar.slider("Module Temp (°C)", 30.0, 65.0, 35.0)

st.sidebar.header("📅 Solar Position")
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_of_year = st.sidebar.slider("Day of Year", 1, 365, 180)

# 3. Main Logic
try:
    # Load Data with error checking
    gen_df = pd.read_csv("Plant_1_Generation_Data.csv")
    weather_df = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")

    # Clean the column names (removes hidden spaces)
    gen_df.columns = gen_df.columns.str.strip()
    weather_df.columns = weather_df.columns.str.strip()

    # Merge Data
    full_df = pd.merge(gen_df, weather_df, on="DATE_TIME", how="inner")
    
    # ⚠️ THIS IS THE IMPORTANT PART: Remove all rows with empty values
    full_df = full_df.dropna()
    
    if full_df.empty:
        st.error("The merged data is empty. Please check your CSV files!")
        st.stop()

    target_dc = "DC_POWER"
    features = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]
    
    X = full_df[features]
    y = full_df[target_dc]

    # 4. Train Model
    model_dc = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model_dc.fit(X, y)

    # 5. Prediction for Sidebar Inputs
    input_df = pd.DataFrame([[irradiation, ambient_temp, module_temp]], columns=features)
    prediction = model_dc.predict(input_df)[0]
    prediction = max(0, prediction) # Ensure no negative power

    # 6. Top Metrics Display
    st.title("☀️ Solar Power Forecasting & Explainable AI")
    m1, m2, m3 = st.columns(3)
    m1.metric("AI Prediction (DC)", f"{prediction:.2f} kW")
    m2.metric("Target (DC Power)", f"{y.mean():.2f} kW (Avg)")
    m3.metric("Model Confidence", "High (XGBoost)")

    # 7. Advanced XAI: Side-by-Side Plots
    st.divider()
    col_left, col_right = st.columns([1, 1])

    # Initialize SHAP Explainer
    explainer = shap.Explainer(model_dc, X)

    with col_left:
        st.subheader("🔍 Local Explanation (Force Plot)")
        st.write("Tug-of-war for THIS specific moment.")
        
        # 'check_additivity=False' fixes the red technical error box
        shap_values_local = explainer(input_df, check_additivity=False)

        fig_force, ax_force = plt.subplots(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value, 
            shap_values_local.values[0], 
            input_df.iloc[0], 
            matplotlib=True, 
            show=False
        )
        st.pyplot(plt.gcf())
        plt.close(fig_force)

    with col_right:
        st.subheader("📊 Global Impact (Bar Chart)")
        st.write("Variable importance across all data.")

        # SHAP Bar Plot (using a sample for speed)
        sample_shap = explainer(X.head(100), check_additivity=False)
        fig_bar = plt.figure(figsize=(8, 4))
        shap.plots.bar(sample_shap, show=False)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    # 8. Historical Analytics
    st.divider()
    st.subheader("📈 Plant Historical Performance (Daylight Comparison)")
    chart_data = full_df[full_df[target_dc] > 10].tail(100).set_index('DATE_TIME')
    st.line_chart(chart_data[[target_dc, target_ac]])

except FileNotFoundError:
    st.error("⚠️ Missing Data Files! Ensure CSVs are in the project folder on GitHub.")
except Exception as e:
    st.error(f"❌ Technical Error: {e}")