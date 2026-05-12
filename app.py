import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Pro Solar XAI Dashboard", layout="wide")

st.title("☀️ Pro-Grade Solar Forecasting & XAI Dashboard")
st.write("Full-chain modeling: Astronomical Position ➔ Weather ➔ DC Power ➔ AC Conversion.")

# 2. Advanced Data Loader
@st.cache_data
def load_data():
    # Load Plant_1 files
    gen = pd.read_csv('Plant_1_Generation_Data.csv')
    weather = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
    
    # Standardize column names to UPPERCASE
    gen.columns = gen.columns.str.upper()
    weather.columns = weather.columns.str.upper()
    
    # Convert to datetime objects
    gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'])
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'])
    
    # Merge datasets on time
    df = pd.merge(gen, weather, on="DATE_TIME", how="inner")
    
    # Engineering Features: Extract Time Geometry
    df['HOUR'] = df['DATE_TIME'].dt.hour
    df['DAY_OF_YEAR'] = df['DATE_TIME'].dt.dayofyear
    
    # Model columns
    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'HOUR', 'DAY_OF_YEAR']
    target_dc = 'DC_POWER'
    target_ac = 'AC_POWER'
    
    return df, features, target_dc, target_ac

# 3. Main Execution Block
try:
    full_df, feature_cols, target_dc, target_ac = load_data()
    X = full_df[feature_cols]
    y_dc = full_df[target_dc]
    y_ac = full_df[target_ac]

    # Model A: Weather & Position to DC Power
    model_dc = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model_dc.fit(X, y_dc)

    # Model B: DC Power to AC Power (Inverter Physics)
    model_ac = xgb.XGBRegressor(n_estimators=100, max_depth=3)
    model_ac.fit(full_df[[target_dc]], y_ac)

    # 4. Sidebar Controls
    st.sidebar.header("🕹️ System Parameters")
    irrad = st.sidebar.slider("Irradiation (W/m²)", 0.0, float(X['IRRADIATION'].max()), 0.5)
    amb_temp = st.sidebar.slider("Ambient Temp (°C)", 0.0, 50.0, 25.0)
    mod_temp = st.sidebar.slider("Module Temp (°C)", 0.0, 70.0, 35.0)
    
    st.sidebar.header("📅 Solar Position")
    hour_input = st.sidebar.slider("Hour of Day", 0, 23, 12)
    day_input = st.sidebar.slider("Day of Year", 1, 365, 180)

    # 5. Multistage Prediction Logic
    input_df = pd.DataFrame([[amb_temp, mod_temp, irrad, hour_input, day_input]], columns=feature_cols)
    
    pred_dc = max(0, model_dc.predict(input_df)[0])
    pred_ac = max(0, model_ac.predict(pd.DataFrame([[pred_dc]], columns=[target_dc]))[0])

    # 6. Performance Metrics (KPIs)
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Forecasted DC Power", f"{pred_dc:,.2f} kW")
    with m2:
        st.metric("Forecasted AC Power", f"{pred_ac:,.2f} kW")
    with m3:
        efficiency = (pred_ac / pred_dc * 100) if pred_dc > 10 else 0
        st.metric("Inverter Efficiency", f"{efficiency:.2f}%")
    with m4:
        load_factor = (pred_ac / 10000) * 100
        st.metric("System Load Factor", f"{load_factor:.1f}%")

        # 6.5 Anomaly Detection (Health Monitoring)
    st.divider()
    st.subheader("🛠️ System Health & Anomaly Detection")
    
    # We find a real data point from the CSV that matches the current slider 'HOUR'
    # to compare our "Ideal AI Prediction" vs "Real Recorded Data"
    actual_data_sample = full_df[full_df['HOUR'] == hour_input].head(1)
    
    if not actual_data_sample.empty:
        actual_dc = actual_data_sample[target_dc].values[0]
        
        # Calculate Deviation
        deviation = abs(pred_dc - actual_dc)
        health_score = max(0, 100 - (deviation / 10000 * 100)) # Normalized to 10MW scale
        
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.write(f"**Ideal AI Expectation:** {pred_dc:,.2f} kW")
            st.write(f"**Real Sensor Reading (Sample):** {actual_dc:,.2f} kW")
            
            if health_score > 90:
                st.success(f"System Health: {health_score:.1f}% - Operating Normally")
            elif health_score > 70:
                st.warning(f"System Health: {health_score:.1f}% - Potential Soiling/Dust Detected")
            else:
                st.error(f"System Health: {health_score:.1f}% - Critical Fault: Check Inverter/Panels")
        
        with h_col2:
            # Simple bar chart showing the gap
            chart_compare = pd.DataFrame({
                "Category": ["AI Prediction", "Actual Sensor"],
                "Power (kW)": [pred_dc, actual_dc]
            })
            st.bar_chart(chart_compare.set_index("Category"))

    # 7. Advanced XAI: Force Plot & Bar Chart
    st.divider()
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🔍 Local Explanation (Force Plot)")
        st.write("Tug-of-war for THIS specific moment.")
        
        explainer = shap.Explainer(model_dc, X)
        shap_values = explainer(input_df)
        
        # Use Matplotlib backend for the Force Plot to ensure it renders in Streamlit
        fig_force, ax_force = plt.subplots(figsize=(10, 3))
        shap.force_plot(explainer.expected_value, shap_values.values[0], input_df.iloc[0], matplotlib=True, show=False)
        st.pyplot(plt.gcf())
        plt.close(fig_force)

    with copyright:
        st.subheader("📊 Global Impact (Bar Chart)")
        st.write("Variable importance across all data.")
        
        fig_bar = plt.figure(figsize=(8, 4))
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    # 8. Historical Analytics
    st.divider()
    st.subheader("📈 Plant Historical Performance (Daylight Comparison)")
    chart_data = full_df[full_df[target_dc] > 10].tail(100).set_index('DATE_TIME')
    st.line_chart(chart_data[[target_dc, target_ac]])

except FileNotFoundError:
    st.error("Missing Data Files! Ensure CSVs are in the project folder.")
except Exception as e:
    st.error(f"Technical Error: {e}")