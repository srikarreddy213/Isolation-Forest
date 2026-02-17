import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Air Quality Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Model Info"])

st.sidebar.markdown("---")
st.sidebar.info("Isolation Forest Model\nContamination: 1%")

# =========================
# HEADER
# =========================
st.title("🔍 Air Quality Anomaly Detection System")
st.markdown("Detect abnormal pollution levels using Machine Learning.")
st.markdown("---")

# =========================
# LOAD MODEL
# =========================
model_path = "isolation_forest_model.pkl"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()


# =========================
# SINGLE PREDICTION
# =========================
if page == "Single Prediction":

    st.subheader("🔮 Single Data Point Prediction")

    col1, col2 = st.columns(2)

    with col1:
        co_gt = st.number_input("CO(GT)", 0.0, 50.0, 2.0)
        nox_gt = st.number_input("NOx(GT)", 0.0, 500.0, 150.0)

    with col2:
        c6h6_gt = st.number_input("C6H6(GT)", 0.0, 100.0, 10.0)
        no2_gt = st.number_input("NO2(GT)", 0.0, 500.0, 100.0)

    if st.button("Predict"):

        input_data = pd.DataFrame({
            'CO(GT)': [co_gt],
            'C6H6(GT)': [c6h6_gt],
            'NOx(GT)': [nox_gt],
            'NO2(GT)': [no2_gt]
        })

        prediction = model.predict(input_data)[0]
        score = model.score_samples(input_data)[0]

        st.markdown("---")

        colA, colB = st.columns(2)

        if prediction == -1:
            colA.error("🚨 ANOMALY DETECTED")
        else:
            colA.success("✅ NORMAL")

        colB.metric("Anomaly Score", f"{score:.4f}")

        st.markdown("### Input Data")
        st.dataframe(input_data)


# =========================
# BATCH PREDICTION
# =========================
elif page == "Batch Prediction":

    st.subheader("📂 Upload CSV for Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV (CO(GT), C6H6(GT), NOx(GT), NO2(GT))",
        type="csv"
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        required_columns = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain required columns.")
        else:

            df_filtered = df[required_columns]

            predictions = model.predict(df_filtered)
            scores = model.score_samples(df_filtered)

            df_filtered["Prediction"] = predictions
            df_filtered["Score"] = scores

            normal_count = (predictions == 1).sum()
            anomaly_count = (predictions == -1).sum()

            st.markdown("### 📊 Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", len(df_filtered))
            col2.metric("Normal", normal_count)
            col3.metric("Anomalies", anomaly_count)

            st.markdown("---")
            st.dataframe(df_filtered, use_container_width=True)

            # Visualization
            st.markdown("### 📈 Anomaly Score Distribution")

            fig, ax = plt.subplots()
            ax.hist(scores[predictions == 1], bins=30, alpha=0.6, label="Normal")
            ax.hist(scores[predictions == -1], bins=30, alpha=0.6, label="Anomaly")
            ax.legend()
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Frequency")

            st.pyplot(fig)


# =========================
# MODEL INFO
# =========================
else:

    st.subheader("📘 Model Information")

    st.write("**Model Type:** Isolation Forest")
    st.write("**Purpose:** Detect anomalies in air quality data")

    params = model.get_params()

    st.markdown("### Parameters")
    for key, value in params.items():
        st.write(f"- {key}: {value}")

    st.markdown("### Features Used")
    st.write("- CO(GT)")
    st.write("- C6H6(GT)")
    st.write("- NOx(GT)")
    st.write("- NO2(GT)")

    st.markdown("---")
    st.info("""
    Prediction = 1 → Normal  
    Prediction = -1 → Anomaly  
    Lower score = More abnormal
    """)
