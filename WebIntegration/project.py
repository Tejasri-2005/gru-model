import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="PrognosAI - RUL Prediction & Maintenance Alerts",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main {
    background-color: #0d1117;
    color: #c9d1d9;
    padding-left: 6rem;
    padding-right: 6rem;
}
.big-title {
    font-size: 80px;
    font-weight: 900;
    text-align: center;
    color: #58a6ff;
    margin-bottom: 0px;
    letter-spacing: 2px;
}
.subtitle {
    font-size: 36px;
    text-align: center;
    color: #ffffff;
    margin-top: 10px;
    margin-bottom: 50px;
}
h2, h3 { color: #58a6ff; }

/* Labels */
label, .stSelectbox > div > div > div > div, .stNumberInput > label, .stFileUploader > label {
    color: #ffffff !important;
    font-size: 17px;
}

/* Inputs */
div[data-baseweb="select"] > div {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-radius: 10px;
    padding: 10px;
}
.stNumberInput > div > div > input {
    background-color: #000000 !important;
    color: #ffffff !important;
}

/* Buttons */
.stButton>button {
    background-color: #238636;
    color: white;
    border-radius: 12px;
    width: 100%;
    height: 60px;
    font-size: 20px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #2ea043;
}

/* Expander headers */
.streamlit-expanderHeader {
    background-color: #161b22;
    color: #58a6ff;
    border-radius: 12px;
    padding: 30px;
    font-size: 32px !important;
    font-weight: 900 !important;
    border: 1px solid #30363d;
    text-align: left;
}
.streamlit-expanderHeader:hover {
    background-color: #21262d;
    border-color: #58a6ff;
}
.streamlit-expanderHeader p {
    font-size: 32px !important;
    margin: 0;
}
.streamlit-expanderContent {
    background-color: #161b22;
    border-radius: 0 0 12px 12px;
    padding: 35px;
    border: 1px solid #30363d;
    border-top: none;
}

/* Larger text inside expanders */
.streamlit-expanderContent p, 
.streamlit-expanderContent span, 
.streamlit-expanderContent div {
    font-size: 20px !important;
    color: #c9d1d9 !important;
}

/* Table styling */
.stDataFrame tbody td {
    font-size: 18px !important;
    color: #c9d1d9;
}
.stDataFrame thead th {
    font-size: 18px !important;
    color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("<div class='big-title'>PrognosAI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Remaining Useful Life Prediction & Maintenance Alerts</div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 19px; color: #8b949e; margin: 40px 0;'>
Deep learning-based predictive maintenance for turbofan engines using NASA CMAPSS data.<br>
Predict RUL, identify failing engines early, and get actionable maintenance recommendations.
</div>
""", unsafe_allow_html=True)

# ================= CONFIGURATION =================
st.markdown("---")
st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🔧 Configure Prediction</h2>", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 2.5, 1])
with col_center:
    st.markdown("<p style='color: #ffffff; text-align: center; margin-bottom: 40px; font-size: 18px;'>Select dataset, upload files, and customize thresholds</p>", unsafe_allow_html=True)

    selected_dataset = st.selectbox(
        "Select Dataset (Domain)",
        ["FD001", "FD002", "FD003", "FD004"]
    )

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #ffffff; font-weight: bold; font-size: 18px;'>Upload Data Files</p>", unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        train_file = st.file_uploader("Train File (optional)", type=["txt"])
    with col_f2:
        test_file = st.file_uploader("Test File (required)", type=["txt"])
    with col_f3:
        rul_file = st.file_uploader("RUL File (required)", type=["txt"])

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #ffffff; font-weight: bold; font-size: 18px;'>Maintenance Thresholds</p>", unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        critical_threshold = st.number_input("Critical Threshold (cycles)", value=30, min_value=1)
    with col_t2:
        warning_threshold = st.number_input("Warning Threshold (cycles)", value=60, min_value=1)

    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    run_button = st.button("🚀 RUN PREDICTION", type="primary", use_container_width=True)

st.markdown("---")

# ================= PREDICTION =================
if run_button:
    if not test_file or not rul_file:
        st.error("⚠️ Please upload Test and RUL files.")
    else:
        with st.spinner("Processing and generating predictions..."):
            BASE_DIR = Path.cwd()
            
            test_df = pd.read_csv(test_file, sep="\s+", header=None)
            full_columns = ['unit', 'cycle'] + [f'setting{i}' for i in range(1,4)] + [f'sensor{i}' for i in range(1,22)]
            test_df.columns = full_columns
            true_rul = pd.read_csv(rul_file, header=None)[0].values
            
            model_path = BASE_DIR / f"best_gru_fd{selected_dataset[-3:]}.h5"
            scaler_path = BASE_DIR / f"scaler_fd{selected_dataset[-3:]}.joblib"
            
            if not model_path.exists() or not scaler_path.exists():
                st.error(f"Model/scaler for {selected_dataset} not found.")
                st.stop()
            
            model = tf.keras.models.load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)
            expected_features = scaler.n_features_in_
            feature_columns = test_df.columns[-expected_features:]
            
            seq_length = 50
            
            grouped = test_df.groupby('unit')
            sequences = []
            for _, group in grouped:
                data = group[feature_columns].values
                if len(data) < seq_length:
                    padded = np.pad(data, ((seq_length - len(data), 0), (0, 0)), 'constant')
                    sequences.append(padded)
                else:
                    sequences.append(data[-seq_length:])
            
            X_test = np.array(sequences)
            X_flat = X_test.reshape(-1, expected_features)
            X_scaled = scaler.transform(X_flat).reshape(X_test.shape)
            
            pred_rul = model.predict(X_scaled, verbose=0).flatten()
            pred_rul = np.clip(pred_rul, 0, None)
            
            results = pd.DataFrame({
                'engine_id': range(1, len(pred_rul)+1),
                'predicted_rul': pred_rul.round(2),
                'true_rul': true_rul,
                'error': (true_rul - pred_rul).round(2),
                'abs_error': np.abs(true_rul - pred_rul).round(2)
            })
            results['status'] = np.where(results['predicted_rul'] < critical_threshold, 'Critical',
                                        np.where(results['predicted_rul'] < warning_threshold, 'Warning', 'Healthy'))
            
            def get_action(row):
                if row['status'] == 'Critical':
                    return "Immediate Maintenance / Ground Engine"
                elif row['status'] == 'Warning':
                    return "Schedule Maintenance Soon / Increased Monitoring"
                else:
                    return "Normal Operation"
            results['recommended_action'] = results.apply(get_action, axis=1)
            
            rmse = np.sqrt((results['error']**2).mean())
            mae = results['abs_error'].mean()
            
            st.session_state.results = results
            st.session_state.dataset = selected_dataset
            st.session_state.rmse = rmse
            st.session_state.mae = mae
            st.session_state.features = feature_columns.tolist()

# ================= RESULTS =================
if 'results' in st.session_state:
    results = st.session_state.results
    dataset = st.session_state.dataset
    rmse = st.session_state.rmse
    mae = st.session_state.mae

    st.markdown("<h2 style='text-align: center; margin: 60px 0; color: #58a6ff;'>📊 Prediction Results</h2>", unsafe_allow_html=True)

    # ---------------- EXPANDER 1 ----------------
    with st.expander("📊 1. Dataset Overview & Summary", expanded=True):
        st.markdown(f"<h3 style='color:#ffffff;'>Dataset: {dataset}</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Engines", len(results))
        col2.metric("Critical Engines", len(results[results['status']=='Critical']))
        col3.metric("Warning Engines", len(results[results['status']=='Warning']))
        col4.metric("Healthy Engines", len(results[results['status']=='Healthy']))
        st.markdown(f"<p style='font-size:20px; color:#c9d1d9;'><b>Features Used ({len(st.session_state.features)}):</b> {', '.join(st.session_state.features)}</p>", unsafe_allow_html=True)
        st.dataframe(results[['engine_id','predicted_rul','true_rul','status','recommended_action']].head(15))
        csv = results.to_csv(index=False).encode()
        st.download_button("📥 Download Overview Data", csv, f"{dataset}_overview.csv", use_container_width=True)

    # ---------------- EXPANDER 2 ----------------
    with st.expander("🚀 2. Fleet Health Visualization", expanded=False):
        st.markdown("<p style='font-size:20px; color:#c9d1d9;'>Visualize predicted vs true RUL and fleet health distribution.</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(results, x="true_rul", y="predicted_rul", color="status",
                             color_discrete_map={"Healthy":"#2ea043","Warning":"#ffa500","Critical":"#ff0000"},
                             title="True RUL vs Predicted RUL")
            max_val = max(results['true_rul'].max(), results['predicted_rul'].max()) + 20
            fig.add_trace(go.Scatter(x=[0,max_val], y=[0,max_val], mode="lines",
                                     line=dict(dash="dash", color="#58a6ff", width=3),
                                     name="Perfect Prediction"))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(results, x="predicted_rul", color="status",
                               color_discrete_map={"Healthy":"#2ea043","Warning":"#ffa500","Critical":"#ff0000"},
                               title="Predicted RUL Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # ================= FIXED MATPLOTLIB PLOT (ONLY CHANGE) =================
        st.markdown("<p style='font-size:20px; color:#c9d1d9;'><b>True vs Predicted RUL (Matplotlib)</b></p>", unsafe_allow_html=True)
        plt.figure(figsize=(10,5))
        plt.plot(results['engine_id'], results['true_rul'], label="True RUL")
        plt.plot(results['engine_id'], results['predicted_rul'], label="Predicted RUL")
        plt.legend()
        plt.title(f"True vs Predicted RUL ({dataset})")
        plt.xlabel("Engine ID")
        plt.ylabel("RUL")
        st.pyplot(plt.gcf())
        plt.close()

        csv = results.to_csv(index=False).encode()
        st.download_button("📥 Download Fleet Health Data", csv, f"{dataset}_fleet_health.csv", use_container_width=True)

    # ---------------- EXPANDER 3 ----------------
    with st.expander("⚠️ 3. Maintenance Alerts & Recommended Actions", expanded=False):
        st.markdown("<p style='font-size:20px; color:#c9d1d9;'>Engines requiring maintenance based on predicted RUL.</p>", unsafe_allow_html=True)
        critical = results[results['status']=="Critical"]
        warning = results[results['status']=="Warning"]
        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Critical", len(critical))
        col2.metric("⚠️ Warning", len(warning))
        col3.metric("✅ Healthy", len(results) - len(critical) - len(warning))
        if not critical.empty:
            st.error("🔴 Critical Engines - Immediate Action Required")
            st.dataframe(critical[['engine_id','predicted_rul','recommended_action']].sort_values("predicted_rul"))
        if not warning.empty:
            st.warning("🟡 Warning Engines - Schedule Maintenance Soon")
            st.dataframe(warning[['engine_id','predicted_rul','recommended_action']].sort_values("predicted_rul"))
        if critical.empty and warning.empty:
            st.success("✅ All engines are Healthy - No immediate action required")
        alerts = results[results['status']!="Healthy"]
        csv = alerts.to_csv(index=False).encode()
        st.download_button("📥 Download Alerts with Actions", csv, f"{dataset}_maintenance_alerts.csv", use_container_width=True)

    # ---------------- EXPANDER 4 ----------------
    with st.expander("📊 4. Status Distribution", expanded=False):
        st.markdown("<p style='font-size:20px; color:#c9d1d9;'>Visual representation of fleet status distribution.</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(results, names="status", title="Fleet Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(results['status'].value_counts().reset_index(),
                         x="status", y="count", color="status",
                         title="Engine Count by Status")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- EXPANDER 5 ----------------
    with st.expander("🔍 5. Final Analytics & Performance", expanded=False):
        st.markdown("<p style='font-size:20px; color:#c9d1d9;'>Detailed analytics of prediction results, including descriptive statistics.</p>", unsafe_allow_html=True)
        st.dataframe(results.describe())
        csv_full = results.to_csv(index=False).encode()
        st.download_button("📥 Download Full Results", csv_full, f"{dataset}_full_results.csv", use_container_width=True)

    # ================= FINAL TABLE =================
    st.subheader("📋 True RUL vs Predicted RUL")
    st.dataframe(
        results[['engine_id','true_rul','predicted_rul','status']].rename(
            columns={'engine_id':'Engine_ID','true_rul':'True_RUL',
                     'predicted_rul':'Predicted_RUL','status':'Health_Status'}
        ),
        use_container_width=True
    )

    st.success("✅ Analysis complete! Explore all sections above.")

