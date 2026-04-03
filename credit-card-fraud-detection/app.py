import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os


st.set_page_config(page_title="Fraud Detector AI", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_assets():
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

@st.cache_data
def load_data():
    data_path = "data/creditcard.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    return None

model, scaler = load_assets()
df = load_data()


st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    .fraud-alert { color: #ff4b4b; font-weight: bold; font-size: 24px; }
    .legit-alert { color: #00fa9a; font-weight: bold; font-size: 24px; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ AI Credit Card Fraud Detection System")
st.markdown("An advanced machine learning pipeline (XGBoost) actively monitoring credit card transactions for fraudulent activity.")

if model is None:
    st.error("⚠️ Pipeline Error: 'best_model.pkl' not found.")
    st.info("Checklist:\n1. Open your terminal.\n2. Navigate to `notebooks/`.\n3. Run `python 02_train.py` to train and save the model.")
    st.stop()


st.sidebar.header("⚙️ Simulation Settings")

sim_mode = st.sidebar.radio("Select Input Mode:", ["Random Legit Transaction", "Random Fraud Transaction", "Manual Entry"])

input_data = None
true_label = None

if sim_mode == "Manual Entry":
    st.sidebar.markdown("### Transaction Details")
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
    time_elapsed = st.sidebar.number_input("Time Elapsed (seconds)", min_value=0.0, value=3600.0)
    
    v1 = st.sidebar.slider("V1 (PCA Feature)", -5.0, 5.0, 0.0)
    v2 = st.sidebar.slider("V2 (PCA Feature)", -5.0, 5.0, 0.0)
    v3 = st.sidebar.slider("V3 (PCA Feature)", -5.0, 5.0, 0.0)
    v4 = st.sidebar.slider("V4 (PCA Feature)", -5.0, 5.0, 0.0)
    
    
    v_features = [0.0] * 28
    v_features[0], v_features[1], v_features[2], v_features[3] = v1, v2, v3, v4
    
    
    scaled_amt = scaler.transform([[amount]])[0][0]
    scaled_time = scaler.transform([[time_elapsed]])[0][0]
    
    input_data = pd.DataFrame([[scaled_time] + v_features + [scaled_amt]], 
                              columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])
    true_label = "Unknown"

elif df is not None:
    
    if sim_mode == "Random Legit Transaction":
        sample = df[df['Class'] == 0].sample(1)
        true_label = 0
    else:
        sample = df[df['Class'] == 1].sample(1)
        true_label = 1
        
    st.sidebar.success(f"Loaded row {sample.index[0]} from Kaggle dataset.")
    
    
    raw_amount = sample['Amount'].values[0]
    
    
    sample_features = sample.drop('Class', axis=1).copy()
    sample_features['Amount'] = scaler.transform(sample_features['Amount'].values.reshape(-1, 1))
    sample_features['Time'] = scaler.transform(sample_features['Time'].values.reshape(-1, 1))
    
    input_data = sample_features
else:
    st.sidebar.error("Dataset not found! Please place 'creditcard.csv' in the 'data' folder to use the Random Sample feature.")


if input_data is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Transaction Analysis")
        
       
        if sim_mode == "Manual Entry":
             display_amt = amount
        else:
             display_amt = raw_amount
             
        st.markdown(f"**Transaction Amount:** ${display_amt:,.2f}")
        if true_label != "Unknown":
            label_text = "FRAUD ⚠️" if true_label == 1 else "LEGITIMATE ✅"
            color = "red" if true_label == 1 else "green"
            st.markdown(f"**Actual Database Label:** <span style='color:{color}'>**{label_text}**</span>", unsafe_allow_html=True)
            
        if st.button("🔍 Run Machine Learning Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing 30-factor PCA matrix..."):
                # Predict
                prob = model.predict_proba(input_data)[0][1]
                pred = int(prob > 0.5)
                
               
                st.markdown("---")
                if pred == 1:
                    st.markdown(f'<div class="metric-card"><span class="fraud-alert">🚨 SYSTEM ALERT: FRAUD DETECTED</span><br>Probability: {prob*100:.2f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><span class="legit-alert">✅ CLEARED: SECURE TRANSACTION</span><br>Probability of Fraud: {prob*100:.2f}%</div>', unsafe_allow_html=True)

                st.session_state['run_shap'] = True

    with col2:
        if st.session_state.get('run_shap', False):
            st.subheader("🧠 Model Explainability (SHAP)")
            st.markdown("Understanding **WHY** the model made this decision.")
            
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            
            
            fig = plt.figure(figsize=(8, 4))
            
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value, 
                shap_values[0],
                feature_names=input_data.columns.tolist(),
                show=False
            )
                
            plt.gcf().set_facecolor('#0e1117')
            plt.gca().set_facecolor('#0e1117')
            plt.gca().tick_params(colors='white')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            st.info("💡 **How to read this plot:** Red bars push the model towards predicting Fraud. Blue bars push the model towards predicting a Legitimate transaction.")

st.markdown("---")
st.markdown("###### Developed for College Submission | Powered by XGBoost, SciKit-Learn, and Streamlit")
