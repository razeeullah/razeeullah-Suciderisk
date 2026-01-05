"""
Unified Clinical Mental Health Assistant
Version: 2.1.0 (Viva Edition - Multi-Model)

This application combines machine learning (ML) models with clinical analysis 
to detect suicide risk and predict mental health conditions based on text input.
"""

import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import re
import json

# Internal module imports
from diagnostic_engine import DiagnosticAssistant
from situational_analyzer import SituationalAnalyzer

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------

# Set base directory for pathing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Mental Health Assistant",
    page_icon="🧠",
    layout="wide"
)

# Custom Enhanced CSS for a professional look
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 1rem; }
    .crisis-banner {
        background-color: #EF4444; color: white; padding: 25px;
        border-radius: 12px; text-align: center; font-weight: 800;
        font-size: 22px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .report-section {
        background-color: #F3F4F6; padding: 20px; border-radius: 10px;
        margin: 15px 0; border-left: 6px solid #3B82F6;
    }
    .risk-high { color: #DC2626; font-weight: bold; font-size: 24px; }
    .risk-low { color: #059669; font-weight: bold; }
    .triage-wellness { border-left-color: #10B981; }
    .triage-checkin { border-left-color: #F59E0B; }
    .triage-therapist { border-left-color: #F97316; }
    .triage-critical { border-left-color: #DC2626; }
    .pom-card {
        background-color: #EFF6FF; padding: 20px; border-radius: 12px;
        border-top: 5px solid #3B82F6; margin-bottom: 20px;
    }
    .grounding-box {
        background-color: #FFF7ED; padding: 20px; border-radius: 12px;
        border: 2px dashed #F97316; margin-top: 15px;
    }
    .stTextArea textarea { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CORE ANALYTIC FUNCTIONS
# ---------------------------------------------------------

@st.cache_resource
def load_all_models():
    """
    Load all trained ML models and vectorizers from the filesystem.
    Models are cached to improve performance.
    """
    models = {}
    try:
        # 1. Suicide Risk Detection Models
        s_models_config = {
            'Linear Regression': 'suicide_linear_regression_model.pkl',
            'Logistic Regression': 'suicide_logistic_regression_model.pkl',
            'SVM': 'suicide_svm_model.pkl'
        }
        s_vec_path = os.path.join(SCRIPT_DIR, 'tfidf_vectorizer.pkl')
        
        if os.path.exists(s_vec_path):
            models['suicide_vec'] = joblib.load(s_vec_path)
            for name, filename in s_models_config.items():
                m_path = os.path.join(SCRIPT_DIR, filename)
                if os.path.exists(m_path):
                    models[f'suicide_{name}'] = joblib.load(m_path)
            
        # 2. Multi-Condition Classification Models
        m_models_config = {
            'Linear Regression': 'multi_linear_regression_model.pkl',
            'Logistic Regression': 'multi_logistic_regression_model.pkl',
            'SVM': 'multi_svm_model.pkl'
        }
        m_vec_path = os.path.join(SCRIPT_DIR, 'multi_tfidf.pkl')
        
        if os.path.exists(m_vec_path):
            models['multi_vec'] = joblib.load(m_vec_path)
            for name, filename in m_models_config.items():
                m_path = os.path.join(SCRIPT_DIR, filename)
                if os.path.exists(m_path):
                    models[f'multi_{name}'] = joblib.load(m_path)
            
        return models
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        return models

def preprocess_text(text):
    """
    Standard text cleaning pipeline for ML inference.
    """
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'" -]', '', text) # Remove special chars
    text = ' '.join(text.split()) # Normalize whitespace
    return text

def predict_suicide_risk(text, model, vectorizer, model_name=""):
    """
    Predict binary suicide risk using the selected ML model.
    """
    cleaned = preprocess_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    
    # Linear Regression yields continuous output; we threshold at 0.5
    if "Linear" in model_name:
        prediction_val = model.predict(text_tfidf)[0]
        prediction = 1 if prediction_val >= 0.5 else 0
        probabilities = [max(0, 1 - prediction_val), min(1, prediction_val)]
    else:
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
    
    risk_label = "SUICIDE RISK DETECTED" if prediction == 1 else "Low Risk"
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    
    return risk_label, confidence, (prediction == 1)

def predict_condition(text, models, model_name="Logistic Regression"):
    """
    Predict multi-class mental health condition.
    """
    model_key = f'multi_{model_name}'
    if model_key not in models:
        return None, None
        
    cleaned = preprocess_text(text)
    vec = models['multi_vec'].transform([cleaned])
    prediction = models[model_key].predict(vec)[0]
    
    # Handle confidence estimation for different model types
    if hasattr(models[model_key], "predict_proba"):
        probs = models[model_key].predict_proba(vec)[0]
        conf = max(probs)
    else:
        # For Ridge/Linear, we use decision_function with softmax as proxy
        d_func = models[model_key].decision_function(vec)[0]
        exp_d = np.exp(d_func - np.max(d_func))
        probs = exp_d / exp_d.sum()
        conf = max(probs)
        
    return prediction, conf

# ---------------------------------------------------------
# UI COMPONENTS
# ---------------------------------------------------------

def display_sidebar_resources():
    """Helper for crisis resource sidebar."""
    st.sidebar.title("🆘 Support Hotlines")
    st.sidebar.markdown("""
    **National Suicide & Crisis Lifeline**
    - Call or Text: **988**
    - Available 24/7 across US/Canada
    
    **Crisis Text Line**
    - Text **HOME** to **741741**
    
    **International Support**
    - [findahelpline.com](https://findahelpline.com)
    """)
    
    st.sidebar.divider()
    st.sidebar.info("Disclaimer: This tool is for clinical research and academic demonstration only. It is not an emergency response system.")

def main():
    # Initialize components
    if 'entries' not in st.session_state:
        st.session_state.entries = []
    if 'assistant' not in st.session_state:
        st.session_state.assistant = DiagnosticAssistant()
    if 'situational' not in st.session_state:
        st.session_state.situational = SituationalAnalyzer()
    
    models = load_all_models()
    display_sidebar_resources()
    
    # Main Header
    st.markdown('<div class="crisis-banner">🆘 EMERGENCY: CALL 988 (USA) OR LOCAL SERVICES - YOU ARE NOT ALONE</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">🧠 Clinical Mental Health Assistant</div>', unsafe_allow_html=True)
    st.write("Advanced diagnostic tool integrating Machine Learning with DSM-5 guidelines.")

    # Application Tabs
    tabs = st.tabs([
        "🎯 Suicide Risk (ML)", 
        "🧬 Condition Prediction (ML)", 
        "📋 Diagnostics (DSM-5)", 
        "🌱 Peace of Mind (Situational)",
        "📊 Dataset Explorer"
    ])
    
    # TAB 1: SUICIDE RISK ANALYSIS
    with tabs[0]:
        available_s_models = [k.replace('suicide_', '') for k in models.keys() if k.startswith('suicide_') and k != 'suicide_vec']
        
        if not available_s_models:
            st.warning("Suicide detection models not found. Please run the training pipeline.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("🎯 ML Suicide Detection")
                input_text = st.text_area("Analyze patient text:", height=200, key="ml_input_viva")
            with col2:
                st.subheader("Model Settings")
                s_model_choice = st.selectbox("Inference Algorithm:", available_s_models, index=available_s_models.index('SVM') if 'SVM' in available_s_models else 0)
                st.info("Compare model outputs to see different classification behaviors.")

            if st.button("🔍 Analyze Risk", type="primary"):
                if not input_text.strip():
                    st.warning("Missing input text."); return
                
                risk, conf, is_high = predict_suicide_risk(input_text, models[f'suicide_{s_model_choice}'], models['suicide_vec'], s_model_choice)
                
                st.markdown("---")
                if is_high:
                    st.markdown(f'<div class="risk-high">🚨 {risk}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">✅ {risk}</div>', unsafe_allow_html=True)
                
                st.metric(f"Confidence Level ({s_model_choice})", f"{conf*100:.2f}%")

    # TAB 2: CONDITION CLASSIFICATION
    with tabs[1]:
        available_m_models = [k.replace('multi_', '') for k in models.keys() if k.startswith('multi_') and k != 'multi_vec']
        
        if not available_m_models:
            st.warning("Multi-class models not found.")
        else:
            st.subheader("🧠 Multi-Condition Condition Classifier")
            multi_input = st.text_area("Patient Journal Entry:", height=200, key="multi_input_viva")
            
            m_col1, m_col2 = st.columns([1, 1])
            with m_col1:
                m_model_choice = st.radio("Classification Engine:", available_m_models, horizontal=True)
            
            if st.button("🔮 Predict Condition"):
                if not multi_input.strip():
                    st.warning("Missing input."); return
                    
                cond, conf = predict_condition(multi_input, models, m_model_choice)
                st.markdown("---")
                st.header(f"🩺 Prediction: {cond}")
                st.metric("Probability Score", f"{conf*100:.1f}%")

    # TAB 3: DSM-5 DIAGNOSTICS (Procedural Logic)
    with tabs[2]:
        st.subheader("📋 Rule-Based Diagnostic Engine")
        st.markdown("Longitudinal analysis based on patterns within the DSM-5 framework.")
        
        col_in, col_set = st.columns([3, 1])
        with col_in:
            dsm_text = st.text_area("Add journal entry for sequence analysis:", height=150)
        with col_set:
            dsm_date = st.date_input("Clinical Date", datetime.now())
            if st.button("➕ Log Entry", use_container_width=True):
                if dsm_text.strip():
                    st.session_state.entries.append({"text": dsm_text, "date": dsm_date.strftime("%Y-%m-%d")})
                    st.success("Entry added.")
        
        if st.session_state.entries:
            st.divider()
            report = st.session_state.assistant.analyze(st.session_state.entries)
            
            # Polar Chart Visualization
            df_plot = pd.DataFrame(report["diagnostics"])
            fig = px.line_polar(df_plot, r='Score', theta='Condition', line_close=True, range_r=[0,100], markers=True)
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)
            
            for diag in sorted(report["diagnostics"], key=lambda x: x['Score'], reverse=True):
                if diag['Score'] > 10:
                    t_class = "triage-critical" if diag['Score'] > 85 else "triage-therapist" if diag['Score'] > 60 else "triage-checkin"
                    st.markdown(f"""
                    <div class="report-section {t_class}">
                        <h4>{diag['Condition']} - Score: {diag['Score']}/100</h4>
                        <p><strong>Guidance:</strong> {diag['Recommended_Action']}</p>
                        <small>Key patterns detected: {', '.join(diag['Evidence_Detected'])}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.button("🗑️ Reset Tracking"):
                st.session_state.entries = []
                st.rerun()

    # TAB 4: SITUATIONAL ANALYSIS
    with tabs[3]:
        st.subheader("🌱 Holistic Situational Analyzer")
        situ_text = st.text_area("Life stressor description:", height=200, placeholder="Describe the current life situation and stressors...")
        
        if st.button("⚖️ Analyze Situation"):
            if not situ_text.strip():
                st.warning("Please enter context.")
            else:
                res = st.session_state.situational.analyze(situ_text)
                st.info(f"**Clinical Insight:** {res['validation']}")
                
                if res['is_emergency']:
                    st.markdown('<div class="grounding-box">', unsafe_allow_html=True)
                    st.warning("### 🧘 Immediate Intervention Required")
                    for step in res['grounding']:
                        st.write(f"- {step}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("PoM Index", f"{res['pom_score']}/100")
                col_m2.progress(res['pom_score']/100)

    # TAB 5: DATASETS
    with tabs[4]:
        st.subheader("📊 Research Dataset Explorer")
        ds_map = {
            "Suicide Detection (Kaggle)": "Suicide_Detection.csv",
            "Student Mental Health": "Student Mental health.csv",
            "Anxiety/Depression Survey": "survey.csv"
        }
        choice = st.selectbox("View Source Data:", list(ds_map.keys()))
        path = os.path.join(SCRIPT_DIR, ds_map[choice])
        
        if os.path.exists(path):
            df_v = pd.read_csv(path, nrows=100)
            st.dataframe(df_v, use_container_width=True)
        else:
            st.error("Dataset not found locally.")

if __name__ == "__main__":
    main()
