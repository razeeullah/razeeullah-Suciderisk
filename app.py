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
# UI CONFIGURATION & CSS (FUTURISTIC GLASS)
# ---------------------------------------------------------
st.set_page_config(page_title="MindSuite AI", page_icon="🧠", layout="wide")

# ---------------------------------------------------------
# CORE ANALYTIC FUNCTIONS
# ---------------------------------------------------------

def calculate_wellness_score(suicide_is_high, dsm_report, pom_score):
    """
    Aggregates clinical and ML markers into a 1-10 wellness score.
    """
    # 1. Inverse DSM severity (Higher score in DSM = Lower Wellness)
    if dsm_report and "diagnostics" in dsm_report:
        dsm_scores = [d['Score'] for d in dsm_report["diagnostics"]]
        avg_dsm = sum(dsm_scores) / len(dsm_scores) if dsm_scores else 0
    else:
        avg_dsm = 0
        
    # Weightage: 40% PoM, 60% Inverse DSM
    wellness_base = (pom_score * 0.4) + ((100 - avg_dsm) * 0.6)
    final_score = (wellness_base / 10)
    
    # 2. Suicide Risk Penalty (Hard cap at 2.9)
    if suicide_is_high:
        final_score = min(final_score, 2.9)
    
    final_score = max(1.0, min(10.0, final_score))
    
    # 3. Categorize
    if final_score >= 8.0:
        category, color = "Good", "#059669"
    elif final_score >= 6.0:
        category, color = "Stable", "#3B82F6"
    elif final_score >= 3.0:
        category, color = "Serious", "#F59E0B"
    else:
        category, color = "Critical", "#DC2626"
        
    return round(final_score, 1), category, color

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
    Predict multi-class mental health condition with full probability distribution.
    """
    model_key = f'multi_{model_name}'
    if model_key not in models:
        return None, None
        
    cleaned = preprocess_text(text)
    vec = models['multi_vec'].transform([cleaned])
    classes = models[model_key].classes_
    
    # Handle confidence estimation for different model types
    if hasattr(models[model_key], "predict_proba"):
        probs = models[model_key].predict_proba(vec)[0]
    else:
        # For Ridge/Linear, we use decision_function with softmax as proxy
        d_func = models[model_key].decision_function(vec)[0]
        exp_d = np.exp(d_func - np.max(d_func))
        probs = exp_d / exp_d.sum()
        
    # Sort transitions for top-N analysis
    prob_map = list(zip(classes, probs))
    prob_map.sort(key=lambda x: x[1], reverse=True)
    
    return prob_map  # Returns list of (class, prob) sorted by confidence

# ---------------------------------------------------------
# UI COMPONENTS
# ---------------------------------------------------------

def display_sidebar_resources():
    """Helper for crisis resource sidebar."""
    st.sidebar.title("🆘 Support Hotlines")
    st.sidebar.markdown("""
    **National Suicide & Crisis Lifeline**
    - Call or Text: **+92323263I2**
    - Available 24/7 across Pakistan
    
    **Crisis Text Line**
    - Text **HOME** to **+92349736283**
    
    **International Support**
    - [findahelpline.com](https://findahelpline.com)
    """)
    
    st.sidebar.divider()
    st.sidebar.info("Disclaimer: This tool is for clinical research and academic demonstration only. It is not an emergency response system.")

def main():
    # ---------------------------------------------------------
# MAIN APP FLOW
# ---------------------------------------------------------
    import auth
    auth.init_db()

    # SESSION STATE INITIALIZATION
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard Overview'
    
    # ---------------------------------------------------------
    # AUTHENTICATION FLOW
    # ---------------------------------------------------------
    if not st.session_state['authenticated']:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: var(--text-main);'>MindSuite AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.7; color: var(--text-muted);'>Advanced Clinical Analytics Platform</p>", unsafe_allow_html=True)
        
        # Centered Pill Container
        st.markdown('<div class="login-pill">', unsafe_allow_html=True)
        
        tab_login, tab_signup = st.tabs(["🔒 Login", "📝 Sign Up"])
        
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user", placeholder="Enter your ID")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Enter Platform", use_container_width=True):
                if auth.login_user(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid Username/Password")
        
        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            new_user = st.text_input("New Username", key="reg_user")
            new_pass = st.text_input("New Password", type="password", key="reg_pass")
            new_email = st.text_input("Email (Optional)", key="reg_email")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Secure Account", use_container_width=True):
                if auth.register_user(new_user, new_pass, new_email):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already exists.")
                    
        st.markdown('</div>', unsafe_allow_html=True)
        return  # STOP EXECUTION HERE IF NOT LOGGED IN

    # ---------------------------------------------------------
    # LOGGED IN DASHBOARD
    # ---------------------------------------------------------
    
    # LOAD MODELS
    with st.spinner("Initializing Neural Engines..."):
        models = load_all_models() # Changed from load_models() to load_all_models() to match existing function name
    
    # Init Analyzer
    if 'assistant' not in st.session_state:
        st.session_state.assistant = DiagnosticAssistant()
    if 'situational' not in st.session_state:
        st.session_state.situational = SituationalAnalyzer()
    if 'entries' not in st.session_state:
        st.session_state.entries = []

    # SIDEBAR NAVIGATION
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['user']}")
        st.markdown("---")
        
        pages = [
            ("Dashboard Overview", "🏠"),
            ("Wellness Analysis", "🏆"),
            ("Suicide Risk (ML)", "🎯"),
            ("Condition Prediction", "🔮"),
            ("DSM-5 Diagnostics", "📋"),
            ("Situational Analyzer", "🌱"),
            ("Dataset Explorer", "📊")
        ]
        
        for page_name, icon in pages:
            is_active = st.session_state.page == page_name
            # The original instruction had a custom event listener which is not directly supported by Streamlit buttons.
            # Reverting to standard Streamlit button behavior for navigation.
            if st.button(f"{icon} {page_name}", key=f"btn_{page_name}", use_container_width=True, 
                         type="primary" if is_active else "secondary"):
                st.session_state.page = page_name
                st.rerun()
                
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['authenticated'] = False
            st.session_state['user'] = None
            st.rerun()

        display_sidebar_resources() # Moved this call here to be part of the authenticated sidebar
    
    # Main Content Area
    st.markdown('<div class="crisis-pill">🆘 EMERGENCY: CALL +923414593933</div>', unsafe_allow_html=True)
    
    # PAGE: OVERVIEW
    if st.session_state.page == "Dashboard Overview":
        st.title("Welcome back, Researcher")
        st.markdown("Here is your clinical analysis overview for today.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="card">
                <h3>Total Logged Entries</h3>
                <div class="stat-value">{len(st.session_state.entries)}</div>
                <p style="opacity: 0.7;">Patient longitudinal data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <h3>Diagnostic Sessions</h3>
                <div class="stat-value">12</div>
                <p style="opacity: 0.7;">Active AI classifications</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="card">
                <h3>Risk Alerts</h3>
                <div class="stat-value" style="color: var(--accent);">0</div>
                <p style="opacity: 0.7;">Requiring immediate attention</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        st.subheader("Recent Activity")
        if st.session_state.entries:
            for entry in st.session_state.entries[-3:]:
                st.markdown(f"""
                <div class="card" style="margin-bottom: 15px; padding: 15px;">
                    <strong>{entry['date']}</strong><br>
                    <small>{entry['text'][:100]}...</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent entries logged. Start by analyzing a patient journal.")

    # PAGE: WELLNESS ANALYSIS
    elif st.session_state.page == "Wellness Analysis":
        st.subheader("🏆 Holistic Wellness Analysis")
        st.markdown("Enter a detailed journal entry or patient description to generate a comprehensive wellness score.")
        
        dash_text = st.text_area("Patient Journal / Description:", height=250, key="dash_input")
        
        if st.button("🚀 Generate Wellness Report", type="primary"):
            if not dash_text.strip():
                st.warning("Please provide input text."); st.stop()
            
            with st.spinner("Analyzing multi-dimensional clinical data..."):
                # 1. Suicide Risk
                s_model_key = 'suicide_SVM' if 'suicide_SVM' in models else [k for k in models.keys() if k.startswith('suicide_') and k != 'suicide_vec'][0] if any(k.startswith('suicide_') for k in models) else None
                is_high_s = False
                if s_model_key and 'suicide_vec' in models:
                    _, _, is_high_s = predict_suicide_risk(dash_text, models[s_model_key], models['suicide_vec'], "SVM")
                
                # 2. Condition Prediction
                m_model_name = 'Logistic Regression' if 'multi_Logistic Regression' in models else [k for k in models.keys() if k.startswith('multi_') and k != 'multi_vec'][0].replace('multi_', '') if any(k.startswith('multi_') for k in models) else None
                m_res = predict_condition(dash_text, models, m_model_name) if m_model_name else [("N/A", 0)]
                cond_pred = m_res[0][0]
                cond_conf = m_res[0][1]
                
                # 3. DSM-5 Analysis
                dsm_report = st.session_state.assistant.analyze([{"text": dash_text, "date": datetime.now().strftime("%Y-%m-%d")}])
                
                # 4. Situational PoM
                situ_res = st.session_state.situational.analyze(dash_text)
                
                # 5. Score
                score, category, color = calculate_wellness_score(is_high_s, dsm_report, situ_res['pom_score'])
                
                st.markdown(f"""
                <div class="card" style="text-align: center; border-top: 8px solid {color};">
                    <div style="font-size: 5rem; font-weight: 900;">{score} / 10</div>
                    <div style="font-size: 1.5rem; color: {color};">Status: {category}</div>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Suicide Risk", "🚨 HIGH" if is_high_s else "✅ Low")
                c2.metric("Prediction", cond_pred)
                c3.metric("PoM Index", f"{situ_res['pom_score']}/100")

    # PAGE: SUICIDE RISK
    elif st.session_state.page == "Suicide Risk (ML)":
        st.subheader("🎯 ML Suicide Detection")
        available_s_models = [k.replace('suicide_', '') for k in models.keys() if k.startswith('suicide_') and k != 'suicide_vec']
        
        col1, col2 = st.columns([2, 1])
        with col1:
            input_text = st.text_area("Analyze patient text:", height=300, key="ml_input_viva")
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("### Settings")
            s_model_choice = st.selectbox("Inference Algorithm:", available_s_models, index=available_s_models.index('SVM') if 'SVM' in available_s_models else 0)
            st.info("Compare model outputs to see different classification behaviors.")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔍 Analyze Risk"):
            if not input_text.strip():
                st.warning("Missing input text."); st.stop()
            
            risk, conf, is_high = predict_suicide_risk(input_text, models[f'suicide_{s_model_choice}'], models['suicide_vec'], s_model_choice)
            
            st.markdown(f"""
            <div class="card" style="border-left: 10px solid {'#ef4444' if is_high else '#10b981'};">
                <h2 style="color: {'#ef4444' if is_high else '#10b981'};">{risk}</h2>
                <div class="stat-value">{conf*100:.2f}%</div>
                <p>Confidence Level ({s_model_choice})</p>
            </div>
            """, unsafe_allow_html=True)

    # PAGE: CONDITION PREDICTION
    elif st.session_state.page == "Condition Prediction":
        st.subheader("🧬 Multi-Condition Classifier")
        available_m_models = [k.replace('multi_', '') for k in models.keys() if k.startswith('multi_') and k != 'multi_vec']
        
        multi_input = st.text_area("Patient Journal Entry:", height=300, key="multi_input_viva")
        m_model_choice = st.radio("Classification Engine:", available_m_models, horizontal=True)
        
        if st.button("🔮 Predict Condition"):
            if not multi_input.strip():
                st.warning("Missing input."); st.stop()
                
            prob_map = predict_condition(multi_input, models, m_model_choice)
            
            # Divide result section
            st.divider()
            
            # Top Result 3D Card
            top_cond, top_prob = prob_map[0]
            st.markdown(f"""
            <div class="card" style="border-left: 10px solid var(--primary); margin-bottom: 30px;">
                <h2 style="color: var(--primary); margin: 0;">Primary Classification: {top_cond}</h2>
                <div class="stat-value">{top_prob*100:.1f}%</div>
                <p style="opacity: 0.7;">Engine: {m_model_choice} | Certainty Index</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Distribution Chart
            st.subheader("📊 Differential Diagnosis Chart")
            df_probs = pd.DataFrame(prob_map, columns=['Condition', 'Probability'])
            df_probs = df_probs.sort_values('Probability', ascending=True) # For horizontal chart
            
            fig = px.bar(df_probs, x='Probability', y='Condition', orientation='h',
                         color='Probability', color_continuous_scale='Viridis',
                         template='plotly_white')
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Certainty Score",
                yaxis_title="",
                showlegend=False,
                height=400,
                font=dict(color='#0F172A')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 3 Insights Breakdown
            st.subheader("🔍 Top Insight Breakdown")
            cols = st.columns(3)
            for i in range(min(3, len(prob_map))):
                with cols[i]:
                    c, p = prob_map[i]
                    st.markdown(f"""
                    <div class="card" style="padding: 20px; text-align: center;">
                        <div style="font-weight: 800; color: var(--text-main);">{c}</div>
                        <div style="font-size: 1.5rem; color: var(--secondary);">{p*100:.1f}%</div>
                        <small style="opacity: 0.6; color: var(--text-muted);">Weighted Confidence</small>
                    </div>
                    """, unsafe_allow_html=True)

    # PAGE: DSM-5 DIAGNOSTICS
    elif st.session_state.page == "DSM-5 Diagnostics":
        st.subheader("📋 Rule-Based Diagnostic Engine")
        col_in, col_set = st.columns([3, 1])
        with col_in:
            dsm_text = st.text_area("Add journal entry for sequence analysis:", height=200)
        with col_set:
            dsm_date = st.date_input("Clinical Date", datetime.now())
            if st.button("➕ Log Entry", use_container_width=True):
                if dsm_text.strip():
                    st.session_state.entries.append({"text": dsm_text, "date": dsm_date.strftime("%Y-%m-%d")})
                    st.success("Entry added.")
        
        if st.session_state.entries:
            st.divider()
            report = st.session_state.assistant.analyze(st.session_state.entries)
            
            df_plot = pd.DataFrame(report["diagnostics"])
            fig = px.line_polar(df_plot, r='Score', theta='Condition', line_close=True, range_r=[0,100], markers=True)
            
            # Glow effect & Premium Fill
            fig.update_traces(
                fill='toself', 
                fillcolor='rgba(99, 102, 241, 0.2)', # Semi-transparent indigo
                line=dict(color='#6366f1', width=3),
                marker=dict(size=10, color='#6366f1', line=dict(color='white', width=1)),
                hovertemplate="<b>%{theta}</b><br>Score: %{r}/100<extra></extra>"
            )
            
            # Light Theme Layout Integration
            fig.update_layout(
                polar=dict(
                    bgcolor='rgba(255,255,255,0.5)',
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 100], 
                        gridcolor='rgba(0,0,0,0.1)',
                        tickfont=dict(color='#64748B', size=10),
                        angle=0,
                        tickangle=0
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(0,0,0,0.1)',
                        linecolor='rgba(0,0,0,0.2)',
                        tickfont=dict(color='#0F172A', size=12)
                    )
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#0F172A',
                margin=dict(t=40, b=40, l=40, r=40),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            for diag in sorted(report["diagnostics"], key=lambda x: x['Score'], reverse=True):
                if diag['Score'] > 10:
                    st.markdown(f"""
                    <div class="card" style="margin-bottom: 20px;">
                        <h4>{diag['Condition']} - {diag['Score']}/100</h4>
                        <p>{diag['Recommended_Action']}</p>
                        <small>Evidence: {', '.join(diag['Evidence_Detected'])}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.button("🗑️ Reset Tracking"):
                st.session_state.entries = []
                st.rerun()

    # PAGE: SITUATIONAL
    elif st.session_state.page == "Situational Analyzer":
        st.subheader("🌱 Holistic Situational Analyzer")
        situ_text = st.text_area("Life stressor description:", height=300, key="situ_input")
        
        if st.button("⚖️ Analyze Situation"):
            if not situ_text.strip():
                st.warning("Please enter context."); st.stop()
                
            res = st.session_state.situational.analyze(situ_text)
            st.success(res['validation'])
            
            if res['is_emergency']:
                st.markdown("""
                <div class="card" style="background: rgba(239, 68, 68, 0.1); border-color: #ef4444;">
                    <h3 style="color: #ef4444;">🧘 Immediate Intervention Required</h3>
                    <ul>
                    """ + "".join([f"<li>{s}</li>" for s in res['grounding']]) + """
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("PoM Index", f"{res['pom_score']}/100")
            col_m2.progress(res['pom_score']/100)

    # PAGE: DATASETS
    elif st.session_state.page == "Dataset Explorer":
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
