"""
Unified Clinical Mental Health Assistant
Version: 2.0.0 (Unified ML & DSM-5)

Combines ML-based suicide risk detection with DSM-5 diagnostic analysis.
"""

import streamlit as st
import joblib
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import re
import json
from diagnostic_engine import DiagnosticAssistant

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Page config
st.set_page_config(
    page_title="Clinical Mental Health Assistant",
    page_icon="🧠",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .crisis-banner {
        background-color: #ff4444; color: white; padding: 20px;
        border-radius: 10px; text-align: center; font-weight: bold;
        font-size: 20px; margin-bottom: 20px;
    }
    .report-section {
        background-color: #f8f9fa; padding: 20px; border-radius: 10px;
        margin: 10px 0; border-left: 5px solid #007bff;
    }
    .risk-high { color: #dc3545; font-weight: bold; font-size: 24px; }
    .risk-low { color: #28a745; font-weight: bold; }
    .triage-wellness { border-left-color: #28a745; }
    .triage-checkin { border-left-color: #ffc107; }
    .triage-therapist { border-left-color: #fd7e14; }
    .triage-critical { border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained ML model and vectorizer."""
    try:
        model_path = os.path.join(SCRIPT_DIR, 'suicide_detection_model.pkl')
        vectorizer_path = os.path.join(SCRIPT_DIR, 'tfidf_vectorizer.pkl')
        
        if not os.path.exists(model_path):
            return None, None
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer
    except Exception as e:
        return None, None


def preprocess_text(text):
    """Clean text for analysis."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'" -]', '', text)
    text = ' '.join(text.split())
    return text


def predict_risk(text, model, vectorizer):
    """Predict suicide risk."""
    cleaned = preprocess_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    if prediction == 1:
        risk = "SUICIDE RISK DETECTED"
        confidence = probabilities[1]
        is_high_risk = True
    else:
        risk = "Low Risk"
        confidence = probabilities[0]
        is_high_risk = False
    
    return risk, confidence, is_high_risk


def display_crisis_resources():
    """Display crisis resources."""
    st.sidebar.title("🆘 Crisis Resources")
    st.sidebar.markdown("""
    ### 📞 988 Suicide & Crisis Lifeline
    **Call or Text: 988** (US/Canada)
    
    Available 24/7 - Free and confidential
    
    ---
    ### 💬 Crisis Text Line
    **Text HOME to 741741**
    
    ---
    ### 🚑 Emergency
    **Call 911**
    
    ---
    ### 🌐 International
    [findahelpline.com](https://findahelpline.com)
    """)

def sidebar_onboarding():
    st.sidebar.title("📋 User Onboarding")
    static_factors = {}
    with st.sidebar.expander("Risk History"):
        if st.checkbox("Family history of Mood Disorders"):
            static_factors["Mood Disorders"] = True
        if st.checkbox("Family history of Psychotic Disorders"):
            static_factors["Psychotic Disorders"] = True
        if st.checkbox("Chronic work/life stress"):
            static_factors["Anxiety Disorders"] = True
        if st.checkbox("Previous trauma experience"):
            static_factors["Trauma-Related"] = True
    return static_factors


def main():
    if 'entries' not in st.session_state:
        st.session_state.entries = []
    if 'assistant' not in st.session_state:
        st.session_state.assistant = DiagnosticAssistant()
    
    model, vectorizer = load_model()
    static_factors = sidebar_onboarding()
    display_crisis_resources()
    
    st.markdown('<div class="crisis-banner">🆘 CRISIS SUPPORT: Call 988 (US) or local emergency - 24/7</div>', unsafe_allow_html=True)
    st.title("🧠 Clinical Mental Health Assistant")
    
    tab_ml, tab_dsm, tab_data = st.tabs(["🎯 Suicide Risk (ML)", "📋 Diagnostics (DSM-5)", "📊 Data Insights"])
    
    with tab_ml:
        if model is None:
            st.warning("Model files not found. Please train the model in the 'train_model.py' script.")
            st.info("The app will proceed with DSM-5 Diagnostics and Data Insights only.")
        else:
            st.markdown("### 🎯 ML Suicide Risk Detection")
            st.markdown("This section uses a Logistic Regression model trained on 1M+ samples.")
            input_text = st.text_area("Analyze text for suicide risk:", height=150, key="ml_input", placeholder="Type or paste text here...")
            if st.button("🔍 Run ML Analysis", type="primary"):
                if not input_text.strip():
                    st.warning("Please enter some text to analyze.")
                    return
                with st.spinner("Analyzing risk..."):
                    risk, conf, is_high = predict_risk(input_text, model, vectorizer)
                    st.markdown("---")
                    if is_high:
                        st.error(f"🚨 {risk}")
                        st.metric("Model Confidence", f"{conf*100:.1f}%")
                        st.warning("⚠️ High Risk detected. Please utilize the crisis resources in the sidebar.")
                    else:
                        st.success(f"✓ {risk}")
                        st.metric("Model Confidence", f"{conf*100:.1f}%")

    with tab_dsm:
        st.markdown("### 📋 DSM-5 Multi-Disorder Diagnostics")
        st.info("Input text entries below to see risk clustering across 7 clinical categories.")
        col1, col2 = st.columns([3, 1])
        with col1:
            dsm_input = st.text_area("Add journal entry or text for tracking:", height=150, key="dsm_input", placeholder="I feel very sad and can't focus on work...")
        with col2:
            dsm_date = st.date_input("Entry Date", datetime.now())
            if st.button("➕ Add & Analyze Entry", use_container_width=True):
                if dsm_input.strip():
                    st.session_state.entries.append({"text": dsm_input, "date": dsm_date.strftime("%Y-%m-%d")})
                    st.success("Entry added to tracking.")
                else:
                    st.warning("Please enter text.")
        
        if st.session_state.entries:
            st.markdown("---")
            report = st.session_state.assistant.analyze(st.session_state.entries, static_factors)
            
            # Radar Chart
            df_plot = pd.DataFrame(report["diagnostics"])
            fig = px.line_polar(df_plot, r='Score', theta='Condition', line_close=True, range_r=[0,100], markers=True, title="DSM-5 Risk Polar Map")
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Triage
            st.markdown("#### Detailed Diagnostic Assessment")
            for diag in sorted(report["diagnostics"], key=lambda x: x['Score'], reverse=True):
                if diag['Score'] > 10:
                    triage_class = "triage-critical" if diag['Score'] > 85 else "triage-therapist" if diag['Score'] > 60 else "triage-checkin" if diag['Score'] > 30 else "triage-wellness"
                    st.markdown(f"""
                    <div class="report-section {triage_class}">
                        <h4 style="margin:0;">{diag['Condition']} ({diag['Score']}/100)</h4>
                        <p style="margin:5px 0;"><strong>Recommendation:</strong> {diag['Recommended_Action']}</p>
                        <small>Evidence: {', '.join(diag['Evidence_Detected']) if diag['Evidence_Detected'] else 'No specific patterns detected.'}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear Tracking History"):
                st.session_state.entries = []
                st.rerun()

    with tab_data:
        st.markdown("### 📊 Mental Health Dataset Insights")
        st.markdown("Explore the statistics behind the training data and community surveys.")
        datasets = {
            "Suicide Detection (Main Model)": "Suicide_Detection.csv",
            "Student Mental Health": "Student Mental health.csv",
            "Survey (General Anxiety/Depression)": "survey.csv",
            "Music & Mental Health": "mxmh_survey_results.csv"
        }
        selected_ds = st.selectbox("Select Dataset to Explore", list(datasets.keys()))
        data_path = os.path.join(SCRIPT_DIR, datasets[selected_ds])
        
        if os.path.exists(data_path):
            try:
                df_view = pd.read_csv(data_path, nrows=500)
                st.write(f"Sample data from `{datasets[selected_ds]}`")
                st.dataframe(df_view.head(10), use_container_width=True)
                
                # Visualizations
                st.markdown("#### Primary Labels Distribution")
                possible_targets = ['class', 'label', 'Treatment', 'Depression', 'Anxiety']
                target = next((col for col in possible_targets if col in df_view.columns), None)
                
                if target:
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.plotly_chart(px.pie(df_view, names=target, title=f"Distribution of '{target}'"), use_container_width=True)
                    with col_p2:
                        st.plotly_chart(px.histogram(df_view, x=target, title=f"Frequency of '{target}'"), use_container_width=True)
                else:
                    st.info("No categorical label column found for visualization.")
                    
            except Exception as e:
                st.error(f"Error loading visualization: {e}")
        else:
            st.error(f"Dataset file `{datasets[selected_ds]}` not found.")


if __name__ == "__main__":
    main()
