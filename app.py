"""
ML-Based Suicide Risk Analyzer - Streamlit Application
Version: 1.0.1 (Standalone ML)

Uses the trained machine learning model for suicide risk detection.
"""

import streamlit as st
import joblib
import os
from datetime import datetime
import re

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Crisis resources
CRISIS_RESOURCES = {
    "988": "988 Suicide & Crisis Lifeline",
    "crisis_text": "Text HOME to 741741",
    "emergency": "911 for immediate emergencies"
}

# Page config
st.set_page_config(
    page_title="Suicide Risk Analyzer",
    page_icon="💙",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .crisis-banner {
        background-color: #ff4444;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 20px;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
        font-size: 24px;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained ML model and vectorizer."""
    try:
        model_path = os.path.join(SCRIPT_DIR, 'suicide_detection_model.pkl')
        vectorizer_path = os.path.join(SCRIPT_DIR, 'tfidf_vectorizer.pkl')
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first by running: `python train_model.py`")
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
    **Call or Text: 988**
    
    Available 24/7 - Free and confidential
    
    ---
    
    ### � Crisis Text Line
    **Text HOME to 741741**
    
    ---
    
    ### 🚑 Emergency
    **Call 911**
    
    For immediate life-threatening emergencies
    
    ---
    
    ### 🌐 International Resources
    Visit: [findahelpline.com](https://findahelpline.com)
    """)


def main():
    # Load model
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Emergency banner
    st.markdown("""
    <div class="crisis-banner">
        🆘 CRISIS SUPPORT: Call 988 (US) or your local emergency number - Available 24/7
    </div>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("💙 ML-Based Suicide Risk Analyzer")
    st.markdown("""
    This tool uses machine learning trained on 100,000 samples to detect suicide risk in text.
    
    **Model Performance:**
    - 93% Accuracy
    - 94% Precision  
    - 92% Recall
    """)
    
    # Display sidebar resources
    display_crisis_resources()
    
    # Disclaimer
    with st.expander("⚠️ Important Disclaimer"):
        st.warning("""
        **This tool is for analytical support only and does NOT replace professional judgment.**
        
        - This is not a medical diagnosis
        - Always consult licensed mental health professionals
        - If someone is in immediate danger, call 911 or go to the nearest emergency room
        - Crisis support is available 24/7 at 988
        """)
    
    # Main input
    st.markdown("---")
    st.markdown("### � Enter Text to Analyze")
    
    input_text = st.text_area(
        "Paste or type text:",
        height=200,
        placeholder="Enter text here for analysis..."
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.rerun()
    
    # Analysis
    if analyze_btn:
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                risk, confidence, is_high_risk = predict_risk(input_text, model, vectorizer)
                
                st.markdown("---")
                
                if is_high_risk:
                    # HIGH RISK - Show crisis resources first
                    st.error("# 🚨 SUICIDE RISK DETECTED 🚨")
                    
                    st.markdown("""
                    <div style="background-color: #ff4444; color: white; padding: 30px; 
                                border-radius: 10px; font-size: 18px; line-height: 2;">
                        <h2 style="color: white;">⚠️ IMMEDIATE SUPPORT AVAILABLE</h2>
                        
                        <p><strong>🔴 Call 988 - Suicide & Crisis Lifeline</strong><br>
                        Available 24/7 - Free and confidential</p>
                        
                        <p><strong>💬 Text HOME to 741741</strong><br>
                        Crisis Text Line</p>
                        
                        <p><strong>🚑 Emergency: Call 911</strong><br>
                        For immediate life-threatening situations</p>
                        
                        <hr style="border-color: white;">
                        
                        <p style="font-size: 24px;">💙 You are not alone. Help is available right now.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Show prediction details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Risk Level", risk, delta=None)
                    with col2:
                        st.metric("Model Confidence", f"{confidence*100:.1f}%")
                    
                    st.warning("""
                    **Recommended Action:**
                    - Do not leave the person alone
                    - Encourage immediate professional contact
                    - Call 988 or visit nearest emergency room
                    - Remove access to lethal means if possible
                    """)
                    
                else:
                    # LOW RISK - Normal display
                    st.success("### Analysis Complete")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'<p class="risk-low">✓ {risk}</p>', unsafe_allow_html=True)
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    st.info("""
                    **Recommendation:**
                    - Standard support and monitoring
                    - Encourage self-care and wellness
                    - Crisis resources available if needed (see sidebar)
                    """)
                
                # Save results option
                st.markdown("---")
                
                report = f"""
SUICIDE RISK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RISK LEVEL: {risk}
CONFIDENCE: {confidence*100:.1f}%

MODEL INFORMATION:
- Algorithm: Logistic Regression with TF-IDF
- Training Dataset: 100,000 samples
- Accuracy: 93%
- Precision: 94%
- Recall: 92%

INPUT TEXT:
{input_text}

DISCLAIMER:
This is an automated analysis and NOT a medical diagnosis.
Always consult with licensed mental health professionals.

CRISIS RESOURCES:
- 988 Suicide & Crisis Lifeline (Call or text 988)
- Crisis Text Line (Text HOME to 741741)
- Emergency Services (Call 911)
"""
                
                st.download_button(
                    "📥 Download Report",
                    data=report,
                    file_name=f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 14px;">
        <p>💙 This tool supports, never replaces, human judgment and compassion.</p>
        <p>Model trained on Suicide_Detection.csv dataset (1M+ samples)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
