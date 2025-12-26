"""
Clinical Sentiment Analyst - Streamlit Web Application

A web-based tool for analyzing text for anxiety markers and suicidal ideation
with built-in safety protocols and crisis intervention features.

⚠️ IMPORTANT: This tool is for analytical support only and does not replace
professional clinical judgment.
"""

import streamlit as st
from clinical_analyzer import ClinicalAnalyzer
from safety_override import SafetyOverride, get_emergency_banner
import json


# Page configuration
st.set_page_config(
    page_title="Clinical Sentiment Analyst",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for calming design
st.markdown("""
<style>
    .main {
        background-color: #f0f4f8;
    }
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
    }
    .crisis-banner {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .report-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-moderate {
        color: #fd7e14;
        font-weight: bold;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-imminent {
        color: #dc3545;
        font-weight: bold;
        font-size: 24px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ClinicalAnalyzer()
    if 'safety' not in st.session_state:
        st.session_state.safety = SafetyOverride()
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None


def display_emergency_banner():
    """Display persistent emergency resource banner."""
    st.markdown("""
    <div class="crisis-banner">
        🆘 CRISIS SUPPORT AVAILABLE 24/7: Call 988 (US) or your local emergency number
    </div>
    """, unsafe_allow_html=True)


def display_disclaimers():
    """Display medical and legal disclaimers."""
    disclaimers = st.session_state.safety.get_disclaimers()
    
    with st.expander("⚠️ Important Disclaimers - Please Read"):
        st.markdown(f"""
        <div class="disclaimer">
            <h4>🏥 Professional Judgment Required</h4>
            <p>{disclaimers['primary']}</p>
            
            <h4>🚨 Emergency Protocol</h4>
            <p>{disclaimers['emergency']}</p>
            
            <h4>🔒 Privacy Notice</h4>
            <p>{disclaimers['privacy']}</p>
            
            <h4>👨‍⚕️ Seek Professional Help</h4>
            <p>{disclaimers['professional_help']}</p>
        </div>
        """, unsafe_allow_html=True)


def display_crisis_resources(region="united_states"):
    """Display crisis resources in sidebar."""
    st.sidebar.title("🆘 Crisis Resources")
    
    resources = st.session_state.safety.get_crisis_resources(region)
    
    if region == "united_states":
        primary = resources["primary_hotline"]
        st.sidebar.markdown(f"""
        ### 📞 {primary['name']}
        **Call: {primary['number']}**
        
        {primary['available']}
        
        {primary['description']}
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Additional Resources")
        
        for hotline in resources["alternative_hotlines"]:
            with st.sidebar.expander(hotline['name']):
                if 'number' in hotline:
                    st.write(f"**Call:** {hotline['number']}")
                if 'text' in hotline:
                    st.write(f"**Text:** {hotline['text']}")
                st.write(hotline['description'])
        
        st.sidebar.markdown("---")
        emergency = resources["emergency"]
        st.sidebar.error(f"🚑 **Emergency: {emergency['number']}**\n\n{emergency['description']}")


def get_risk_css_class(risk_level):
    """Get CSS class for risk level."""
    return f"risk-{risk_level.lower()}"


def display_analysis_result(result, override_triggered, emergency_msg):
    """Display analysis results with appropriate formatting."""
    
    if override_triggered:
        # SAFETY OVERRIDE: Display emergency message prominently
        st.error("# 🚨 IMMEDIATE SUPPORT NEEDED 🚨")
        st.markdown(f"""
        <div style="background-color: #ff4444; color: white; padding: 30px; 
                    border-radius: 10px; font-size: 18px; line-height: 1.8;">
        {emergency_msg.replace('\n', '<br>')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.warning("📋 **Analysis suppressed due to safety protocols. Crisis resources prioritized.**")
        
    else:
        # Display full analysis for Low/Moderate risk
        st.markdown(f"""
        <div class="report-section">
            <h2>Analysis Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk level
        col1, col2 = st.columns([2, 1])
        with col1:
            risk_class = get_risk_css_class(result.risk_level)
            st.markdown(f"""
            <div class="report-section">
                <h3>Risk Level</h3>
                <p class="{risk_class}">{result.risk_level.upper()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="report-section">
                <h3>Confidence</h3>
                <p><strong>{result.confidence * 100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Primary emotions
        st.markdown(f"""
        <div class="report-section">
            <h3>Primary Emotions Detected</h3>
            <p>{', '.join(result.primary_emotions)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key phrases
        if result.key_phrases:
            st.markdown("""
            <div class="report-section">
                <h3>Key Phrases Identified</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, phrase in enumerate(result.key_phrases, 1):
                st.markdown(f"{i}. {phrase}")
        
        # Detailed scores
        st.markdown("""
        <div class="report-section">
            <h3>Detailed Analysis Scores</h3>
        </div>
        """, unsafe_allow_html=True)
        
        score_cols = st.columns(3)
        score_items = list(result.detailed_scores.items())
        
        for idx, (category, score) in enumerate(score_items):
            with score_cols[idx % 3]:
                display_name = category.replace('_', ' ').title()
                st.metric(display_name, f"{score:.2f}")
        
        # Recommended action
        action_color = "#dc3545" if result.risk_level in ["High", "Imminent"] else "#fd7e14" if result.risk_level == "Moderate" else "#28a745"
        
        st.markdown(f"""
        <div class="report-section" style="border-left: 5px solid {action_color};">
            <h3>Recommended Action</h3>
            <p style="font-size: 16px;">{result.recommended_action}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show crisis resources for Moderate/High
        if result.risk_level in ["Moderate", "High"]:
            st.warning("💙 **Crisis resources are available in the sidebar. Please review them.**")


def main():
    """Main application function."""
    
    # Initialize
    initialize_session_state()
    
    # Display emergency banner
    display_emergency_banner()
    
    # Title
    st.title("💙 Clinical Sentiment Analyst")
    st.markdown("""
    This tool analyzes text for markers of anxiety and suicidal ideation using 
    clinical psychology frameworks including the Columbia Scale.
    """)
    
    # Display disclaimers
    display_disclaimers()
    
    # Display crisis resources in sidebar
    region = st.sidebar.selectbox(
        "Select Region for Crisis Resources",
        ["united_states", "international"],
        index=0
    )
    display_crisis_resources(region)
    
    # Main input area
    st.markdown("---")
    st.markdown("### 📝 Enter Text for Analysis")
    
    input_text = st.text_area(
        "Paste or type the text you want to analyze:",
        height=200,
        placeholder="Enter text here for sentiment and risk analysis..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.analysis_complete = False
        st.session_state.current_result = None
        st.rerun()
    
    # Perform analysis
    if analyze_button:
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                # Run analysis
                result = st.session_state.analyzer.analyze(input_text)
                
                # Check safety override
                override_triggered, emergency_msg, processed_result = st.session_state.safety.create_safety_report(
                    result.risk_level,
                    {
                        "risk_level": result.risk_level,
                        "primary_emotions": result.primary_emotions,
                        "key_phrases": result.key_phrases,
                        "recommended_action": result.recommended_action,
                        "detailed_scores": result.detailed_scores,
                        "confidence": result.confidence
                    },
                    region
                )
                
                # Store results
                st.session_state.analysis_complete = True
                st.session_state.current_result = (result, override_triggered, emergency_msg)
    
    # Display results
    if st.session_state.analysis_complete and st.session_state.current_result:
        st.markdown("---")
        result, override_triggered, emergency_msg = st.session_state.current_result
        display_analysis_result(result, override_triggered, emergency_msg)
        
        # Download report option (for non-override cases)
        if not override_triggered:
            st.markdown("---")
            report_text = st.session_state.analyzer.format_report(result)
            st.download_button(
                label="📥 Download Analysis Report",
                data=report_text,
                file_name=f"clinical_analysis_{result.timestamp.replace(':', '-')}.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 14px;">
        <p>This tool is designed to support, not replace, clinical judgment.</p>
        <p>Always consult with licensed mental health professionals for proper evaluation.</p>
        <p>💙 If you're in crisis, help is available right now.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
