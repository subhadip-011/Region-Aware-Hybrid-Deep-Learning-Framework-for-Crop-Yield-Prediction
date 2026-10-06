# streamlit_app/app.py

import streamlit as st

#
# PAGE CONFIGURATION
#

st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS
#

st.markdown("""
<style>

.main-title {
    font-size: 3rem;
    font-weight: bold;
    color: #2E7D32;
    text-align: center;
    margin-bottom: 0.5rem;
}

.sub-title {
    font-size: 1.3rem;
    color: #555;
    text-align: center;
    margin-bottom: 2rem;
}

.feature-card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 1rem;
    border-left: 5px solid #2E7D32;
    margin-bottom: 1rem;
}

.metric-box {
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 100%
    );

    padding: 1.5rem;

    border-radius: 1rem;

    text-align: center;

    color: white;
}

.footer {
    text-align: center;
    color: gray;
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

#
# HEADER
#

st.markdown(
    '<div class="main-title">'
    '🌾 Crop Yield Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Explainable Region-Aware Hybrid Deep Learning Framework '
    'for Crop Yield Prediction'
    '</div>',
    unsafe_allow_html=True
)

#
# HERO SECTION
#

col1, col2 = st.columns([2, 1])

with col1:

    st.markdown("""
    ## 🌱 Project Overview

    This intelligent agricultural prediction platform uses:

    - Machine Learning
    - Hybrid Deep Learning
    - Explainable AI (SHAP)
    - Climate Analytics
    - Soil Intelligence
    - Region-Aware Modeling

    to predict crop yield accurately using
    environmental and agricultural parameters.

    ---
    
    ### 🎯 Main Objectives

    - Predict crop yield using AI
    - Analyze climate impact
    - Improve agricultural decisions
    - Provide explainable predictions
    - Support smart farming systems

    """)

with col2:

    st.markdown("""
    <div class="metric-box">

    <h2>🏆 Best Model</h2>

    <h1>Hybrid RF + ANN</h1>

    <hr>

    <h3>R² Score</h3>

    <h1>0.9426</h1>

    </div>
    """, unsafe_allow_html=True)

#
# FEATURES SECTION
#

st.markdown("---")

st.markdown("## 🚀 Platform Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:

    st.markdown("""
    <div class="feature-card">

    <h3>🔮 Yield Prediction</h3>

    Predict crop yield using:
    
    - Random Forest
    - XGBoost
    - Hybrid Deep Learning

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">

    <h3>📈 Advanced Analytics</h3>

    Interactive agricultural analysis:

    - Climate analysis
    - Soil analysis
    - Crop analysis
    - Trend analysis

    </div>
    """, unsafe_allow_html=True)

with feature_col2:

    st.markdown("""
    <div class="feature-card">

    <h3>🔍 Explainable AI</h3>

    SHAP-based model interpretability:

    - Feature importance
    - Model transparency
    - Prediction explanation

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">

    <h3>🌍 Region-Aware System</h3>

    Region-specific agricultural intelligence:

    - Agro-zone analysis
    - State-level prediction
    - Climate-aware features

    </div>
    """, unsafe_allow_html=True)

#
# MODEL PERFORMANCE
#

st.markdown("---")

st.markdown("## 📊 Model Performance")

performance_col1, performance_col2, performance_col3 = st.columns(3)

with performance_col1:

    st.metric(
        label="🌲 Random Forest",
        value="0.9364",
        delta="Best ML Model"
    )

with performance_col2:

    st.metric(
        label="⚡ XGBoost",
        value="0.9355",
        delta="High Accuracy"
    )

with performance_col3:

    st.metric(
        label="🧠 Hybrid RF + ANN",
        value="0.9426",
        delta="Best Overall"
    )

#
# NAVIGATION GUIDE
#

st.markdown("---")

st.markdown("## 🧭 Navigation Guide")

st.info("""
Use the left sidebar to navigate between pages:

📊 Dashboard → Overview and statistics

🔮 Predict → Predict crop yield

📈 Analysis → EDA and SHAP analysis

📚 Documentation → Project methodology and workflow
""")

#
# TECHNOLOGY STACK
#

st.markdown("---")

st.markdown("## 💻 Technology Stack")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:

    st.success("""
    ### 🤖 Machine Learning
    
    - Random Forest
    - XGBoost
    - Scikit-learn
    """)

with tech_col2:

    st.success("""
    ### 🧠 Deep Learning
    
    - TensorFlow
    - Keras
    - ANN Hybrid Model
    """)

with tech_col3:

    st.success("""
    ### 📊 Visualization
    
    - Streamlit
    - Plotly
    - SHAP
    - Matplotlib
    """)

#
# FOOTER
#

st.markdown("---")

st.markdown("""
<div class="footer">

🌾 Explainable Hybrid AI System for Smart Agriculture

<br><br>

Built using Machine Learning, Deep Learning,
SHAP Explainability, and Streamlit

<br><br>

© 2026 Crop Yield Prediction Project

</div>
""", unsafe_allow_html=True)