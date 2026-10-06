# streamlit_app/pages/4_Documentation.py

import streamlit as st

#
# PAGE CONFIGURATION
#

st.set_page_config(
    page_title="Documentation",
    page_icon="📚",
    layout="wide"
)

#
# CUSTOM CSS
#

st.markdown("""
<style>

.main-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2E7D32;
    text-align: center;
    margin-bottom: 1rem;
}

.section-title {
    font-size: 1.6rem;
    font-weight: bold;
    color: #2E7D32;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.info-box {
    background-color: #f8f9fa;
    padding: 1.2rem;
    border-radius: 1rem;
    border-left: 5px solid #2E7D32;
    margin-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)

#
# HEADER
#

st.markdown(
    '<div class="main-title">'
    '📚 Project Documentation'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
This section contains the complete documentation
for the AI-powered crop yield prediction framework.
""")

#
# PROJECT OVERVIEW
#

st.markdown(
    '<div class="section-title">'
    '🌾 Project Overview'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

<b>Project Title:</b>

Explainable Region-Aware Hybrid Deep Learning Framework
for Crop Yield Prediction

<br><br>

<b>Objective:</b>

Develop an intelligent agricultural prediction system
that predicts crop yield using:

- Climate parameters
- Soil nutrients
- Regional information
- Machine Learning
- Hybrid Deep Learning
- Explainable AI (SHAP)

</div>
""", unsafe_allow_html=True)

#
# PROBLEM STATEMENT
#

st.markdown(
    '<div class="section-title">'
    '🎯 Problem Statement'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
Agricultural productivity is highly influenced by:

- Climate variability
- Soil fertility
- Water availability
- Regional conditions

Traditional yield estimation methods are often:

- inaccurate
- time-consuming
- non-scalable

This project aims to solve these issues
using Artificial Intelligence and Explainable AI.
""")

#
# DATASET INFORMATION
#

st.markdown(
    '<div class="section-title">'
    '📊 Dataset Information'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
### Dataset Features

The dataset contains:

#### 🌤️ Climate Features
- Rainfall
- Temperature
- Humidity

#### 🧪 Soil Features
- Soil Nitrogen
- Soil Phosphorus
- Soil Potassium
- Soil pH

#### 🌍 Regional Features
- State
- Agro-zone
- Crop type

#### 🎯 Target Variable
- Crop Yield (kg/ha)
""")

#
# DATA PREPROCESSING
#

st.markdown(
    '<div class="section-title">'
    '🛠️ Data Preprocessing'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
### Preprocessing Steps

1. Missing value handling
2. Outlier treatment
3. Label encoding
4. Feature scaling
5. Feature engineering
6. Train-test splitting

### Feature Engineering

Additional engineered features:

- Soil fertility index
- Water availability
- Climate stress
- Temperature-rainfall interaction
- Agro-zone encoding
""")

#
# MACHINE LEARNING MODELS
#

st.markdown(
    '<div class="section-title">'
    '🤖 Machine Learning Models'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
### Models Used

| Model | Purpose |
|---|---|
| Linear Regression | Baseline model |
| Decision Tree | Non-linear learning |
| Random Forest | Best ML model |
| XGBoost | Boosting-based prediction |
| SVR | Support Vector Regression |
| ANN | Deep Learning |
| Hybrid RF + ANN | Final proposed model |

### Best Model

🏆 Hybrid RF + ANN

- R² Score: 0.9426
- RMSE: 5475.78
- MAE: 1199.49
""")

#
# HYBRID MODEL ARCHITECTURE
#

st.markdown(
    '<div class="section-title">'
    '🧠 Hybrid Model Architecture'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
### Hybrid Framework

The final prediction system combines:

#### 🌲 Random Forest
- Handles tabular relationships
- Captures non-linear patterns
- Robust against overfitting

#### 🧠 Artificial Neural Network (ANN)
- Learns complex feature interactions
- Improves generalization
- Deep representation learning

### Ensemble Strategy

Final prediction:

Hybrid Prediction =
0.5 × ANN +
0.3 × RF +
0.2 × XGBoost
""")

#
# SHAP EXPLAINABILITY
#

st.markdown(
    '<div class="section-title">'
    '🔍 Explainable AI (SHAP)'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
### Why SHAP?

SHAP (SHapley Additive exPlanations)
was used to explain:

- Feature importance
- Prediction reasoning
- Model transparency

### Key Insights

Most influential features:

- Crop type
- Rainfall
- Water availability
- Temperature
- Soil fertility

### Benefits

- Better interpretability
- Transparent AI decisions
- Trustworthy predictions
""")

#
# STREAMLIT APPLICATION
#

st.markdown(
    '<div class="section-title">'
    '💻 Streamlit Web Application'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
### Features of the Web Application

- Multi-page architecture
- Real-time prediction
- Interactive visualizations
- SHAP explainability
- Downloadable reports
- Climate analysis dashboard

### Pages

| Page | Description |
|---|---|
| Dashboard | Dataset and model overview |
| Predict | Yield prediction |
| Analysis | EDA and SHAP analysis |
| Documentation | Project explanation |
""")

#
# TECHNOLOGY STACK
#

st.markdown(
    '<div class="section-title">'
    '⚙️ Technology Stack'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
### Programming & Frameworks

- Python
- Streamlit
- TensorFlow
- Scikit-learn
- XGBoost

### Visualization

- Plotly
- Matplotlib
- Seaborn
- SHAP

### Data Processing

- Pandas
- NumPy
""")

#
# FUTURE IMPROVEMENTS
#

st.markdown(
    '<div class="section-title">'
    '🚀 Future Improvements'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
### Possible Enhancements

- Real-time weather API integration
- Satellite image analysis
- Mobile application
- IoT sensor integration
- Advanced deep learning models
- Multi-language support
- Real-time farmer advisory system
""")

#
# CONCLUSION
#

st.markdown(
    '<div class="section-title">'
    '✅ Conclusion'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
The proposed Explainable Region-Aware Hybrid
Deep Learning Framework successfully predicts
crop yield using climate, soil, and regional features.

The integration of:

- Machine Learning
- Deep Learning
- Explainable AI
- Streamlit deployment

makes the system highly suitable for
smart agriculture applications.
""")

#
# FOOTER
#

st.markdown("---")

st.markdown("""
<div style="text-align:center; color:gray;">

📚 AI-Powered Agricultural Intelligence System

<br><br>

Hybrid Deep Learning + SHAP Explainability

</div>
""", unsafe_allow_html=True)