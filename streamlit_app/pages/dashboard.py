# streamlit_app/pages/1_Dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#
# PAGE CONFIGURATION
#

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
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

.metric-card {
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 100%
    );

    padding: 1.5rem;

    border-radius: 1rem;

    color: white;

    text-align: center;
}

.section-title {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2E7D32;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)

#
# LOAD DATA
#

@st.cache_data
def load_data():

    df = pd.read_csv(
        'data/processed/processed_data.csv'
    )

    return df

#
# LOAD MODEL RESULTS
#

@st.cache_data
def load_model_results():

    results = pd.DataFrame({

        'Model': [
            'Linear Regression',
            'Decision Tree',
            'Random Forest',
            'XGBoost',
            'SVR',
            'ANN',
            'Hybrid RF + ANN'
        ],

        'R² Score': [
            0.1698,
            0.9127,
            0.9364,
            0.9355,
            -0.0983,
            0.9400,
            0.9426
        ],

        'RMSE': [
            20823.87,
            6750.30,
            5763.55,
            5804.22,
            23952.03,
            5600.00,
            5475.78
        ]
    })

    return results

#
# MAIN FUNCTION
#

def main():

    #
    # TITLE
    #

    st.markdown(
        '<div class="main-title">'
        '📊 Crop Yield Prediction Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    Interactive dashboard for monitoring:

    - Dataset statistics
    - Model performance
    - Agricultural insights
    - Climate analytics
    """)

    #
    # LOAD DATA
    #

    with st.spinner(
        "Loading dataset..."
    ):

        df = load_data()

        results_df = load_model_results()

    #
    # LOAD LABEL ENCODERS
    #

    label_encoders = joblib.load(
        'models/saved/label_encoders.pkl'
    )

    #
    # REMOVE INVALID LABELS
    #

    df = df[
        (df['crop_encoded'] >= 0)
        &
        (df['state_encoded'] >= 0)
    ]

    #
    # KPI METRICS
    #

    st.markdown(
        '<div class="section-title">'
        '📌 Dataset Overview'
        '</div>',
        unsafe_allow_html=True
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:

        st.markdown(f"""
        <div class="metric-card">

        <h3>Total Samples</h3>

        <h1>{len(df):,}</h1>

        </div>
        """, unsafe_allow_html=True)

    with metric_col2:

        st.markdown(f"""
        <div class="metric-card">

        <h3>Total Features</h3>

        <h1>{df.shape[1]}</h1>

        </div>
        """, unsafe_allow_html=True)

    with metric_col3:

        st.markdown(f"""
        <div class="metric-card">

        <h3>Crop Types</h3>

        <h1>{df['crop_encoded'].nunique()}</h1>

        </div>
        """, unsafe_allow_html=True)

    with metric_col4:

        st.markdown(f"""
        <div class="metric-card">

        <h3>States</h3>

        <h1>{df['state_encoded'].nunique()}</h1>

        </div>
        """, unsafe_allow_html=True)

    #
    # MODEL PERFORMANCE
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🤖 Model Performance Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    fig = px.bar(

        results_df,

        x='Model',

        y='R² Score',

        color='R² Score',

        title='Model Accuracy Comparison',

        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        xaxis_tickangle=-20
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    #
    # RMSE ANALYSIS
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📉 RMSE Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    fig = px.line(

        results_df,

        x='Model',

        y='RMSE',

        markers=True,

        title='RMSE Comparison Across Models'
    )

    fig.update_layout(
        xaxis_tickangle=-20
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    #
    # CROP DISTRIBUTION
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🌾 Crop Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    crop_counts = (

        df['crop_encoded']

        .value_counts()

        .reset_index()
    )

    crop_counts.columns = [
        'crop_encoded',
        'count'
    ]

    # Decode crop names
    crop_counts['crop_name'] = (

        label_encoders['crop']

        .inverse_transform(

            crop_counts['crop_encoded'].astype(int)
        )
    )

    fig = px.pie(

        crop_counts,

        values='count',

        names='crop_name',

        title='Crop Distribution'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    #
    # STATE-WISE ANALYSIS
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📍 State-wise Yield Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    state_yield = (

        df.groupby('state_encoded')['yield_kg_ha']

        .mean()

        .reset_index()
    )

    # Decode state names
    state_yield['state_name'] = (

        label_encoders['state']

        .inverse_transform(

            state_yield['state_encoded'].astype(int)
        )
    )

    fig = px.bar(

        state_yield,

        x='state_name',

        y='yield_kg_ha',

        color='yield_kg_ha',

        title='Average Yield by State',

        labels={

            'state_name': 'State',

            'yield_kg_ha': 'Average Yield (kg/ha)'
        },

        color_continuous_scale='Turbo'
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    #
    # CLIMATE ANALYSIS
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🌤️ Climate vs Yield Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    fig = px.scatter(

        df.sample(
            min(2000, len(df))
        ),

        x='rainfall_mm',

        y='yield_kg_ha',

        color='temperature_C',

        title='Rainfall vs Crop Yield',

        opacity=0.7
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    #
    # SOIL ANALYSIS
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🧪 Soil Nutrient Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    soil_df = pd.DataFrame({

        'Nutrient': [
            'Nitrogen',
            'Phosphorus',
            'Potassium'
        ],

        'Average': [

            df['soil_N_kg_ha'].mean(),

            df['soil_P_kg_ha'].mean(),

            df['soil_K_kg_ha'].mean()
        ]
    })

    fig = px.bar(

        soil_df,

        x='Nutrient',

        y='Average',

        color='Average',

        title='Average Soil Nutrient Levels'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    #
    # CORRELATION HEATMAP
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🔗 Correlation Heatmap'
        '</div>',
        unsafe_allow_html=True
    )

    numeric_cols = (
        df.select_dtypes(
            include=[np.number]
        )
    )

    correlation_matrix = numeric_cols.corr()

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    sns.heatmap(

        correlation_matrix,

        cmap='coolwarm',

        center=0,

        linewidths=0.5,

        ax=ax
    )

    st.pyplot(fig)

    #
    # DATASET PREVIEW
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📄 Dataset Preview'
        '</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    #
    # FOOTER
    #

    st.markdown("---")

    st.markdown("""
    <div style="text-align:center; color:gray;">

    🌾 AI-Powered Crop Yield Dashboard

    <br><br>

    Hybrid Deep Learning + Climate Analytics

    </div>
    """, unsafe_allow_html=True)

#
# RUN DASHBOARD
#

if __name__ == "__main__":

    main()