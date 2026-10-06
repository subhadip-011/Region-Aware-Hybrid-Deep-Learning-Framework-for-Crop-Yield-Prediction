# streamlit_app/pages/3_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#
# PAGE CONFIGURATION
#

st.set_page_config(
    page_title="Analysis",
    page_icon="📈",
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
    font-size: 1.5rem;
    font-weight: bold;
    color: #2E7D32;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.metric-card {
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 100%
    );

    padding: 1rem;

    border-radius: 1rem;

    color: white;

    text-align: center;
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
# MAIN FUNCTION
#

def main():

    #
    # TITLE
    #

    st.markdown(
        '<div class="main-title">'
        '📈 Agricultural Data Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    Comprehensive agricultural analysis using:

    - Climate data
    - Soil data
    - Crop yield trends
    - Machine Learning insights
    """)

    #
    # LOAD DATA
    #

    with st.spinner(
        "Loading dataset..."
    ):

        df = load_data()

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
    # DATASET METRICS
    #

    st.markdown(
        '<div class="section-title">'
        '📊 Dataset Statistics'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(f"""
        <div class="metric-card">

        <h3>Total Samples</h3>

        <h1>{len(df):,}</h1>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">

        <h3>Total Features</h3>

        <h1>{df.shape[1]}</h1>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">

        <h3>Crop Types</h3>

        <h1>{df['crop_encoded'].nunique()}</h1>

        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown(f"""
        <div class="metric-card">

        <h3>States</h3>

        <h1>{df['state_encoded'].nunique()}</h1>

        </div>
        """, unsafe_allow_html=True)

    #
    # YIELD DISTRIBUTION
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🌾 Yield Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    fig = px.histogram(

        df,

        x='yield_kg_ha',

        nbins=50,

        title='Crop Yield Distribution',

        color_discrete_sequence=['green']
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    #
    # CROP-WISE ANALYSIS
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🌱 Crop-wise Yield Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    crop_yield = (

        df.groupby('crop_encoded')['yield_kg_ha']

        .mean()

        .reset_index()
    )

    # Decode crop names safely
    crop_yield['crop_name'] = (

        label_encoders['crop']

        .inverse_transform(

            crop_yield['crop_encoded'].astype(int)
        )
    )

    fig = px.bar(

        crop_yield,

        x='crop_name',

        y='yield_kg_ha',

        color='yield_kg_ha',

        title='Average Yield by Crop',

        labels={

            'crop_name': 'Crop Type',

            'yield_kg_ha': 'Average Yield (kg/ha)'
        },

        color_continuous_scale='Viridis'
    )

    fig.update_layout(

        xaxis_tickangle=-45,

        title_x=0.3
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

    # Decode state names safely
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

        xaxis_tickangle=-45,

        title_x=0.3
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
        '🌤️ Climate Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(

            df.sample(
                min(2000, len(df))
            ),

            x='rainfall_mm',

            y='yield_kg_ha',

            color='temperature_C',

            title='Rainfall vs Yield',

            opacity=0.7
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(

            df.sample(
                min(2000, len(df))
            ),

            x='temperature_C',

            y='yield_kg_ha',

            color='humidity_pct',

            title='Temperature vs Yield',

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

        'Average Value': [

            df['soil_N_kg_ha'].mean(),

            df['soil_P_kg_ha'].mean(),

            df['soil_K_kg_ha'].mean()
        ]
    })

    fig = px.bar(

        soil_df,

        x='Nutrient',

        y='Average Value',

        color='Average Value',

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
    # YEARLY TREND ANALYSIS
    #

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📅 Yearly Yield Trend'
        '</div>',
        unsafe_allow_html=True
    )

    yearly_trend = (

        df.groupby('year')['yield_kg_ha']

        .mean()

        .reset_index()
    )

    fig = px.line(

        yearly_trend,

        x='year',

        y='yield_kg_ha',

        markers=True,

        title='Yearly Yield Trend'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

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

    model_df = pd.DataFrame({

        'Model': [

            'Random Forest',

            'XGBoost',

            'ANN',

            'Hybrid RF + ANN'
        ],

        'R² Score': [

            0.9364,

            0.9355,

            0.9400,

            0.9426
        ]
    })

    fig = px.line(

        model_df,

        x='Model',

        y='R² Score',

        markers=True,

        title='Model Accuracy Comparison'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

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

    📈 Agricultural Intelligence Analysis System

    <br><br>

    AI + Climate Analytics + Soil Intelligence

    </div>
    """, unsafe_allow_html=True)

#
# RUN PAGE
#

if __name__ == "__main__":

    main()