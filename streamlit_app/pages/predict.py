# streamlit_app/pages/predict.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

#
# PAGE CONFIGURATION
#

st.set_page_config(
    page_title="Predict Crop Yield",
    page_icon="🔮",
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

.prediction-box {
    background: linear-gradient(
        135deg,
        #11998e 0%,
        #38ef7d 100%
    );

    padding: 2rem;

    border-radius: 1rem;

    color: white;

    text-align: center;

    margin-top: 1rem;
}

.info-box {
    background-color: #f5f5f5;

    padding: 1rem;

    border-radius: 0.8rem;

    margin-top: 1rem;
}

</style>
""", unsafe_allow_html=True)

#
# LOAD MODELS
#

@st.cache_resource
def load_models():

    best_model = joblib.load(
        'models/saved/best_model.pkl'
    )

    scaler = joblib.load(
        'models/saved/scaler.pkl'
    )

    label_encoders = joblib.load(
        'models/saved/label_encoders.pkl'
    )

    feature_columns = joblib.load(
        'models/saved/feature_columns.pkl'
    )

    return (
        best_model,
        scaler,
        label_encoders,
        feature_columns
    )

#
# MAIN FUNCTION
#

def main():

    #
    # TITLE
    #

    st.markdown(
        '<div class="main-title">'
        '🔮 Crop Yield Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    Predict agricultural crop yield using the best trained Machine Learning model.

    The prediction system analyzes:

    - Climate conditions
    - Soil nutrients
    - Region information
    - Crop type
    - Environmental factors
    """)

    #
    # LOAD MODELS
    #

    (
        best_model,
        scaler,
        label_encoders,
        feature_columns
    ) = load_models()

    #
    # SIDEBAR INPUTS
    #

    st.sidebar.header(
        "📥 Enter Agricultural Parameters"
    )

    #
    # CLIMATE INPUTS
    #

    rainfall = st.sidebar.slider(
        "Rainfall (mm)",
        0.0,
        5000.0,
        1200.0
    )

    temperature = st.sidebar.slider(
        "Temperature (°C)",
        0.0,
        50.0,
        28.0
    )

    humidity = st.sidebar.slider(
        "Humidity (%)",
        0.0,
        100.0,
        70.0
    )

    #
    # SOIL INPUTS
    #

    soil_n = st.sidebar.slider(
        "Nitrogen (N)",
        0.0,
        300.0,
        120.0
    )

    soil_p = st.sidebar.slider(
        "Phosphorus (P)",
        0.0,
        300.0,
        80.0
    )

    soil_k = st.sidebar.slider(
        "Potassium (K)",
        0.0,
        300.0,
        100.0
    )

    soil_ph = st.sidebar.slider(
        "Soil pH",
        0.0,
        14.0,
        6.5
    )

    #
    # YEAR
    #

    year = st.sidebar.slider(
        "Year",
        2000,
        2035,
        2025
    )

    #
    # DROPDOWN OPTIONS
    #

    crop_options = list(
        label_encoders['crop'].classes_
    )

    state_options = list(
        label_encoders['state'].classes_
    )

    district_options = list(
        label_encoders['district'].classes_
    )

    agro_options = list(
        label_encoders['agro_zone'].classes_
    )

    #
    # CATEGORICAL INPUTS
    #

    crop = st.sidebar.selectbox(
        "Crop Type",
        crop_options
    )

    state = st.sidebar.selectbox(
        "State",
        state_options
    )

    district = st.sidebar.selectbox(
        "District",
        district_options
    )

    agro_zone = st.sidebar.selectbox(
        "Agro Climatic Zone",
        agro_options
    )

    #
    # FEATURE ENGINEERING
    #

    temp_rainfall = (
        temperature * rainfall
    )

    soil_fertility = (
        (soil_n + soil_p + soil_k) / 3
    )

    water_availability = (
        rainfall * humidity
    )

    climate_stress = (
        temperature / (humidity + 1)
    )

    soil_ph_quality = (
        abs(7 - soil_ph)
    )

    humidity_temp_ratio = (
        humidity / (temperature + 1)
    )

    #
    # ENCODE CATEGORICAL VALUES
    #

    crop_encoded = (
        label_encoders['crop']
        .transform([crop])[0]
    )

    state_encoded = (
        label_encoders['state']
        .transform([state])[0]
    )

    district_encoded = (
        label_encoders['district']
        .transform([district])[0]
    )

    agro_encoded = (
        label_encoders['agro_zone']
        .transform([agro_zone])[0]
    )

    #
    # CREATE INPUT DATAFRAME
    #

    input_df = pd.DataFrame({

        'rainfall_mm': [rainfall],

        'temperature_C': [temperature],

        'humidity_pct': [humidity],

        'soil_N_kg_ha': [soil_n],

        'soil_P_kg_ha': [soil_p],

        'soil_K_kg_ha': [soil_k],

        'soil_pH': [soil_ph],

        'year': [year],

        'state_encoded': [state_encoded],

        'district_encoded': [district_encoded],

        'crop_encoded': [crop_encoded],

        'agro_zone_encoded': [agro_encoded],

        'temp_rainfall': [temp_rainfall],

        'soil_fertility': [soil_fertility],

        'water_availability': [water_availability],

        'climate_stress': [climate_stress],

        'soil_ph_quality': [soil_ph_quality],

        'humidity_temp_ratio': [humidity_temp_ratio]
    })

    #
    # ENSURE COLUMN ORDER
    #

    input_df = input_df[
        feature_columns
    ]

    #
    # SCALE INPUT
    #

    input_scaled = scaler.transform(
        input_df
    )

    #
    # PREDICTION BUTTON
    #

    if st.button(
        "🚀 Predict Crop Yield"
    ):

        prediction = (
            best_model.predict(
                input_scaled
            )[0]
        )

        #
        # PREDICTION RESULT
        #

        st.markdown(f"""
        <div class="prediction-box">

        <h2>🌾 Predicted Crop Yield</h2>

        <h1>{prediction:,.2f} kg/ha</h1>

        </div>
        """, unsafe_allow_html=True)

        #
        # YIELD CATEGORY
        #

        if prediction < 3000:

            category = "Low Yield"

        elif prediction < 7000:

            category = "Moderate Yield"

        else:

            category = "High Yield"

        st.markdown(f"""
        <div class="info-box">

        <h3>📊 Yield Category</h3>

        <h2>{category}</h2>

        </div>
        """, unsafe_allow_html=True)

        #
        # INPUT SUMMARY
        #

        st.markdown("## 📋 Input Summary")

        summary_df = pd.DataFrame({

            'Feature': [

                'Crop',

                'State',

                'District',

                'Agro Zone',

                'Rainfall',

                'Temperature',

                'Humidity',

                'Nitrogen',

                'Phosphorus',

                'Potassium',

                'Soil pH'
            ],

            'Value': [

                crop,

                state,

                district,

                agro_zone,

                rainfall,

                temperature,

                humidity,

                soil_n,

                soil_p,

                soil_k,

                soil_ph
            ]
        })

        st.dataframe(
            summary_df,
            use_container_width=True
        )

        #
        # FEATURE VISUALIZATION
        #

        st.markdown("## 📈 Environmental Parameters")

        viz_df = pd.DataFrame({

            'Feature': [

                'Rainfall',

                'Temperature',

                'Humidity',

                'Nitrogen',

                'Phosphorus',

                'Potassium'
            ],

            'Value': [

                rainfall,

                temperature,

                humidity,

                soil_n,

                soil_p,

                soil_k
            ]
        })

        fig = px.bar(

            viz_df,

            x='Feature',

            y='Value',

            color='Value',

            title='Input Environmental Conditions',

            color_continuous_scale='Viridis'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        #
        # SUCCESS MESSAGE
        #

        st.success(
            "Prediction generated successfully using the best trained ML model."
        )

#
# RUN PAGE
#

if __name__ == "__main__":

    main()