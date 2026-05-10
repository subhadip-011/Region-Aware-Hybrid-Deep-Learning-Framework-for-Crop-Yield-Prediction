# Fixed Complete Preprocessor Code for Crop Yield Prediction

# src/data/preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings
import os

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Complete Data Preprocessing Pipeline
    """

    def __init__(self, data_path='data/raw/FAO_Crop_data.csv'):
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None

    # =========================================================
    # LOAD DATA
    # =========================================================

    def load_data(self):
        """Load dataset"""

        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)

        self.data = pd.read_csv(self.data_path)

        print(f"✅ Data Loaded Successfully")
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Columns: {self.data.columns.tolist()}")

        return self.data

    # =========================================================
    # EXPLORE DATA
    # =========================================================

    def explore_data(self):
        """Explore dataset"""

        print("\n" + "=" * 60)
        print("DATA EXPLORATION")
        print("=" * 60)

        print("\nMissing Values:")
        print(self.data.isnull().sum())

        print("\nData Types:")
        print(self.data.dtypes)

        print("\nFirst 5 Rows:")
        print(self.data.head())

    # =========================================================
    # HANDLE MISSING VALUES
    # =========================================================

    def handle_missing_values(self):
        """Fill missing values"""

        print("\n" + "=" * 60)
        print("HANDLING MISSING VALUES")
        print("=" * 60)

        missing_before = self.data.isnull().sum().sum()
        print(f"Missing Values Before: {missing_before}")

        # Numerical columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if self.data[col].isnull().sum() > 0:
                median_value = self.data[col].median()
                self.data[col] = self.data[col].fillna(median_value)
                print(f"Filled {col} with median: {median_value}")

        # Categorical columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if self.data[col].isnull().sum() > 0:
                mode_value = self.data[col].mode()[0]
                self.data[col] = self.data[col].fillna(mode_value)
                print(f"Filled {col} with mode: {mode_value}")

        missing_after = self.data.isnull().sum().sum()

        print(f"✅ Missing Values After: {missing_after}")

        return self.data

    # =========================================================
    # REMOVE DUPLICATES
    # =========================================================

    def remove_duplicates(self):
        """Remove duplicate rows"""

        print("\n" + "=" * 60)
        print("REMOVING DUPLICATES")
        print("=" * 60)

        before = self.data.shape[0]

        self.data = self.data.drop_duplicates()

        after = self.data.shape[0]

        print(f"✅ Removed {before - after} duplicate rows")

        return self.data

    # =========================================================
    # HANDLE OUTLIERS
    # =========================================================

    def handle_outliers(self):
        """Handle outliers using IQR method"""

        print("\n" + "=" * 60)
        print("HANDLING OUTLIERS")
        print("=" * 60)

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:

            if col == 'yield_kg_ha':
                continue

            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (
                (self.data[col] < lower_bound) |
                (self.data[col] > upper_bound)
            ).sum()

            self.data[col] = self.data[col].clip(lower_bound, upper_bound)

            if outliers > 0:
                print(f"Capped {outliers} outliers in {col}")

        print("✅ Outlier Handling Completed")

        return self.data

    # =========================================================
    # ENCODE CATEGORICAL VARIABLES
    # =========================================================

    def encode_categorical(self):
        """Encode categorical variables"""

        print("\n" + "=" * 60)
        print("ENCODING CATEGORICAL VARIABLES")
        print("=" * 60)

        categorical_cols = ['state', 'district', 'crop']

        for col in categorical_cols:

            if col in self.data.columns:

                self.data[col] = self.data[col].fillna('Unknown')

                encoder = LabelEncoder()

                self.data[f'{col}_encoded'] = encoder.fit_transform(
                    self.data[col].astype(str)
                )

                self.label_encoders[col] = encoder

                print(f"Encoded {col} successfully")

        return self.data

    # =========================================================
    # CREATE REGION FEATURES
    # =========================================================

    def create_region_features(self):
        """Create agro climatic zone features"""

        print("\n" + "=" * 60)
        print("CREATING REGION FEATURES")
        print("=" * 60)

        zone_mapping = {
            'Northern': ['Punjab', 'Haryana', 'Uttar Pradesh', 'Uttarakhand', 'Himachal Pradesh'],
            'Southern': ['Tamil Nadu', 'Kerala', 'Karnataka', 'Andhra Pradesh', 'Telangana'],
            'Eastern': ['West Bengal', 'Odisha', 'Bihar', 'Jharkhand', 'Assam'],
            'Western': ['Gujarat', 'Maharashtra', 'Rajasthan', 'Goa'],
            'Central': ['Madhya Pradesh', 'Chhattisgarh']
        }

        def get_zone(state):
            for zone, states in zone_mapping.items():
                if state in states:
                    return zone
            return 'Other'

        self.data['agro_zone'] = self.data['state'].apply(get_zone)

        zone_encoder = LabelEncoder()

        self.data['agro_zone_encoded'] = zone_encoder.fit_transform(
            self.data['agro_zone']
        )

        self.label_encoders['agro_zone'] = zone_encoder

        print("✅ Agro Climatic Zones Created")

        return self.data

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================

    def feature_engineering(self):
        """Create advanced features"""

        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        # Temperature and rainfall interaction
        self.data['temp_rainfall'] = (
            self.data['temperature_C'] * self.data['rainfall_mm']
        )

        # Soil fertility score
        self.data['soil_fertility'] = (
            self.data['soil_N_kg_ha'] * 0.4 +
            self.data['soil_P_kg_ha'] * 0.3 +
            self.data['soil_K_kg_ha'] * 0.3
        )

        # Water availability
        self.data['water_availability'] = (
            self.data['rainfall_mm'] * self.data['humidity_pct'] / 1000
        )

        # Climate stress
        self.data['climate_stress'] = (
            np.maximum(0, self.data['temperature_C'] - 30) *
            np.maximum(0, 100 - self.data['humidity_pct']) / 100
        )

        # Soil pH quality
        self.data['soil_ph_quality'] = (
            1 - np.abs(self.data['soil_pH'] - 6.75) / 3.5
        )

        self.data['soil_ph_quality'] = self.data['soil_ph_quality'].clip(0, 1)

        # Safe humidity temperature ratio
        self.data['humidity_temp_ratio'] = np.where(
            self.data['temperature_C'] != 0,
            self.data['humidity_pct'] / self.data['temperature_C'],
            0
        )

        # Replace infinity values
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill remaining NaN values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            self.data[col] = self.data[col].fillna(self.data[col].median())

        print("✅ Feature Engineering Completed")

        return self.data

    # =========================================================
    # PREPARE FEATURES
    # =========================================================

    def prepare_features(self):
        """Prepare final feature list"""

        print("\n" + "=" * 60)
        print("PREPARING FEATURES")
        print("=" * 60)

        base_features = [
            'rainfall_mm',
            'temperature_C',
            'humidity_pct',
            'soil_N_kg_ha',
            'soil_P_kg_ha',
            'soil_K_kg_ha',
            'soil_pH',
            'year'
        ]

        encoded_features = [
            'state_encoded',
            'district_encoded',
            'crop_encoded',
            'agro_zone_encoded'
        ]

        engineered_features = [
            'temp_rainfall',
            'soil_fertility',
            'water_availability',
            'climate_stress',
            'soil_ph_quality',
            'humidity_temp_ratio'
        ]

        self.feature_columns = (
            base_features +
            encoded_features +
            engineered_features
        )

        self.feature_columns = [
            col for col in self.feature_columns
            if col in self.data.columns
        ]

        print(f"Using {len(self.feature_columns)} Features")

        return self.data[self.feature_columns]

    # =========================================================
    # SCALE FEATURES
    # =========================================================

    def scale_features(self, X):
        """Scale features safely"""

        print("\n" + "=" * 60)
        print("SCALING FEATURES")
        print("=" * 60)

        # Remove infinity
        X = X.replace([np.inf, -np.inf], np.nan)

        # Fill NaN
        X = X.fillna(X.median())

        # Final safety check
        print(f"NaN Values Before Scaling: {X.isnull().sum().sum()}")

        X_scaled = self.scaler.fit_transform(X)

        print("✅ Feature Scaling Completed")

        return X_scaled

    # =========================================================
    # SPLIT DATA
    # =========================================================

    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split dataset"""

        print("\n" + "=" * 60)
        print("SPLITTING DATA")
        print("=" * 60)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

        val_ratio = val_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio,
            random_state=42
        )

        print(f"Training Shape: {X_train.shape}")
        print(f"Validation Shape: {X_val.shape}")
        print(f"Test Shape: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    # =========================================================
    # SAVE ARTIFACTS
    # =========================================================

    def save_artifacts(self, save_path='models/saved/'):
        """Save scaler and encoders"""

        os.makedirs(save_path, exist_ok=True)

        joblib.dump(self.scaler, f'{save_path}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{save_path}/label_encoders.pkl')
        joblib.dump(self.feature_columns, f'{save_path}/feature_columns.pkl')

        print("✅ Saved Preprocessing Artifacts")

    # =========================================================
    # COMPLETE PIPELINE
    # =========================================================

    def run_pipeline(self):
        """Run complete preprocessing pipeline"""

        print("\n" + "=" * 70)
        print("STARTING COMPLETE PREPROCESSING PIPELINE")
        print("=" * 70)

        # Load
        self.load_data()

        # Explore
        self.explore_data()

        # Clean
        self.handle_missing_values()
        self.remove_duplicates()
        self.handle_outliers()

        # Encode
        self.encode_categorical()

        # Region Features
        self.create_region_features()

        # Feature Engineering
        self.feature_engineering()

        # Prepare Features
        X = self.prepare_features()

        # Target Variable
        y = self.data['yield_kg_ha'].copy()

        # Remove invalid target rows
        valid_rows = ~y.isnull()

        X = X[valid_rows]
        y = y[valid_rows]

        # Scale
        X_scaled = self.scale_features(X)

        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X_scaled,
            y
        )

        # Save artifacts
        self.save_artifacts()

        # Save processed dataset
        os.makedirs('data/processed', exist_ok=True)

        processed_data = pd.DataFrame(
            X_scaled,
            columns=self.feature_columns
        )

        processed_data['yield_kg_ha'] = y.values

        processed_data.to_csv(
            'data/processed/processed_data.csv',
            index=False
        )

        print("✅ Processed Data Saved")

        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 70)

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':

    preprocessor = DataPreprocessor(
        'data/raw/FAO_Crop_data.csv'
    )

    X_train, X_val, X_test, y_train, y_val, y_test = (
        preprocessor.run_pipeline()
    )

    print("\nFinal Dataset Shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

