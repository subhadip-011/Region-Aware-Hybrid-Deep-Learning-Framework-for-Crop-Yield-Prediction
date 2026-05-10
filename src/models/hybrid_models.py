#src/models/hybrid_model.py

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model


# =========================================================
# LOAD DATA
# =========================================================

print("\n" + "=" * 70)
print("HYBRID MODEL")
print("=" * 70)

processed_data = pd.read_csv(
    'data/processed/processed_data.csv'
)

feature_columns = [
    col for col in processed_data.columns
    if col != 'yield_kg_ha'
]

X = processed_data[feature_columns].values

y = processed_data['yield_kg_ha'].values

# =========================================================
# SPLIT DATA
# =========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

# =========================================================
# HANDLE NaN VALUES
# =========================================================

X_test = np.nan_to_num(X_test)

# =========================================================
# LOAD MODELS
# =========================================================

rf_model = joblib.load(
    'models/saved/random_forest.pkl'
)

ann_model = load_model(
    'models/saved/ann_model.keras'
)

print("✅ Models Loaded Successfully")

# =========================================================
# PREDICTIONS
# =========================================================

rf_predictions = rf_model.predict(X_test)

ann_predictions = ann_model.predict(X_test).flatten()

# =========================================================
# HYBRID PREDICTION
# =========================================================

hybrid_predictions = (
    0.5 * rf_predictions +
    0.5 * ann_predictions
)

# =========================================================
# EVALUATE HYBRID MODEL
# =========================================================

mae = mean_absolute_error(y_test, hybrid_predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, hybrid_predictions)
)

r2 = r2_score(y_test, hybrid_predictions)

print("\n" + "=" * 70)
print("HYBRID MODEL PERFORMANCE")
print("=" * 70)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

print("\n" + "=" * 70)
print("HYBRID MODEL COMPLETED")
print("=" * 70)

# Run ANN Model


# Final Hybrid Formula
'''python
Hybrid prediction formula:

0.5 × Random Forest + 0.5 × ANN
'''