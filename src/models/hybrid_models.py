# src/models/hybrid_model.py

import os
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


print("\n" + "=" * 70)
print("HYBRID MODEL")
print("=" * 70)


# Load processed dataset
processed_data = pd.read_csv(
    'data/processed/processed_data.csv'
)

feature_columns = [
    col
    for col in processed_data.columns
    if col != 'yield_kg_ha'
]

X = processed_data[
    feature_columns
].values

y = processed_data[
    'yield_kg_ha'
].values


print(
    f"\nDataset Shape: "
    f"{processed_data.shape}"
)

print(
    f"Number of Features: "
    f"{len(feature_columns)}"
)


# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)


# Handle NaN and infinite values
X_train = np.nan_to_num(
    X_train,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_val = np.nan_to_num(
    X_val,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_test = np.nan_to_num(
    X_test,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# Load trained models
rf_model = joblib.load(
    'models/saved/random_forest.pkl'
)

ann_model = load_model(
    'models/saved/ann_model.keras'
)

print("\nModels Loaded Successfully")


# Generate validation predictions
rf_val_predictions = rf_model.predict(
    X_val
)

ann_val_predictions = ann_model.predict(
    X_val,
    verbose=0
).flatten()


# Find the best hybrid weight using validation data
weights = np.arange(
    0.0,
    1.01,
    0.05
)

best_weight = None
best_val_r2 = -np.inf

for rf_weight in weights:

    ann_weight = 1.0 - rf_weight

    hybrid_val_predictions = (
        rf_weight * rf_val_predictions
        + ann_weight * ann_val_predictions
    )

    val_r2 = r2_score(
        y_val,
        hybrid_val_predictions
    )

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_weight = rf_weight


best_ann_weight = 1.0 - best_weight


print("\nBest Hybrid Weights:")
print(
    f"Random Forest : {best_weight:.2f}"
)

print(
    f"ANN           : {best_ann_weight:.2f}"
)

print(
    f"Validation R² : {best_val_r2:.4f}"
)


# Generate final test predictions
rf_test_predictions = rf_model.predict(
    X_test
)

ann_test_predictions = ann_model.predict(
    X_test,
    verbose=0
).flatten()


# Final hybrid prediction
hybrid_test_predictions = (
    best_weight * rf_test_predictions
    + best_ann_weight * ann_test_predictions
)


# Evaluate hybrid model
mae = mean_absolute_error(
    y_test,
    hybrid_test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        hybrid_test_predictions
    )
)

r2 = r2_score(
    y_test,
    hybrid_test_predictions
)


# Evaluate individual models on the same test set
rf_test_r2 = r2_score(
    y_test,
    rf_test_predictions
)

ann_test_r2 = r2_score(
    y_test,
    ann_test_predictions
)

rf_test_mae = mean_absolute_error(
    y_test,
    rf_test_predictions
)

ann_test_mae = mean_absolute_error(
    y_test,
    ann_test_predictions
)

rf_test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_test_predictions
    )
)

ann_test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ann_test_predictions
    )
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print("\nRandom Forest:")
print(
    f"MAE  : {rf_test_mae:.2f}"
)
print(
    f"RMSE : {rf_test_rmse:.2f}"
)
print(
    f"R²   : {rf_test_r2:.4f}"
)

print("\nANN:")
print(
    f"MAE  : {ann_test_mae:.2f}"
)
print(
    f"RMSE : {ann_test_rmse:.2f}"
)
print(
    f"R²   : {ann_test_r2:.4f}"
)

print("\nHybrid Model:")
print(
    f"MAE  : {mae:.2f}"
)
print(
    f"RMSE : {rmse:.2f}"
)
print(
    f"R²   : {r2:.4f}"
)

print(
    f"\nRF Weight  : {best_weight:.2f}"
)

print(
    f"ANN Weight : {best_ann_weight:.2f}"
)

print(
    f"Hybrid Validation R² : "
    f"{best_val_r2:.4f}"
)


# Save hybrid information
os.makedirs(
    'models/saved',
    exist_ok=True
)

hybrid_results = pd.DataFrame({
    'Model': [
        'Random Forest',
        'ANN',
        'Hybrid'
    ],
    'Test R²': [
        rf_test_r2,
        ann_test_r2,
        r2
    ],
    'Test RMSE': [
        rf_test_rmse,
        ann_test_rmse,
        rmse
    ],
    'Test MAE': [
        rf_test_mae,
        ann_test_mae,
        mae
    ]
})

hybrid_results.to_csv(
    'models/saved/hybrid_model_comparison.csv',
    index=False
)


print(
    "\nHybrid comparison saved -> "
    "models/saved/hybrid_model_comparison.csv"
)


print("\n" + "=" * 70)
print("HYBRID MODEL COMPLETED")
print("=" * 70)