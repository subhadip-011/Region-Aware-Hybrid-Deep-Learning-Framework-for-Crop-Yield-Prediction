import os
import pickle
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import load_model


#
# Configuration
#

RAW_PATH = "data/raw/FAO_Crop_data.csv"
PROCESSED_PATH = "data/processed/processed_data.csv"

LSTM_MODEL_PATH = "models/saved/lstm_model.keras"
LSTM_X_SCALER_PATH = "models/saved/lstm_scaler.pkl"
LSTM_Y_SCALER_PATH = "models/saved/lstm_target_scaler.pkl"

RF_MODEL_PATH = "models/saved/rf_lstm_random_forest.pkl"
HYBRID_MODEL_PATH = "models/saved/rf_lstm_hybrid.pkl"
RESULTS_PATH = "models/saved/rf_lstm_hybrid_results.csv"

SEQUENCE_LENGTH = 2


#
# Load data
#

print("=" * 70)
print("RF + REGION-AWARE LSTM HYBRID MODEL")
print("=" * 70)

raw = pd.read_csv(RAW_PATH)
processed = pd.read_csv(PROCESSED_PATH)

print(f"Raw dataset shape       : {raw.shape}")
print(f"Processed dataset shape : {processed.shape}")


#
# Validate row alignment
#

if len(raw) != len(processed):
    raise ValueError(
        "Raw and processed datasets do not have the same number of rows."
    )

raw["agro_zone_encoded"] = processed["agro_zone_encoded"]


#
# Configuration
#

group_columns = [
    "state",
    "district",
    "crop"
]

sequence_features = [
    "rainfall_mm",
    "temperature_C",
    "humidity_pct",
    "soil_N_kg_ha",
    "soil_P_kg_ha",
    "soil_K_kg_ha",
    "soil_pH"
]

target_column = "yield_kg_ha"

rf_features = [
    column
    for column in processed.columns
    if column != target_column
]


#
# Clean raw data
#

raw = raw.dropna(
    subset=[
        "year",
        target_column
    ]
).copy()

raw["year"] = pd.to_numeric(
    raw["year"],
    errors="coerce"
)

raw[target_column] = pd.to_numeric(
    raw[target_column],
    errors="coerce"
)

raw = raw.dropna(
    subset=[
        "year",
        target_column
    ]
).copy()

raw = raw.sort_values(
    group_columns + ["year"]
)


#
# Prepare sequence features
#

for column in sequence_features:

    raw[column] = pd.to_numeric(
        raw[column],
        errors="coerce"
    )

    raw[column] = (
        raw.groupby(group_columns)[column]
        .transform(
            lambda x: x.interpolate(
                method="linear",
                limit_direction="both"
            )
        )
    )

    raw[column] = raw[column].fillna(
        raw[column].median()
    )


#
# Create chronological sequences
#

X_sequences = []
y_values = []
target_years = []
target_indices = []
region_values = []


for _, group in raw.groupby(group_columns):

    group = group.sort_values("year")

    if len(group) < SEQUENCE_LENGTH + 1:
        continue

    group_rows = list(group.index)

    for i in range(
        SEQUENCE_LENGTH,
        len(group)
    ):

        historical = group.iloc[
            i - SEQUENCE_LENGTH:i
        ]

        target_row = group.iloc[i]

        years = historical["year"].values
        target_year = int(target_row["year"])

        # Require truly consecutive years.
        if not (
            years[1] - years[0] == 1
            and target_year - years[1] == 1
        ):
            continue

        sequence = []

        for _, row in historical.iterrows():

            sequence.append([
                row["rainfall_mm"],
                row["temperature_C"],
                row["humidity_pct"],
                row["soil_N_kg_ha"],
                row["soil_P_kg_ha"],
                row["soil_K_kg_ha"],
                row["soil_pH"],
                row[target_column],
                row["agro_zone_encoded"]
            ])

        X_sequences.append(sequence)

        y_values.append(
            target_row[target_column]
        )

        target_years.append(
            target_year
        )

        target_indices.append(
            target_row.name
        )

        region_values.append(
            target_row["agro_zone_encoded"]
        )


X_lstm = np.array(
    X_sequences,
    dtype=np.float32
)

y_lstm = np.array(
    y_values,
    dtype=np.float32
)

target_years = np.array(
    target_years
)

target_indices = np.array(
    target_indices
)

region_values = np.array(
    region_values,
    dtype=np.float32
)


print()
print(
    f"Generated sequences : {len(X_lstm)}"
)


#
# Chronological masks
#

train_mask = target_years <= 2022
val_mask = target_years == 2023
test_mask = target_years == 2024


print()
print("Chronological sequence split:")
print(
    f"Training   : {train_mask.sum()}"
)
print(
    f"Validation : {val_mask.sum()}"
)
print(
    f"Testing    : {test_mask.sum()}"
)


#
# Load LSTM scalers
#

with open(
    LSTM_X_SCALER_PATH,
    "rb"
) as file:

    lstm_x_scaler = pickle.load(file)


with open(
    LSTM_Y_SCALER_PATH,
    "rb"
) as file:

    lstm_y_scaler = pickle.load(file)


#
# Scale LSTM inputs
#

n_features = X_lstm.shape[2]

X_lstm_2d = X_lstm.reshape(
    -1,
    n_features
)

X_lstm_scaled = lstm_x_scaler.transform(
    X_lstm_2d
).reshape(
    X_lstm.shape
)


#
# Load trained region-aware LSTM
#

print()
print("Loading region-aware LSTM...")

lstm_model = load_model(
    LSTM_MODEL_PATH
)


#
# LSTM predictions
#

print("Generating LSTM predictions...")

lstm_predictions_scaled = lstm_model.predict(
    [
        X_lstm_scaled,
        region_values.reshape(-1, 1)
    ],
    verbose=0
).ravel()

lstm_predictions = (
    lstm_y_scaler.inverse_transform(
        lstm_predictions_scaled.reshape(-1, 1)
    ).ravel()
)


#
# Prepare processed RF data
#
# Raw and processed files have the same row order.
#

processed_clean = processed.copy()

processed_clean = processed_clean.replace(
    [np.inf, -np.inf],
    np.nan
)

processed_clean[rf_features] = (
    processed_clean[rf_features]
    .fillna(
        processed_clean[rf_features].median()
    )
)


#
# Get RF training rows
#
# RF is trained only on observations through 2022.
#

raw_years = pd.to_numeric(
    pd.read_csv(RAW_PATH)["year"],
    errors="coerce"
)

rf_train_mask = (
    raw_years <= 2022
) & (
    processed_clean[target_column].notna()
)

rf_train_indices = np.where(
    rf_train_mask
)[0]


X_rf_train = processed_clean.loc[
    rf_train_indices,
    rf_features
]

y_rf_train = processed_clean.loc[
    rf_train_indices,
    target_column
]


#
# Train chronological Random Forest
#

print()
print("Training chronological Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=4,
    max_samples=0.8,
    max_features=1.0,
    max_depth=10,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_rf_train,
    y_rf_train
)


#
# RF predictions for LSTM target rows
#

print("Generating Random Forest predictions...")

rf_target_positions = target_indices.astype(int)

X_rf_targets = processed_clean.loc[
    rf_target_positions,
    rf_features
]

rf_predictions = rf_model.predict(
    X_rf_targets
)


#
# Extract split predictions
#

y_train = y_lstm[train_mask]
y_val = y_lstm[val_mask]
y_test = y_lstm[test_mask]

rf_train_pred = rf_predictions[train_mask]
rf_val_pred = rf_predictions[val_mask]
rf_test_pred = rf_predictions[test_mask]

lstm_train_pred = lstm_predictions[train_mask]
lstm_val_pred = lstm_predictions[val_mask]
lstm_test_pred = lstm_predictions[test_mask]


#
# Validation-based hybrid weight selection
#

print()
print("Searching RF + LSTM hybrid weights...")

best_weight = None
best_r2 = -np.inf

weight_results = []

for rf_weight in np.arange(
    0.0,
    1.01,
    0.05
):

    lstm_weight = 1.0 - rf_weight

    hybrid_val_pred = (
        rf_weight * rf_val_pred
        + lstm_weight * lstm_val_pred
    )

    r2 = r2_score(
        y_val,
        hybrid_val_pred
    )

    weight_results.append([
        rf_weight,
        lstm_weight,
        r2
    ])

    if r2 > best_r2:

        best_r2 = r2
        best_weight = rf_weight


best_rf_weight = float(
    best_weight
)

best_lstm_weight = (
    1.0 - best_rf_weight
)


print()
print("=" * 70)
print("BEST HYBRID WEIGHTS")
print("=" * 70)

print(
    f"Random Forest : {best_rf_weight:.2f}"
)

print(
    f"LSTM          : {best_lstm_weight:.2f}"
)

print(
    f"Validation R² : {best_r2:.4f}"
)


#
# Final hybrid predictions
#

hybrid_train_pred = (
    best_rf_weight * rf_train_pred
    + best_lstm_weight * lstm_train_pred
)

hybrid_val_pred = (
    best_rf_weight * rf_val_pred
    + best_lstm_weight * lstm_val_pred
)

hybrid_test_pred = (
    best_rf_weight * rf_test_pred
    + best_lstm_weight * lstm_test_pred
)


#
# Evaluation function
#

def calculate_metrics(
    y_true,
    predictions
):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return mae, rmse, r2


#
# Calculate metrics
#

rf_test_metrics = calculate_metrics(
    y_test,
    rf_test_pred
)

lstm_test_metrics = calculate_metrics(
    y_test,
    lstm_test_pred
)

hybrid_test_metrics = calculate_metrics(
    y_test,
    hybrid_test_pred
)


#
# Print final comparison
#

print()
print("=" * 70)
print("2024 TEST SET COMPARISON")
print("=" * 70)

print()
print("Random Forest")
print(
    f"MAE  : {rf_test_metrics[0]:.2f}"
)
print(
    f"RMSE : {rf_test_metrics[1]:.2f}"
)
print(
    f"R²   : {rf_test_metrics[2]:.4f}"
)

print()
print("Region-Aware LSTM")
print(
    f"MAE  : {lstm_test_metrics[0]:.2f}"
)
print(
    f"RMSE : {lstm_test_metrics[1]:.2f}"
)
print(
    f"R²   : {lstm_test_metrics[2]:.4f}"
)

print()
print("RF + LSTM HYBRID")
print(
    f"MAE  : {hybrid_test_metrics[0]:.2f}"
)
print(
    f"RMSE : {hybrid_test_metrics[1]:.2f}"
)
print(
    f"R²   : {hybrid_test_metrics[2]:.4f}"
)


#
# Save RF model
#

os.makedirs(
    "models/saved",
    exist_ok=True
)

joblib.dump(
    rf_model,
    RF_MODEL_PATH
)


#
# Save hybrid configuration
#

hybrid_config = {
    "rf_weight": best_rf_weight,
    "lstm_weight": best_lstm_weight,
    "sequence_length": SEQUENCE_LENGTH,
    "test_year": 2024
}

with open(
    HYBRID_MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        hybrid_config,
        file
    )


#
# Save results
#

results = pd.DataFrame({

    "model": [
        "Random Forest",
        "Region-Aware LSTM",
        "RF + LSTM Hybrid"
    ],

    "MAE": [
        rf_test_metrics[0],
        lstm_test_metrics[0],
        hybrid_test_metrics[0]
    ],

    "RMSE": [
        rf_test_metrics[1],
        lstm_test_metrics[1],
        hybrid_test_metrics[1]
    ],

    "R2": [
        rf_test_metrics[2],
        lstm_test_metrics[2],
        hybrid_test_metrics[2]
    ],

    "RF_weight": [
        1.0,
        0.0,
        best_rf_weight
    ],

    "LSTM_weight": [
        0.0,
        1.0,
        best_lstm_weight
    ]
})

results.to_csv(
    RESULTS_PATH,
    index=False
)


print()
print("=" * 70)
print("RF + LSTM HYBRID SAVED")
print("=" * 70)

print(
    RF_MODEL_PATH
)

print(
    HYBRID_MODEL_PATH
)

print(
    RESULTS_PATH
)

print()
print("=" * 70)
print("RF + LSTM HYBRID TRAINING COMPLETED")
print("=" * 70)