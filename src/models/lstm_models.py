import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# Configuration
# ============================================================

RAW_PATH = "data/raw/FAO_Crop_data.csv"
PROCESSED_PATH = "data/processed/processed_data.csv"

MODEL_PATH = "models/saved/lstm_model.keras"
X_SCALER_PATH = "models/saved/lstm_scaler.pkl"
Y_SCALER_PATH = "models/saved/lstm_target_scaler.pkl"
METRICS_PATH = "models/saved/lstm_metrics.csv"

SEQUENCE_LENGTH = 2
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("REGION-AWARE LSTM TIME-SERIES CROP YIELD MODEL")
print("=" * 70)

raw = pd.read_csv(RAW_PATH)
processed = pd.read_csv(PROCESSED_PATH)

print(f"Raw dataset shape       : {raw.shape}")
print(f"Processed dataset shape : {processed.shape}")


# ============================================================
# Attach agro-climatic region information
# ============================================================

if len(raw) != len(processed):
    raise ValueError(
        "Raw and processed datasets have different numbers of rows."
    )

if "agro_zone_encoded" not in processed.columns:
    raise ValueError(
        "agro_zone_encoded was not found in processed_data.csv"
    )

raw["agro_zone_encoded"] = processed["agro_zone_encoded"]


# ============================================================
# Configuration
# ============================================================

group_columns = [
    "state",
    "district",
    "crop"
]

numeric_features = [
    "rainfall_mm",
    "temperature_C",
    "humidity_pct",
    "soil_N_kg_ha",
    "soil_P_kg_ha",
    "soil_K_kg_ha",
    "soil_pH"
]

target_column = "yield_kg_ha"

required_columns = (
    group_columns
    + numeric_features
    + [
        "year",
        "agro_zone_encoded",
        target_column
    ]
)

missing_columns = [
    col for col in required_columns
    if col not in raw.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# Basic cleaning
# ============================================================

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
).reset_index(drop=True)


print(f"Rows with valid yield : {len(raw)}")


# ============================================================
# Feature preparation
# ============================================================

for column in numeric_features:
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


# ============================================================
# Generate valid time-series sequences
#
# Each sequence:
#
# Year t-2
# Year t-1
#       ↓
# Predict Year t
#
# Historical yield is included as an input.
# ============================================================

X_sequences = []
y_values = []
target_years = []
region_values = []
crop_values = []

for _, group in raw.groupby(group_columns):

    group = group.sort_values("year").reset_index(drop=True)

    if len(group) < SEQUENCE_LENGTH + 1:
        continue

    for i in range(SEQUENCE_LENGTH, len(group)):

        historical = group.iloc[
            i - SEQUENCE_LENGTH:i
        ]

        target_row = group.iloc[i]

        years = historical["year"].values
        target_year = int(target_row["year"])

        # Require consecutive historical years.
        # Example:
        # 2020 -> 2021 -> predict 2022
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

        target_years.append(target_year)

        region_values.append(
            target_row["agro_zone_encoded"]
        )

        crop_values.append(
            target_row["crop"]
        )


X = np.array(
    X_sequences,
    dtype=np.float32
)

y = np.array(
    y_values,
    dtype=np.float32
)

target_years = np.array(
    target_years
)

region_values = np.array(
    region_values
)

crop_values = np.array(
    crop_values
)


print(f"Generated valid sequences : {len(X)}")


# ============================================================
# Chronological split
#
# Train      : target year <= 2022
# Validation : target year == 2023
# Test       : target year == 2024
# ============================================================

train_mask = target_years <= 2022
val_mask = target_years == 2023
test_mask = target_years == 2024

X_train = X[train_mask]
X_val = X[val_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_val = y[val_mask]
y_test = y[test_mask]

print()
print("Chronological split:")
print(f"Training sequences   : {len(X_train)}")
print(f"Validation sequences : {len(X_val)}")
print(f"Testing sequences    : {len(X_test)}")


if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError(
        "One of the chronological splits contains no sequences."
    )


# ============================================================
# Scale sequence features
#
# Fit ONLY on training data.
# ============================================================

n_features = X_train.shape[2]

X_scaler = StandardScaler()

X_train_2d = X_train.reshape(
    -1,
    n_features
)

X_val_2d = X_val.reshape(
    -1,
    n_features
)

X_test_2d = X_test.reshape(
    -1,
    n_features
)

X_scaler.fit(X_train_2d)

X_train = X_scaler.transform(
    X_train_2d
).reshape(X_train.shape)

X_val = X_scaler.transform(
    X_val_2d
).reshape(X_val.shape)

X_test = X_scaler.transform(
    X_test_2d
).reshape(X_test.shape)


# ============================================================
# Scale target
#
# Yield values can be very large, so target scaling
# improves neural-network optimization.
# ============================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
).ravel()

y_val_scaled = y_scaler.transform(
    y_val.reshape(-1, 1)
).ravel()

y_test_scaled = y_scaler.transform(
    y_test.reshape(-1, 1)
).ravel()


# ============================================================
# Build region-aware LSTM
# ============================================================

sequence_input = Input(
    shape=(SEQUENCE_LENGTH, n_features),
    name="sequence_input"
)

x = LSTM(
    64,
    activation="tanh",
    return_sequences=False
)(sequence_input)

x = Dropout(0.20)(x)

x = Dense(
    32,
    activation="relu"
)(x)

x = Dropout(0.15)(x)


# ============================================================
# Explicit region input
#
# Agro-climatic zone is supplied separately so the model
# receives regional context in addition to the sequence.
# ============================================================

region_input = Input(
    shape=(1,),
    name="region_input"
)

region_dense = Dense(
    8,
    activation="relu"
)(region_input)


# ============================================================
# Combine temporal + regional information
# ============================================================

combined = Concatenate()([
    x,
    region_dense
])

combined = Dense(
    32,
    activation="relu"
)(combined)

combined = Dropout(0.15)(combined)

output = Dense(
    1,
    activation="linear",
    name="yield_output"
)(combined)


model = Model(
    inputs=[
        sequence_input,
        region_input
    ],
    outputs=output
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)


print()
print("Model architecture:")
model.summary()


# ============================================================
# Callbacks
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=1e-5,
    verbose=1
)


# ============================================================
# Train
# ============================================================

print()
print("Training region-aware LSTM...")

history = model.fit(
    [
        X_train,
        region_values[train_mask].reshape(-1, 1)
    ],
    y_train_scaled,

    validation_data=(
        [
            X_val,
            region_values[val_mask].reshape(-1, 1)
        ],
        y_val_scaled
    ),

    epochs=150,
    batch_size=32,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# Prediction function
# ============================================================

def predict_original_scale(
    X_data,
    regions
):
    predictions_scaled = model.predict(
        [
            X_data,
            regions.reshape(-1, 1)
        ],
        verbose=0
    ).ravel()

    predictions = y_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).ravel()

    return predictions


# ============================================================
# Evaluate
# ============================================================

def evaluate_model(
    name,
    X_data,
    y_true,
    regions
):

    predictions = predict_original_scale(
        X_data,
        regions
    )

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

    print()
    print("=" * 60)
    print(f"{name} PERFORMANCE")
    print("=" * 60)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return mae, rmse, r2


train_metrics = evaluate_model(
    "TRAIN",
    X_train,
    y_train,
    region_values[train_mask]
)

val_metrics = evaluate_model(
    "VALIDATION",
    X_val,
    y_val,
    region_values[val_mask]
)

test_metrics = evaluate_model(
    "TEST",
    X_test,
    y_test,
    region_values[test_mask]
)


# ============================================================
# Save model and scalers
# ============================================================

os.makedirs(
    "models/saved",
    exist_ok=True
)

model.save(
    MODEL_PATH
)

with open(
    X_SCALER_PATH,
    "wb"
) as file:

    pickle.dump(
        X_scaler,
        file
    )

with open(
    Y_SCALER_PATH,
    "wb"
) as file:

    pickle.dump(
        y_scaler,
        file
    )


# ============================================================
# Save metrics
# ============================================================

metrics_df = pd.DataFrame({
    "split": [
        "train",
        "validation",
        "test"
    ],

    "MAE": [
        train_metrics[0],
        val_metrics[0],
        test_metrics[0]
    ],

    "RMSE": [
        train_metrics[1],
        val_metrics[1],
        test_metrics[1]
    ],

    "R2": [
        train_metrics[2],
        val_metrics[2],
        test_metrics[2]
    ]
})

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_PATH)
print(X_SCALER_PATH)
print(Y_SCALER_PATH)
print(METRICS_PATH)

print()
print("=" * 70)
print("REGION-AWARE LSTM TRAINING COMPLETED")
print("=" * 70)