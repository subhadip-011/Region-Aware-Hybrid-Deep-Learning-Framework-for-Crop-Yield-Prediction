# Complete ANN Model and Hybrid Model Code

# src/models/ann_model.py

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# =========================================================
# LOAD DATA
# =========================================================

print("\n" + "=" * 70)
print("ARTIFICIAL NEURAL NETWORK TRAINING")
print("=" * 70)

processed_data = pd.read_csv(
    'data/processed/processed_data.csv'
)

print(f"Dataset Shape: {processed_data.shape}")

# =========================================================
# FEATURES AND TARGET
# =========================================================

feature_columns = [
    col for col in processed_data.columns
    if col != 'yield_kg_ha'
]

X = processed_data[feature_columns].values

y = processed_data['yield_kg_ha'].values

# =========================================================
# TRAIN TEST SPLIT
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

X_train = np.nan_to_num(X_train)
X_val = np.nan_to_num(X_val)
X_test = np.nan_to_num(X_test)

# =========================================================
# BUILD ANN MODEL
# =========================================================

model = Sequential()

# Input Layer
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))

# Hidden Layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

# Output Layer
model.add(Dense(1, activation='linear'))

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# =========================================================
# EARLY STOPPING
# =========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining ANN Model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =========================================================
# PREDICTIONS
# =========================================================

train_pred = model.predict(X_train).flatten()
val_pred = model.predict(X_val).flatten()
test_pred = model.predict(X_test).flatten()

# =========================================================
# EVALUATION FUNCTION
# =========================================================


def evaluate(y_true, y_pred, dataset_name):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"{dataset_name} PERFORMANCE")
    print("=" * 60)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return mae, rmse, r2


# =========================================================
# EVALUATE MODEL
# =========================================================

train_metrics = evaluate(y_train, train_pred, 'TRAIN')
val_metrics = evaluate(y_val, val_pred, 'VALIDATION')
test_metrics = evaluate(y_test, test_pred, 'TEST')

# =========================================================
# OVERFITTING CHECK
# =========================================================

train_r2 = train_metrics[2]
test_r2 = test_metrics[2]

difference = train_r2 - test_r2

print(f"\nTrain-Test Difference: {difference:.4f}")

if difference > 0.15:
    print("⚠️ Possible Overfitting Detected")
else:
    print("✅ Good Generalization")

# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs('models/saved', exist_ok=True)

model.save('models/saved/ann_model.keras')

print("\n✅ ANN Model Saved")
print("📁 models/saved/ann_model.keras")

print("\n" + "=" * 70)
print("ANN TRAINING COMPLETED")
print("=" * 70)
print(os.listdir('models/saved'))
