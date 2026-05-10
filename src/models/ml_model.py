# Fixed Complete ML Training Pipeline
# src/models/ml_models.py

import numpy as np
import pandas as pd
import joblib
import time
import os

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split


class MLModelTrainer:
    """
    Train and evaluate machine learning models
    """

    def __init__(self, X_train, X_val, X_test,
                 y_train, y_val, y_test,
                 feature_names=None):

        # Safety against NaN and infinite values
        self.X_train = np.nan_to_num(X_train)
        self.X_val = np.nan_to_num(X_val)
        self.X_test = np.nan_to_num(X_test)

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.feature_names = feature_names

        self.models = {}
        self.results = {}

    # =====================================================
    # EVALUATE MODEL
    # =====================================================

    def evaluate_model(self, model):
        """Calculate evaluation metrics"""

        y_train_pred = model.predict(self.X_train)
        y_val_pred = model.predict(self.X_val)
        y_test_pred = model.predict(self.X_test)

        metrics = {
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'train_r2': r2_score(self.y_train, y_train_pred),

            'val_mae': mean_absolute_error(self.y_val, y_val_pred),
            'val_rmse': np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
            'val_r2': r2_score(self.y_val, y_val_pred),

            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'test_r2': r2_score(self.y_test, y_test_pred)
        }

        return metrics

    # =====================================================
    # PRINT METRICS
    # =====================================================

    def print_metrics(self, model_name, metrics):
        """Print model metrics"""

        print("\n" + "=" * 60)
        print(f"{model_name} PERFORMANCE")
        print("=" * 60)

        print(f"Train MAE  : {metrics['train_mae']:.2f}")
        print(f"Train RMSE : {metrics['train_rmse']:.2f}")
        print(f"Train R²   : {metrics['train_r2']:.4f}")

        print()

        print(f"Validation MAE  : {metrics['val_mae']:.2f}")
        print(f"Validation RMSE : {metrics['val_rmse']:.2f}")
        print(f"Validation R²   : {metrics['val_r2']:.4f}")

        print()

        print(f"Test MAE  : {metrics['test_mae']:.2f}")
        print(f"Test RMSE : {metrics['test_rmse']:.2f}")
        print(f"Test R²   : {metrics['test_r2']:.4f}")

        # Overfitting detection
        if metrics['train_r2'] - metrics['test_r2'] > 0.15:
            print("\n⚠️ Warning: Possible Overfitting Detected")

    # =====================================================
    # LINEAR REGRESSION
    # =====================================================

    def train_linear_regression(self):

        print("\nTraining Linear Regression...")

        start_time = time.time()

        model = LinearRegression()

        model.fit(self.X_train, self.y_train)

        training_time = time.time() - start_time

        metrics = self.evaluate_model(model)
        metrics['training_time'] = training_time

        self.models['Linear Regression'] = model
        self.results['Linear Regression'] = metrics

        self.print_metrics('Linear Regression', metrics)

    # =====================================================
    # DECISION TREE
    # =====================================================

    def train_decision_tree(self):

        print("\nTraining Decision Tree...")

        start_time = time.time()

        model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )

        model.fit(self.X_train, self.y_train)

        training_time = time.time() - start_time

        metrics = self.evaluate_model(model)
        metrics['training_time'] = training_time

        self.models['Decision Tree'] = model
        self.results['Decision Tree'] = metrics

        self.print_metrics('Decision Tree', metrics)

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    def train_random_forest(self):

        print("\nTraining Random Forest...")

        start_time = time.time()

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)

        training_time = time.time() - start_time

        metrics = self.evaluate_model(model)
        metrics['training_time'] = training_time

        self.models['Random Forest'] = model
        self.results['Random Forest'] = metrics

        self.print_metrics('Random Forest', metrics)

        # Feature Importance
        if self.feature_names is not None:

            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                by='Importance',
                ascending=False
            )

            print("\nTop 10 Important Features:")
            print(importance_df.head(10))

    # =====================================================
    # XGBOOST
    # =====================================================

    def train_xgboost(self):

        print("\nTraining XGBoost...")

        start_time = time.time()

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)

        training_time = time.time() - start_time

        metrics = self.evaluate_model(model)
        metrics['training_time'] = training_time

        self.models['XGBoost'] = model
        self.results['XGBoost'] = metrics

        self.print_metrics('XGBoost', metrics)

    # =====================================================
    # SUPPORT VECTOR REGRESSOR
    # =====================================================

    def train_svr(self):

        print("\nTraining Support Vector Regressor...")

        start_time = time.time()

        model = SVR(
            kernel='rbf',
            C=100,
            gamma='scale'
        )

        model.fit(self.X_train, self.y_train)

        training_time = time.time() - start_time

        metrics = self.evaluate_model(model)
        metrics['training_time'] = training_time

        self.models['SVR'] = model
        self.results['SVR'] = metrics

        self.print_metrics('SVR', metrics)

    # =====================================================
    # COMPARE MODELS
    # =====================================================

    def compare_models(self):

        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        comparison = []

        for name, metrics in self.results.items():

            comparison.append({
                'Model': name,
                'Test R²': metrics['test_r2'],
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'Training Time': metrics['training_time']
            })

        comparison_df = pd.DataFrame(comparison)

        comparison_df = comparison_df.sort_values(
            by='Test R²',
            ascending=False
        )

        print(comparison_df.to_string(index=False))

        best_model_name = comparison_df.iloc[0]['Model']

        print(f"\n🏆 Best Model: {best_model_name}")

        return comparison_df

    # =====================================================
    # SAVE MODELS
    # =====================================================

    def save_models(self, save_path='models/saved/'):
        """Save trained models"""

        os.makedirs(save_path, exist_ok=True)

        # Save all models
        for name, model in self.models.items():

            filename = name.lower().replace(' ', '_') + '.pkl'

            full_path = os.path.join(save_path, filename)

            joblib.dump(model, full_path)

            print(f"✅ Saved {name} -> {full_path}")

        # Save best model separately
        best_model_name = max(
            self.results,
            key=lambda x: self.results[x]['test_r2']
        )

        best_model = self.models[best_model_name]

        best_model_path = os.path.join(save_path, 'best_model.pkl')

        joblib.dump(best_model, best_model_path)

        print(f"\n🏆 Best Model Saved: {best_model_name}")
        print(f"📁 Location: {best_model_path}")


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == '__main__':

    print("\n" + "=" * 70)
    print("CROP YIELD PREDICTION - MACHINE LEARNING TRAINING")
    print("=" * 70)

    # Load processed dataset
    processed_data = pd.read_csv(
        'data/processed/processed_data.csv'
    )

    print(f"\nProcessed Dataset Shape: {processed_data.shape}")

    # Features and target
    feature_columns = [
        col for col in processed_data.columns
        if col != 'yield_kg_ha'
    ]

    X = processed_data[feature_columns].values

    y = processed_data['yield_kg_ha'].values

    # Split dataset
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

    # Create trainer object
    trainer = MLModelTrainer(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names=feature_columns
    )

    # Train Models
    trainer.train_linear_regression()
    trainer.train_decision_tree()
    trainer.train_random_forest()
    trainer.train_xgboost()
    trainer.train_svr()

    # Compare Models
    comparison = trainer.compare_models()

    # Save Models
    trainer.save_models()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)