# src/models/ml_models.py

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from xgboost import XGBRegressor


class MLModelTrainer:
    """
    Train and evaluate machine learning models
    for crop yield prediction.
    """

    def __init__(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names=None
    ):

        # Handle NaN and infinite values
        self.X_train = np.nan_to_num(
            X_train,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        self.X_val = np.nan_to_num(
            X_val,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        self.X_test = np.nan_to_num(
            X_test,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.feature_names = feature_names

        self.models = {}
        self.results = {}

    # Model evaluation
    def evaluate_model(self, model):
        """
        Calculate Train, Validation and Test metrics.
        """

        y_train_pred = model.predict(self.X_train)
        y_val_pred = model.predict(self.X_val)
        y_test_pred = model.predict(self.X_test)

        metrics = {
            'train_mae': mean_absolute_error(
                self.y_train,
                y_train_pred
            ),

            'train_rmse': np.sqrt(
                mean_squared_error(
                    self.y_train,
                    y_train_pred
                )
            ),

            'train_r2': r2_score(
                self.y_train,
                y_train_pred
            ),

            'val_mae': mean_absolute_error(
                self.y_val,
                y_val_pred
            ),

            'val_rmse': np.sqrt(
                mean_squared_error(
                    self.y_val,
                    y_val_pred
                )
            ),

            'val_r2': r2_score(
                self.y_val,
                y_val_pred
            ),

            'test_mae': mean_absolute_error(
                self.y_test,
                y_test_pred
            ),

            'test_rmse': np.sqrt(
                mean_squared_error(
                    self.y_test,
                    y_test_pred
                )
            ),

            'test_r2': r2_score(
                self.y_test,
                y_test_pred
            )
        }

        return metrics

    # Print model metrics
    def print_metrics(self, model_name, metrics):

        print("\n" + "=" * 65)
        print(f"{model_name} PERFORMANCE")
        print("=" * 65)

        print(
            f"Train MAE       : "
            f"{metrics['train_mae']:.2f}"
        )

        print(
            f"Train RMSE      : "
            f"{metrics['train_rmse']:.2f}"
        )

        print(
            f"Train R²        : "
            f"{metrics['train_r2']:.4f}"
        )

        print()

        print(
            f"Validation MAE  : "
            f"{metrics['val_mae']:.2f}"
        )

        print(
            f"Validation RMSE : "
            f"{metrics['val_rmse']:.2f}"
        )

        print(
            f"Validation R²   : "
            f"{metrics['val_r2']:.4f}"
        )

        print()

        print(
            f"Test MAE        : "
            f"{metrics['test_mae']:.2f}"
        )

        print(
            f"Test RMSE       : "
            f"{metrics['test_rmse']:.2f}"
        )

        print(
            f"Test R²         : "
            f"{metrics['test_r2']:.4f}"
        )

        # Overfitting detection
        train_test_gap = (
            metrics['train_r2']
            - metrics['test_r2']
        )

        print(
            f"\nTrain-Test R² Difference : "
            f"{train_test_gap:.4f}"
        )

        if train_test_gap > 0.15:
            print(
                "Warning: Possible "
                "Overfitting Detected"
            )
        else:
            print(
                "Generalization looks reasonable"
            )

    # Linear Regression
    def train_linear_regression(self):

        print("\nTraining Linear Regression...")

        start_time = time.time()

        model = LinearRegression()

        model.fit(
            self.X_train,
            self.y_train
        )

        training_time = time.time() - start_time

        metrics = self.evaluate_model(model)
        metrics['training_time'] = training_time

        self.models['Linear Regression'] = model
        self.results['Linear Regression'] = metrics

        self.print_metrics(
            'Linear Regression',
            metrics
        )

    # Decision Tree
    def train_decision_tree(self):

        print("\nTraining Decision Tree...")

        start_time = time.time()

        model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        training_time = time.time() - start_time

        metrics = self.evaluate_model(model)
        metrics['training_time'] = training_time

        self.models['Decision Tree'] = model
        self.results['Decision Tree'] = metrics

        self.print_metrics(
            'Decision Tree',
            metrics
        )

    # Optimized Random Forest
    def train_random_forest(self):

        print("\nTraining Optimized Random Forest...")

        start_time = time.time()

        # Random Forest hyperparameter search space
        param_distributions = {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [0.7, 1.0, 'sqrt'],
            'max_samples': [0.8, 1.0],
            'bootstrap': [True]
        }

        # Base Random Forest model
        rf = RandomForestRegressor(
            random_state=42,
            n_jobs=1
        )

        # Faster hyperparameter optimization
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=10,
            scoring='r2',
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            return_train_score=False
        )

        print(
            "\nSearching for the best "
            "Random Forest parameters..."
        )

        search.fit(
            self.X_train,
            self.y_train
        )

        model = search.best_estimator_

        training_time = time.time() - start_time

        print("\nBest Random Forest Parameters:")

        for parameter, value in search.best_params_.items():
            print(
                f"{parameter:20s}: {value}"
            )

        print(
            f"\nBest 3-Fold CV R²: "
            f"{search.best_score_:.4f}"
        )

        metrics = self.evaluate_model(model)

        metrics['training_time'] = training_time
        metrics['cv_r2'] = search.best_score_

        self.models['Random Forest'] = model
        self.results['Random Forest'] = metrics

        self.print_metrics(
            'Optimized Random Forest',
            metrics
        )

        # Feature importance
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

            print(
                importance_df.head(10)
                .to_string(index=False)
            )

            os.makedirs(
                'models/saved',
                exist_ok=True
            )

            importance_df.to_csv(
                'models/saved/'
                'random_forest_feature_importance.csv',
                index=False
            )

            print(
                "\nFeature importance saved."
            )

    # XGBoost
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

        model.fit(
            self.X_train,
            self.y_train
        )

        training_time = time.time() - start_time

        metrics = self.evaluate_model(model)
        metrics['training_time'] = training_time

        self.models['XGBoost'] = model
        self.results['XGBoost'] = metrics

        self.print_metrics(
            'XGBoost',
            metrics
        )

    # Support Vector Regressor
    def train_svr(self):

        print("\nTraining Support Vector Regressor...")

        start_time = time.time()

        model = SVR(
            kernel='rbf',
            C=100,
            gamma='scale'
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        training_time = time.time() - start_time

        metrics = self.evaluate_model(model)
        metrics['training_time'] = training_time

        self.models['SVR'] = model
        self.results['SVR'] = metrics

        self.print_metrics(
            'SVR',
            metrics
        )

    # Compare models
    def compare_models(self):

        print("\n" + "=" * 85)
        print("MODEL COMPARISON")
        print("=" * 85)

        comparison = []

        for name, metrics in self.results.items():

            comparison.append({
                'Model': name,

                'Validation R²':
                    metrics['val_r2'],

                'Validation RMSE':
                    metrics['val_rmse'],

                'Validation MAE':
                    metrics['val_mae'],

                'Test R²':
                    metrics['test_r2'],

                'Test RMSE':
                    metrics['test_rmse'],

                'Test MAE':
                    metrics['test_mae'],

                'Training Time':
                    metrics['training_time']
            })

        comparison_df = pd.DataFrame(
            comparison
        )

        # Select the model using validation R².
        # The test set is reserved for final evaluation.
        comparison_df = comparison_df.sort_values(
            by='Validation R²',
            ascending=False
        )

        print(
            comparison_df.to_string(
                index=False
            )
        )

        best_model_name = (
            comparison_df.iloc[0]['Model']
        )

        print(
            "\nBest Model based on "
            "Validation R²:"
        )

        print(
            f"   {best_model_name}"
        )

        print(
            f"\nValidation R²: "
            f"{comparison_df.iloc[0]['Validation R²']:.4f}"
        )

        return comparison_df

    # Save trained models
    def save_models(
        self,
        save_path='models/saved/'
    ):

        os.makedirs(
            save_path,
            exist_ok=True
        )

        # Save all trained models
        for name, model in self.models.items():

            filename = (
                name.lower()
                .replace(' ', '_')
                + '.pkl'
            )

            full_path = os.path.join(
                save_path,
                filename
            )

            joblib.dump(
                model,
                full_path
            )

            print(
                f"Saved {name} -> "
                f"{full_path}"
            )

        # Select the best model using validation R²
        best_model_name = max(
            self.results,
            key=lambda x:
            self.results[x]['val_r2']
        )

        best_model = self.models[
            best_model_name
        ]

        best_model_path = os.path.join(
            save_path,
            'best_model.pkl'
        )

        joblib.dump(
            best_model,
            best_model_path
        )

        print(
            f"\nBest Model Saved: "
            f"{best_model_name}"
        )

        print(
            f"Location: "
            f"{best_model_path}"
        )

    # Save model comparison
    def save_comparison(
        self,
        comparison_df,
        save_path='models/saved/'
    ):

        os.makedirs(
            save_path,
            exist_ok=True
        )

        comparison_path = os.path.join(
            save_path,
            'model_comparison.csv'
        )

        comparison_df.to_csv(
            comparison_path,
            index=False
        )

        print(
            f"\nModel comparison saved -> "
            f"{comparison_path}"
        )


if __name__ == '__main__':

    print("\n")
    print("=" * 75)
    print(
        "CROP YIELD PREDICTION - "
        "MACHINE LEARNING TRAINING"
    )
    print("=" * 75)

    # Load processed dataset
    processed_data = pd.read_csv(
        'data/processed/processed_data.csv'
    )

    print(
        f"\nProcessed Dataset Shape: "
        f"{processed_data.shape}"
    )

    # Separate features and target
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
        f"Number of Features: "
        f"{len(feature_columns)}"
    )

    # Split dataset into 70% training,
    # 15% validation and 15% testing
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42
        )
    )

    print("\nDataset Split:")

    print(
        f"Training   : "
        f"{len(X_train)} "
        f"({len(X_train) / len(X) * 100:.1f}%)"
    )

    print(
        f"Validation : "
        f"{len(X_val)} "
        f"({len(X_val) / len(X) * 100:.1f}%)"
    )

    print(
        f"Testing    : "
        f"{len(X_test)} "
        f"({len(X_test) / len(X) * 100:.1f}%)"
    )

    # Create trainer
    trainer = MLModelTrainer(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names=feature_columns
    )

    # Train models
    trainer.train_linear_regression()
    trainer.train_decision_tree()
    trainer.train_random_forest()
    trainer.train_xgboost()
    trainer.train_svr()

    # Compare models
    comparison = trainer.compare_models()

    # Save comparison
    trainer.save_comparison(
        comparison
    )

    # Save models
    trainer.save_models()

    print("\n")
    print("=" * 75)
    print(
        "TRAINING COMPLETED SUCCESSFULLY"
    )
    print("=" * 75)