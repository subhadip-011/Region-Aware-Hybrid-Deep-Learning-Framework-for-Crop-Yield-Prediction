# src/explainability/shap_analyzer.py

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from src.utils.logger import logger


class SHAPExplainer:
    """
    Comprehensive SHAP Analysis Module
    for Explainable Crop Yield Prediction
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self, model, feature_names):

        self.model = model

        self.feature_names = feature_names

        self.explainer = None

        self.shap_values = None

        self.X_data = None

    # =====================================================
    # CREATE EXPLAINER
    # =====================================================

    def create_explainer(
        self,
        X_background,
        model_type='tree'
    ):
        """
        Create SHAP explainer based on model type
        """

        logger.info(
            f"Creating SHAP explainer for {model_type} model"
        )

        try:

            if model_type == 'tree':

                self.explainer = shap.TreeExplainer(
                    self.model
                )

            elif model_type == 'linear':

                self.explainer = shap.LinearExplainer(
                    self.model,
                    X_background
                )

            else:
                # Deep learning models
                self.explainer = shap.GradientExplainer(
                    self.model,
                    X_background
                )

            logger.info(
                f"{model_type} SHAP explainer created successfully"
            )

            return self.explainer

        except Exception as e:

            logger.error(
                f"Error creating SHAP explainer: {e}"
            )

            raise e

    # =====================================================
    # CALCULATE SHAP VALUES
    # =====================================================

    def calculate_shap_values(self, X):
        """
        Calculate SHAP values
        """

        try:

            logger.info(
                f"Calculating SHAP values for {len(X)} samples"
            )

            self.X_data = X

            self.shap_values = self.explainer.shap_values(X)

            # Safety for tree-based models
            if isinstance(self.shap_values, list):

                self.shap_values = self.shap_values[0]

            logger.info(
                "SHAP values calculated successfully"
            )

            return self.shap_values

        except Exception as e:

            logger.error(
                f"Error calculating SHAP values: {e}"
            )

            raise e

    # =====================================================
    # SUMMARY PLOT
    # =====================================================

    def summary_plot(self, save_path=None):
        """
        Create SHAP summary plot
        """

        try:

            plt.figure(figsize=(12, 8))

            shap.summary_plot(
                self.shap_values,
                self.X_data,
                feature_names=self.feature_names,
                show=False
            )

            plt.tight_layout()

            if save_path:

                plt.savefig(
                    save_path,
                    dpi=300,
                    bbox_inches='tight'
                )

                logger.info(
                    f"Summary plot saved to {save_path}"
                )

            plt.show()

        except Exception as e:

            logger.error(
                f"Error generating summary plot: {e}"
            )

    # =====================================================
    # BAR PLOT
    # =====================================================

    def bar_plot(self, save_path=None):
        """
        Create SHAP bar plot
        """

        try:

            plt.figure(figsize=(10, 6))

            shap.summary_plot(
                self.shap_values,
                self.X_data,
                feature_names=self.feature_names,
                plot_type='bar',
                show=False
            )

            plt.tight_layout()

            if save_path:

                plt.savefig(
                    save_path,
                    dpi=300,
                    bbox_inches='tight'
                )

                logger.info(
                    f"Bar plot saved to {save_path}"
                )

            plt.show()

        except Exception as e:

            logger.error(
                f"Error generating bar plot: {e}"
            )

    # =====================================================
    # WATERFALL PLOT
    # =====================================================

    def waterfall_plot(
        self,
        sample_idx,
        X_sample,
        save_path=None
    ):
        """
        Create SHAP waterfall plot
        """

        try:

            shap_values_sample = (
                self.shap_values[sample_idx]
            )

            expected_value = np.array(
                self.explainer.expected_value
            ).mean()

            explanation = shap.Explanation(

                values=shap_values_sample,

                base_values=expected_value,

                data=X_sample.iloc[sample_idx].values,

                feature_names=self.feature_names
            )

            plt.figure(figsize=(12, 6))

            shap.waterfall_plot(
                explanation,
                show=False
            )

            plt.tight_layout()

            if save_path:

                plt.savefig(
                    save_path,
                    dpi=300,
                    bbox_inches='tight'
                )

                logger.info(
                    f"Waterfall plot saved to {save_path}"
                )

            plt.show()

        except Exception as e:

            logger.error(
                f"Error generating waterfall plot: {e}"
            )

    # =====================================================
    # DEPENDENCE PLOT
    # =====================================================

    def dependence_plot(
        self,
        feature_name,
        save_path=None
    ):
        """
        Create SHAP dependence plot
        """

        try:

            shap.dependence_plot(
                feature_name,
                self.shap_values,
                self.X_data,
                feature_names=self.feature_names,
                show=False
            )

            plt.tight_layout()

            if save_path:

                plt.savefig(
                    save_path,
                    dpi=300,
                    bbox_inches='tight'
                )

                logger.info(
                    f"Dependence plot saved to {save_path}"
                )

            plt.show()

        except Exception as e:

            logger.error(
                f"Error generating dependence plot: {e}"
            )

    # =====================================================
    # FORCE PLOT
    # =====================================================

    def force_plot(self, sample_idx):
        """
        Create SHAP force plot
        """

        try:

            shap.initjs()

            expected_value = np.array(
                self.explainer.expected_value
            ).mean()

            return shap.force_plot(
                expected_value,
                self.shap_values[sample_idx],
                self.X_data.iloc[sample_idx],
                feature_names=self.feature_names
            )

        except Exception as e:

            logger.error(
                f"Error generating force plot: {e}"
            )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    def feature_importance_df(self):
        """
        Return feature importance DataFrame
        """

        try:

            importance_df = pd.DataFrame({

                'feature': self.feature_names,

                'shap_importance': np.abs(
                    self.shap_values
                ).mean(axis=0)

            }).sort_values(
                'shap_importance',
                ascending=False
            )

            logger.info(
                "Feature importance DataFrame created"
            )

            return importance_df

        except Exception as e:

            logger.error(
                f"Error generating feature importance: {e}"
            )

    # =====================================================
    # INTERACTIVE FEATURE IMPORTANCE
    # =====================================================

    def interactive_importance_plot(
        self,
        top_n=15
    ):
        """
        Interactive Plotly feature importance plot
        """

        try:

            importance_df = (
                self.feature_importance_df()
                .head(top_n)
            )

            fig = px.bar(
                importance_df,
                x='shap_importance',
                y='feature',
                orientation='h',
                title='SHAP Feature Importance',
                color='shap_importance',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(
                yaxis={
                    'categoryorder':
                    'total ascending'
                }
            )

            fig.show()

        except Exception as e:

            logger.error(
                f"Error generating interactive plot: {e}"
            )

    # =====================================================
    # SINGLE PREDICTION ANALYSIS
    # =====================================================

    def analyze_prediction(
        self,
        sample_idx,
        X_sample,
        y_true=None,
        y_pred=None
    ):
        """
        Detailed explanation for single prediction
        """

        try:

            shap_values_sample = (
                self.shap_values[sample_idx]
            )

            contributions = pd.DataFrame({

                'feature': self.feature_names,

                'value':
                X_sample.iloc[sample_idx].values,

                'shap_value':
                shap_values_sample,

                'abs_contribution':
                np.abs(shap_values_sample)

            }).sort_values(
                'abs_contribution',
                ascending=False
            )

            print("\n" + "=" * 70)

            print(
                f"SHAP PREDICTION ANALYSIS "
                f"(Sample {sample_idx})"
            )

            print("=" * 70)

            if y_true is not None:

                print(
                    f"\nActual Yield     : "
                    f"{y_true:.2f} kg/ha"
                )

            if y_pred is not None:

                print(
                    f"Predicted Yield  : "
                    f"{y_pred:.2f} kg/ha"
                )

            print("\nTop Contributing Features:\n")

            for _, row in (
                contributions.head(5).iterrows()
            ):

                direction = (
                    "↑ Positive"
                    if row['shap_value'] > 0
                    else "↓ Negative"
                )

                print(
                    f"{direction} | "
                    f"{row['feature']} = "
                    f"{row['value']:.2f} "
                    f"(SHAP: "
                    f"{row['shap_value']:+.2f})"
                )

            return contributions

        except Exception as e:

            logger.error(
                f"Error analyzing prediction: {e}"
            )

    # =====================================================
    # SAVE FEATURE IMPORTANCE CSV
    # =====================================================

    def save_feature_importance(
        self,
        save_path
    ):
        """
        Save feature importance CSV
        """

        try:

            importance_df = (
                self.feature_importance_df()
            )

            importance_df.to_csv(
                save_path,
                index=False
            )

            logger.info(
                f"Feature importance saved to "
                f"{save_path}"
            )

        except Exception as e:

            logger.error(
                f"Error saving feature importance: {e}"
            )
# =====================================================
# MAIN EXECUTION
# =====================================================
# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    import os
    import joblib

    # -------------------------------------------------
    # VISUALIZATION DIRECTORY
    # -------------------------------------------------

    VISUALIZATION_DIR = 'notebooks/visualizations'

    os.makedirs(
        VISUALIZATION_DIR,
        exist_ok=True
    )

    logger.info(
        "Starting SHAP analysis..."
    )

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------

    data = pd.read_csv(
        'data/processed/processed_data.csv'
    )

    logger.info(
        f"Dataset loaded successfully: {data.shape}"
    )

    # -------------------------------------------------
    # FEATURES
    # -------------------------------------------------

    X = data.drop(
        columns=['yield_kg_ha']
    )

    # Keep only numerical features
    X = X.select_dtypes(
        include=[np.number]
    )

    logger.info(
        f"Feature matrix shape: {X.shape}"
    )

    # -------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------

    model = joblib.load(
        'models/saved/random_forest.pkl'
    )

    logger.info(
        "Random Forest model loaded successfully"
    )

    # -------------------------------------------------
    # CREATE SHAP EXPLAINER
    # -------------------------------------------------

    explainer = SHAPExplainer(
        model=model,
        feature_names=X.columns.tolist()
    )

    explainer.create_explainer(
        X_background=X.sample(
            100,
            random_state=42
        ),
        model_type='tree'
    )

    logger.info(
        "SHAP explainer created successfully"
    )

    # -------------------------------------------------
    # SAMPLE DATA FOR SHAP
    # -------------------------------------------------

    X_sample = X.sample(
        500,
        random_state=42
    )

    logger.info(
        f"Using sample size: {len(X_sample)}"
    )

    # -------------------------------------------------
    # CALCULATE SHAP VALUES
    # -------------------------------------------------

    explainer.calculate_shap_values(
        X_sample
    )

    logger.info(
        "SHAP values calculated successfully"
    )

    # -------------------------------------------------
    # SUMMARY PLOT
    # -------------------------------------------------

    explainer.summary_plot(
        save_path=(
            f'{VISUALIZATION_DIR}/'
            'shap_summary_plot.png'
        )
    )

    logger.info(
        "SHAP summary plot saved"
    )

    # -------------------------------------------------
    # BAR PLOT
    # -------------------------------------------------

    explainer.bar_plot(
        save_path=(
            f'{VISUALIZATION_DIR}/'
            'shap_bar_plot.png'
        )
    )

    logger.info(
        "SHAP bar plot saved"
    )

    # -------------------------------------------------
    # DEPENDENCE PLOT
    # -------------------------------------------------

    top_feature = X.columns[0]

    explainer.dependence_plot(
        feature_name=top_feature,
        save_path=(
            f'{VISUALIZATION_DIR}/'
            'shap_dependence_plot.png'
        )
    )

    logger.info(
        f"Dependence plot created for "
        f"{top_feature}"
    )

    # -------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------

    importance_df = (
        explainer.feature_importance_df()
    )

    print("\n" + "=" * 70)

    print(
        "TOP 10 SHAP IMPORTANT FEATURES"
    )

    print("=" * 70)

    print(
        importance_df.head(10)
    )

    # -------------------------------------------------
    # SAVE FEATURE IMPORTANCE
    # -------------------------------------------------

    importance_df.to_csv(
        f'{VISUALIZATION_DIR}/'
        'shap_feature_importance.csv',
        index=False
    )

    logger.info(
        "SHAP feature importance CSV saved"
    )

    # -------------------------------------------------
    # SINGLE PREDICTION ANALYSIS
    # -------------------------------------------------

    sample_idx = 0

    prediction = model.predict(
        X_sample.iloc[[sample_idx]]
    )[0]

    explainer.analyze_prediction(
        sample_idx=sample_idx,
        X_sample=X_sample,
        y_pred=prediction
    )

    logger.info(
        "Single prediction analysis completed"
    )

    # -------------------------------------------------
    # WATERFALL PLOT
    # -------------------------------------------------

    explainer.waterfall_plot(
        sample_idx=sample_idx,
        X_sample=X_sample,
        save_path=(
            f'{VISUALIZATION_DIR}/'
            'shap_waterfall_plot.png'
        )
    )

    logger.info(
        "Waterfall plot generated"
    )

    # -------------------------------------------------
    # SAVE INTERACTIVE IMPORTANCE PLOT
    # -------------------------------------------------

    explainer.interactive_importance_plot()

    logger.info(
        "Interactive SHAP plot generated"
    )

    # -------------------------------------------------
    # FINAL SUCCESS MESSAGE
    # -------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "✅ SHAP ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print(
        f"\nVisualizations saved in:\n"
        f"{VISUALIZATION_DIR}"
    )

    logger.info(
        "Complete SHAP analysis pipeline finished"
    )