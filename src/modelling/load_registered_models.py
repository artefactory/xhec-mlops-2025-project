"""
Script to load and use registered models from MLflow Model Registry.

This demonstrates how to load the baseline and production models
for inference and comparison.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


def load_registered_model(model_name: str, stage: str = None):
    """
    Load a registered model from MLflow Model Registry.

    Args:
        model_name: Name of the registered model
        stage: Stage of the model (None, Staging, Production)

    Returns:
        Loaded model
    """
    if stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"

    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Loaded {model_name} from {stage or 'latest'}")
    return model


def predict_with_registered_model(
    model_name: str, data: pd.DataFrame, stage: str = None
):
    """
    Make predictions using a registered model.

    Args:
        model_name: Name of the registered model
        data: Input data for prediction
        stage: Stage of the model

    Returns:
        Predictions
    """
    model = load_registered_model(model_name, stage)

    # Note: In production, you would need to preprocess the data
    # using the same scaler and feature engineering pipeline
    predictions = model.predict(data)

    return predictions


def compare_registered_models(data: pd.DataFrame):
    """
    Compare predictions from baseline and production models.

    Args:
        data: Input data for comparison
    """
    print("=== COMPARING REGISTERED MODELS ===")

    # Load both models
    baseline_model = load_registered_model("abalone_age_baseline")
    production_model = load_registered_model("abalone_age_production", "Staging")

    # Make predictions
    baseline_pred = baseline_model.predict(data)
    production_pred = production_model.predict(data)

    print("\nPredictions comparison:")
    print(f"Baseline (Linear Regression): {baseline_pred[0]:.2f}")
    print(f"Production (Random Forest):   {production_pred[0]:.2f}")
    print(f"Difference: {abs(baseline_pred[0] - production_pred[0]):.2f}")

    return {"baseline": baseline_pred, "production": production_pred}


def list_registered_models():
    """
    List all registered models and their versions.
    """
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()

    print("=== REGISTERED MODELS ===")
    registered_models = client.search_registered_models()

    for model in registered_models:
        print(f"\n📦 {model.name}:")
        for version in model.latest_versions:
            stage_emoji = (
                "🚀"
                if version.current_stage == "Staging"
                else "📊"
                if version.current_stage == "Production"
                else "🔬"
            )
            print(
                f"  {stage_emoji} Version {version.version}: {version.current_stage or 'None (Baseline)'}"
            )
            print(f"     Description: {version.description}")
            print(f"     Tags: {dict(version.tags) if version.tags else 'None'}")


if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")

    # List registered models
    list_registered_models()

    # Example: Create sample data for prediction
    sample_data = pd.DataFrame(
        {
            "Sex": ["M"],
            "Length": [0.5],
            "Diameter": [0.4],
            "Height": [0.15],
            "Whole weight": [0.6],
            "Shucked weight": [0.3],
            "Viscera weight": [0.15],
            "Shell weight": [0.15],
        }
    )

    print("\n=== SAMPLE PREDICTION ===")
    print(f"Sample data: {sample_data.iloc[0].to_dict()}")

    # Note: This is a simplified example. In production, you would need
    # to apply the same preprocessing pipeline (feature engineering, scaling)
    # that was used during training.
    print("\n⚠️  Note: This example assumes preprocessed data.")
    print("In production, apply the same preprocessing pipeline used during training.")
