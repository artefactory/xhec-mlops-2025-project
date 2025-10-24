"""Inference functions for the web service."""

import pickle as pkl
import pandas as pd
from pathlib import Path
from .app_config import config
from .schemas import AbaloneFeatures, PredictionResponse
from .preprocessing import preprocess_single_sample


def load_model(model_path: Path = None):
    """Load the trained model from disk.

    Args:
        model_path: Path to the pickled model file

    Returns:
        The loaded model

    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if model_path is None:
        model_path = config.model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please train a model first using the /train endpoint or running the training pipeline."
        )

    with open(model_path, "rb") as f:
        model = pkl.load(f)

    return model


def prepare_features(features: AbaloneFeatures) -> pd.DataFrame:
    """Convert AbaloneFeatures to a DataFrame with the correct column names and order.

    Uses the same preprocessing pipeline as training to ensure consistency.

    Args:
        features: Input features from the API request

    Returns:
        DataFrame with encoded features ready for prediction
    """
    # Convert Pydantic model to dictionary
    features_dict = {
        "Sex": features.sex,
        "Length": features.length,
        "Diameter": features.diameter,
        "Height": features.height,
        "Whole weight": features.whole_weight,
        "Shucked weight": features.shucked_weight,
        "Viscera weight": features.viscera_weight,
        "Shell weight": features.shell_weight,
    }

    # Use the unified preprocessing function
    processed_df = preprocess_single_sample(features_dict)

    return processed_df


def run_inference(features: AbaloneFeatures, model=None) -> PredictionResponse:
    """Run inference on a single abalone sample.

    Args:
        features: Input features for prediction
        model: Pre-loaded model (optional, will load from disk if not provided)

    Returns:
        PredictionResponse with predicted rings and age
    """
    # Load model if not provided
    if model is None:
        model = load_model()

    # Prepare features
    X = prepare_features(features)

    # Make prediction
    predicted_rings = float(model.predict(X)[0])

    # Calculate age (rings + 1.5 years)
    predicted_age = predicted_rings + 1.5

    # Create response
    response = PredictionResponse(
        predicted_rings=round(predicted_rings, 2),
        predicted_age=round(predicted_age, 2),
        input_features=features,
    )

    return response


def run_batch_inference(
    features_list: list[AbaloneFeatures], model=None
) -> list[PredictionResponse]:
    """Run inference on multiple abalone samples.

    Args:
        features_list: List of input features for prediction
        model: Pre-loaded model (optional, will load from disk if not provided)

    Returns:
        List of PredictionResponse objects
    """
    # Load model if not provided
    if model is None:
        model = load_model()

    # Process all samples
    predictions = []
    for features in features_list:
        prediction = run_inference(features, model=model)
        predictions.append(prediction)

    return predictions
