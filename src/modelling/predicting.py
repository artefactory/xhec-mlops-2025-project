"""
Prediction module for abalone age prediction.

This module handles model loading and prediction functionality
for the trained models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any
from .utils import load_model_and_preprocessors
from .preprocessing import (
    get_feature_columns,
    engineer_features,
    encode_categorical_features,
)


def load_trained_model(model_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a trained model and its preprocessors.

    Args:
        model_dir: Directory containing the pickled model and preprocessors

    Returns:
        Dictionary with model, scaler, and label_encoder
    """
    model, scaler, label_encoder = load_model_and_preprocessors(model_dir)

    return {"model": model, "scaler": scaler, "label_encoder": label_encoder}


def preprocess_new_data(
    data: Union[pd.DataFrame, Dict[str, Any]], label_encoder: Any
) -> np.ndarray:
    """
    Preprocess new data for prediction using the same preprocessing pipeline.

    Args:
        data: New data as DataFrame or dictionary
        label_encoder: LabelEncoder used during training

    Returns:
        Preprocessed features array
    """
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Encode categorical features
    df, _ = encode_categorical_features(df)

    # Engineer features
    df = engineer_features(df)

    # Get feature columns
    feature_cols = get_feature_columns()

    # Select features
    X = df[feature_cols]

    return X.values


def predict_abalone_age(
    data: Union[pd.DataFrame, Dict[str, Any]], model_dir: Union[str, Path]
) -> Dict[str, Any]:
    """
    Predict abalone age from new data.

    Args:
        data: New data as DataFrame or dictionary with features
        model_dir: Directory containing the trained model

    Returns:
        Dictionary with prediction and confidence
    """
    # Load model and preprocessors
    model_components = load_trained_model(model_dir)
    model = model_components["model"]
    scaler = model_components["scaler"]
    label_encoder = model_components["label_encoder"]

    # Preprocess data
    X = preprocess_new_data(data, label_encoder)

    # Scale features
    X_scaled = scaler.transform(X)

    # Make prediction
    prediction = model.predict(X_scaled)

    # Calculate confidence (for Random Forest, use prediction variance)
    if hasattr(model, "estimators_"):
        # Random Forest - calculate prediction variance
        predictions = np.array([tree.predict(X_scaled) for tree in model.estimators_])
        confidence = 1.0 / (1.0 + np.var(predictions, axis=0))
    else:
        # Linear Regression - use R² as confidence proxy
        confidence = np.array([0.5])  # Default confidence

    return {
        "prediction": float(prediction[0]),
        "confidence": float(confidence[0]),
        "model_type": type(model).__name__,
    }


def predict_batch(data: pd.DataFrame, model_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Predict abalone age for a batch of data.

    Args:
        data: DataFrame with multiple samples
        model_dir: Directory containing the trained model

    Returns:
        DataFrame with predictions and confidence scores
    """
    # Load model and preprocessors
    model_components = load_trained_model(model_dir)
    model = model_components["model"]
    scaler = model_components["scaler"]
    label_encoder = model_components["label_encoder"]

    # Preprocess data
    X = preprocess_new_data(data, label_encoder)

    # Scale features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)

    # Calculate confidence
    if hasattr(model, "estimators_"):
        # Random Forest - calculate prediction variance
        all_predictions = np.array(
            [tree.predict(X_scaled) for tree in model.estimators_]
        )
        confidence = 1.0 / (1.0 + np.var(all_predictions, axis=0))
    else:
        # Linear Regression - use R² as confidence proxy
        confidence = np.full(len(predictions), 0.5)

    # Create results DataFrame
    results = data.copy()
    results["predicted_age"] = predictions
    results["confidence"] = confidence

    return results


def validate_prediction_input(data: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
    """
    Validate input data for prediction.

    Args:
        data: Input data to validate

    Returns:
        True if valid, False otherwise
    """
    required_features = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
    ]

    if isinstance(data, dict):
        # Check if all required features are present
        missing_features = [feat for feat in required_features if feat not in data]
        if missing_features:
            print(f"Missing required features: {missing_features}")
            return False
    else:
        # DataFrame
        missing_features = [
            feat for feat in required_features if feat not in data.columns
        ]
        if missing_features:
            print(f"Missing required features: {missing_features}")
            return False

    return True


def get_model_info(model_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a trained model.

    Args:
        model_dir: Directory containing the trained model

    Returns:
        Dictionary with model information
    """
    model_components = load_trained_model(model_dir)
    model = model_components["model"]

    info = {
        "model_type": type(model).__name__,
        "model_dir": str(model_dir),
        "feature_count": len(get_feature_columns()),
    }

    # Add model-specific information
    if hasattr(model, "feature_importances_"):
        info["has_feature_importance"] = True
        info["feature_importance"] = dict(
            zip(get_feature_columns(), model.feature_importances_)
        )
    else:
        info["has_feature_importance"] = False

    if hasattr(model, "coef_"):
        info["has_coefficients"] = True
        info["coefficients"] = dict(zip(get_feature_columns(), model.coef_))
    else:
        info["has_coefficients"] = False

    return info
