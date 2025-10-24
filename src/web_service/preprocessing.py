"""
Unified preprocessing functions for both training and inference.
This ensures exact same preprocessing is applied in both cases.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle as pkl
from pathlib import Path


def preprocess_data(
    df: pd.DataFrame, fit_encoder: bool = False, encoder_path: Path = None
) -> tuple:
    """
    Apply consistent preprocessing to the dataset.

    Args:
        df: Input DataFrame with raw data
        fit_encoder: If True, fit a new encoder. If False, load existing encoder.
        encoder_path: Path to save/load the encoder

    Returns:
        tuple: (processed_df, label_encoder)
    """
    df = df.copy()

    if encoder_path is None:
        encoder_path = Path("src/web_service/local_objects/label_encoder.pkl")

    if fit_encoder:
        # Training mode: fit new encoder
        label_encoder = LabelEncoder()
        df["Sex_encoded"] = label_encoder.fit_transform(df["Sex"])

        # Save encoder for inference
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        with open(encoder_path, "wb") as f:
            pkl.dump(label_encoder, f)
    else:
        # Inference mode: load existing encoder
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at {encoder_path}. "
                "Please train a model first."
            )

        with open(encoder_path, "rb") as f:
            label_encoder = pkl.load(f)

        df["Sex_encoded"] = label_encoder.transform(df["Sex"])

    # Remove original Sex column
    df = df.drop(["Sex"], axis=1)

    # Ensure columns are in the correct order
    if "Rings" in df.columns:
        # Training data - separate features and target
        feature_columns = [col for col in df.columns if col != "Rings"]
        # Reorder to match expected order
        expected_order = [
            "Sex_encoded",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
        ]
        # Only include columns that exist
        feature_columns = [col for col in expected_order if col in feature_columns]
        df = df[feature_columns + ["Rings"]]
    else:
        # Inference data - only features
        expected_order = [
            "Sex_encoded",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
        ]
        # Only include columns that exist
        feature_columns = [col for col in expected_order if col in df.columns]
        df = df[feature_columns]

    return df, label_encoder


def preprocess_single_sample(features_dict: dict) -> pd.DataFrame:
    """
    Preprocess a single sample for inference.

    Args:
        features_dict: Dictionary with feature values

    Returns:
        DataFrame ready for model prediction
    """
    # Create DataFrame from single sample
    df = pd.DataFrame([features_dict])

    # Apply same preprocessing (inference mode)
    processed_df, _ = preprocess_data(df, fit_encoder=False)

    return processed_df


def prepare_training_data(data_path: Path) -> tuple:
    """
    Load and preprocess training data.

    Args:
        data_path: Path to the CSV data file

    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series
    """
    # Load raw data
    df = pd.read_csv(data_path)

    # Preprocess (training mode - fit new encoder)
    processed_df, encoder = preprocess_data(df, fit_encoder=True)

    # Split features and target
    y = processed_df["Rings"]
    X = processed_df.drop(["Rings"], axis=1)

    return X, y, encoder
