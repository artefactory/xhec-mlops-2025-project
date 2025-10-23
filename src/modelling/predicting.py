"""Module for making predictions with the trained model."""

from pathlib import Path

import pandas as pd
from utils import load_pickle_object


def load_model(model_path=None):
    """
    Load a trained model from pickle file.

    Args:
        model_path: Path to the pickled model. If None, uses default path.

    Returns:
        Loaded sklearn Pipeline model
    """
    if model_path is None:
        model_path = (
            Path(__file__).parent.parent / "web_service" / "local_objects" / "model.pkl"
        )

    model = load_pickle_object(model_path)
    return model


def predict(model, X):
    """
    Make predictions using the trained model.

    Args:
        model: Trained sklearn Pipeline model
        X: Features to predict on (pandas DataFrame or numpy array)

    Returns:
        Array of predictions
    """
    predictions = model.predict(X)
    return predictions


def predict_from_csv(model_path, csv_path):
    """
    Load data from CSV and make predictions.

    Args:
        model_path: Path to the pickled model
        csv_path: Path to the CSV file with features

    Returns:
        pandas DataFrame with features and predictions
    """
    model = load_model(model_path)
    df = pd.read_csv(csv_path)

    # Remove target column if present
    if "Rings" in df.columns:
        df = df.drop(columns=["Rings"])

    predictions = predict(model, df)
    df["Predicted_Rings"] = predictions

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Make predictions using trained model."
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to the model file"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the CSV file with features"
    )
    args = parser.parse_args()

    result = predict_from_csv(args.model, args.data)
    print("\nPredictions:")
    print(result.head(10))
    print(f"\nTotal predictions made: {len(result)}")
