"""
Utility functions for model serialization and common operations.

This module provides functions to pickle models, encoders, and other objects
for deployment and inference.
"""

import pickle
from pathlib import Path
from typing import Any, Union


def pickle_object(obj: Any, file_path: Union[str, Path]) -> None:
    """
    Pickle an object to a file.

    Args:
        obj: Object to pickle
        file_path: Path where to save the pickled object

    Raises:
        FileNotFoundError: If the directory doesn't exist
        PermissionError: If unable to write to the file
    """
    file_path = Path(file_path)

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        print(f"Object successfully pickled to: {file_path}")
    except Exception as e:
        print(f"Error pickling object to {file_path}: {e}")
        raise


def load_pickled_object(file_path: Union[str, Path]) -> Any:
    """
    Load a pickled object from a file.

    Args:
        file_path: Path to the pickled file

    Returns:
        The unpickled object

    Raises:
        FileNotFoundError: If the file doesn't exist
        pickle.UnpicklingError: If the file is corrupted
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Pickled file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        print(f"Object successfully loaded from: {file_path}")
        return obj
    except Exception as e:
        print(f"Error loading pickled object from {file_path}: {e}")
        raise


def save_model_and_preprocessors(
    model: Any, scaler: Any, label_encoder: Any, output_dir: Union[str, Path]
) -> None:
    """
    Save model and preprocessors to the deployment directory.

    Args:
        model: Trained model to save
        scaler: StandardScaler used for feature scaling
        label_encoder: LabelEncoder used for categorical encoding
        output_dir: Directory to save the objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "model.pkl"
    pickle_object(model, model_path)

    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    pickle_object(scaler, scaler_path)

    # Save label encoder
    encoder_path = output_dir / "label_encoder.pkl"
    pickle_object(label_encoder, encoder_path)

    print(f"Model and preprocessors saved to: {output_dir}")


def load_model_and_preprocessors(model_dir: Union[str, Path]) -> tuple[Any, Any, Any]:
    """
    Load model and preprocessors from the deployment directory.

    Args:
        model_dir: Directory containing the pickled objects

    Returns:
        Tuple of (model, scaler, label_encoder)
    """
    model_dir = Path(model_dir)

    # Load model
    model_path = model_dir / "model.pkl"
    model = load_pickled_object(model_path)

    # Load scaler
    scaler_path = model_dir / "scaler.pkl"
    scaler = load_pickled_object(scaler_path)

    # Load label encoder
    encoder_path = model_dir / "label_encoder.pkl"
    label_encoder = load_pickled_object(encoder_path)

    print(f"Model and preprocessors loaded from: {model_dir}")
    return model, scaler, label_encoder


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path

    Returns:
        Path object of the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
