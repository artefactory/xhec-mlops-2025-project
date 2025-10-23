"""Utility functions for model serialization."""

import pickle
from pathlib import Path


def pickle_object(obj, filepath):
    """
    Serialize and save a Python object using pickle.

    Args:
        obj: Python object to serialize (e.g., trained model, preprocessor)
        filepath: Path where the pickled object should be saved
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

    print(f"Object saved to {filepath}")


def load_pickle_object(filepath):
    """
    Load a pickled Python object.

    Args:
        filepath: Path to the pickled object file

    Returns:
        Deserialized Python object
    """
    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    return obj
