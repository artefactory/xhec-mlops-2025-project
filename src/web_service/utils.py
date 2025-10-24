"""
Utility functions for the web service.

This module provides helper functions for loading pickled objects
and other common utilities.
"""

import pickle
from pathlib import Path
from typing import Any, Union


def load_object(file_path: Union[str, Path]) -> Any:
    """
    Load a pickled object from file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        The unpickled object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pickle.UnpicklingError: If the file cannot be unpickled
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    
    return obj
