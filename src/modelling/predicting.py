import pandas as pd
from pathlib import Path
from utils import load_pickle
import numpy as np
from typing import Optional
from src.modelling.preprocessing import preprocess_data


def load_model(model_path: Path):
    return load_pickle(model_path)


def predict(data: pd.DataFrame, model_path: Optional[Path], model=None):
    if model is None:
        model = load_model(model_path)
    return model.predict(data)


def predict_on_unseen_data(
    input__data_filepath: Optional[str],
    data: Optional[pd.DataFrame],
    model_path: Optional[str],
    model=None,
) -> np.ndarray:
    """Using the pre train model to predict new data"""
    if input__data_filepath and not data:
        data = preprocess_data(filepath=input__data_filepath)
    if not data and not input__data_filepath:
        raise ValueError("Either 'data' or 'input__data_filepath' must be provided.")
    if "rings" in data.columns:
        data = data.drop(columns=["rings"])

    if model is None:
        if model_path is None:
            raise ValueError("Either 'model' or 'model_path' must be provided.")
        model = load_model(model_path)

    y_pred = predict(data=data, model=model)
    return y_pred
