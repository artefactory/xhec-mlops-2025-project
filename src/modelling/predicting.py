import pandas as pd
from pathlib import Path
from src.modelling.utils import load_pickle
import numpy as np
from src.modelling.preprocessing import preprocess_data


def load_model(model_path: Path):
    return load_pickle(model_path)


def predict(data: pd.DataFrame, model):
    return model.predict(data)


def predict_on_unseen_data(
    input__data_filepath,
    model_path,
) -> np.ndarray:
    """Using the pre train model to predict new data"""
    data = preprocess_data(input__data_filepath)
    if "rings" in data.columns:
        data = data.drop(columns=["rings"])

    model = load_model(model_path)

    y_pred = predict(data=data, model=model)
    return y_pred
