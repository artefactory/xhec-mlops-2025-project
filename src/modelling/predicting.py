import pandas as pd
from pathlib import Path
from utils import load_pickle


def load_model(model_path: Path):
    return load_pickle(model_path)


def predict(model, data: pd.DataFrame):
    return model.predict(data)
