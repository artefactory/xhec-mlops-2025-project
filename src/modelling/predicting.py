# Predicting function for a Linear Regression model

from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator

from .config import MODEL_DIR
from .utils import load_from_pickle


def predict(model: Optional[BaseEstimator], new_features: pd.DataFrame) -> pd.Series:
    """Make predictions using the trained Linear Regression model.

    Args:
        model: The trained Linear Regression model.
        new_features (pd.DataFrame): The input features for prediction.

    Returns:
        pd.Series: The predicted target variable.
    """
    if model is None:
        model = load_from_pickle(MODEL_DIR / "model.pkl")

    predictions = model.predict(new_features)
    return pd.Series(predictions)
