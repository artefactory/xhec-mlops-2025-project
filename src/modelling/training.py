"""Module for training the abalone age prediction model."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def train_model(preprocessor, X_train, y_train):
    """
    Train a linear regression model with preprocessing.

    Args:
        preprocessor: sklearn ColumnTransformer for data preprocessing
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained sklearn Pipeline containing preprocessor and model
    """
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary containing evaluation metrics (MAE, RMSE, R2)
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    return metrics
