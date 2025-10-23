# Training function for a Linear Regression model

import pandas as pd
from prefect import task
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task(name="train-model")
def train_model(x: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """Train a Linear Regression model on the provided features and target.

    Args:
        x (pd.DataFrame): The input features.
        y (pd.Series): The target variable.

    Returns:
        LinearRegression: The trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(x, y)
    return model


@task(name="evaluate-model")
def evaluate_model(model: LinearRegression, x: pd.DataFrame, y: pd.Series) -> float:
    """Evaluate the model using Mean Squared Error (MSE).

    Args:
        model (LinearRegression): The trained Linear Regression model.
        x (pd.DataFrame): The input features.
        y (pd.Series): The true target variable.

    Returns:
        float: The Mean Squared Error of the model's predictions.
    """
    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    return mse
