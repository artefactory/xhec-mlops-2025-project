"""
Training module for abalone age prediction with MLflow integration.

This module handles model training, evaluation, and MLflow experiment tracking
for Linear Regression and Random Forest models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from typing import Dict, Any
import warnings

warnings.filterwarnings("ignore")


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list,
    run_name: str = "Linear_Regression_Baseline",
) -> Dict[str, Any]:
    """
    Train a Linear Regression model with MLflow tracking.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_cols: List of feature column names
        run_name: Name for the MLflow run

    Returns:
        Dictionary with model and metrics
    """
    print("=== LINEAR REGRESSION TRAINING ===")

    with mlflow.start_run(run_name=run_name):
        # Model parameters
        model_params = {"fit_intercept": True}

        # Train model
        lr = LinearRegression(**model_params)
        lr.fit(X_train, y_train)

        # Predictions
        y_pred = lr.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="r2")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        print("Linear Regression - Metrics:")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"R² Cross-validation (5-fold): {cv_mean:.3f} (+/- {cv_std * 2:.3f})")

        # MLflow logging
        mlflow.log_params(model_params)
        mlflow.log_metrics(
            {
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
                "cv_r2_mean": cv_mean,
                "cv_r2_std": cv_std,
            }
        )

        # Log model
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(lr, "model", signature=signature)

        # Log metadata
        mlflow.log_param("features", feature_cols)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("model_type", "LinearRegression")

        print("✅ Linear Regression model logged in MLflow!")

        return {
            "model": lr,
            "metrics": {
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
                "cv_r2_mean": cv_mean,
                "cv_r2_std": cv_std,
            },
            "predictions": y_pred,
        }


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list,
    run_name: str = "Random_Forest_Default",
    model_params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Train a Random Forest model with MLflow tracking.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_cols: List of feature column names
        run_name: Name for the MLflow run
        model_params: Model parameters (default: basic Random Forest)

    Returns:
        Dictionary with model and metrics
    """
    print("=== RANDOM FOREST TRAINING ===")

    # Default parameters
    if model_params is None:
        model_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }

    with mlflow.start_run(run_name=run_name):
        # Train model
        rf = RandomForestRegressor(**model_params)
        rf.fit(X_train, y_train)

        # Predictions
        y_pred = rf.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        print("Random Forest - Metrics:")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"R² Cross-validation (5-fold): {cv_mean:.3f} (+/- {cv_std * 2:.3f})")

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\nFeature importance:")
        print(feature_importance.head(10))

        # MLflow logging
        mlflow.log_params(model_params)
        mlflow.log_metrics(
            {
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
                "cv_r2_mean": cv_mean,
                "cv_r2_std": cv_std,
            }
        )

        # Log model
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(rf, "model", signature=signature)

        # Log metadata
        mlflow.log_param("features", feature_cols)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("model_type", "RandomForestRegressor")

        # Log feature importance (sanitize feature names for MLflow)
        for feature, importance in zip(feature_cols, rf.feature_importances_):
            # Replace special characters with underscores for MLflow compatibility
            sanitized_feature = (
                feature.replace("*", "_").replace(" ", "_").replace("-", "_")
            )
            mlflow.log_metric(f"feature_importance_{sanitized_feature}", importance)

        print("✅ Random Forest model logged in MLflow!")

        return {
            "model": rf,
            "metrics": {
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
                "cv_r2_mean": cv_mean,
                "cv_r2_std": cv_std,
            },
            "predictions": y_pred,
            "feature_importance": feature_importance,
        }


def train_optimized_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list,
) -> Dict[str, Any]:
    """
    Train an optimized Random Forest model with MLflow tracking.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_cols: List of feature column names

    Returns:
        Dictionary with model and metrics
    """
    print("=== OPTIMIZED RANDOM FOREST TRAINING ===")

    # Optimized parameters
    best_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": 42,
    }

    return train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
        run_name="Best_Random_Forest_Optimized",
        model_params=best_params,
    )


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all models (Linear Regression, Random Forest, Optimized Random Forest).

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_cols: List of feature column names

    Returns:
        Dictionary with all trained models and their results
    """
    print("=== TRAINING ALL MODELS ===")

    results = {}

    # Train Linear Regression
    results["linear_regression"] = train_linear_regression(
        X_train, y_train, X_test, y_test, feature_cols
    )

    # Train Random Forest
    results["random_forest"] = train_random_forest(
        X_train, y_train, X_test, y_test, feature_cols
    )

    # Train Optimized Random Forest
    results["optimized_random_forest"] = train_optimized_random_forest(
        X_train, y_train, X_test, y_test, feature_cols
    )

    # Compare models
    print("\n=== MODEL COMPARISON ===")
    for model_name, result in results.items():
        metrics = result["metrics"]
        print(f"{model_name}:")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  CV R²: {metrics['cv_r2_mean']:.3f} (+/- {metrics['cv_r2_std']:.3f})")
        print()

    return results
