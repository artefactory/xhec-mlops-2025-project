from sklearn.ensemble import RandomForestRegressor
from .utils import pickle_object
from prefect import task
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


@task
def train(
    X,
    y,
    X_test=None,
    y_test=None,
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    savepath=None,
):
    # Configure MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    
    with mlflow.start_run(run_name="abalone-age-prediction"):
        # Log parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs
        })
        
        # Train model
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        rf.fit(X, y)
        
        # Log model
        mlflow.sklearn.log_model(rf, "model")
        
        # Evaluate and log metrics if test data provided
        if X_test is not None and y_test is not None:
            y_pred = rf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_metrics({
                "mse": mse,
                "rmse": np.sqrt(mse),
                "r2_score": r2
            })
            
            print(f"📊 Model metrics - MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f}")

        # Save model locally if path provided
        if savepath:
            pickle_object(rf, os.path.join(savepath, "model.pkl"))

        return rf
