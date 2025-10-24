"""
Prefect tasks for abalone age prediction pipeline.

This module contains Prefect tasks that wrap the existing functionality
to enable workflow orchestration and monitoring.
"""

from pathlib import Path
from prefect import task, get_run_logger
import mlflow
import mlflow.sklearn

from .preprocessing import preprocess_data, get_feature_columns
from .training import train_all_models
from .utils import save_model_and_preprocessors


@task(name="setup_mlflow")
def setup_mlflow_task(experiment_name: str = "abalone_age_prediction") -> None:
    """
    Setup MLflow tracking task.

    Args:
        experiment_name: Name of the MLflow experiment
    """
    logger = get_run_logger()
    
    # Set tracking URI to local file system
    mlflow.set_tracking_uri("file:./mlruns")

    # Set experiment
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow experiment: {mlflow.get_experiment_by_name(experiment_name)}")


@task(name="load_and_preprocess_data")
def load_and_preprocess_data_task(trainset_path: str):
    """
    Load and preprocess data task.
    
    Args:
        trainset_path: Path to the training dataset
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, feature_columns)
    """
    logger = get_run_logger()
    logger.info("Loading and preprocessing data...")
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = (
        preprocess_data(trainset_path, test_size=0.2, random_state=42)
    )

    # Get feature columns
    feature_cols = get_feature_columns()
    
    logger.info(f"Using {len(feature_cols)} features for training")
    logger.info(f"Training set shape: {X_train_scaled.shape}")
    logger.info(f"Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, feature_cols


@task(name="train_models")
def train_models_task(X_train_scaled, y_train, X_test_scaled, y_test, feature_cols):
    """
    Train all models task.
    
    Args:
        X_train_scaled, X_test_scaled, y_train, y_test: Training and test data
        feature_cols: List of feature column names
        
    Returns:
        Dictionary of model results
    """
    logger = get_run_logger()
    logger.info("Training models...")
    
    results = train_all_models(
        X_train_scaled, y_train, X_test_scaled, y_test, feature_cols
    )
    
    logger.info(f"Trained {len(results)} models")
    
    return results


@task(name="select_best_model")
def select_best_model_task(results):
    """
    Select best model based on R² score task.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        Tuple of (best_model_name, best_model, best_metrics)
    """
    logger = get_run_logger()
    logger.info("Selecting best model...")
    
    best_model_name = max(results.keys(), key=lambda k: results[k]["metrics"]["r2"])
    best_model = results[best_model_name]["model"]
    best_metrics = results[best_model_name]["metrics"]
    
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best R² score: {best_metrics['r2']:.3f}")
    
    return best_model_name, best_model, best_metrics


@task(name="save_best_model")
def save_best_model_task(best_model, scaler, label_encoder, output_dir):
    """
    Save best model and preprocessors task.
    
    Args:
        best_model: The best trained model
        scaler: Fitted scaler
        label_encoder: Fitted label encoder
        output_dir: Directory to save the model
        
    Returns:
        Path to saved model
    """
    logger = get_run_logger()
    logger.info(f"Saving best model to {output_dir}...")
    
    save_model_and_preprocessors(
        model=best_model,
        scaler=scaler,
        label_encoder=label_encoder,
        output_dir=output_dir,
    )
    
    model_path = Path(output_dir) / "model.pkl"
    logger.info(f"Model saved to: {model_path}")
    
    return str(model_path)


@task(name="log_model_comparison")
def log_model_comparison_task(results, best_model_name):
    """
    Log model comparison results task.
    
    Args:
        results: Dictionary of model results
        best_model_name: Name of the best model
    """
    logger = get_run_logger()
    logger.info("=== MODEL COMPARISON ===")
    
    for model_name, result in results.items():
        metrics = result["metrics"]
        marker = "🏆" if model_name == best_model_name else "  "
        logger.info(f"{marker} {model_name}:")
        logger.info(f"    RMSE: {metrics['rmse']:.3f}")
        logger.info(f"    R²: {metrics['r2']:.3f}")
        logger.info(f"    MAE: {metrics['mae']:.3f}")
        logger.info(f"    CV R²: {metrics['cv_r2_mean']:.3f} (+/- {metrics['cv_r2_std']:.3f})")
