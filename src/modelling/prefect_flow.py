"""
Prefect flow for abalone age prediction pipeline.

This module contains the main Prefect flow that orchestrates the complete
training pipeline using Prefect tasks.
"""

from pathlib import Path
from prefect import flow, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

from .prefect_tasks import (
    setup_mlflow_task,
    load_and_preprocess_data_task,
    train_models_task,
    select_best_model_task,
    save_best_model_task,
    log_model_comparison_task,
)


@flow(
    name="abalone_training_pipeline",
    task_runner=ConcurrentTaskRunner(),
    description="Complete training pipeline for abalone age prediction with MLflow integration"
)
def abalone_training_flow(
    trainset_path: str, 
    output_dir: str = "src/web_service/local_objects"
) -> dict:
    """
    Complete training pipeline with Prefect and MLflow integration.

    Args:
        trainset_path: Path to the training dataset
        output_dir: Directory to save the trained model and preprocessors
        
    Returns:
        Dictionary with training results and model path
    """
    logger = get_run_logger()
    logger.info("=== ABALONE AGE PREDICTION TRAINING PIPELINE ===")

    # Setup MLflow
    setup_mlflow_task()

    # Load and preprocess data
    logger.info("1. PREPROCESSING DATA")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, feature_cols = (
        load_and_preprocess_data_task(trainset_path)
    )

    # Train all models with MLflow tracking
    logger.info("2. TRAINING MODELS")
    results = train_models_task(X_train_scaled, y_train, X_test_scaled, y_test, feature_cols)

    # Select best model based on R² score
    logger.info("3. SELECTING BEST MODEL")
    best_model_name, best_model, best_metrics = select_best_model_task(results)

    # Save best model and preprocessors
    logger.info(f"4. SAVING MODEL TO {output_dir}")
    model_path = save_best_model_task(best_model, scaler, label_encoder, output_dir)

    # Log model comparison
    log_model_comparison_task(results, best_model_name)

    # Print summary
    logger.info("=== TRAINING COMPLETE ===")
    logger.info(f"✅ Best model: {best_model_name}")
    logger.info(f"✅ Model saved to: {model_path}")
    logger.info("✅ MLflow experiment: abalone_age_prediction")
    logger.info(f"✅ Features used: {len(feature_cols)}")
    logger.info(f"✅ Training samples: {len(X_train_scaled)}")
    logger.info(f"✅ Test samples: {len(X_test_scaled)}")

    return {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "model_path": model_path,
        "feature_count": len(feature_cols),
        "training_samples": len(X_train_scaled),
        "test_samples": len(X_test_scaled),
        "all_results": results
    }


@flow(
    name="abalone_training_with_ui",
    task_runner=ConcurrentTaskRunner(),
    description="Training pipeline with MLflow UI launcher"
)
def abalone_training_with_ui_flow(
    trainset_path: str, 
    output_dir: str = "src/web_service/local_objects"
) -> dict:
    """
    Complete training pipeline with MLflow UI launcher.

    Args:
        trainset_path: Path to the training dataset
        output_dir: Directory to save the trained model and preprocessors
        
    Returns:
        Dictionary with training results and model path
    """
    logger = get_run_logger()
    
    # Run main training pipeline
    results = abalone_training_flow(trainset_path, output_dir)

    # Launch MLflow UI
    logger.info("=== LAUNCHING MLFLOW UI ===")
    try:
        import subprocess
        import webbrowser
        import time
        import os

        # Start MLflow UI in background
        process = subprocess.Popen(
            ["mlflow", "ui"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )

        # Wait a moment for the server to start
        time.sleep(3)

        # Open browser
        webbrowser.open("http://localhost:5000")

        logger.info("✅ MLflow UI launched at: http://localhost:5000")
        logger.info("📊 You can now view your experiments in the browser!")
        logger.info("🛑 Press Ctrl+C to stop the MLflow UI server")

        # Keep the process running
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("🛑 Stopping MLflow UI server...")
            process.terminate()

    except Exception as e:
        logger.error(f"❌ Could not launch MLflow UI: {e}")
        logger.info("💡 You can manually run: mlflow ui")

    return results
