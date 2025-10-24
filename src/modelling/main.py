"""
Main training pipeline for abalone age prediction with MLflow and Prefect integration.

This module orchestrates the complete training flow: data loading, preprocessing,
model training, evaluation, and saving for deployment using Prefect flows and tasks.
"""

import argparse
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from .preprocessing import preprocess_data, get_feature_columns
from .training import train_all_models
from .utils import save_model_and_preprocessors

warnings.filterwarnings("ignore")


@task(name="setup_mlflow")
def setup_mlflow(experiment_name: str = "abalone_age_prediction") -> None:
    """
    Setup MLflow tracking.

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


def main(trainset_path: str, output_dir: str = "src/web_service/local_objects") -> None:
    """
    Complete training pipeline with MLflow integration.

    Args:
        trainset_path: Path to the training dataset
        output_dir: Directory to save the trained model and preprocessors
    """
    print("=== ABALONE AGE PREDICTION TRAINING PIPELINE ===")

    # Setup MLflow
    setup_mlflow()

    # Preprocess data
    print("\n1. PREPROCESSING DATA")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = (
        preprocess_data(trainset_path, test_size=0.2, random_state=42)
    )

    # Get feature columns
    feature_cols = get_feature_columns()
    print(f"Using {len(feature_cols)} features for training")

    # Train all models with MLflow tracking
    print("\n2. TRAINING MODELS")
    results = train_all_models(
        X_train_scaled, y_train, X_test_scaled, y_test, feature_cols
    )

    # Select best model based on R² score
    print("\n3. SELECTING BEST MODEL")
    best_model_name = max(results.keys(), key=lambda k: results[k]["metrics"]["r2"])
    best_model = results[best_model_name]["model"]

    print(f"Best model: {best_model_name}")
    print(f"Best R² score: {results[best_model_name]['metrics']['r2']:.3f}")

    # Save best model and preprocessors
    print(f"\n4. SAVING MODEL TO {output_dir}")
    save_model_and_preprocessors(
        model=best_model,
        scaler=scaler,
        label_encoder=label_encoder,
        output_dir=output_dir,
    )

    # Print summary
    print("\n=== TRAINING COMPLETE ===")
    print(f"✅ Best model: {best_model_name}")
    print(f"✅ Model saved to: {output_dir}")
    print("✅ MLflow experiment: abalone_age_prediction")
    print(f"✅ Features used: {len(feature_cols)}")
    print(f"✅ Training samples: {len(X_train_scaled)}")
    print(f"✅ Test samples: {len(X_test_scaled)}")

    # Print model comparison
    print("\n=== MODEL COMPARISON ===")
    for model_name, result in results.items():
        metrics = result["metrics"]
        marker = "🏆" if model_name == best_model_name else "  "
        print(f"{marker} {model_name}:")
        print(f"    RMSE: {metrics['rmse']:.3f}")
        print(f"    R²: {metrics['r2']:.3f}")
        print(f"    MAE: {metrics['mae']:.3f}")
        print(
            f"    CV R²: {metrics['cv_r2_mean']:.3f} (+/- {metrics['cv_r2_std']:.3f})"
        )
        print()


def main_with_mlflow_ui_launch(
    trainset_path: str, output_dir: str = "src/web_service/local_objects"
) -> None:
    """
    Complete training pipeline with MLflow UI launcher.

    Args:
        trainset_path: Path to the training dataset
        output_dir: Directory to save the trained model and preprocessors
    """
    # Run main training pipeline
    main(trainset_path, output_dir)

    # Launch MLflow UI
    print("\n=== LAUNCHING MLFLOW UI ===")
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

        print("✅ MLflow UI launched at: http://localhost:5000")
        print("📊 You can now view your experiments in the browser!")
        print("🛑 Press Ctrl+C to stop the MLflow UI server")

        # Keep the process running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping MLflow UI server...")
            process.terminate()

    except Exception as e:
        print(f"❌ Could not launch MLflow UI: {e}")
        print("💡 You can manually run: mlflow ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train abalone age prediction models with MLflow integration."
    )
    parser.add_argument(
        "trainset_path", type=str, help="Path to the training dataset (CSV file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/web_service/local_objects",
        help="Directory to save the trained model and preprocessors",
    )
    parser.add_argument(
        "--launch-ui", action="store_true", help="Launch MLflow UI after training"
    )

    args = parser.parse_args()

    if args.launch_ui:
        main_with_mlflow_ui_launch(args.trainset_path, args.output_dir)
    else:
        main(args.trainset_path, args.output_dir)
