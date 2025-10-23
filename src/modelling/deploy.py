"""
Deployment script for the Abalone training pipeline.

This script creates a Prefect deployment for automated model retraining.
Run this script to set up scheduled training runs.
"""

from prefect import serve

from .config import DATA_DIR
from .main import main as training_flow


def create_deployment():
    """Create and serve the training pipeline deployment.

    This deployment will retrain the model on a schedule.
    The schedule is set to run daily at 2 AM UTC.
    """
    # Create deployment with schedule
    deployment = training_flow.to_deployment(
        name="abalone-training-daily",
        version="1.0",
        tags=["production", "ml", "abalone", "training"],
        description="Daily automated training pipeline for Abalone age prediction model",
        parameters={"trainset_path": DATA_DIR / "abalone.csv"},
        # Schedule: Run daily at 1 AM UTC
        cron="0 1 * * *",
    )

    # Serve the deployment
    # This keeps the deployment running and listening for scheduled runs
    serve(deployment)


if __name__ == "__main__":
    print("Starting Abalone Training Pipeline Deployment...")
    print("Schedule: Daily at 2 AM UTC")
    print("Press Ctrl+C to stop the deployment")
    create_deployment()
