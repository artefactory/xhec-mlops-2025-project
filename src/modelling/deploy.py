"""Create and serve Prefect deployment for automated model retraining."""

from pathlib import Path
from prefect import serve
from train_flow import training_flow


def create_deployment():
    """Create a Prefect deployment for regular model retraining.
    The deployment will run every Sunday at 2 AM to retrain the model.
    """
    # Default paths
    default_trainset_path = Path("data/abalone.csv")
    default_output_dir = Path("src/web_service/local_objects")

    deployment = training_flow.to_deployment(
        name="abalone-weekly-retraining",
        parameters={
            "trainset_path": default_trainset_path,
            "output_dir": default_output_dir,
            "test_size": 0.2,
            "random_state": 42,
        },
        cron="0 2 * * 0",  # Every Sunday at 2 AM
        tags=["ml", "training", "abalone", "weekly"],
        description="Automated weekly retraining of the abalone age prediction model",
    )

    return deployment


if __name__ == "__main__":
    print("Creating and serving Prefect deployment...")
    print("Schedule: Every Sunday at 2:00 AM")
    print("Prefect UI: http://127.0.0.1:4200")
    print("\nPress Ctrl+C to stop the deployment server\n")

    deployment = create_deployment()
    serve(deployment)
