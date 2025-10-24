"""
Script to create Prefect deployment for abalone training pipeline.

This script creates a deployment that can be used to schedule
regular model retraining.
"""

import os
from pathlib import Path
from prefect import serve
from prefect.server.schemas.schedules import CronSchedule

from .prefect_flow import abalone_training_flow


def create_deployment():
    """
    Create Prefect deployment for abalone training pipeline.
    """
    # Get the dataset path
    dataset_path = Path("abalone.csv").absolute()
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Create deployment using the new API (without schedule for now)
    deployment = abalone_training_flow.to_deployment(
        name="abalone-training-deployment",
        version="1.0.0",
        description="Abalone age prediction model training deployment",
        parameters={
            "trainset_path": str(dataset_path),
            "output_dir": "src/web_service/local_objects"
        },
        tags=["ml", "training", "abalone"]
    )
    
    return deployment


def create_deployment_with_ui():
    """
    Create Prefect deployment with MLflow UI launcher.
    """
    from .prefect_flow import abalone_training_with_ui_flow
    
    # Get the dataset path
    dataset_path = Path("abalone.csv").absolute()
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Create deployment using the new API (without schedule for now)
    deployment = abalone_training_with_ui_flow.to_deployment(
        name="abalone-training-with-ui-deployment",
        version="1.0.0",
        description="Abalone training deployment with MLflow UI launcher",
        parameters={
            "trainset_path": str(dataset_path),
            "output_dir": "src/web_service/local_objects"
        },
        tags=["ml", "training", "abalone", "ui"]
    )
    
    return deployment


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Prefect deployment for abalone training")
    parser.add_argument(
        "--with-ui", 
        action="store_true", 
        help="Create deployment with MLflow UI launcher"
    )
    parser.add_argument(
        "--serve", 
        action="store_true", 
        help="Serve the deployment immediately"
    )
    
    args = parser.parse_args()
    
    if args.with_ui:
        deployment = create_deployment_with_ui()
        print("Created deployment with MLflow UI launcher")
    else:
        deployment = create_deployment()
        print("Created standard deployment")
    
    if args.serve:
        print("Serving deployment...")
        serve(deployment)
    else:
        print("Deployment created. Use --serve to start serving.")
        print("Or use: prefect deployment apply <deployment-name>")
