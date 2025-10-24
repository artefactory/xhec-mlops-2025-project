"""
Script to run the Prefect flow for abalone training.

This script provides a command-line interface to run the training pipeline
using Prefect flows.
"""

import argparse
import asyncio
from pathlib import Path

from .prefect_flow import abalone_training_flow, abalone_training_with_ui_flow


def main():
    """
    Main function to run the Prefect flow.
    """
    parser = argparse.ArgumentParser(
        description="Run abalone age prediction training with Prefect"
    )
    parser.add_argument(
        "trainset_path", 
        type=str, 
        help="Path to the training dataset (CSV file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/web_service/local_objects",
        help="Directory to save the trained model and preprocessors",
    )
    parser.add_argument(
        "--launch-ui", 
        action="store_true", 
        help="Launch MLflow UI after training"
    )
    parser.add_argument(
        "--async-run", 
        action="store_true", 
        help="Run flow asynchronously"
    )

    args = parser.parse_args()

    # Check if dataset exists
    if not Path(args.trainset_path).exists():
        print(f"❌ Dataset not found at: {args.trainset_path}")
        return 1

    # Choose the appropriate flow
    if args.launch_ui:
        flow_func = abalone_training_with_ui_flow
        print("🚀 Running training pipeline with MLflow UI launcher...")
    else:
        flow_func = abalone_training_flow
        print("🚀 Running training pipeline...")

    try:
        if args.async_run:
            # Run asynchronously
            result = asyncio.run(flow_func(args.trainset_path, args.output_dir))
        else:
            # Run synchronously
            result = flow_func(args.trainset_path, args.output_dir)

        print("\n✅ Training completed successfully!")
        print(f"📊 Best model: {result['best_model_name']}")
        print(f"📈 R² score: {result['best_metrics']['r2']:.3f}")
        print(f"📁 Model saved to: {result['model_path']}")
        
        return 0

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
