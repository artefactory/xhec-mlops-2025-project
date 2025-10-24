# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from prefect import flow
from src.modelling.preprocessing import prepare_data
from src.modelling.training import train
from pathlib import Path


@flow(name="training-pipeline")
def main(trainset_path: Path = Path("data/abalone.csv")) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Prepare data (subflow: load, encode, split)
    print("Preparing data...")
    X_train, _, y_train, _ = prepare_data(trainset_path)
    # (Optional) Pickle encoder if need be

    # Train model
    print("Training the model...")
    train(X_train, y_train, savepath="src/web_service/local_objects")
    # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument(
        "--trainset_path",
        type=str,
        default="data/abalone.csv",
        help="Path to the training set (default: data/abalone.csv)",
    )
    args = parser.parse_args()
    main(Path(args.trainset_path))
