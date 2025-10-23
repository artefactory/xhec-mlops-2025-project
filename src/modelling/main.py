# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from loguru import logger
from prefect import flow

from .config import DATA_DIR
from .preprocessing import preprocess_data
from .training import train_model
from .utils import load_data, save_to_pickle


@flow(name="training-pipeline", log_prints=True)
def main(trainset_path: Path = DATA_DIR / "abalone.csv") -> None:
    """Train a model using the data at the given path and save the model (pickle).

    This is the main training flow that orchestrates all the training tasks.

    Args:
        trainset_path: Path to the training dataset
    """
    # Read data
    logger.info(f"Loading data from {trainset_path}")
    df = load_data(trainset_path)

    # Preprocess data
    logger.info("Preprocessing data")
    x, y, scaler = preprocess_data(df)

    # (Optional) Pickle encoder if need be
    logger.info("Saving scaler to src/web_service/local_objects/scaler.pkl")
    save_to_pickle(scaler, "src/web_service/local_objects/scaler.pkl")

    # Train model
    logger.info("Training model")
    model = train_model(x, y)

    # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder
    logger.info("Saving trained model to src/web_service/local_objects/model.pkl")
    save_to_pickle(model, "src/web_service/local_objects/model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(Path(args.trainset_path))
