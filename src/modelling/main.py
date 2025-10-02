# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data

    # Preprocess data

    # (Optional) Pickle encoder if need be

    # Train model

    # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
