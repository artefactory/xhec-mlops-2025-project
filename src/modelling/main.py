# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path
from preprocessing import preprocess_data
from training import train_model


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    df = preprocess_data(trainset_path)
    output_dir = (
        Path(__file__).resolve().parent.parent / "web_service" / "local_objects"
    )

    train_model(df, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
