"""Main training flow for the abalone age prediction model."""

import argparse
from pathlib import Path

from preprocessing import create_preprocessor, load_data, prepare_features, split_data
from training import evaluate_model, train_model
from utils import pickle_object


def main(trainset_path: str) -> None:
    """
    Train a model using the data at the given path and save the model (pickle).

    Args:
        trainset_path: Path to the training dataset CSV file
    """
    print("=" * 50)
    print("Starting model training pipeline")
    print("=" * 50)

    # Read data
    print(f"\n1. Loading data from {trainset_path}...")
    df = load_data(trainset_path)
    print(f"   Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Preprocess data
    print("\n2. Preparing features and target...")
    X, y = prepare_features(df)

    print("\n3. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")

    print("\n4. Creating preprocessor...")
    preprocessor = create_preprocessor()

    # Train model
    print("\n5. Training model...")
    model = train_model(preprocessor, X_train, y_train)
    print("   Model training completed!")

    # Evaluate model
    print("\n6. Evaluating model on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"   MAE  : {metrics['mae']:.3f}")
    print(f"   RMSE : {metrics['rmse']:.3f}")
    print(f"   R²   : {metrics['r2']:.3f}")

    # Pickle model
    print("\n7. Saving model...")
    model_path = (
        Path(__file__).parent.parent / "web_service" / "local_objects" / "model.pkl"
    )
    pickle_object(model, model_path)

    print("\n" + "=" * 50)
    print("Training pipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
