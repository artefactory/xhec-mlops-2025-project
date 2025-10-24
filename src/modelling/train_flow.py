"""Prefect-based training flow for the Abalone age prediction model."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from prefect import flow, task
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess_data
from utils import pickle_object


@task(name="Load and Preprocess Data", retries=2, retry_delay_seconds=10)
def load_and_preprocess_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess the abalone dataset.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Preprocessed DataFrame
    """
    df = preprocess_data(csv_path)
    return df


@task(name="Prepare Features and Target")
def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target.

    Args:
        df: Preprocessed dataframe

    Returns:
        Tuple of (X, y)
    """
    X = df.drop(columns=["rings"])
    y = df["rings"]
    return X, y


@task(name="Split Train Test")
def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


@task(name="Build Pipeline")
def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build the sklearn pipeline with preprocessing and model.

    Args:
        X: Features dataframe to determine column types

    Returns:
        Sklearn Pipeline
    """
    dummy_cols = [c for c in X.columns if c.startswith("sex_")]
    num_cols = [c for c in X.columns if c not in dummy_cols]

    preproc = ColumnTransformer(
        [
            ("scale_num", StandardScaler(), num_cols),
            ("pass_dummies", "passthrough", dummy_cols),
        ]
    )

    model = Pipeline([("preprocess", preproc), ("regressor", LinearRegression())])

    return model


@task(name="Train Model")
def train_model_task(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Train the model pipeline.

    Args:
        model: Sklearn pipeline
        X_train: Training features
        y_train: Training target

    Returns:
        Fitted pipeline
    """
    model.fit(X_train, y_train)
    return model


@task(name="Evaluate Model")
def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model performance on test set.

    Args:
        model: Fitted pipeline
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return metrics


@task(name="Save Model")
def save_model(model: Pipeline, output_dir: Path) -> None:
    """Save the trained model to disk.

    Args:
        model: Fitted pipeline
        output_dir: Directory to save the model
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"
    pickle_object(model, model_path)


@flow(name="Train Abalone Age Prediction Model", log_prints=True)
def training_flow(
    trainset_path: Path,
    output_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Complete training flow for the abalone age prediction model.

    Args:
        trainset_path: Path to the training dataset CSV
        output_dir: Directory to save the model (default: src/web_service/local_objects)
        test_size: Proportion of test set
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with evaluation metrics
    """
    print("Starting Abalone Age Prediction training flow...")

    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent / "web_service" / "local_objects"
        )

    # Load and preprocess data
    df = load_and_preprocess_data(trainset_path)
    print(f"Loaded and preprocessed {len(df)} samples")

    # Prepare features and target
    X, y = prepare_features_target(df)
    print(f"Prepared features: {X.shape}, target: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Split data - Train: {len(X_train)}, Test: {len(X_test)}")

    # Build pipeline
    model = build_pipeline(X_train)
    print("Built preprocessing and model pipeline")

    # Train model
    model = train_model_task(model, X_train, y_train)
    print("Model trained")

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Metrics:")
    print(f"  - MAE: {metrics['mae']:.4f}")
    print(f"  - MSE: {metrics['mse']:.4f}")
    print(f"  - R²: {metrics['r2']:.4f}")

    # Save model
    save_model(model, output_dir)
    print(f"Model saved to {output_dir / 'model.pkl'}")

    print("Training flow completed successfully!")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model using Prefect flow with the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()

    metrics = training_flow(Path(args.trainset_path))
