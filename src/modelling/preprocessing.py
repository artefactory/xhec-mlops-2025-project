"""Module for data preprocessing."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(filepath):
    """
    Load the abalone dataset from a CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        pandas DataFrame with the loaded data
    """
    df = pd.read_csv(filepath)
    return df


def prepare_features(df):
    """
    Prepare features and target from the dataframe.
    Creates an Age variable (Rings + 1.5) but uses Rings as the target.

    Args:
        df: pandas DataFrame with the abalone data

    Returns:
        Tuple of (X, y) where X contains features and y contains target (Rings)
    """
    # Create Age variable (though not used as target)
    df["Age"] = df["Rings"] + 1.5

    # Drop both Rings and Age from features, use Rings as target
    X = df.drop(columns=["Rings", "Age"])
    y = df["Rings"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.

    Args:
        X: Features
        y: Target
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def create_preprocessor():
    """
    Create a preprocessing pipeline for the abalone dataset.

    Returns:
        sklearn ColumnTransformer with OneHotEncoder for 'Sex' and StandardScaler for numerical features
    """
    categorical = ["Sex"]
    numerical = [
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical),
            ("num", StandardScaler(), numerical),
        ]
    )

    return preprocessor
