"""
Preprocessing module for abalone age prediction.

This module handles data loading, feature engineering, and preprocessing
for the abalone dataset with 27 engineered features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the abalone dataset from CSV file.

    Args:
        data_path: Path to the CSV file

    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features (Sex) using LabelEncoder.

    Args:
        df: DataFrame with categorical features

    Returns:
        DataFrame with encoded features
    """
    le = LabelEncoder()
    df["Sex_encoded"] = le.fit_transform(df["Sex"])

    print("Sex mapping:")
    for i, sex in enumerate(le.classes_):
        print(f"{sex}: {i}")

    return df, le


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 27 engineered features from the original dataset.

    Args:
        df: DataFrame with basic features

    Returns:
        DataFrame with engineered features
    """
    print("=== FEATURE ENGINEERING ===")

    # 1. Log transformations for weights
    df["log_Whole weight"] = np.log1p(df["Whole weight"])
    df["log_Shucked weight"] = np.log1p(df["Shucked weight"])
    df["log_Viscera weight"] = np.log1p(df["Viscera weight"])
    df["log_Shell weight"] = np.log1p(df["Shell weight"])

    # 2. Ratios between weights
    df["ratio_shell_whole"] = df["Shell weight"] / (df["Whole weight"] + 1e-8)
    df["ratio_shucked_whole"] = df["Shucked weight"] / (df["Whole weight"] + 1e-8)
    df["ratio_viscera_whole"] = df["Viscera weight"] / (df["Whole weight"] + 1e-8)

    # 3. Geometric volume (approximation)
    df["geo_volume"] = df["Length"] * df["Diameter"] * df["Height"]

    # 4. Densities
    df["density_whole"] = df["Whole weight"] / (df["geo_volume"] + 1e-8)
    df["density_shell"] = df["Shell weight"] / (df["geo_volume"] + 1e-8)
    df["density_shucked"] = df["Shucked weight"] / (df["geo_volume"] + 1e-8)

    # 5. Interactions between dimensions
    df["Length*Diameter"] = df["Length"] * df["Diameter"]
    df["Diameter*Height"] = df["Diameter"] * df["Height"]

    # 6. Polynomial features
    df["Length_sq"] = df["Length"] ** 2
    df["Diameter_sq"] = df["Diameter"] ** 2
    df["Height_sq"] = df["Height"] ** 2

    # 7. Geometric ratios
    df["len_to_diam"] = df["Length"] / (df["Diameter"] + 1e-8)
    df["ht_to_diam"] = df["Height"] / (df["Diameter"] + 1e-8)
    df["wt_per_len"] = df["Whole weight"] / (df["Length"] + 1e-8)

    print(f"Created {27} engineered features")
    return df


def get_feature_columns() -> List[str]:
    """
    Get the list of feature columns used for training.

    Returns:
        List of feature column names
    """
    return [
        # Basic features
        "Sex_encoded",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        # Engineered features
        "log_Whole weight",
        "log_Shucked weight",
        "log_Viscera weight",
        "log_Shell weight",
        "ratio_shell_whole",
        "ratio_shucked_whole",
        "ratio_viscera_whole",
        "geo_volume",
        "density_whole",
        "density_shell",
        "density_shucked",
        "Length*Diameter",
        "Diameter*Height",
        "Length_sq",
        "Diameter_sq",
        "Height_sq",
        "len_to_diam",
        "ht_to_diam",
        "wt_per_len",
    ]


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target variable.

    Args:
        df: DataFrame with all features

    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["Rings"]

    print(f"Features selected: {len(feature_cols)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y


def split_and_scale_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Split data into train/test and apply StandardScaler.

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    print("=== DATA SPLITTING AND SCALING ===")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("StandardScaler normalization applied")
    print(f"Mean of scaled features (train): {X_train_scaled.mean(axis=0)[:5]}...")
    print(f"Std of scaled features (train): {X_train_scaled.std(axis=0)[:5]}...")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def preprocess_data(
    data_path: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder
]:
    """
    Complete preprocessing pipeline.

    Args:
        data_path: Path to the CSV file
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder)
    """
    # Load data
    df = load_data(data_path)

    # Encode categorical features
    df, label_encoder = encode_categorical_features(df)

    # Engineer features
    df = engineer_features(df)

    # Prepare features and target
    X, y = prepare_features_and_target(df)

    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size, random_state
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder
