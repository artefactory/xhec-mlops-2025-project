# Preprocessing functions for the abalone dataset.

from typing import Optional, Tuple

import pandas as pd
from prefect import task
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NUMERICAL_COLS


def onehot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the the categorical variable 'Sex' in the abalone dataset.

    Args:
        df (pd.DataFrame): The input dataframe with all columns.

    Returns:
        pd.DataFrame: The dataframe with the 'Sex' column one-hot encoded.
    """
    df = df.copy()
    sex_encoder = OneHotEncoder(drop="first", sparse_output=False)
    sex_encoded = sex_encoder.fit_transform(df[["Sex"]])
    sex_feature_names = sex_encoder.get_feature_names_out(["Sex"])
    sex_df = pd.DataFrame(sex_encoded, columns=sex_feature_names, index=df.index)
    df = df.drop("Sex", axis=1)
    df_encoded = pd.concat([df, sex_df], axis=1)
    return df_encoded


def scale(
    df: pd.DataFrame, scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale the numerical features in the abalone dataset using StandardScaler.

    Args:
        df (pd.DataFrame): The input dataframe with all columns.
        scaler (Optional[StandardScaler], optional): An existing StandardScaler to use for transformation.
            If None, a new scaler will be created and fitted. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: A tuple containing the scaled dataframe and the scaler used.
    """
    df = df.copy()
    numerical_cols = NUMERICAL_COLS
    if scaler is None:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    return df, scaler


@task(name="preprocess-data")
def preprocess_data(
    df: pd.DataFrame, scaler: Optional[StandardScaler] = None, with_target: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series], StandardScaler]:
    """Preprocess the abalone dataset by one-hot encoding categorical variables and scaling numerical features.

    Args:
        df (pd.DataFrame): The input dataframe with all columns.
        scaler (Optional[StandardScaler], optional): An existing StandardScaler to use for scaling.
            If None, a new scaler will be created and fitted. Defaults to None.
        with_target (bool, optional): Whether to separate the target variable 'Rings' from the features. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.Series], StandardScaler]: A tuple containing the preprocessed dataframe,
            the target variable (if with_target is True), and the scaler used.
    """
    df = df.copy()
    df = onehot(df)

    if with_target:
        y = df["Rings"]
        df = df.drop("Rings", axis=1)
    else:
        y = None
    df, scaler = scale(df, scaler)

    return df, y, scaler
