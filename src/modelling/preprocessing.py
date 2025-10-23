import pandas as pd
import numpy as np
from pathlib import Path


def to_snake(s: str) -> str:
    s = s.strip().lower().replace("/", " per ").replace("-", " ")
    s = "".join(ch if ch.isalnum() or ch.isspace() else "_" for ch in s)
    s = "_".join(s.split())
    while "__" in s:
        s = s.replace("__", "_")
    return s


def remove_iqr_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df_clean = df.copy()
    for col in cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean


def preprocess_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [to_snake(c) for c in df.columns]
    if "age" in df.columns:
        df = df.drop(columns=["age"])

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df_clean = remove_iqr_outliers(df, numeric_cols)

    df_clean = pd.get_dummies(df_clean, columns=["sex"], drop_first=True)
    return df_clean
