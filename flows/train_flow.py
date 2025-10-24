# flows/train_flow.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from prefect import flow, task

DATA_PATH = Path("assets/data/abalone.csv")      # adjust if your dataset lives elsewhere
MODELS_DIR = Path("assets/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

@task
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@task
def split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    # Example: predict "Rings" from all other numeric columns
    y = df["Rings"]
    X = df.drop(columns=["Rings"])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@task
def train_model(X_train, y_train, n_estimators: int = 300, random_state: int = 42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

@task
def evaluate(model, X_test, y_test) -> float:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return mae

@task
def persist(model, outdir: Path = MODELS_DIR, name: str = "rf_abalone.joblib") -> str:
    outpath = outdir / name
    joblib.dump(model, outpath)
    return str(outpath)

@flow(name="train-flow")
def train_flow(
    data_path: str = str(DATA_PATH),
    test_size: float = 0.2,
    n_estimators: int = 300,
):
    df = load_data(Path(data_path))
    X_train, X_test, y_train, y_test = split(df, test_size=test_size)
    model = train_model(X_train, y_train, n_estimators=n_estimators)
    mae = evaluate(model, X_test, y_test)
    model_path = persist(model)
    print(f"Saved model -> {model_path} | MAE={mae:.4f}")
    return {"mae": mae, "model_path": model_path}

if __name__ == "__main__":
    train_flow()
