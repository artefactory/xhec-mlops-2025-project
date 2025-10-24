from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import pickle_object


def train_model(df, output_dir: Path):
    X = df.drop(columns=["rings"])
    y = df["rings"]

    dummy_cols = [c for c in X.columns if c.startswith("sex_")]
    num_cols = [c for c in X.columns if c not in dummy_cols]

    preproc = ColumnTransformer(
        [
            ("scale_num", StandardScaler(), num_cols),
            ("pass_dummies", "passthrough", dummy_cols),
        ]
    )

    model = Pipeline([("preprocess", preproc), ("regressor", LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    print("Model performance:", metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_object(model, output_dir / "model.pkl")
    print(f"Model saved to {output_dir / 'model.pkl'}")

    return model, metrics
