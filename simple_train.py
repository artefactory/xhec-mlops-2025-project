#!/usr/bin/env python3
"""Simple training script that uses the unified preprocessing."""

import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle as pkl
import os
from src.web_service.preprocessing import prepare_training_data

# Add the project root to the path to enable imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    # Use same paths as API
    data_path = Path("data/abalone.csv")
    model_path = Path("src/web_service/local_objects/model.pkl")

    print("🚀 Training model with unified preprocessing...")

    # Load and preprocess data using the SAME function as the API
    X, y, encoder = prepare_training_data(data_path)

    print("Loaded and preprocessed data:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature columns: {list(X.columns)}")
    print(f"  Target range: {y.min():.1f} - {y.max():.1f} rings")

    # Split data (same as API)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Train model (same as API)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("🤖 Training Random Forest...")
    rf.fit(X_train, y_train)

    # Evaluate
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"  Training R²: {train_score:.4f}")
    print(f"  Test R²: {test_score:.4f}")

    # Save model (same as API)
    os.makedirs(model_path.parent, exist_ok=True)
    with open(model_path, "wb") as f:
        pkl.dump(rf, f)

    print(f"✅ Model saved to: {model_path}")

    # Test the model with a sample
    sample_prediction = rf.predict(X_test.iloc[0:1])
    actual_value = y_test.iloc[0]
    print("🧪 Sample test:")
    print(
        f"  Predicted: {sample_prediction[0]:.2f} rings ({sample_prediction[0] + 1.5:.2f} years)"
    )
    print(f"  Actual: {actual_value:.0f} rings ({actual_value + 1.5:.1f} years)")

    print("🎉 Training complete! The model is ready for the API.")


if __name__ == "__main__":
    main()
