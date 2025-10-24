"""Configuration for the FastAPI application."""

from pathlib import Path
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Application configuration."""

    app_name: str = "Abalone Age Prediction API"
    app_version: str = "0.0.1"
    app_description: str = (
        "API for predicting abalone age based on physical measurements"
    )

    # Model paths
    model_path: Path = Path("src/web_service/local_objects/model.pkl")

    # Data paths
    data_path: Path = Path("data/abalone.csv")

    # Model parameters - these must match the training data column order exactly
    feature_columns: list = [
        "Sex_encoded",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
    ]

    target_column: str = "Rings"

    # Sex encoding mapping
    sex_mapping: dict = {"M": 0, "F": 1, "I": 2}  # Male, Female, Infant

    class Config:
        env_prefix = "ABALONE_"


# Create a singleton instance
config = AppConfig()
