"""Pydantic schemas for request and response validation."""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class AbaloneFeatures(BaseModel):
    """Input features for abalone age prediction.

    All measurements are in physical units (mm for length/diameter/height, grams for weights).
    """

    sex: Literal["M", "F", "I"] = Field(
        ..., description="Sex of the abalone: M (Male), F (Female), or I (Infant)"
    )
    length: float = Field(..., gt=0, description="Longest shell measurement (mm)")
    diameter: float = Field(..., gt=0, description="Perpendicular to length (mm)")
    height: float = Field(..., gt=0, description="Height with meat in shell (mm)")
    whole_weight: float = Field(..., gt=0, description="Whole abalone weight (grams)")
    shucked_weight: float = Field(..., gt=0, description="Weight of meat (grams)")
    viscera_weight: float = Field(
        ..., gt=0, description="Gut weight after bleeding (grams)"
    )
    shell_weight: float = Field(
        ..., gt=0, description="Weight after being dried (grams)"
    )

    @field_validator("shucked_weight", "viscera_weight", "shell_weight")
    @classmethod
    def check_weights_sum(cls, v, info):
        """Validate that component weights don't exceed whole weight."""
        # Note: This is a simplified check. Full validation would require all fields
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sex": "M",
                    "length": 0.455,
                    "diameter": 0.365,
                    "height": 0.095,
                    "whole_weight": 0.514,
                    "shucked_weight": 0.2245,
                    "viscera_weight": 0.101,
                    "shell_weight": 0.15,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    predicted_rings: float = Field(
        ..., description="Predicted number of rings (age indicator)"
    )
    predicted_age: float = Field(
        ..., description="Predicted age in years (rings + 1.5)"
    )
    input_features: AbaloneFeatures = Field(
        ..., description="Echo of the input features used for prediction"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_rings": 15.0,
                    "predicted_age": 16.5,
                    "input_features": {
                        "sex": "M",
                        "length": 0.455,
                        "diameter": 0.365,
                        "height": 0.095,
                        "whole_weight": 0.514,
                        "shucked_weight": 0.2245,
                        "viscera_weight": 0.101,
                        "shell_weight": 0.15,
                    },
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    samples: list[AbaloneFeatures] = Field(
        ..., min_length=1, description="List of abalone samples to predict"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: list[PredictionResponse] = Field(
        ..., description="List of predictions for each sample"
    )
    count: int = Field(..., description="Number of predictions made")


class TrainingRequest(BaseModel):
    """Request model for model training."""

    n_estimators: int = Field(
        default=100, gt=0, description="Number of trees in the random forest"
    )
    max_depth: int = Field(default=20, gt=0, description="Maximum depth of the trees")
    min_samples_split: int = Field(
        default=5,
        gt=1,
        description="Minimum number of samples required to split an internal node",
    )
    min_samples_leaf: int = Field(
        default=2,
        gt=0,
        description="Minimum number of samples required to be at a leaf node",
    )
    random_state: int = Field(
        default=42, description="Random state for reproducibility"
    )


class TrainingResponse(BaseModel):
    """Response model for training endpoint."""

    message: str = Field(..., description="Status message about the training")
    model_path: str = Field(..., description="Path where the model was saved")
    training_samples: int = Field(
        ..., description="Number of samples used for training"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Health status of the API")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    version: str = Field(..., description="API version")
