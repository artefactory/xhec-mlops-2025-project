"""
Pydantic models for the web service.

This module defines the request and response models for the abalone age prediction API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class AbaloneInput(BaseModel):
    """
    Input model for abalone age prediction.
    
    Attributes:
        Sex: Sex of the abalone (M, F, or I for infant)
        Length: Longest shell measurement (mm)
        Diameter: Perpendicular to length (mm)
        Height: With meat in shell (mm)
        Whole_weight: Whole abalone weight (grams)
        Shucked_weight: Weight of meat (grams)
        Viscera_weight: Gut weight after bleeding (grams)
        Shell_weight: After being dried (grams)
    """
    
    Sex: Literal["M", "F", "I"] = Field(
        ...,
        description="Sex of the abalone: M (Male), F (Female), or I (Infant)",
        examples=["M"]
    )
    Length: float = Field(
        ...,
        gt=0,
        description="Longest shell measurement in mm",
        examples=[0.455]
    )
    Diameter: float = Field(
        ...,
        gt=0,
        description="Perpendicular to length in mm",
        examples=[0.365]
    )
    Height: float = Field(
        ...,
        gt=0,
        description="With meat in shell in mm",
        examples=[0.095]
    )
    Whole_weight: float = Field(
        ...,
        gt=0,
        description="Whole abalone weight in grams",
        examples=[0.514]
    )
    Shucked_weight: float = Field(
        ...,
        gt=0,
        description="Weight of meat in grams",
        examples=[0.2245]
    )
    Viscera_weight: float = Field(
        ...,
        gt=0,
        description="Gut weight after bleeding in grams",
        examples=[0.101]
    )
    Shell_weight: float = Field(
        ...,
        gt=0,
        description="Shell weight after being dried in grams",
        examples=[0.15]
    )
    
    @field_validator('Sex')
    @classmethod
    def validate_sex(cls, v):
        """Validate that Sex is one of M, F, or I."""
        if v not in ['M', 'F', 'I']:
            raise ValueError('Sex must be M, F, or I')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "Sex": "M",
                "Length": 0.455,
                "Diameter": 0.365,
                "Height": 0.095,
                "Whole_weight": 0.514,
                "Shucked_weight": 0.2245,
                "Viscera_weight": 0.101,
                "Shell_weight": 0.15
            }
        }


class PredictionOutput(BaseModel):
    """
    Output model for abalone age prediction.
    
    Attributes:
        predicted_rings: Predicted number of rings (age indicator)
        predicted_age: Predicted age in years (rings + 1.5)
        confidence: Confidence score of the prediction (0-1)
        model_type: Type of model used for prediction
    """
    
    predicted_rings: float = Field(
        ...,
        description="Predicted number of rings",
        examples=[9.5]
    )
    predicted_age: float = Field(
        ...,
        description="Predicted age in years (rings + 1.5)",
        examples=[11.0]
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score of the prediction",
        examples=[0.85]
    )
    model_type: str = Field(
        ...,
        description="Type of model used",
        examples=["RandomForestRegressor"]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_rings": 9.5,
                "predicted_age": 11.0,
                "confidence": 0.85,
                "model_type": "RandomForestRegressor"
            }
        }
