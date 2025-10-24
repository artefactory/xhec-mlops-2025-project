"""
Web service library module.

This module contains the models and inference logic for the FastAPI service.
"""

from .models import AbaloneInput, PredictionOutput
from .inference import AbalonePredictor, get_predictor

__all__ = [
    "AbaloneInput",
    "PredictionOutput",
    "AbalonePredictor",
    "get_predictor",
]

