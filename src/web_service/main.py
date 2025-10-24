"""
FastAPI application for abalone age prediction.

This module provides a REST API for predicting abalone age based on
physical measurements.
"""

from fastapi import FastAPI, HTTPException
from .lib.models import AbaloneInput, PredictionOutput
from .lib.inference import get_predictor


app = FastAPI(
    title="Abalone Age Prediction API",
    description="API for predicting the age of abalones based on physical measurements. "
                "The age is estimated from the number of rings in the shell (rings + 1.5 years).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        get_predictor()
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"⚠ Warning: Failed to load model on startup: {e}")


@app.get("/", tags=["Health"])
def home() -> dict:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with health status
    """
    return {
        "health_check": "App up and running!",
        "service": "Abalone Age Prediction API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
def health() -> dict:
    """
    Detailed health check endpoint.
    
    Returns:
        Dictionary with detailed health status
    """
    try:
        predictor = get_predictor()
        return {
            "status": "healthy",
            "model_loaded": predictor.model is not None,
            "model_type": type(predictor.model).__name__ if predictor.model else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/predict", response_model=PredictionOutput, status_code=200, tags=["Prediction"])
def predict(payload: AbaloneInput) -> PredictionOutput:
    """
    Predict abalone age from physical measurements.
    
    Args:
        payload: AbaloneInput with physical measurements
        
    Returns:
        PredictionOutput with predicted age and confidence
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Get predictor instance
        predictor = get_predictor()
        
        # Convert Pydantic model to dict
        input_data = payload.model_dump()
        
        # Make prediction
        result = predictor.predict(input_data)
        
        # Return prediction
        return PredictionOutput(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
def model_info() -> dict:
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary with model information
    """
    try:
        predictor = get_predictor()
        return {
            "model_type": type(predictor.model).__name__,
            "model_dir": str(predictor.model_dir),
            "features_count": len(predictor._get_feature_columns()),
            "feature_names": predictor._get_feature_columns()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )
