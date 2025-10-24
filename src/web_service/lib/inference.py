"""
Inference module for the web service.

This module handles model loading and prediction logic for the FastAPI service.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from ..utils import load_object


class AbalonePredictor:
    """
    Predictor class for abalone age prediction.
    
    This class loads the trained model and preprocessors, and provides
    methods for making predictions on new data.
    """
    
    def __init__(self, model_dir: str = "src/web_service/local_objects"):
        """
        Initialize the predictor by loading model and preprocessors.
        
        Args:
            model_dir: Directory containing the model and preprocessor files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, scaler, and label encoder from disk."""
        try:
            self.model = load_object(self.model_dir / "model.pkl")
            self.scaler = load_object(self.model_dir / "scaler.pkl")
            self.label_encoder = load_object(self.model_dir / "label_encoder.pkl")
            print(f"✓ Successfully loaded model: {type(self.model).__name__}")
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to load model artifacts: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading model artifacts: {e}")
    
    def _preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Dictionary with abalone features
            
        Returns:
            Preprocessed feature array
        """
        # Create DataFrame from input
        df = pd.DataFrame([{
            "Sex": input_data["Sex"],
            "Length": input_data["Length"],
            "Diameter": input_data["Diameter"],
            "Height": input_data["Height"],
            "Whole weight": input_data["Whole_weight"],
            "Shucked weight": input_data["Shucked_weight"],
            "Viscera weight": input_data["Viscera_weight"],
            "Shell weight": input_data["Shell_weight"]
        }])
        
        # Encode Sex feature
        df["Sex_encoded"] = self.label_encoder.transform(df["Sex"])
        
        # Engineer features (same as in training)
        df = self._engineer_features(df)
        
        # Select features in correct order
        feature_cols = self._get_feature_columns()
        X = df[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features (must match training preprocessing).
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with engineered features
        """
        # Log transformations
        df["log_Whole weight"] = np.log1p(df["Whole weight"])
        df["log_Shucked weight"] = np.log1p(df["Shucked weight"])
        df["log_Viscera weight"] = np.log1p(df["Viscera weight"])
        df["log_Shell weight"] = np.log1p(df["Shell weight"])
        
        # Ratios between weights
        df["ratio_shell_whole"] = df["Shell weight"] / (df["Whole weight"] + 1e-8)
        df["ratio_shucked_whole"] = df["Shucked weight"] / (df["Whole weight"] + 1e-8)
        df["ratio_viscera_whole"] = df["Viscera weight"] / (df["Whole weight"] + 1e-8)
        
        # Geometric volume
        df["geo_volume"] = df["Length"] * df["Diameter"] * df["Height"]
        
        # Densities
        df["density_whole"] = df["Whole weight"] / (df["geo_volume"] + 1e-8)
        df["density_shell"] = df["Shell weight"] / (df["geo_volume"] + 1e-8)
        df["density_shucked"] = df["Shucked weight"] / (df["geo_volume"] + 1e-8)
        
        # Interactions between dimensions
        df["Length*Diameter"] = df["Length"] * df["Diameter"]
        df["Diameter*Height"] = df["Diameter"] * df["Height"]
        
        # Polynomial features
        df["Length_sq"] = df["Length"] ** 2
        df["Diameter_sq"] = df["Diameter"] ** 2
        df["Height_sq"] = df["Height"] ** 2
        
        # Geometric ratios
        df["len_to_diam"] = df["Length"] / (df["Diameter"] + 1e-8)
        df["ht_to_diam"] = df["Height"] / (df["Diameter"] + 1e-8)
        df["wt_per_len"] = df["Whole weight"] / (df["Length"] + 1e-8)
        
        return df
    
    def _get_feature_columns(self):
        """Get list of feature columns in correct order."""
        return [
            "Sex_encoded",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "log_Whole weight",
            "log_Shucked weight",
            "log_Viscera weight",
            "log_Shell weight",
            "ratio_shell_whole",
            "ratio_shucked_whole",
            "ratio_viscera_whole",
            "geo_volume",
            "density_whole",
            "density_shell",
            "density_shucked",
            "Length*Diameter",
            "Diameter*Height",
            "Length_sq",
            "Diameter_sq",
            "Height_sq",
            "len_to_diam",
            "ht_to_diam",
            "wt_per_len",
        ]
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for abalone age.
        
        Args:
            input_data: Dictionary with abalone features
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess input
        X_scaled = self._preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        # Calculate confidence
        confidence = self._calculate_confidence(X_scaled)
        
        return {
            "predicted_rings": float(prediction),
            "predicted_age": float(prediction + 1.5),
            "confidence": float(confidence),
            "model_type": type(self.model).__name__
        }
    
    def _calculate_confidence(self, X_scaled: np.ndarray) -> float:
        """
        Calculate prediction confidence.
        
        Args:
            X_scaled: Scaled feature array
            
        Returns:
            Confidence score between 0 and 1
        """
        if hasattr(self.model, "estimators_"):
            # Random Forest - use prediction variance
            predictions = np.array([
                tree.predict(X_scaled)[0] for tree in self.model.estimators_
            ])
            variance = np.var(predictions)
            confidence = 1.0 / (1.0 + variance)
        else:
            # Other models - default confidence
            confidence = 0.75
        
        return confidence


# Global predictor instance
_predictor = None


def get_predictor() -> AbalonePredictor:
    """
    Get or create the global predictor instance.
    
    Returns:
        AbalonePredictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = AbalonePredictor()
    return _predictor
