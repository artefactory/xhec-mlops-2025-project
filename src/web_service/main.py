"""FastAPI application for Abalone Age Prediction."""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pathlib import Path
import sys
from src.web_service.app_config import config
from src.web_service.schemas import (
    AbaloneFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    TrainingRequest,
    TrainingResponse,
    HealthResponse,
)
from src.web_service.inference import run_inference, run_batch_inference, load_model

# Add the project root to the path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Initialize FastAPI app
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description=config.app_description,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Global model cache
_model_cache = None


def get_model():
    """Get or load the model (with caching)."""
    global _model_cache
    if _model_cache is None:
        try:
            _model_cache = load_model()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            )
    return _model_cache


def clear_model_cache():
    """Clear the model cache (useful after training)."""
    global _model_cache
    _model_cache = None


@app.get("/", response_class=HTMLResponse, tags=["Home"])
async def home():
    """Home page with API information and usage examples."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Abalone Age Prediction API</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .content {
                padding: 40px;
            }
            .section {
                margin-bottom: 40px;
            }
            .section h2 {
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.8em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }
            .section h3 {
                color: #764ba2;
                margin-top: 20px;
                margin-bottom: 10px;
                font-size: 1.3em;
            }
            .endpoint {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .method {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 5px;
                font-weight: bold;
                margin-right: 10px;
                font-size: 0.9em;
            }
            .get {
                background: #28a745;
                color: white;
            }
            .post {
                background: #007bff;
                color: white;
            }
            .code {
                background: #2d2d2d;
                color: #f8f8f2;
                padding: 20px;
                border-radius: 5px;
                overflow-x: auto;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .feature-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .feature-card h3 {
                color: white;
                margin: 0 0 10px 0;
            }
            .links {
                display: flex;
                gap: 20px;
                justify-content: center;
                margin-top: 30px;
            }
            .btn {
                display: inline-block;
                padding: 15px 30px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
                transition: transform 0.2s, background 0.2s;
            }
            .btn:hover {
                background: #764ba2;
                transform: translateY(-2px);
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background: #667eea;
                color: white;
            }
            tr:hover {
                background: #f5f5f5;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🦪 Abalone Age Prediction API</h1>
                <p>Machine Learning API for predicting abalone age based on physical measurements</p>
            </div>

            <div class="content">
                <div class="section">
                    <h2>📚 About</h2>
                    <p>
                        This API provides machine learning predictions for abalone age based on physical characteristics.
                        The model uses a Random Forest Regressor trained on the classic UCI Abalone dataset.
                        The age of abalone is determined by the number of rings through microscopic examination,
                        and this API predicts it using non-invasive physical measurements.
                    </p>
                </div>

                <div class="section">
                    <h2>🚀 Quick Start</h2>
                    <div class="links">
                        <a href="/docs" class="btn">📖 API Documentation (Swagger)</a>
                        <a href="/redoc" class="btn">📋 API Documentation (ReDoc)</a>
                        <a href="/health" class="btn">💚 Health Check</a>
                    </div>
                </div>

                <div class="section">
                    <h2>🔌 Available Endpoints</h2>

                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/</strong>
                        <p>This home page with API information</p>
                    </div>

                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/health</strong>
                        <p>Health check endpoint - verify API and model status</p>
                    </div>

                    <div class="endpoint">
                        <span class="method post">POST</span>
                        <strong>/predict</strong>
                        <p>Predict abalone age for a single sample</p>
                    </div>

                    <div class="endpoint">
                        <span class="method post">POST</span>
                        <strong>/predict/batch</strong>
                        <p>Predict abalone age for multiple samples at once</p>
                    </div>

                    <div class="endpoint">
                        <span class="method post">POST</span>
                        <strong>/train</strong>
                        <p>Train a new model with custom hyperparameters</p>
                    </div>
                </div>

                <div class="section">
                    <h2>📊 Input Features</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Type</th>
                            <th>Description</th>
                            <th>Example</th>
                        </tr>
                        <tr>
                            <td><strong>sex</strong></td>
                            <td>string</td>
                            <td>M (Male), F (Female), or I (Infant)</td>
                            <td>"M"</td>
                        </tr>
                        <tr>
                            <td><strong>length</strong></td>
                            <td>float</td>
                            <td>Longest shell measurement (mm)</td>
                            <td>0.455</td>
                        </tr>
                        <tr>
                            <td><strong>diameter</strong></td>
                            <td>float</td>
                            <td>Perpendicular to length (mm)</td>
                            <td>0.365</td>
                        </tr>
                        <tr>
                            <td><strong>height</strong></td>
                            <td>float</td>
                            <td>Height with meat in shell (mm)</td>
                            <td>0.095</td>
                        </tr>
                        <tr>
                            <td><strong>whole_weight</strong></td>
                            <td>float</td>
                            <td>Whole abalone weight (grams)</td>
                            <td>0.514</td>
                        </tr>
                        <tr>
                            <td><strong>shucked_weight</strong></td>
                            <td>float</td>
                            <td>Weight of meat (grams)</td>
                            <td>0.2245</td>
                        </tr>
                        <tr>
                            <td><strong>viscera_weight</strong></td>
                            <td>float</td>
                            <td>Gut weight after bleeding (grams)</td>
                            <td>0.101</td>
                        </tr>
                        <tr>
                            <td><strong>shell_weight</strong></td>
                            <td>float</td>
                            <td>Weight after being dried (grams)</td>
                            <td>0.15</td>
                        </tr>
                    </table>
                </div>

                <div class="section">
                    <h2>💻 Example Usage</h2>

                    <h3>Single Prediction (cURL)</h3>
                    <div class="code">curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "sex": "M",
    "length": 0.455,
    "diameter": 0.365,
    "height": 0.095,
    "whole_weight": 0.514,
    "shucked_weight": 0.2245,
    "viscera_weight": 0.101,
    "shell_weight": 0.15
  }'</div>

                    <h3>Python Example</h3>
                    <div class="code">import requests

url = "http://localhost:8000/predict"
data = {
    "sex": "M",
    "length": 0.455,
    "diameter": 0.365,
    "height": 0.095,
    "whole_weight": 0.514,
    "shucked_weight": 0.2245,
    "viscera_weight": 0.101,
    "shell_weight": 0.15
}

response = requests.post(url, json=data)
print(response.json())</div>

                    <h3>Expected Response</h3>
                    <div class="code">{
  "predicted_rings": 15.0,
  "predicted_age": 16.5,
  "input_features": {
    "sex": "M",
    "length": 0.455,
    ...
  }
}</div>
                </div>

                <div class="section">
                    <h2>⚡ Features</h2>
                    <div class="feature-grid">
                        <div class="feature-card">
                            <h3>🎯 Accurate Predictions</h3>
                            <p>Random Forest model trained on real abalone data</p>
                        </div>
                        <div class="feature-card">
                            <h3>⚡ Fast Inference</h3>
                            <p>Optimized for low-latency predictions</p>
                        </div>
                        <div class="feature-card">
                            <h3>📦 Batch Processing</h3>
                            <p>Process multiple samples in one request</p>
                        </div>
                        <div class="feature-card">
                            <h3>🔧 Retrainable</h3>
                            <p>Train new models via API endpoint</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health status of the API and model availability."""
    model_loaded = False
    try:
        model = get_model()
        model_loaded = model is not None
    except HTTPException:
        pass

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=config.app_version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: AbaloneFeatures):
    """Predict the age of an abalone based on physical measurements.

    The model predicts the number of rings, which can be used to estimate age.
    Age = Rings + 1.5 years

    Args:
        features: Physical measurements of the abalone

    Returns:
        Predicted number of rings and estimated age
    """
    try:
        model = get_model()
        prediction = run_inference(features, model=model)
        return prediction
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Predict the age of multiple abalones in a single request.

    This endpoint is more efficient for processing multiple samples as it loads
    the model once and reuses it for all predictions.

    Args:
        request: Batch request containing multiple abalone samples

    Returns:
        List of predictions for each sample
    """
    try:
        model = get_model()
        predictions = run_batch_inference(request.samples, model=model)
        return BatchPredictionResponse(predictions=predictions, count=len(predictions))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest = TrainingRequest()):
    """Train a new Random Forest model with the specified hyperparameters.

    This endpoint triggers the training pipeline which:
    1. Loads the data from the configured data path
    2. Preprocesses the data (encodes categorical features, splits into train/test)
    3. Trains a Random Forest Regressor with the provided hyperparameters
    4. Saves the trained model to disk

    After training, the model cache is cleared and the new model will be used
    for subsequent predictions.

    Args:
        request: Training configuration with hyperparameters

    Returns:
        Training status and model information
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        import pickle as pkl
        import os
        from .preprocessing import prepare_training_data

        # Load and preprocess data using unified preprocessing
        X, y, encoder = prepare_training_data(config.data_path)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model with custom parameters
        rf = RandomForestRegressor(
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            min_samples_leaf=request.min_samples_leaf,
            random_state=request.random_state,
            n_jobs=-1,
        )

        rf.fit(X_train, y_train)

        # Save model
        model_path = config.model_path
        os.makedirs(model_path.parent, exist_ok=True)
        with open(model_path, "wb") as f:
            pkl.dump(rf, f)

        # Clear model cache to load the newly trained model
        clear_model_cache()

        return TrainingResponse(
            message="Model trained successfully",
            model_path=str(config.model_path),
            training_samples=len(X_train),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
