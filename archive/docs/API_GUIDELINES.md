# Abalone Age Prediction Web Service

A FastAPI-based web service for predicting abalone age using machine learning.

## Features

- 🎯 **Single & Batch Predictions**: Predict age for one or multiple abalones
- 🔧 **Model Training**: Train new models via API with custom hyperparameters
- 📊 **Pydantic Validation**: Strong input/output validation
- 📖 **Interactive Documentation**: Auto-generated Swagger UI and ReDoc
- 💚 **Health Checks**: Monitor API and model availability
- 🎨 **Beautiful UI**: Modern home page with usage examples

## Quick Start

### 1. Install Dependencies

```bash
# From project root
pip install -e .
```

Or with uv:

```bash
uv pip install -e .
```

### 2. Train a Model (if not already trained)

```bash
python src/modelling/main.py
```

This will create `src/web_service/local_objects/model.pkl`.

### 3. Start the Server

```bash
# From project root
uvicorn src.web_service.main:app --reload --host 0.0.0.0 --port 8000
```

Or run directly:

```bash
python src/web_service/main.py
```

### 4. Access the API

- **Home Page**: http://localhost:8000/
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### GET `/`
Home page with API information and usage examples.

**Response**: HTML page

### GET `/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.0.1"
}
```

### POST `/predict`
Predict abalone age for a single sample.

**Request Body**:
```json
{
  "sex": "M",
  "length": 0.455,
  "diameter": 0.365,
  "height": 0.095,
  "whole_weight": 0.514,
  "shucked_weight": 0.2245,
  "viscera_weight": 0.101,
  "shell_weight": 0.15
}
```

**Response**:
```json
{
  "predicted_rings": 15.0,
  "predicted_age": 16.5,
  "input_features": { ... }
}
```

### POST `/predict/batch`
Predict abalone age for multiple samples.

**Request Body**:
```json
{
  "samples": [
    {
      "sex": "M",
      "length": 0.455,
      ...
    },
    {
      "sex": "F",
      "length": 0.53,
      ...
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [ ... ],
  "count": 2
}
```

### POST `/train`
Train a new model with custom hyperparameters.

**Request Body** (all fields optional):
```json
{
  "n_estimators": 100,
  "max_depth": 20,
  "min_samples_split": 5,
  "min_samples_leaf": 2,
  "random_state": 42
}
```

**Response**:
```json
{
  "message": "Model trained successfully",
  "model_path": "src/web_service/local_objects/model.pkl",
  "training_samples": 3341
}
```

## Input Features

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `sex` | string | M (Male), F (Female), or I (Infant) | "M" |
| `length` | float | Longest shell measurement (mm) | 0.455 |
| `diameter` | float | Perpendicular to length (mm) | 0.365 |
| `height` | float | Height with meat in shell (mm) | 0.095 |
| `whole_weight` | float | Whole abalone weight (grams) | 0.514 |
| `shucked_weight` | float | Weight of meat (grams) | 0.2245 |
| `viscera_weight` | float | Gut weight after bleeding (grams) | 0.101 |
| `shell_weight` | float | Weight after being dried (grams) | 0.15 |

## Usage Examples

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sex": "M",
    "length": 0.455,
    "diameter": 0.365,
    "height": 0.095,
    "whole_weight": 0.514,
    "shucked_weight": 0.2245,
    "viscera_weight": 0.101,
    "shell_weight": 0.15
  }'
```

### Using Python

```python
import requests

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
print(response.json())
```

### Batch Predictions

```python
import requests

url = "http://localhost:8000/predict/batch"
data = {
    "samples": [
        {
            "sex": "M",
            "length": 0.455,
            "diameter": 0.365,
            "height": 0.095,
            "whole_weight": 0.514,
            "shucked_weight": 0.2245,
            "viscera_weight": 0.101,
            "shell_weight": 0.15
        },
        {
            "sex": "F",
            "length": 0.53,
            "diameter": 0.42,
            "height": 0.135,
            "whole_weight": 0.677,
            "shucked_weight": 0.2565,
            "viscera_weight": 0.1415,
            "shell_weight": 0.21
        }
    ]
}

response = requests.post(url, json=data)
result = response.json()
print(f"Processed {result['count']} samples")
for pred in result['predictions']:
    print(f"Predicted age: {pred['predicted_age']} years")
```

## Project Structure

```
src/web_service/
├── __init__.py           # Package initialization
├── main.py               # FastAPI application
├── app_config.py         # Application configuration
├── schemas.py            # Pydantic models for validation
├── inference.py          # Inference functions
├── README.md             # This file
└── local_objects/        # Directory for saved models
    └── model.pkl         # Trained model
```

## Configuration

Configuration is managed in `app_config.py` using Pydantic Settings. You can override settings using environment variables with the `ABALONE_` prefix:

```bash
export ABALONE_MODEL_PATH="path/to/model.pkl"
export ABALONE_DATA_PATH="path/to/data.csv"
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `422`: Validation error (invalid input)
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)

## Development

### Running with Auto-reload

```bash
uvicorn src.web_service.main:app --reload
```

### Running Tests

```bash
pytest tests/
```

### Linting

```bash
ruff check src/web_service/
```

## Model Information

- **Algorithm**: Random Forest Regressor
- **Features**: 8 physical measurements
- **Target**: Number of rings (age indicator)
- **Framework**: scikit-learn

Age is calculated as: **Age = Rings + 1.5 years**

## License

Part of the X-HEC MLOps Project
