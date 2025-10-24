# Step 4: FastAPI + Docker

This step creates a production-ready REST API for serving ML predictions and containerizes it with Docker.

## 🎯 What's Implemented

- **FastAPI Application**: Complete REST API with prediction endpoints
- **Docker Containerization**: Production-ready containerized deployment
- **Unified Preprocessing**: Consistent data processing for training and inference
- **Health Monitoring**: API health checks and status endpoints

## 📁 Files Created/Modified

```
src/web_service/           # FastAPI application
├── main.py               # Main API with all endpoints
├── schemas.py            # Pydantic models for validation
├── inference.py          # Prediction logic
├── preprocessing.py      # Unified preprocessing pipeline
├── app_config.py         # Configuration management
└── local_objects/        # Model and encoder storage

bin/run_services.sh       # Service startup script
Dockerfile.app            # Docker configuration
```

## 🚀 Setup from Scratch

### Step 1: Install Dependencies
```bash
# Install the project with all dependencies
# Note: 'xhec-mlops-project-student' is the package name defined in pyproject.toml
pip install -e .
```

### Step 2: Train the Model
```bash
# Train a model (creates model.pkl and label_encoder.pkl)
python simple_train.py
```

### Step 3: Test API Locally
```bash
# Start the FastAPI server
./bin/run_services.sh

# In another terminal, test the API
python test_api_client.py
```

### Step 4: Docker Deployment
```bash
# Build the Docker image
docker build -f Dockerfile.app -t abalone-api .

# Run the container with specified port binding
docker run -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
```

## 🔌 API Endpoints

- **`GET /`** - Interactive home page with documentation
- **`GET /health`** - API health status and model availability
- **`POST /predict`** - Single abalone age prediction
- **`POST /predict/batch`** - Batch predictions for multiple samples
- **`POST /train`** - Train new model with custom parameters

## 📊 Example Usage

**Health Check:**
```bash
curl http://localhost:8001/health
```

**Make Prediction:**
```bash
curl -X POST http://localhost:8001/predict \
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

**Expected Response:**
```json
{
  "predicted_rings": 15.0,
  "predicted_age": 16.5,
  "input_features": { ... }
}
```

## 🌐 Access Points

When running in Docker:
- **Home Page**: http://localhost:8001/
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

## ✅ Verification Steps

1. **Local Testing**: Ensure API works locally before Docker
2. **Docker Build**: Verify Docker image builds successfully
3. **Container Run**: Test API endpoints work in container
4. **Port Binding**: Confirm correct port mapping (8001 externally)

## 🔧 Troubleshooting

- **Model Missing**: Run `python simple_train.py` to create model files
- **Port Conflicts**: Ensure ports 8001 and 4201 are available
- **Docker Issues**: Check Docker is running and has sufficient resources
