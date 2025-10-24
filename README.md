<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

Welcome to our MLOps project! In this hands-on project, we built a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Our Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Out Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)



## 👥 Team Information

**Team Members:**
- Soumyabrata Bairagi (roy.bensimon@polytechnique.edu)
- Samuel Rajzman (samuel.rajzman@hec.edu)
- Roy Bensimon (roy.bensimon@hec.edu)
- Vassili de Rosen (vassili.de-rosen@hec.edu)
- Adam Berdah (adam.berdah@hec.edu)

## 🛠️ Development Environment

### Prerequisites
- Python 3.10 or 3.11
- UV package manager
- Git
- GitHub account

### Complete Setup Instructions

1. **Install UV Package Manager:**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone and Setup Repository:**
   ```bash
   # Clone your forked repository
   git clone https://github.com/YOUR_USERNAME/xhec-mlops-2025-project.git
   cd xhec-mlops-2025-project

   # Install all dependencies
   uv sync --extra dev

   # Install pre-commit hooks
   uv run pre-commit install
   ```

3. **Verify Installation:**
   ```bash
   # Check if everything is working
   uv run pre-commit run --all-files
   uv run ruff check .
   uv run pytest --version
   ```

### Development Workflow

**Daily Development:**
```bash
# Start your development session
uv sync --extra dev

# Run code quality checks
uv run pre-commit run --all-files

# Run tests
uv run pytest

# Run linting
uv run ruff check .
```

**Adding New Dependencies:**
```bash
# Add a new dependency
uv add <package>==<version>

# Add a dev dependency
uv add --dev <package>==<version>

# Sync environment
uv sync
```

### Code Quality Tools

**Automated Tools:**
- **Pre-commit hooks:** Run automatically on every commit
- **Ruff:** Fast Python linting and formatting
- **Black:** Code formatting (via ruff)
- **isort:** Import sorting
- **Pytest:** Testing framework

**Manual Commands:**
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Run tests with coverage
uv run pytest --cov=.

# Check all pre-commit hooks
uv run pre-commit run --all-files
```

### CI/CD Pipeline

**Automated Checks:**
- ✅ Code quality (ruff, pre-commit)
- ✅ Testing (pytest with coverage)
- ✅ Multi-Python support (3.10, 3.11)
- ✅ Dependency management (UV)

**GitHub Actions:**
- Runs on every push and pull request
- Tests on multiple Python versions
- Generates coverage reports
- Uploads to Codecov

### Troubleshooting

**Common Issues:**

1. **UV not found:**
   ```bash
   # Add UV to your PATH
   export PATH="$HOME/.cargo/bin:$PATH"
   # Or restart your terminal
   ```

2. **Pre-commit hooks failing:**
   ```bash
   # Update pre-commit hooks
   uv run pre-commit autoupdate
   uv run pre-commit install
   ```

3. **Dependencies not installing:**
   ```bash
   # Clear UV cache and reinstall
   uv cache clean
   uv sync --extra dev
   ```

4. **Python version issues:**
   ```bash
   # Check Python version
   python --version
   # Should be 3.10 or 3.11
   ```

**Getting Help:**
- Check the [UV documentation](https://docs.astral.sh/uv/)
- Review [Pre-commit documentation](https://pre-commit.com/)
- Check GitHub Actions logs for CI issues


## 🎯 Final Deliverables Checklist

When you're done, your repository should contain:

✅ **Automated Training Pipeline**
- [x] Prefect workflows for model training
- [x] Separate modules for training and inference
- [x] Reproducible model and encoder generation

✅ **Automated Deployment**
- [x] Prefect deployment for regular retraining

✅ **Production API**
- [x] Working REST API for predictions
- [x] Pydantic input validation
- [x] Docker containerization

✅ **Professional Documentation**
- [x] Updated README with team info
- [x] Clear setup and run instructions
- [x] Complete development environment setup
- [x] Troubleshooting guide
- [x] All TODOs removed from code

## 🔄 Prefect Workflow Management

### Prerequisites for Prefect

1. **Ensure Dependencies are Installed:**
   ```bash
   # Activate virtual environment
   source .venv/bin/activate
   
   # Install/update dependencies
   uv sync --extra dev
   
   # Verify Prefect and MLflow are installed
   python -c "import prefect; import mlflow; print('✅ Dependencies OK')"
   ```

### Running Training with Prefect

1. **Start Prefect Server (Terminal 1):**
   ```bash
   # Start Prefect server in background
   prefect server start
   ```
   - Server will be available at http://localhost:4200
   - Keep this terminal running

2. **Run Training Flow (Terminal 2):**
   ```bash
   # Basic training
   python -m src.modelling.run_prefect_flow abalone.csv
   
   # Training with MLflow UI launcher
   python -m src.modelling.run_prefect_flow abalone.csv --launch-ui
   
   # Asynchronous execution
   python -m src.modelling.run_prefect_flow abalone.csv --async-run
   ```

3. **View Prefect UI:**
   - Open http://localhost:4200 in your browser
   - Navigate to "Runs" to see flow executions
   - Click on individual runs to see detailed logs
   - Monitor task execution in real-time

### Creating and Managing Deployments

1. **Create Deployment:**
   ```bash
   # Standard deployment
   python -m src.modelling.create_deployment
   
   # Deployment with MLflow UI
   python -m src.modelling.create_deployment --with-ui
   ```

2. **Apply and Serve Deployment:**
   ```bash
   # Apply deployment to Prefect server
   python -m src.modelling.create_deployment --serve
   
   # Or manually apply
   prefect deployment apply abalone_training_pipeline/abalone-training-deployment
   ```

3. **Run Deployment:**
   ```bash
   # Execute deployment manually
   prefect deployment run "abalone_training_pipeline/abalone-training-deployment"
   ```

4. **Start Agent (for scheduled runs):**
   ```bash
   # Start agent to execute scheduled deployments
   prefect agent start --pool default-agent-pool
   ```

### Prefect UI Features

- **Dashboard**: Overview of all flow runs and system health
- **Flow Runs**: Monitor all training executions with detailed logs
- **Task Runs**: Task-level monitoring and debugging
- **Deployments**: Manage and configure automated deployments
- **Logs**: Real-time logging and error tracking
- **Schedules**: Configure automated retraining schedules

### MLflow Integration

1. **Start MLflow UI:**
   ```bash
   # Start MLflow UI (separate terminal)
   mlflow ui
   ```
   - Available at http://localhost:5000
   - View experiments, models, and metrics

2. **View Training Results:**
   - Navigate to the "abalone_age_prediction" experiment
   - Compare model performance metrics
   - Download trained models and artifacts

### Troubleshooting Prefect

**Common Issues:**

1. **"No module named 'mlflow'" Error:**
   ```bash
   # Ensure virtual environment is activated
   source .venv/bin/activate
   
   # Reinstall dependencies
   uv sync --extra dev
   ```

2. **Prefect Server Connection Issues:**
   ```bash
   # Check if server is running
   curl http://localhost:4200/api/health
   
   # Restart server if needed
   prefect server start
   ```

3. **Deployment Not Found:**
   ```bash
   # List available deployments
   prefect deployment ls
   
   # Apply deployment if missing
   python -m src.modelling.create_deployment
   ```

4. **Flow Run Failures:**
   - Check Prefect UI logs for detailed error messages
   - Verify dataset path exists: `ls abalone.csv`
   - Ensure all dependencies are installed

---

## 🚀 FastAPI Deployment & Prediction Service

### Running the API Locally

1. **Ensure the Model Files Exist:**
   ```bash
   # Model artifacts should be in:
   ls src/web_service/local_objects/
   # Should contain: model.pkl, scaler.pkl, label_encoder.pkl
   ```

2. **Start the API:**
   ```bash
   # Activate virtual environment
   source .venv/bin/activate
   
   # Run FastAPI application
   uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001 --reload
   ```
   
   - API will be available at http://localhost:8001
   - Interactive docs at http://localhost:8001/docs
   - Alternative docs at http://localhost:8001/redoc

3. **Test the API:**
   ```bash
   # Health check
   curl http://localhost:8001/health
   
   # Make a prediction
   curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Sex": "M",
       "Length": 0.455,
       "Diameter": 0.365,
       "Height": 0.095,
       "Whole_weight": 0.514,
       "Shucked_weight": 0.2245,
       "Viscera_weight": 0.101,
       "Shell_weight": 0.15
     }'
   ```

### Running with Docker

#### Building the Docker Image

```bash
# Build the Docker image
docker build -t abalone-prediction-api -f Dockerfile.app .

# Verify the image was created
docker images | grep abalone-prediction-api
```

#### Running the Docker Container

```bash
# Run the container with port mapping
docker run -d \
  --name abalone-api \
  -p 0.0.0.0:8000:8001 \
  -p 0.0.0.0:4200:4201 \
  abalone-prediction-api

# Check if container is running
docker ps | grep abalone-api

# View logs
docker logs abalone-api -f
```

**Port Mapping:**
- **8000** (host) → **8001** (container): FastAPI application
- **4200** (host) → **4201** (container): Prefect server

#### Access Services

Once the container is running:
- **FastAPI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prefect UI**: http://localhost:4200

#### Docker Management Commands

```bash
# Stop the container
docker stop abalone-api

# Start the container
docker start abalone-api

# Restart the container
docker restart abalone-api

# Remove the container
docker rm -f abalone-api

# View container logs
docker logs abalone-api

# Execute commands inside container
docker exec -it abalone-api bash

# Remove the image
docker rmi abalone-prediction-api
```

### API Endpoints

#### Health Check
```bash
GET /
GET /health
```

#### Make Predictions
```bash
POST /predict
```

**Request Body:**
```json
{
  "Sex": "M",
  "Length": 0.455,
  "Diameter": 0.365,
  "Height": 0.095,
  "Whole_weight": 0.514,
  "Shucked_weight": 0.2245,
  "Viscera_weight": 0.101,
  "Shell_weight": 0.15
}
```

**Response:**
```json
{
  "predicted_rings": 9.5,
  "predicted_age": 11.0,
  "confidence": 0.85,
  "model_type": "RandomForestRegressor"
}
```

#### Model Information
```bash
GET /model/info
```

### Testing the API with Examples

```bash
# Example 1: Young Male Abalone
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Sex": "M",
    "Length": 0.35,
    "Diameter": 0.265,
    "Height": 0.09,
    "Whole_weight": 0.2255,
    "Shucked_weight": 0.0995,
    "Viscera_weight": 0.0485,
    "Shell_weight": 0.07
  }'

# Example 2: Female Abalone
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Sex": "F",
    "Length": 0.53,
    "Diameter": 0.42,
    "Height": 0.135,
    "Whole_weight": 0.677,
    "Shucked_weight": 0.2565,
    "Viscera_weight": 0.1415,
    "Shell_weight": 0.21
  }'

# Example 3: Infant Abalone
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Sex": "I",
    "Length": 0.075,
    "Diameter": 0.055,
    "Height": 0.01,
    "Whole_weight": 0.002,
    "Shucked_weight": 0.0010,
    "Viscera_weight": 0.0005,
    "Shell_weight": 0.0015
  }'
```

### API Features

- ✅ **Automatic Model Loading**: Model, scaler, and label encoder loaded on startup
- ✅ **Input Validation**: Pydantic models ensure data integrity
- ✅ **Feature Engineering**: Automatic creation of 27 engineered features
- ✅ **Confidence Scores**: Prediction confidence based on model variance
- ✅ **Health Monitoring**: Built-in health check endpoints
- ✅ **Interactive Documentation**: Swagger UI and ReDoc
- ✅ **Docker Support**: Production-ready containerization

### Troubleshooting API

**Common Issues:**

1. **Port Already in Use:**
   ```bash
   # Find process using the port
   lsof -i :8001
   
   # Kill the process
   kill -9 <PID>
   ```

2. **Model Files Not Found:**
   ```bash
   # Ensure model files exist
   ls -la src/web_service/local_objects/
   
   # If missing, run training pipeline
   python -m src.modelling.run_prefect_flow abalone.csv
   ```

3. **Docker Build Failures:**
   ```bash
   # Clean Docker cache
   docker system prune -a
   
   # Rebuild with no cache
   docker build --no-cache -t abalone-prediction-api -f Dockerfile.app .
   ```

4. **Container Won't Start:**
   ```bash
   # Check logs for errors
   docker logs abalone-api
   
   # Check if ports are available
   lsof -i :8000
   lsof -i :4200
   ```

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
