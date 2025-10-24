<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)

**A complete MLOps pipeline for predicting abalone age using physical measurements**

[Overview](#-project-overview) •
[Quick Start](#-quick-start) •
[Dataset](#-about-the-dataset) •
[Pipeline](#-project-pipeline) •
[Documentation](#-detailed-documentation)

</div>

---

## 🎯 Project Overview

This project demonstrates a complete MLOps workflow for predicting the age of abalone (a type of sea snail) using physical measurements. Instead of the traditional time-consuming method of counting shell rings under a microscope, we use machine learning to predict age from easily measurable features.

**Key Features:**
- 📊 Comprehensive exploratory data analysis
- 🤖 Automated ML training pipeline with Prefect
- 🔄 Scheduled model retraining
- 🚀 FastAPI-based prediction service
- 🐳 Fully containerized with Docker
- 📈 Experiment tracking with MLflow
- ✅ CI/CD with linting and formatting

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11
- Docker (for containerized deployment)
- Dataset from [Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

### Installation

1. **Clone the repository and set up environment:**
```bash
git clone <repository-url>
cd xhec-mlops-project
uv sync
source .venv/bin/activate
```

2. **Install pre-commit hooks:**
```bash
pre-commit install
pre-commit run --all-files
```

3. **Download the dataset if not already available:**
   - Get the dataset from [Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)
   - Place `abalone.csv` in the `data/` directory

---

💬 First, verify everything works locally in your environment.  
Then, run the same API inside Docker for deployment testing.



### 🚀 Part 1: Run the Complete Pipeline Locally

For development or debugging.

```bash
# 1. Explore the data (optional)
jupyter notebook notebooks/eda.ipynb

# 2. Train the model
python src/modelling/train_flow.py data/abalone.csv

# 3. Start the API locally
uvicorn src.web_service.main:app --reload --port 8080

# 4. Test predictions
curl -X POST http://localhost:8080/predict_all

# 5. Test predictions with streamlit (optional)
uvicorn src.web_service.main:app --reload --port 8080
# In a different terminal now, in parallele, run:
streamlit run src/web_service/streamlit.py 
```


### 🐳 Part 2: Run the Project with Docker

```bash
# 1. Build the Docker image
docker build -t mlops-api -f Dockerfile.app .

# 2. Run the container
docker run -dp 0.0.0.0:8000:8080 mlops-api

# 3. Access the API
http://localhost:8000/docs

# 4. Test predictions again
curl -X POST http://localhost:8000/predict_all
```

---

## Note

Prefect server is not included in the Docker. It can be ran seperately using the instructions given below in the detailed documentation!

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Our Goal**: Use easier-to-obtain physical measurements to predict age automatically.

### Dataset Details

- **Samples**: 4,177 abalone
- **Features**: 8 physical measurements + 1 categorical (sex)
- **Target**: Number of rings (age ≈ rings + 1.5 years)

**Features:**
- `Sex`: M (Male), F (Female), I (Infant)
- `Length`: Longest shell measurement (mm)
- `Diameter`: Perpendicular to length (mm)
- `Height`: Height with meat in shell (mm)
- `Whole weight`: Entire abalone weight (grams)
- `Shucked weight`: Weight of meat (grams)
- `Viscera weight`: Gut weight after bleeding (grams)
- `Shell weight`: Weight after being dried (grams)
- `Rings`: Age indicator (+1.5 = age in years)

---

## 🏗️ Project Pipeline

This project follows a complete MLOps workflow:

```
1. Data Exploration (EDA) → 2. Preprocessing → 3. Model Training → 4. API Deployment
        ↓                         ↓                    ↓                    ↓
   notebooks/              src/modelling/        Prefect + MLflow      FastAPI + Docker
```

### Architecture Overview

```
├── data/                          # Raw and processed datasets
├── notebooks/                     # Exploratory analysis
│   ├── eda.ipynb                 # Data exploration
│   └── model.ipynb               # Initial modeling
├── src/
│   ├── modelling/                # Training pipeline
│   │   ├── train_flow.py         # Prefect orchestrated training
│   │   ├── preprocessing.py      # Data preprocessing
│   │   ├── training.py           # Model training logic
│   │   └── predicting.py         # Inference utilities
│   └── web_service/              # API service
│       ├── main.py               # FastAPI application
│       ├── lib/models.py         # Pydantic models
│       └── local_objects/        # Saved models
├── bin/
│   └── run_services.sh           # Deployment automation
└── Dockerfile.app                # Docker configuration
```

---

## 📖 Detailed Documentation

### Stage 1: Exploratory Data Analysis

**Location**: `notebooks/`

Comprehensive exploration revealing:
- **Data Quality**: No missing values, no duplicates, balanced sex distribution
- **Distributions**: Right-skewed for weights, normally distributed for dimensions
- **Correlations**: Shell weight (0.63), diameter (0.58), and length (0.56) best predict age
- **Insights**: Moderate R² (~0.55) indicates physical measurements capture ~55% of age variance

**Key Findings:**
- 164 outliers removed using IQR method
- High multicollinearity among size features (0.90-0.99)
- Infants significantly smaller and younger than adults

**Run EDA:**
```bash
jupyter notebook notebooks/eda.ipynb
```

**View MLflow experiments:**
```bash
cd notebooks
mlflow ui
# Navigate to http://localhost:5000
```

### Stage 2: Data Preprocessing & Feature Engineering

**Location**: `src/modelling/`

**Preprocessing Steps:**
1. Column standardization to snake_case
2. Outlier removal using IQR method (1.5 × IQR)
3. One-hot encoding for `sex` (drop_first=True)
4. Feature scaling with StandardScaler
5. Pipeline creation for consistent transformations

**Model Performance (Linear Regression):**
- **MAE**: ~1.54-1.55 rings
- **MSE**: ~4.46-4.53
- **R²**: ~0.54-0.55

**Train manually:**
```bash
python src/modelling/main.py data/abalone.csv
```

### Stage 3: Automated Training with Prefect

**Location**: `src/modelling/train_flow.py`, `src/modelling/deploy.py`

The Prefect pipeline orchestrates:
1. ✅ Load and preprocess data (with retry logic)
2. ✅ Feature-target separation
3. ✅ Train-test split (80/20)
4. ✅ Pipeline building (ColumnTransformer + LinearRegression)
5. ✅ Model training
6. ✅ Evaluation (MAE, MSE, R²)
7. ✅ Model persistence (pickle)

#### Option 1: One-Time Training

```bash
python src/modelling/train_flow.py data/abalone_clean.csv
```

#### Option 2: Scheduled Retraining

**Terminal 1 - Start Prefect Server:**
```bash
cd src
prefect server start
# UI available at http://127.0.0.1:4200
```

**Terminal 2 - Create Deployment:**
```bash
python src/modelling/deploy.py
# Keep this running to maintain the deployment
```

**Deployment Details:**
- **Name**: `abalone-weekly-retraining`
- **Schedule**: Every Sunday at 2:00 AM
- **Trigger manually**:
  ```bash
  prefect deployment run 'Train Abalone Age Prediction Model/abalone-weekly-retraining'
  ```

**Monitor in Prefect UI:**
- **Flows**: View all registered flows
- **Flow Runs**: Execution history with detailed logs
- **Deployments**: Manage scheduled runs
- **Task Timeline**: DAG visualization

**Benefits:**
- 🔍 Complete observability of each pipeline step
- 🔄 Automatic retry logic for transient failures
- 📅 Automated weekly retraining
- 📊 Centralized metrics tracking
- 📝 Structured logging

### Stage 4: API Deployment with FastAPI

**Location**: `src/web_service/`

#### API Endpoints

**`GET /`** - Health Check
```bash
curl http://localhost:8080/
# Response: {"health_check": "App up and running!"}
```

**`POST /predict_all`** - Batch Predictions
```bash
curl -X POST http://localhost:8080/predict_all
# Response: {"predictions": [10.5, 8.2, 12.1, ...]}
```

#### Local Development

**1. Configure paths in `src/web_service/utils.py`:**
```python
class Paths:
    path_data = "data/abalone_clean.csv"
    path_model = "src/web_service/local_objects/model.pkl"
```

**2. Start the API:**
```bash
uvicorn src.web_service.main:app --reload --port 8080
```

**3. Access documentation:**
- Interactive docs: http://localhost:8080/docs
- Alternative docs: http://localhost:8080/redoc
- OpenAPI schema: http://localhost:8080/openapi.json

**4. Test with Python:**
```python
import requests

# Health check
response = requests.get("http://localhost:8080/")
print(response.json())

# Make predictions
response = requests.post("http://localhost:8080/predict_all")
print(response.json()["predictions"])
```

**5. Test with streamlit (optional):**
```bash
uvicorn src.web_service.main:app --reload --port 8080
# In parallele, run in another terminal:
streamlit run src/web_service/streamlit.py 
```


#### Docker Deployment

**1. Build the image:**
```bash
docker build -t mlops-api -f Dockerfile.app .
```

**2. Run the container:**
```bash
docker run -d -p 0.0.0.0:8000:8080 mlops-api
```

**3. Access the API:**
- Base URL: http://localhost:8000
- Interactive docs: http://localhost:8000/docs

**4. Manage containers:**
```bash
# List running containers
docker ps

# View logs
docker logs <container_id>

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>
```

**Automated deployment:**
```bash
chmod +x bin/run_services.sh
./bin/run_services.sh
```

---

## 🛠️ Development Tools

### Code Quality

This project uses:
- **Ruff**: Fast Python linter and formatter
- **Pre-commit**: Automated code quality checks before commits
- **Type hints**: For better code documentation

```bash
# Run linting
ruff check .

# Run formatting
ruff format .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Experiment Tracking

**MLflow** tracks all model experiments:
```bash
mlflow ui
# Navigate to http://localhost:5000
```

**Tracked information:**
- Model parameters (scaler, test_size, random_state)
- Metrics (MAE, MSE, R²)
- Model artifacts
- Feature coefficients

---

## 📊 Model Performance

### Current Baseline (Linear Regression)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | ~1.55 rings | Average prediction error |
| **MSE** | ~4.5 | Variance of errors |
| **R²** | ~0.55 | Explains 55% of age variance |

### Feature Importance (Correlation with Rings)

1. **Shell weight**: 0.628 (strongest)
2. **Diameter**: 0.575
3. **Length**: 0.557
4. **Height**: 0.557
5. **Whole weight**: 0.540

### Key Insights

- Physical measurements capture moderate predictive power
- Environmental factors and genetics likely contribute to remaining variance
- High multicollinearity among size features suggests dimensionality reduction could help
- Non-linear models may capture additional patterns

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: Module not found errors
```bash
# Solution: Ensure you're in the project root
cd /path/to/xhec-mlops-project
source .venv/bin/activate
```

**Issue**: Port already in use
```bash
# Solution: Find and kill the process
lsof -i :8080
kill -9 <PID>
```

**Issue**: Model file not found
```bash
# Solution: Train the model first
python src/modelling/train_flow.py data/abalone.csv
```

**Issue**: Docker container exits immediately
```bash
# Solution: Check logs for errors
docker logs <container_id>
```

**Issue**: Predictions fail with shape errors
```bash
# Solution: Ensure preprocessing matches training data
# Check that src/web_service/utils.py points to abalone_clean.csv
```

---

## 🚀 Next Steps & Future Enhancements

### Model Improvements
- [ ] Try non-linear models (Random Forest, XGBoost, Neural Networks)
- [ ] Feature engineering (ratios, polynomial features)
- [ ] Handle multicollinearity (PCA, feature selection)
- [ ] Cross-validation for robust evaluation
- [ ] Hyperparameter tuning with Optuna

### API Enhancements
- [ ] Single prediction endpoint with JSON input
- [ ] File upload for batch predictions
- [ ] Model versioning and A/B testing
- [ ] Caching for frequent predictions
- [ ] Prometheus metrics endpoint
- [ ] Authentication (API keys, OAuth2)
- [ ] Rate limiting

### Infrastructure
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Automated testing suite
- [ ] Load testing and performance optimization
- [ ] Monitoring and alerting

### MLOps
- [ ] MLflow model registry integration
- [ ] Data drift detection
- [ ] Model performance monitoring
- [ ] Automated model rollback
- [ ] Feature store integration

---

## 📚 Additional Resources

- **Dataset Source**: [UCI Machine Learning Repository - Abalone Dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)
- **Prefect Documentation**: https://docs.prefect.io
- **FastAPI Documentation**: https://fastapi.tiangolo.com
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **Docker Documentation**: https://docs.docker.com

---

## 👥 Authors

This project was developed by:

| Name                     | Email                          |
|--------------------------|--------------------------------|
| verkrst                  | verakrstic002@gmail.com       |
| Shay0909                 | Shaymaa.bakkass@gmail.com     |
| nandanasreeraj123        | nandanasreeraj@gmail.com      |
| magdalenajankowska       | magdalenaj1000@gmail.com      |
| Sheetal-12               | sheetal.popat@hec.edu         |

---

## 📄 License

This project is part of the X-HEC MLOps course.

---

<div align="center">

**Built with ❤️ for the X-HEC MLOps Course**

</div>
