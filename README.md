<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

Welcome to your MLOps project! In this hands-on project, you'll build a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Our Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Our Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Dataset Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

PR_0:

PR_0:

## 🛠 Development Setup

To set up your environment:

```bash
uv sync
source .venv/bin/activate
pre-commit install
pre-commit run --all-files
```

This project uses:

- Ruff for linting and formatting

- Pre-commit for code quality checks

- Prefect for pipeline orchestration

- FastAPI for serving predictions


PR_3:

## 🔄 Training Pipeline with Prefect

### Option 1: Run Training Flow Directly (One-Time)

**Using the new Prefect flow:**
```bash
python src/modelling/train_flow.py data/abalone.csv
```

**Using the original script (still works):**
```bash
python src/modelling/main.py data/abalone.csv
```

### Option 2: Automated Retraining with Prefect Deployment

#### Start Prefect Server

In a **separate terminal**, start the Prefect server:
```bash
prefect server start
```

The Prefect UI will be available at: **http://127.0.0.1:4200**

#### Create and Run the Deployment

In another terminal, activate your virtual environment and run:
```bash
python src/modelling/deploy.py
```

This will:
- ✅ Create a deployment named "abalone-weekly-retraining"
- ✅ Schedule it to run **every Sunday at 2:00 AM**
- ✅ Keep the deployment server running (leave this terminal open)

**Note:** Keep this terminal running to maintain the deployment. Press `Ctrl+C` to stop.

#### Manually Trigger a Deployment Run

You can trigger the deployment without waiting for the schedule:
```bash
# List all deployments
prefect deployment ls

# Trigger the deployment
prefect deployment run 'Train Abalone Age Prediction Model/abalone-weekly-retraining'
```

### Visualize Flows in Prefect UI

1. Open **http://127.0.0.1:4200** in your browser
2. Navigate to:
   - **Flows** → See all registered flows
   - **Flow Runs** → View execution history with detailed logs
   - **Deployments** → Manage scheduled runs
   - **Work Pools & Workers** → Monitor deployment infrastructure

3. Click on any flow run to see:
   - 📊 Task execution timeline and DAG visualization
   - 📝 Detailed logs for each task
   - ⚙️ Input parameters and output results
   - 📈 Performance metrics (MAE, MSE, R²)
   - ⏱️ Task duration and retry information

### Monitor Flow Runs
```bash
# View recent flow runs
prefect flow-run ls

# Get details of a specific run
prefect flow-run inspect <flow-run-id>

# View logs
prefect flow-run logs <flow-run-id>
```

## 🏗️ Understanding the Training Pipeline Architecture

The Prefect training pipeline (`train_flow.py`) consists of these orchestrated tasks:

1. **Load and Preprocess Data** (with retry logic)
   - Reads CSV file
   - Converts columns to snake_case
   - Calculates age from rings
   - Removes IQR outliers
   - One-hot encodes the 'sex' column

2. **Prepare Features and Target**
   - Separates features (X) from target (y = rings)

3. **Split Train Test**
   - Creates 80/20 train-test split with reproducible random seed

4. **Build Pipeline**
   - Creates sklearn ColumnTransformer for numeric scaling
   - Passes through one-hot encoded sex columns
   - Wraps preprocessing + LinearRegression in Pipeline

5. **Train Model**
   - Fits the complete pipeline on training data

6. **Evaluate Model**
   - Calculates MAE, MSE, and R² metrics on test set

7. **Save Model**
   - Pickles the trained pipeline to `src/web_service/local_objects/model.pkl`

### Key Benefits of the Prefect Implementation

- ✅ **Observability**: See exactly which tasks succeeded/failed and why
- ✅ **Retry Logic**: Automatically retry data loading if it fails (network issues, etc.)
- ✅ **Scheduling**: Automated weekly retraining without manual intervention
- ✅ **Logging**: Centralized, structured logs for all tasks
- ✅ **Parameterization**: Easy to change test_size, random_state, or paths
- ✅ **Monitoring**: Track model performance metrics over time in the UI
- ✅ **Reproducibility**: All parameters and results are logged for each run
