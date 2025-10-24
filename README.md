<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

Welcome to your MLOps project! In this hands-on project, you'll build a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Your Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Your Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)


## 🚀 Quick Start

### Prerequisites
- GitHub account
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (for dataset download)
- Python 3.10 or 3.11

### Setup Steps

1. **Fork this repository**
   - ⚠️ **Important**: Uncheck "Copy the `main` branch only" to get all project branches

2. **Add your team members** as admins to your forked repository

3. **Set up your development environment**:
   ```bash
   # Create and activate a virtual environment
   uv sync
   source venv/bin/activate # on Windows: venv\Scripts\activate

   # Install pre-commit hooks for code quality
    uv pip install pre-commit
    uv run pre-commit install
   ```

## 📋 What You'll Build

By the end of this project, you'll have created:

### 🤖 **Automated ML Pipeline**
- Training workflows using Prefect
- Automatic model retraining on schedule
- Reproducible model and data processing

### 🌐 **Prediction API**
- REST API for real-time predictions
- Input validation with Pydantic
- Docker containerization

### 📊 **Production-Ready Code**
- Clean, well-documented code
- Automated testing and formatting
- Proper error handling

## 📝 How to Work on This Project

### The Branch-by-Branch Approach

This project is organized into numbered branches, each representing a step in building your MLOps system. Think of it like a guided tutorial where each branch teaches you something new!

**Here's how it works**:

1. **Each branch = One pull request** with specific tasks
2. **Follow the numbers** (branch_0, branch_1, etc.) in order
3. **Read the PR instructions** (PR_0.md, PR_1.md, etc.) before starting
4. **Complete all TODOs** in that branch's code
5. **Create a pull request** when done
6. **Merge and move to the next branch**

### Step-by-Step Workflow

For each numbered branch:

```bash
# Switch to the branch
git checkout branch_number_i

# Get latest changes (except for branch_1)
git pull origin main
# Note: A VIM window might open - just type ":wq" to close it

# Push your branch
git push
```

Then:
1. 📖 Read the PR_i.md file carefully
2. 💻 Complete all the TODOs in the code
3. 🔧 Test your changes
4. 📤 Open **ONE** pull request to your main branch
5. ✅ Merge the pull request
6. 🔄 Move to the next branch

> **💡 Pro Tip**: Always integrate your previous work when starting a new branch (except branch_1)!

### 🔍 Understanding Pull Requests

Pull Requests (PRs) are how you propose and review changes before merging them into your main codebase. They're essential for team collaboration!

**Important**: When creating a PR, make sure you're merging into YOUR forked repository, not the original:

❌ **Wrong** (merging to original repo):
![PR Wrong](assets/PR_wrong.png)

✅ **Correct** (merging to your fork):
![PR Right](assets/PR_right.png)

## 🔄 Running the Training Pipeline with Prefect

This project uses [Prefect](https://www.prefect.io/) to orchestrate the ML training pipeline with flows and tasks for better observability and monitoring.

### Starting the Prefect Server

Before running the training pipeline, you need to configure and start the Prefect server:

1. **Configure the Prefect API URL** (first time setup):
   ```bash
   prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
   ```

2. **Start the Prefect server**:
   ```bash
   uv run prefect server start
   ```

   Keep this terminal window open while you work with Prefect.

3. **Access the Prefect UI**:

   Open your browser and navigate to `http://localhost:4200`

### Running the Training Flow

You can run the training pipeline using `uv run` (recommended) or directly with Python:

1. **Using uv run** (recommended - automatically manages dependencies):
   ```bash
   uv run src/modelling/main.py --trainset_path data/abalone.csv
   ```

   Or with the default path:
   ```bash
   uv run src/modelling/main.py
   ```

2. **Using Python directly** (requires activated virtual environment):
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   python src/modelling/main.py --trainset_path data/abalone.csv
   ```

### Viewing Flow Runs in the UI

Once you've started the Prefect server and run the training pipeline, you can monitor and inspect your runs:

1. **Open the Prefect UI** at `http://localhost:4200`
2. **Navigate to Flow Runs**:
   - Click on "Runs" in the left sidebar
   - You'll see all your `training-pipeline` executions
3. **Inspect a Flow Run**:
   - Click on any flow run to see:
     - Execution status and duration
     - The `prepare-data` subflow with its tasks (load_data, encode_sex, splitting_data)
     - The `train` task showing model training
     - Detailed logs for each task
     - Visual execution timeline

**Pipeline Structure**:
- **Main Flow**: `training-pipeline` - orchestrates the entire process
- **Subflow**: `prepare-data` - handles data loading, encoding, and splitting
- **Tasks**: Individual operations (load_data, encode_sex, splitting_data, train, pickle_object)

All flow runs are tracked and can be reviewed in the UI, even after completion!

## 💡 Development Tips

### Managing Dependencies

Use uv to manage dependencies. Install or update packages with:

```bash
uv add <package>==<version>
```

Then sync the environment and regenerate the dependency files:

```bash
uv sync
```

### Code Quality
- The pre-commit hooks will automatically format your code
- Remove all TODOs and unused code before final submission
- Use clear variable names and add docstrings

### Continuous Integration

This project includes automated CI workflows that run on every push and pull request:

**What the CI checks:**
- ✅ Python 3.10 and 3.11 compatibility
- ✅ Pre-commit hooks (formatting, linting, file checks)
- ✅ Ruff linting and formatting
- ✅ Pytest with coverage reporting

The CI workflow automatically validates your code quality and ensures compatibility across Python versions. Make sure all CI checks pass before merging your pull requests!

## 📊 Evaluation Criteria

Your project will be evaluated on:

### 🔍 **Code Quality**
- Clean, readable code structure
- Proper naming conventions
- Good use of docstrings and type hints

### 🎨 **Code Formatting**
- Consistent style (automated with pre-commit)
- Professional presentation

### ⚙️ **Functionality**
- Code runs without errors
- All requirements implemented correctly

### 📖 **Documentation & Reproducibility**
- Clear README with setup instructions
- Team member names and GitHub usernames
- Step-by-step instructions to run everything

### 🤝 **Collaboration**
- Effective use of Pull Requests
- Good teamwork and communication

---

## 🎯 Final Deliverables Checklist

When you're done, your repository should contain:

✅ **Automated Training Pipeline**
- [ ] Prefect workflows for model training
- [ ] Separate modules for training and inference
- [ ] Reproducible model and encoder generation

✅ **Automated Deployment**
- [ ] Prefect deployment for regular retraining

✅ **Production API**
- [ ] Working REST API for predictions
- [ ] Pydantic input validation
- [ ] Docker containerization

✅ **Professional Documentation**
- [ ] Updated README with team info
- [ ] Clear setup and run instructions
- [ ] All TODOs removed from code

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
