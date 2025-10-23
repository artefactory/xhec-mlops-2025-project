<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 👥 Team Members

- **Augustin Naton** - [@auggy-ntn](https://github.com/auggy-ntn)
- **Louis Peretie** - [@PeretieLouis](https://github.com/PeretieLouis)
- **Grégoire Bidault** - [@Gregoire-Bidault](https://github.com/gregoire-bidault)
- **Gustave Triomphe** - [@gustave-triomphe](https://github.com/gustave-triomphe)
- **Sofia Casalini** - [@sofiacasalini](https://github.com/sofiacasalini)

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
- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

### Setup Steps

1. **Fork this repository**
   - ⚠️ **Important**: Uncheck "Copy the `main` branch only" to get all project branches

2. **Add your team members** as admins to your forked repository

3. **Set up your development environment**:
   ```bash
   # Install uv if you haven't already
   # On macOS/Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # On Windows:
   # powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Sync dependencies and create virtual environment
   uv sync

   # Activate the virtual environment
   source .venv/bin/activate  # on macOS/Linux
   # .venv\Scripts\activate   # on Windows

   # Install pre-commit hooks for code quality
   uv run pre-commit install
   ```

4. **Verify your setup**:
   ```bash
   # Check Python version
   python --version  # Should show Python 3.11.x

   # Test pre-commit hooks
   uv run pre-commit run --all-files
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

## 💡 Development Tips

### Managing Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

**Adding new dependencies:**

```bash
# Add a production dependency
uv add <package>

# Add a development dependency
uv add --dev <package>

# Add a specific version
uv add <package>==<version>
```

**Syncing dependencies:**

After pulling changes or modifying dependencies, sync your environment:

```bash
uv sync
```

**Important**: The `uv.lock` file is tracked in git to ensure reproducible builds across all team members.

### Code Quality

This project uses automated code quality tools:

- **ruff**: Fast Python linter and formatter
- **pre-commit**: Runs checks automatically before each commit
- **nbstripout**: Strips output from Jupyter notebooks before committing

The pre-commit hooks will automatically:
- Format your code with ruff
- Check for linting issues
- Sort imports
- Strip notebook outputs
- Validate YAML, TOML, and JSON files

**Manual checks:**

```bash
# Run all pre-commit hooks manually
uv run pre-commit run --all-files

# Format code with ruff
uv run ruff format .

# Check for linting issues
uv run ruff check .
```

**Best practices:**
- Remove all TODOs and unused code before final submission
- Use clear variable names and add docstrings
- Write type hints for function parameters and return values

## � Running the ML Pipeline with Prefect

This project uses [Prefect](https://www.prefect.io/) for ML workflow orchestration, providing observability, scheduling, and automated retraining capabilities.

### Starting the Prefect Server

Before running any workflows, start the Prefect server:

```bash
# Set the Prefect API URL (first time only)
uv run prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api

# Verify SQLite is installed (required for Prefect backend)
sqlite3 --version

# Start the Prefect server
uv run prefect server start --host 0.0.0.0
```

**Access the Prefect UI**: http://0.0.0.0:4200

Keep this terminal running while you work with Prefect!

### Running the Training Pipeline

In a **new terminal**, run the training workflow:

```bash
# Activate virtual environment
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows

# Run the training pipeline
uv run python -m src.modelling.main data/abalone.csv
```

This will:
- Load and preprocess the data
- Train the model
- Save the model and scaler to `src/web_service/local_objects/`

**View the flow run** in the Prefect UI at http://0.0.0.0:4200/runs

### Creating Automated Deployments

To set up automated retraining on a schedule:

```bash
# Run the deployment script (keeps running)
uv run python -m src.modelling.deploy
```

This creates a deployment that:
- Runs the training pipeline daily at 2 AM UTC
- Can be manually triggered from the UI
- Provides full observability of each training run

**Managing deployments:**
- View deployments: http://0.0.0.0:4200/deployments
- Click "Quick Run" to trigger an immediate training run
- Toggle the schedule on/off
- View deployment history and logs

**To stop the deployment**: Press `Ctrl+C` in the terminal running the deployment

### Prefect UI Features

The Prefect dashboard provides:
- **Flow Runs**: View all pipeline executions with status and logs
- **Deployments**: Manage scheduled training runs
- **Work Queues**: Monitor task execution
- **Flow Run Graph**: Visualize task dependencies and execution flow
- **Logs**: Detailed execution logs for debugging

**Useful commands:**

```bash
# Reset the Prefect database (if needed)
uv run prefect server database reset

# View deployments via CLI
uv run prefect deployment ls

# View flow runs via CLI
uv run prefect flow-run ls
```

## �📊 Evaluation Criteria

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
