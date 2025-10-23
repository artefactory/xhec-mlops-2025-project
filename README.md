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
   # Install UV (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone your forked repository
   git clone https://github.com/YOUR_USERNAME/xhec-mlops-2025-project.git
   cd xhec-mlops-2025-project

   # Install all dependencies (including dev dependencies)
   uv sync --extra dev

   # Install pre-commit hooks for code quality
   uv run pre-commit install

   # Verify everything is working
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

## 👥 Team Information

**Team Members:**
- Soumyabrata Bairagi (roy.bensimon@polytechnique.edu)
- Samuel Rajzman (samuel.rajzman@hec.edu)
- Roy Bensimon (roy.bensimon@hec.edu)
- Vassili de Rosen (vassili.-rosen@hec.edu)
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
- [x] Updated README with team info
- [x] Clear setup and run instructions
- [x] Complete development environment setup
- [x] Troubleshooting guide
- [ ] All TODOs removed from code (in progress)

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
