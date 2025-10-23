# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)

## 📋 Overview

MLOps project to predict abalone age using physical measurements instead of counting shell rings under a microscope.

**Dataset**: [Abalone Dataset on Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

---

## 🚀 Environment Setup

### Prerequisites
- Python 3.10 or 3.11
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/xhec-mlops-2025-project.git
cd xhec-mlops-2025-project

# Install dependencies
uv sync --all-extras

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install pre-commit hooks
uv run pre-commit install
```

### Download Dataset

1. Download `abalone.csv` from [Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)
2. Place it in the project root directory

---

## 🔧 Development

### Managing Dependencies

```bash
# Add a new package
uv add <package>==<version>

# Sync environment
uv sync
```

### Code Quality

Pre-commit hooks automatically run on each commit to:
- Format code with `ruff`
- Check YAML, TOML, and JSON files
- Remove trailing whitespace
- Ensure files end with newline

To run manually:
```bash
uv run pre-commit run --all-files
```

### Running Tests

```bash
uv run pytest
```

---

## 📚 Project Structure

This project follows a branch-based workflow. Each numbered branch (`0/environment_setup`, `1/eda_and_modelling_notebooks`, etc.) represents a development milestone.

---

## 👥 Contributors

- **Thibaud**
