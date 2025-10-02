<div align="center">

# xhec-mlops-project-student

[![CI status](https://github.com/artefactory/xhec-mlops-project-student/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/xhec-mlops-project-student/actions/workflows/ci.yaml?query=branch%3Amaster)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)]()

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

This repository has for purpose to industrialize the [Abalone age prediction](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) Kaggle contest.

<details>
<summary>Details on the Abalone Dataset</summary>

The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age.

**Goal**: predict the age of abalone (column "Rings") from physical measurements ("Shell weight", "Diameter", etc...)

You can download the dataset on the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

</details>

## Table of Contents

- [xhec-mlops-project-student](#xhec-mlops-project-student)
  - [Table of Contents](#table-of-contents)
  - [Deliverables and notation](#deliverables-and-notation)
    - [Deliverables](#deliverables)
    - [Notation](#notation)
  - [Steps to reproduce to build the deliverable](#steps-to-reproduce-to-build-the-deliverable)
    - [Pull requests in this project](#pull-requests-in-this-project)
    - [Tips to work on this project](#tips-to-work-on-this-project)

## Deliverables and notation

### Deliverables

The deliverable of this project is a copy of this repository with the industrialization of the Abalone age prediction model.
The industrialization takes the form of an API (which runs locally) that can be used to make predictions on new data.

### Notation

The work will be noted on the following criteria:

- **Clarity** and quality of code 
  - good module structure
  - naming conventions
  - (bonus) docstrings, formatting, type hints
- **Reproducibility** and clarity of instructions to run the code
  - Having a clear README.md with the steps to reproduce to test the code
  - Having a working docker image with the required features (see bellow)
- Having the following **features** in your project
  - Clear README with:
    - context of the project
    - clear steps to reproduce to run the code
  - A working API which can be used to make predictions on new data
    - The API can run on a docker container
    - The API has validation on input data (use Pydantic)
  - The code to get the trained model and encoder is in a separate module and must reproducible (not necessarily in a docker container)
  - The workflows to train the model and to make the inference (prediction of the age of abalone) are in separate modules and use Prefect `flow` and `task` objects

## Steps to reproduce to build the deliverable

To help you with the structure and order of steps to perform in this project, we have created different pull requests opened in this repository. You can follow the order of the pull requests to build your project.

> [!NOTE]
> There are "TODO" in the code of the pull requests or inside this repository. Each "TODO" corresponds to a task to perform to build the project.

You can follow the following steps:

- If not done already, create a GitHub account
- Fork this repository (one person per group)
- Add the different members of your group as admin to your forked repository
- For each opened pull request:
  - Do as many commits as necessary to perform the task of the pull request
  - Merge the pull request in your the main branch of your forked repository

### Pull requests in this project

Github [Pull Requests](https://docs.github.com/articles/about-pull-requests) are a way to propose changes to a repository. They have for purpose to integrate the work of *feature branches* into the main branch of the repository.

Each Pull Request has a number and objectives. Follow the order and associated objectives of the pull requests to build your project.
Each pull request is associated to a feature branch which contains empty files or files with some code to help you with the task of the pull request. 

Here is the workflow you should follow to build your project:

1. Work on the feature branch corresponding to the current pull request (`git checkout <branch_name>`)
2. Commit and push your changes (**each person in the group must do at least one commit!**) in as many commits as necessary to perform the task of the pull request
3. When finished, merge the pull request in the main branch of your forked repository (`master`)
4. Go to the next pull request and repeat steps 1 to 3


### Tips to work on this project

- Use a virtual environment to install the dependencies of the project (conda or virtualenv for instance)

- Once your virtual environment is activated, install the pre-commit hooks to automatically format your code before each commit:

```bash
pip install pre-commit
pre-commit install
```

This will guarantee that your code is formatted correctly and of good quality before each commit.

- Use a `requirements.in` file to list the dependencies of your project. You can use the following command to generate a `requirements.txt` file from a `requirements.in` file:

```bash
pip-compile requirements.in
```
