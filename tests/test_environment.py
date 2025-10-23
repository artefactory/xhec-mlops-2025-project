"""Basic tests to verify environment setup."""


def test_python_version():
    """Test that Python version is correct."""
    import sys

    assert sys.version_info >= (3, 10), "Python version should be 3.10 or higher"
    assert sys.version_info < (3, 12), "Python version should be less than 3.12"


def test_imports():
    """Test that required packages can be imported."""
    import pandas  # noqa: F401
    import pydantic  # noqa: F401
    import sklearn  # noqa: F401
    import numpy  # noqa: F401

    assert True, "All required packages imported successfully"
