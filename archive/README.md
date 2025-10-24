# Archive Directory

This directory contains files that are not essential for the core FastAPI + Docker functionality but may be useful for reference or future development.

## 📁 Directory Structure

### `streamlit/`
Contains Streamlit-related files for web interface development:
- `streamlit_app.py` - Main Streamlit application
- `demo_streamlit.py` - Streamlit demo
- `run_streamlit.sh` - Script to run Streamlit
- `test_streamlit.py` - Streamlit tests
- `STREAMLIT_README.md` - Streamlit documentation
- `.streamlit/` - Streamlit configuration

### `testing/`
Contains testing-related files:
- `pytest.ini` - Pytest configuration
- `run_tests.py` - Test runner script
- `tests/` - Test directory with test files

### `docs/`
Contains additional documentation:
- `API_GUIDELINES.md` - API development guidelines

## 🎯 Purpose

These files were moved here to keep the main repository clean and focused on the essential FastAPI + Docker implementation. They can be restored to the main directory if needed for future development.

## 🔄 Restoring Files

To restore any of these files to the main directory:
```bash
# Example: Restore Streamlit app
cp archive/streamlit/streamlit_app.py .

# Example: Restore testing setup
cp -r archive/testing/* .
```
