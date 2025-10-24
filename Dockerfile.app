# Multi-stage Dockerfile for Abalone Age Prediction API
# Uses Python 3.11 and installs dependencies using uv

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY src/ ./src/
COPY bin/ ./bin/

# Make the run script executable
RUN chmod +x bin/run_services.sh

# Expose ports
# Port 8001 for FastAPI (mapped to 8000 on host)
# Port 4201 for Prefect (mapped to 4200 on host)
EXPOSE 8001 4201

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the services using the shell script
CMD ["./bin/run_services.sh"]
