# Dockerfile for Abalone Age Prediction API
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY simple_train.py ./
COPY bin/ ./bin/

# Create directory for model storage
RUN mkdir -p src/web_service/local_objects

# Make run script executable
RUN chmod +x bin/run_services.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the service
CMD ["./bin/run_services.sh"]
