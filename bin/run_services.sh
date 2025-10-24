#!/bin/bash

# Script to run both Prefect server and FastAPI application
# Prefect runs in background, FastAPI runs in foreground

echo "🚀 Starting Prefect server on port 4201..."
prefect server start --host 0.0.0.0 --port 4201 &

echo "🚀 Starting FastAPI application on port 8001..."
uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
