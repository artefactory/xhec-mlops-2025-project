#!/bin/bash

# Script to run the Abalone Age Prediction API service

echo "🦪 Starting Abalone Age Prediction API Service..."
echo "=================================================="

# Check if model exists, if not train one
if [ ! -f "src/web_service/local_objects/model.pkl" ]; then
    echo "⚠️  Model not found! Training a new model..."
    python simple_train.py
    if [ $? -eq 0 ]; then
        echo "✅ Model trained successfully!"
    else
        echo "❌ Model training failed!"
        exit 1
    fi
fi

# Start the FastAPI server
echo "🚀 Starting FastAPI server on port 8000..."
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🏠 Home Page: http://localhost:8000/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

exec uvicorn src.web_service.main:app --host 0.0.0.0 --port 8000
