#!/bin/bash

# MLOps Deployment Script
# Complete deployment with Docker Compose, Prefect, and MLflow

echo "🚀 Starting MLOps Complete Deployment"
echo "======================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ docker compose not found. Please install Docker with Compose support."
    exit 1
fi

echo "✅ Docker environment ready"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p src/web_service/local_objects

# Check if data exists
if [ ! -f "data/abalone.csv" ]; then
    echo "⚠️  Dataset not found. Please download abalone.csv to data/ directory"
    echo "📥 Download from: https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset"
    exit 1
fi

echo "✅ Data directory ready"

# Build and start services
echo "🐳 Building and starting Docker services..."
docker compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check API
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API service is healthy"
else
    echo "❌ API service is not responding"
fi

# Check Streamlit
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Streamlit service is healthy"
else
    echo "❌ Streamlit service is not responding"
fi

# Check Prefect
if curl -f http://localhost:4200/api/health > /dev/null 2>&1; then
    echo "✅ Prefect service is healthy"
else
    echo "❌ Prefect service is not responding"
fi

# Check MLflow
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ MLflow service is healthy"
else
    echo "❌ MLflow service is not responding"
fi

echo ""
echo "🎉 MLOps Deployment Complete!"
echo "=============================="
echo ""
echo "📊 Services Available:"
echo "  🌐 Streamlit Frontend: http://localhost:8501"
echo "  🔧 API Backend:        http://localhost:8000"
echo "  📚 API Documentation:  http://localhost:8000/docs"
echo "  🔄 Prefect UI:         http://localhost:4200"
echo "  📈 MLflow UI:          http://localhost:5000"
echo ""
echo "🚀 Next Steps:"
echo "  1. Open Streamlit: http://localhost:8501"
echo "  2. Test predictions in the interface"
echo "  3. Monitor training in Prefect: http://localhost:4200"
echo "  4. Track experiments in MLflow: http://localhost:5000"
echo ""
echo "🛑 To stop services: docker compose down"
