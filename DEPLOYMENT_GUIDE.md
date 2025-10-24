# 🚀 MLOps Final Deployment Guide

## 📋 Overview

This guide covers the complete deployment of the MLOps Abalone Age Prediction system with all production-ready components.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   Prefect      │
│   Frontend      │    │   Backend       │    │   Orchestrator │
│   Port: 8501    │    │   Port: 8000    │    │   Port: 4200   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   MLflow        │
                    │   Tracking      │
                    │   Port: 5000    │
                    └─────────────────┘
```

## 🚀 Quick Deployment

### Prerequisites
- Docker and Docker Compose installed
- Dataset `abalone.csv` in `data/` directory
- Git repository cloned

### One-Command Deployment
```bash
# Make script executable and run
chmod +x deploy.sh
./deploy.sh
```

This will:
- ✅ Build all Docker containers
- ✅ Start all services
- ✅ Verify health checks
- ✅ Display access URLs

## 🔧 Manual Deployment

### Step 1: Prepare Environment
```bash
# Create directories
mkdir -p data src/web_service/local_objects

# Download dataset (if not present)
# Place abalone.csv in data/ directory
```

### Step 2: Start Services
```bash
# Start all services
docker-compose up --build -d

# Check logs
docker-compose logs -f
```

### Step 3: Verify Deployment
```bash
# Check service health
curl http://localhost:8000/health    # API
curl http://localhost:8501/_stcore/health  # Streamlit
curl http://localhost:4200/api/health      # Prefect
curl http://localhost:5000/health         # MLflow
```

## 📊 Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Streamlit** | http://localhost:8501 | Interactive frontend |
| **API** | http://localhost:8000 | REST API backend |
| **API Docs** | http://localhost:8000/docs | Swagger documentation |
| **Prefect** | http://localhost:4200 | Workflow orchestration |
| **MLflow** | http://localhost:5000 | Model tracking |

## 🔄 Automated Training

### Prefect Deployment
```bash
# Create Prefect deployment
python deploy_prefect.py

# View in Prefect UI
open http://localhost:4200
```

**Schedule**: Daily at 2 AM UTC
**Tags**: mlops, training, abalone, production

### Manual Training
```bash
# Run training pipeline
python src/modelling/main.py

# With custom parameters
python src/modelling/main.py --trainset_path data/abalone.csv
```

## 📈 Model Tracking

### MLflow Integration
- **Automatic logging** of model parameters and metrics
- **Model versioning** and artifact storage
- **Experiment tracking** for model comparison

### View Experiments
```bash
# Open MLflow UI
open http://localhost:5000

# View experiments, runs, and models
```

## 🧪 Testing

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sex": "M",
    "length": 0.455,
    "diameter": 0.365,
    "height": 0.095,
    "whole_weight": 0.514,
    "shucked_weight": 0.2245,
    "viscera_weight": 0.101,
    "shell_weight": 0.15
  }'
```

### Frontend Testing
1. Open http://localhost:8501
2. Adjust sliders in sidebar
3. Click "Prédire l'âge"
4. Verify prediction results

## 🛠️ Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check Docker status
docker ps

# View logs
docker-compose logs [service_name]

# Restart services
docker-compose restart
```

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8000
netstat -tulpn | grep :8501
netstat -tulpn | grep :4200
netstat -tulpn | grep :5000

# Modify ports in docker-compose.yml if needed
```

#### Data Issues
```bash
# Verify dataset
ls -la data/abalone.csv

# Check data format
head -5 data/abalone.csv
```

### Logs and Monitoring
```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f api
docker-compose logs -f streamlit
docker-compose logs -f prefect
docker-compose logs -f mlflow
```

## 🛑 Stopping Services

### Graceful Shutdown
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop and remove images
docker-compose down --rmi all
```

## 📊 Production Considerations

### Performance
- **API Response Time**: < 100ms
- **Concurrent Users**: 100+ simultaneous
- **Model Accuracy**: R² > 0.8

### Monitoring
- **Health Checks**: Automatic every 30s
- **Logs**: Structured logging with timestamps
- **Metrics**: Response time, error rate, throughput

### Scaling
- **Horizontal Scaling**: Multiple API instances
- **Load Balancing**: Nginx or similar
- **Database**: PostgreSQL for production data

## 🎯 Success Criteria

✅ **All services running** and healthy  
✅ **API responding** to prediction requests  
✅ **Streamlit interface** functional  
✅ **Prefect workflows** executing  
✅ **MLflow tracking** model experiments  
✅ **Docker containers** properly orchestrated  

## 🎉 Deployment Complete!

Your MLOps system is now fully deployed with:
- 🤖 **Automated ML Pipeline** (Prefect)
- 🌐 **Production API** (FastAPI + Docker)
- 🎨 **Interactive Frontend** (Streamlit)
- 📊 **Model Tracking** (MLflow)
- 🐳 **Container Orchestration** (Docker Compose)

**Ready for production use!** 🚀
