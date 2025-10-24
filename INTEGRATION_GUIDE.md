# 🚀 Guide d'Intégration Streamlit + FastAPI

## 📋 Vue d'ensemble

Ce guide explique comment utiliser l'application complète MLOps avec l'interface Streamlit et l'API FastAPI intégrées.

## 🏗️ Architecture

```
┌─────────────────┐    HTTP     ┌─────────────────┐    ML     ┌─────────────────┐
│   Streamlit     │ ──────────► │   FastAPI       │ ────────► │   ML Pipeline   │
│   Frontend      │             │   Backend       │           │   (Prefect)     │
│   Port: 8501    │             │   Port: 8001    │           │                 │
└─────────────────┘             └─────────────────┘           └─────────────────┘
```

## 🚀 Démarrage Rapide

### 1. Démarrer l'API FastAPI
```bash
cd /home/alcyenna/xhec-mlops-2025-project
source venv/bin/activate
uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Démarrer Streamlit (nouveau terminal)
```bash
cd /home/alcyenna/xhec-mlops-2025-project
source venv/bin/activate
streamlit run streamlit_app.py
```

### 3. Accéder aux interfaces
- 🌐 **Streamlit** : http://localhost:8501
- 📚 **API Docs** : http://localhost:8001/docs
- 🔍 **Health Check** : http://localhost:8001/

## 🎯 Fonctionnalités

### Interface Streamlit
- ✅ **Saisie interactive** des caractéristiques d'abalone
- ✅ **Sliders** pour ajuster les valeurs
- ✅ **Prédictions en temps réel** via l'API
- ✅ **Visualisations** des données
- ✅ **Gestion d'erreurs** robuste

### API FastAPI
- ✅ **Endpoint de santé** : `/health`
- ✅ **Prédiction simple** : `/predict`
- ✅ **Prédiction batch** : `/predict/batch`
- ✅ **Documentation interactive** : `/docs`
- ✅ **Validation des données** avec Pydantic

## 📊 Utilisation

### Via Streamlit (Recommandé)
1. Ouvrez http://localhost:8501
2. Ajustez les sliders dans la sidebar
3. Cliquez sur "Prédire l'âge"
4. Consultez les résultats et visualisations

### Via API Directe
```bash
# Test de santé
curl -X GET http://localhost:8001/

# Prédiction
curl -X POST http://localhost:8001/predict \
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

## 🔧 Configuration

### Variables d'environnement
- `API_BASE_URL` : URL de base de l'API (défaut: http://localhost:8001)
- `MODEL_PATH` : Chemin vers le modèle ML
- `DATA_PATH` : Chemin vers les données d'entraînement

### Ports
- **Streamlit** : 8501
- **FastAPI** : 8001
- **Prefect UI** : 4200 (si démarré)

## 🧪 Tests

### Tests unitaires
```bash
# Tests Streamlit
python -m pytest test_streamlit.py -v

# Tests API
python -m pytest tests/test_api.py -v

# Tous les tests
python -m pytest test_streamlit.py tests/test_api.py -v
```

### Tests d'intégration
```bash
# Démarrer l'API
uvicorn src.web_service.main:app --port 8001 &

# Tester la connexion
curl -X GET http://localhost:8001/

# Démarrer Streamlit
streamlit run streamlit_app.py
```

## 🐳 Docker (À venir)

L'intégration Docker complète sera ajoutée dans la prochaine étape :
- Containerisation de l'API
- Containerisation de Streamlit
- Orchestration avec Docker Compose
- Intégration Prefect + MLflow

## 🚨 Dépannage

### Problèmes courants

#### API non accessible
```bash
# Vérifier que l'API est démarrée
curl -X GET http://localhost:8001/

# Vérifier les logs
uvicorn src.web_service.main:app --port 8001 --reload
```

#### Erreurs de connexion Streamlit
- Vérifiez que l'API est démarrée sur le port 8001
- Vérifiez l'URL dans `streamlit_app.py`
- Consultez les logs Streamlit

#### Erreurs de prédiction
- Vérifiez le format des données d'entrée
- Consultez la documentation API : http://localhost:8001/docs
- Vérifiez que le modèle est chargé

## 📈 Performance

### Métriques recommandées
- **Latence API** : < 100ms pour une prédiction
- **Throughput** : > 100 prédictions/minute
- **Disponibilité** : > 99.9%

### Monitoring
- Health checks automatiques
- Logs structurés
- Métriques de performance

## 🎉 Résultat Final

✅ **Pipeline ML** : Prefect + scikit-learn
✅ **API Backend** : FastAPI + Pydantic
✅ **Interface Frontend** : Streamlit
✅ **Tests** : pytest + couverture
✅ **Documentation** : Swagger + guides
✅ **Intégration** : Frontend ↔ Backend

**L'application MLOps est maintenant complète et fonctionnelle !** 🚀
