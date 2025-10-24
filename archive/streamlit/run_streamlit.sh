#!/bin/bash

# Script de lancement pour l'interface Streamlit
# Usage: ./run_streamlit.sh

echo "🐚 Démarrage de l'interface Streamlit pour la prédiction d'âge des abalones..."

# Vérifier si l'environnement virtuel est activé
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Environnement virtuel activé: $VIRTUAL_ENV"
else
    echo "⚠️  Aucun environnement virtuel détecté. Assurez-vous d'avoir activé votre environnement."
fi

# Vérifier si Streamlit est installé
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit n'est pas installé. Installation en cours..."
    pip install streamlit plotly requests
fi

# Vérifier si l'API FastAPI est accessible
echo "🔍 Vérification de l'API FastAPI..."
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✅ API FastAPI accessible sur http://localhost:8001"
else
    echo "⚠️  API FastAPI non accessible. Assurez-vous qu'elle est démarrée."
    echo "   Vous pouvez la démarrer avec: uvicorn src.web_service.main:app --reload"
fi

echo ""
echo "🚀 Lancement de Streamlit..."
echo "   Interface disponible sur: http://localhost:8501"
echo "   Appuyez sur Ctrl+C pour arrêter"
echo ""

# Lancer Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
