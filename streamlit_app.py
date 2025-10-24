"""
Streamlit Interface for Abalone Age Prediction
Interacts with the FastAPI backend to make predictions
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Configuration de la page
st.set_page_config(
    page_title="🐚 Abalone Age Prediction",
    page_icon="🐚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration de l'API
API_BASE_URL = "http://localhost:8001"  # URL de l'API FastAPI
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/"


def check_api_health() -> bool:
    """Vérifier si l'API est accessible"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def make_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Faire une prédiction via l'API"""
    try:
        response = requests.post(PREDICT_ENDPOINT, json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None


def load_sample_data() -> pd.DataFrame:
    """Charger des données d'exemple pour la démonstration"""
    try:
        df = pd.read_csv("data/abalone.csv")
        return df.sample(n=min(100, len(df)))  # Échantillon de 100 lignes max
    except FileNotFoundError:
        st.warning("Fichier de données non trouvé. Utilisation de données simulées.")
        return pd.DataFrame(
            {
                "Sex": ["M", "F", "I"] * 10,
                "Length": np.random.uniform(0.1, 0.8, 30),
                "Diameter": np.random.uniform(0.1, 0.7, 30),
                "Height": np.random.uniform(0.01, 0.3, 30),
                "Whole weight": np.random.uniform(0.1, 2.0, 30),
                "Shucked weight": np.random.uniform(0.05, 1.0, 30),
                "Viscera weight": np.random.uniform(0.01, 0.5, 30),
                "Shell weight": np.random.uniform(0.01, 0.5, 30),
                "Rings": np.random.randint(1, 30, 30),
            }
        )


def main():
    """Interface principale Streamlit"""

    # En-tête
    st.title("🐚 Prédiction de l'Âge des Abalones")
    st.markdown(
        """
    Cette application utilise un modèle de machine learning pour prédire l'âge d'une abalone
    (escargot de mer) en se basant sur ses caractéristiques physiques.
    """
    )

    # Vérification de l'API
    with st.spinner("Vérification de la connexion à l'API..."):
        api_healthy = check_api_health()

    if not api_healthy:
        st.error(
            "❌ L'API FastAPI n'est pas accessible. Assurez-vous qu'elle est démarrée sur http://localhost:8001"
        )
        st.stop()
    else:
        st.success("✅ Connexion à l'API établie")

    # Sidebar pour la saisie des données
    st.sidebar.header("📊 Caractéristiques de l'Abalone")

    # Saisie des données
    sex = st.sidebar.selectbox(
        "Sexe", options=["M", "F", "I"], help="M = Mâle, F = Femelle, I = Immature"
    )

    length = st.sidebar.slider(
        "Longueur (mm)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.001,
        help="Longueur maximale de la coquille",
    )

    diameter = st.sidebar.slider(
        "Diamètre (mm)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.001,
        help="Diamètre perpendiculaire à la longueur",
    )

    height = st.sidebar.slider(
        "Hauteur (mm)",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.001,
        help="Hauteur avec la viande à l'intérieur",
    )

    whole_weight = st.sidebar.slider(
        "Poids total (g)",
        min_value=0.0,
        max_value=3.0,
        value=0.8,
        step=0.001,
        help="Poids total de l'abalone",
    )

    shucked_weight = st.sidebar.slider(
        "Poids de la viande (g)",
        min_value=0.0,
        max_value=2.0,
        value=0.3,
        step=0.001,
        help="Poids de la viande après avoir retiré la coquille",
    )

    viscera_weight = st.sidebar.slider(
        "Poids des viscères (g)",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.001,
        help="Poids des viscères après avoir retiré la coquille",
    )

    shell_weight = st.sidebar.slider(
        "Poids de la coquille (g)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.001,
        help="Poids de la coquille après séchage",
    )

    # Bouton de prédiction
    if st.sidebar.button("🔮 Prédire l'Âge", type="primary"):
        # Préparation des données pour l'API
        input_data = {
            "sex": sex,
            "length": length,
            "diameter": diameter,
            "height": height,
            "whole_weight": whole_weight,
            "shucked_weight": shucked_weight,
            "viscera_weight": viscera_weight,
            "shell_weight": shell_weight,
        }

        # Affichage des données saisies
        st.subheader("📋 Données saisies")
        df_input = pd.DataFrame([input_data])
        st.dataframe(df_input, use_container_width=True)

        # Prédiction
        with st.spinner("🔮 Calcul de la prédiction..."):
            prediction = make_prediction(input_data)

        if prediction:
            st.subheader("🎯 Résultat de la Prédiction")

            # Affichage du résultat
            predicted_age = prediction.get("predicted_age", "N/A")
            confidence = prediction.get("confidence", "N/A")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Âge Prédit (anneaux)",
                    value=f"{predicted_age:.1f}",
                    help="Nombre d'anneaux de croissance prédit",
                )

            with col2:
                st.metric(
                    label="Confiance",
                    value=f"{confidence:.2%}"
                    if isinstance(confidence, (int, float))
                    else "N/A",
                    help="Niveau de confiance de la prédiction",
                )

            # Visualisation
            st.subheader("📊 Visualisation")

            # Graphique de comparaison avec les données d'exemple
            sample_data = load_sample_data()

            fig = px.scatter(
                sample_data,
                x="Whole weight",
                y="Rings",
                color="Sex",
                title="Comparaison avec les données d'entraînement",
                labels={"Whole weight": "Poids Total (g)", "Rings": "Âge (anneaux)"},
                hover_data=["Length", "Diameter", "Height"],
            )

            # Ajouter le point de prédiction
            fig.add_trace(
                go.Scatter(
                    x=[whole_weight],
                    y=[predicted_age],
                    mode="markers",
                    marker=dict(size=15, color="red", symbol="star"),
                    name="Votre prédiction",
                    hovertemplate=f"<b>Votre abalone</b><br>Poids: {whole_weight:.3f}g<br>Âge prédit: {predicted_age:.1f} anneaux<extra></extra>",
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    # Section d'analyse des données
    st.subheader("📈 Analyse des Données")

    # Chargement des données d'exemple
    sample_data = load_sample_data()

    if not sample_data.empty:
        # Statistiques descriptives
        st.subheader("📊 Statistiques Descriptives")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Distribution par sexe:**")
            sex_counts = sample_data["Sex"].value_counts()
            st.bar_chart(sex_counts)

        with col2:
            st.write("**Distribution de l'âge:**")
            st.histogram(sample_data["Rings"], bins=20)

        # Corrélations
        st.subheader("🔗 Corrélations")
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        corr_matrix = sample_data[numeric_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matrice de corrélation des caractéristiques numériques",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Tableau des données
        st.subheader("📋 Données d'Exemple")
        st.dataframe(sample_data.head(10), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666;'>
        <p>🐚 MLOps Project - Prédiction de l'Âge des Abalones</p>
        <p>Interface Streamlit intégrée avec FastAPI</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
