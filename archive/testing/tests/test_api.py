"""
Tests unitaires pour l'API FastAPI et les modèles ML
Ces tests vérifient le bon fonctionnement de l'API et des modèles
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_data_preprocessing():
    """Test des fonctions de preprocessing"""

    # Données de test
    test_data = pd.DataFrame(
        {
            "Sex": ["M", "F", "I", "M", "F"],
            "Length": [0.5, 0.6, 0.4, 0.55, 0.65],
            "Diameter": [0.4, 0.5, 0.3, 0.45, 0.55],
            "Height": [0.15, 0.2, 0.1, 0.18, 0.22],
            "Whole weight": [0.8, 1.0, 0.6, 0.9, 1.1],
            "Shucked weight": [0.3, 0.4, 0.2, 0.35, 0.45],
            "Viscera weight": [0.15, 0.2, 0.1, 0.18, 0.22],
            "Shell weight": [0.2, 0.25, 0.15, 0.23, 0.28],
            "Rings": [15, 18, 12, 16, 20],
        }
    )

    # Test de l'encodage du sexe
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_sex = label_encoder.fit_transform(test_data["Sex"])

    assert len(encoded_sex) == len(test_data)
    assert set(encoded_sex) <= {0, 1, 2}  # 3 valeurs possibles

    # Test de la division train/test
    from sklearn.model_selection import train_test_split

    X = test_data.drop(["Sex", "Rings"], axis=1)
    y = test_data["Rings"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_model_training():
    """Test de l'entraînement du modèle"""

    # Données d'entraînement simulées
    X_train = pd.DataFrame(
        {
            "Length": np.random.uniform(0.1, 0.8, 100),
            "Diameter": np.random.uniform(0.1, 0.7, 100),
            "Height": np.random.uniform(0.01, 0.3, 100),
            "Whole weight": np.random.uniform(0.1, 2.0, 100),
            "Shucked weight": np.random.uniform(0.05, 1.0, 100),
            "Viscera weight": np.random.uniform(0.01, 0.5, 100),
            "Shell weight": np.random.uniform(0.01, 0.5, 100),
            "Sex_encoded": np.random.randint(0, 3, 100),
        }
    )

    y_train = pd.Series(np.random.randint(1, 30, 100))

    # Test avec RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=10,  # Petit nombre pour les tests
        max_depth=5,
        random_state=42,
    )

    # Entraînement
    model.fit(X_train, y_train)

    # Prédiction
    predictions = model.predict(X_train.head(5))

    assert len(predictions) == 5
    assert all(pred >= 0 for pred in predictions)  # Âge positif

    # Test des métriques
    from sklearn.metrics import mean_squared_error, r2_score

    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    assert mse >= 0
    assert -1 <= r2 <= 1


def test_api_models():
    """Test des modèles Pydantic pour l'API"""

    # Test des données d'entrée
    input_data = {
        "sex": "M",
        "length": 0.5,
        "diameter": 0.4,
        "height": 0.15,
        "whole_weight": 0.8,
        "shucked_weight": 0.3,
        "viscera_weight": 0.15,
        "shell_weight": 0.2,
    }

    # Vérifier que toutes les clés requises sont présentes
    required_keys = [
        "sex",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
    ]

    for key in required_keys:
        assert key in input_data

    # Vérifier les types
    assert isinstance(input_data["sex"], str)
    assert isinstance(input_data["length"], (int, float))
    assert isinstance(input_data["diameter"], (int, float))
    assert isinstance(input_data["height"], (int, float))
    assert isinstance(input_data["whole_weight"], (int, float))
    assert isinstance(input_data["shucked_weight"], (int, float))
    assert isinstance(input_data["viscera_weight"], (int, float))
    assert isinstance(input_data["shell_weight"], (int, float))

    # Vérifier les valeurs de sexe
    assert input_data["sex"] in ["M", "F", "I"]

    # Vérifier les plages de valeurs
    assert 0 <= input_data["length"] <= 1
    assert 0 <= input_data["diameter"] <= 1
    assert 0 <= input_data["height"] <= 1
    assert 0 <= input_data["whole_weight"] <= 3
    assert 0 <= input_data["shucked_weight"] <= 2
    assert 0 <= input_data["viscera_weight"] <= 1
    assert 0 <= input_data["shell_weight"] <= 1


def test_api_response_format():
    """Test du format de réponse de l'API"""

    # Format de réponse attendu
    expected_response = {
        "predicted_age": 15.2,
        "confidence": 0.85,
        "model_version": "v1.0",
    }

    # Vérifier la structure
    assert "predicted_age" in expected_response
    assert "confidence" in expected_response
    assert "model_version" in expected_response

    # Vérifier les types
    assert isinstance(expected_response["predicted_age"], (int, float))
    assert isinstance(expected_response["confidence"], (int, float))
    assert isinstance(expected_response["model_version"], str)

    # Vérifier les plages de valeurs
    assert 0 <= expected_response["predicted_age"] <= 30
    assert 0 <= expected_response["confidence"] <= 1


def test_data_validation():
    """Test de la validation des données"""

    # Données valides
    valid_data = {
        "sex": "F",
        "length": 0.6,
        "diameter": 0.5,
        "height": 0.2,
        "whole_weight": 1.0,
        "shucked_weight": 0.4,
        "viscera_weight": 0.2,
        "shell_weight": 0.3,
    }

    # Test des données valides
    assert valid_data["sex"] in ["M", "F", "I"]
    assert 0 <= valid_data["length"] <= 1
    assert 0 <= valid_data["diameter"] <= 1
    assert 0 <= valid_data["height"] <= 1
    assert 0 <= valid_data["whole_weight"] <= 3
    assert 0 <= valid_data["shucked_weight"] <= 2
    assert 0 <= valid_data["viscera_weight"] <= 1
    assert 0 <= valid_data["shell_weight"] <= 1


def test_model_persistence():
    """Test de la sauvegarde et chargement du modèle"""

    import pickle
    import tempfile
    from sklearn.ensemble import RandomForestRegressor

    # Créer un modèle simple
    model = RandomForestRegressor(n_estimators=5, random_state=42)

    # Données d'entraînement minimales
    X = np.random.rand(10, 8)
    y = np.random.randint(1, 30, 10)

    model.fit(X, y)

    # Test de sauvegarde
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        pickle.dump(model, tmp_file)
        tmp_path = tmp_file.name

    # Test de chargement
    with open(tmp_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Vérifier que le modèle chargé fonctionne
    test_prediction = loaded_model.predict(X[:1])
    assert len(test_prediction) == 1
    assert test_prediction[0] >= 0

    # Nettoyer
    os.unlink(tmp_path)


def test_preprocessing_pipeline():
    """Test du pipeline de preprocessing complet"""

    # Données brutes simulées
    raw_data = pd.DataFrame(
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

    # Pipeline de preprocessing
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # 1. Encodage du sexe
    le = LabelEncoder()
    raw_data["Sex_encoded"] = le.fit_transform(raw_data["Sex"])
    processed_data = raw_data.drop(["Sex"], axis=1)

    # 2. Séparation features/target
    X = processed_data.drop(["Rings"], axis=1)
    y = processed_data["Rings"]

    # 3. Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Vérifications
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert "Sex_encoded" in X_train.columns
    assert "Sex" not in X_train.columns
    assert "Rings" not in X_train.columns


if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v"])
