#!/usr/bin/env python3
"""
Script de test complet pour le projet MLOps
Exécute tous les tests et génère un rapport
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Exécuter tous les tests du projet"""

    print("🧪 Exécution des Tests MLOps - Prédiction d'Âge des Abalones")
    print("=" * 60)

    # Vérifier que nous sommes dans le bon répertoire
    if not Path("streamlit_app.py").exists():
        print("❌ Erreur: Exécutez ce script depuis le répertoire du projet")
        sys.exit(1)

    # Vérifier que pytest est installé
    try:
        import pytest

        print(f"✅ pytest version: {pytest.__version__}")
    except ImportError:
        print("❌ pytest n'est pas installé. Installation...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])

    # Créer le dossier tests s'il n'existe pas
    Path("tests").mkdir(exist_ok=True)

    # Liste des fichiers de test
    test_files = ["test_streamlit.py", "tests/test_api.py"]

    print("\n📋 Tests à exécuter:")
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"   ✅ {test_file}")
        else:
            print(f"   ❌ {test_file} (manquant)")

    print("\n🚀 Exécution des tests...")

    # Exécuter les tests Streamlit
    print("\n1️⃣ Tests Streamlit:")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_streamlit.py", "-v", "--tb=short"],
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("Erreurs:", result.stderr)

        if result.returncode == 0:
            print("✅ Tests Streamlit: PASSÉS")
        else:
            print("❌ Tests Streamlit: ÉCHOUÉS")

    except Exception as e:
        print(f"❌ Erreur lors des tests Streamlit: {e}")

    # Exécuter les tests API
    print("\n2️⃣ Tests API:")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_api.py", "-v", "--tb=short"],
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("Erreurs:", result.stderr)

        if result.returncode == 0:
            print("✅ Tests API: PASSÉS")
        else:
            print("❌ Tests API: ÉCHOUÉS")

    except Exception as e:
        print(f"❌ Erreur lors des tests API: {e}")

    # Exécuter tous les tests ensemble
    print("\n3️⃣ Tests Complets:")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "test_streamlit.py",
                "tests/test_api.py",
                "-v",
                "--tb=short",
                "--tb=line",
            ],
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("Erreurs:", result.stderr)

        if result.returncode == 0:
            print("🎉 TOUS LES TESTS: PASSÉS")
        else:
            print("⚠️  CERTAINS TESTS: ÉCHOUÉS")

    except Exception as e:
        print(f"❌ Erreur lors des tests complets: {e}")

    # Générer un rapport de couverture
    print("\n4️⃣ Rapport de Couverture:")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "test_streamlit.py",
                "tests/test_api.py",
                "--cov=src",
                "--cov-report=term-missing",
            ],
            capture_output=True,
            text=True,
        )

        print(result.stdout)

    except Exception as e:
        print(f"⚠️  Rapport de couverture non disponible: {e}")

    print("\n" + "=" * 60)
    print("📊 Résumé des Tests:")
    print("   - Tests Streamlit: Interface utilisateur")
    print("   - Tests API: Modèles et preprocessing")
    print("   - Tests d'intégration: Pipeline complet")
    print("\n💡 Prochaines étapes:")
    print("   1. Corriger les tests échoués")
    print("   2. Ajouter plus de tests si nécessaire")
    print("   3. Intégrer avec l'API FastAPI")
    print("   4. Tester le déploiement Docker")


if __name__ == "__main__":
    run_tests()
