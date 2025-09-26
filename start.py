#!/usr/bin/env python3
"""
🚀 GraphRAG Knowledge Graph - Lancement Rapide
Démarre automatiquement l'API et l'interface Streamlit
"""

import subprocess
import time
import os
import sys
import webbrowser
import requests
from pathlib import Path

def check_python_version():
    """Vérifier la version Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11+ requis. Version actuelle:", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} détecté")
    return True

def check_env_file():
    """Vérifier si .env existe"""
    if not Path(".env").exists():
        print("❌ Fichier .env manquant!")
        print("📝 Copiez .env.example vers .env et configurez vos paramètres")
        return False
    print("✅ Fichier .env trouvé")
    return True

def check_dependencies():
    """Vérifier les dépendances essentielles"""
    try:
        import fastapi
        import streamlit
        import neo4j
        import openai
        print("✅ Dépendances principales installées")
        return True
    except ImportError as e:
        print(f"❌ Dépendances manquantes: {e}")
        print("📦 Exécutez: pip install -r requirements.txt")
        return False

def start_api():
    """Démarrer l'API FastAPI"""
    print("🚀 Démarrage de l'API FastAPI...")
    api_dir = Path("KnowledgeGraphRagAPI")
    if not api_dir.exists():
        print("❌ Répertoire KnowledgeGraphRagAPI introuvable")
        return None
    
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"]
    process = subprocess.Popen(cmd, cwd=api_dir)
    
    # Attendre que l'API soit prête
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ API FastAPI démarrée sur http://localhost:8000")
                return process
        except:
            pass
        time.sleep(1)
        print(f"⏳ Attente API ({i+1}/{max_retries})...")
    
    print("❌ Échec du démarrage de l'API")
    return None

def start_streamlit():
    """Démarrer l'interface Streamlit"""
    print("🎨 Démarrage de l'interface Streamlit...")
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_rag_simple.py", "--server.port=8501"]
    process = subprocess.Popen(cmd)
    
    # Attendre un peu puis ouvrir le navigateur
    time.sleep(3)
    print("✅ Interface Streamlit démarrée sur http://localhost:8501")
    return process

def main():
    """Fonction principale"""
    print("🧠 GraphRAG Knowledge Graph - Lancement Rapide")
    print("=" * 50)
    
    # Vérifications préalables
    if not check_python_version():
        return
    
    if not check_env_file():
        return
    
    if not check_dependencies():
        return
    
    # Démarrage des services
    api_process = start_api()
    if not api_process:
        print("❌ Impossible de démarrer l'API")
        return
    
    streamlit_process = start_streamlit()
    if not streamlit_process:
        print("❌ Impossible de démarrer Streamlit")
        api_process.terminate()
        return
    
    print("\n" + "=" * 50)
    print("🎉 GraphRAG démarré avec succès!")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🎨 Interface Utilisateur: http://localhost:8501")
    print("=" * 50)
    print("\n💡 Conseils:")
    print("- Uploadez d'abord vos documents via l'interface")
    print("- Utilisez l'onglet 'Recherche RAG' pour poser des questions")
    print("- Explorez le graphe via l'onglet 'Graphe'")
    print("\n⏹️ Appuyez sur Ctrl+C pour arrêter")
    
    # Ouvrir le navigateur
    try:
        webbrowser.open("http://localhost:8501")
    except:
        pass
    
    # Attendre l'interruption
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt en cours...")
        streamlit_process.terminate()
        api_process.terminate()
        print("✅ Services arrêtés")

if __name__ == "__main__":
    main()