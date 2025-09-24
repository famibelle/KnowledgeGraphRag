#!/usr/bin/env python3
"""
Script pour démarrer l'API Knowledge Graph RAG
"""

import subprocess
import sys
import os
from pathlib import Path

def start_api():
    """Démarre l'API FastAPI avec uvicorn"""
    
    # Aller dans le dossier de l'API
    api_dir = Path(__file__).parent / "KnowledgeGraphRagAPI"
    
    if not api_dir.exists():
        print(f"❌ Dossier API non trouvé: {api_dir}")
        return
    
    print(f"🚀 Démarrage de l'API depuis: {api_dir}")
    print("📍 URL: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    print("-" * 50)
    
    try:
        # Changer vers le dossier de l'API
        os.chdir(api_dir)
        
        # Démarrer uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 API arrêtée par l'utilisateur")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du démarrage: {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")

if __name__ == "__main__":
    start_api()