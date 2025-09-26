#!/bin/bash

# 🚀 GraphRAG Knowledge Graph - Setup Automatique
# Ce script configure automatiquement l'environnement de développement

set -e  # Arrêter en cas d'erreur

echo "🧠 GraphRAG Knowledge Graph - Installation Automatique"
echo "=================================================="

# Vérifier Python
if ! command -v python &> /dev/null; then
    echo "❌ Python n'est pas installé. Installez Python 3.11+ d'abord."
    exit 1
fi

PYTHON_VERSION=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "✅ Python $PYTHON_VERSION détecté"

# Créer l'environnement virtuel
if [ ! -d ".venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python -m venv .venv
else
    echo "✅ Environnement virtuel existant trouvé"
fi

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source .venv/bin/activate || source .venv/Scripts/activate

# Mettre à jour pip
echo "⬆️ Mise à jour de pip..."
python -m pip install --upgrade pip

# Installer les dépendances globales
echo "📚 Installation des dépendances Streamlit..."
pip install -r requirements.txt

# Installer les dépendances backend
echo "⚙️ Installation des dépendances FastAPI..."
cd KnowledgeGraphRagAPI
pip install -r requirements.txt
cd ..

# Créer .env si il n'existe pas
if [ ! -f ".env" ]; then
    echo "🔐 Création du fichier .env..."
    cp .env.example .env
    echo "⚠️ IMPORTANT: Éditez .env avec vos clés API et paramètres Neo4j!"
else
    echo "✅ Fichier .env existant trouvé"
fi

echo ""
echo "🎉 Installation terminée avec succès!"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Éditez .env avec vos paramètres:"
echo "   - OPENAI_API_KEY=your_key"
echo "   - NEO4J_URI=your_uri"
echo "   - NEO4J_PASSWORD=your_password"
echo ""
echo "2. Démarrez Neo4j et initialisez la base:"
echo "   cd KnowledgeGraphRagAPI"
echo "   python -m uvicorn main:app --reload"
echo "   curl -X POST http://localhost:8000/initialize_db"
echo ""
echo "3. Lancez l'interface Streamlit:"
echo "   streamlit run streamlit_rag_simple.py"
echo ""
echo "4. Ouvrez votre navigateur:"
echo "   - Interface: http://localhost:8501"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "🚀 Prêt à explorer vos documents avec GraphRAG!"