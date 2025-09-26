@echo off
REM 🚀 GraphRAG Knowledge Graph - Setup Automatique Windows
REM Ce script configure automatiquement l'environnement de développement

echo 🧠 GraphRAG Knowledge Graph - Installation Automatique
echo ==================================================

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé. Installez Python 3.11+ d'abord.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% détecté

REM Créer l'environnement virtuel
if not exist ".venv" (
    echo 📦 Création de l'environnement virtuel...
    python -m venv .venv
) else (
    echo ✅ Environnement virtuel existant trouvé
)

REM Activer l'environnement virtuel
echo 🔧 Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat

REM Mettre à jour pip
echo ⬆️ Mise à jour de pip...
python -m pip install --upgrade pip

REM Installer les dépendances globales
echo 📚 Installation des dépendances Streamlit...
pip install -r requirements.txt

REM Installer les dépendances backend
echo ⚙️ Installation des dépendances FastAPI...
cd KnowledgeGraphRagAPI
pip install -r requirements.txt
cd ..

REM Créer .env si il n'existe pas
if not exist ".env" (
    echo 🔐 Création du fichier .env...
    copy .env.example .env
    echo ⚠️ IMPORTANT: Éditez .env avec vos clés API et paramètres Neo4j!
) else (
    echo ✅ Fichier .env existant trouvé
)

echo.
echo 🎉 Installation terminée avec succès!
echo.
echo 📋 Prochaines étapes:
echo 1. Éditez .env avec vos paramètres:
echo    - OPENAI_API_KEY=your_key
echo    - NEO4J_URI=your_uri
echo    - NEO4J_PASSWORD=your_password
echo.
echo 2. Démarrez Neo4j et initialisez la base:
echo    cd KnowledgeGraphRagAPI
echo    python -m uvicorn main:app --reload
echo    curl -X POST http://localhost:8000/initialize_db
echo.
echo 3. Lancez l'interface Streamlit:
echo    streamlit run streamlit_rag_simple.py
echo.
echo 4. Ouvrez votre navigateur:
echo    - Interface: http://localhost:8501
echo    - API Docs: http://localhost:8000/docs
echo.
echo 🚀 Prêt à explorer vos documents avec GraphRAG!
pause