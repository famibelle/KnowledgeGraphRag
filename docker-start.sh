#!/bin/bash
# 🚀 GraphRAG Knowledge Graph - Docker Startup Script

set -e

echo "🧠 GraphRAG Knowledge Graph - Starting PoC Demo"
echo "=============================================="

# Vérifier les variables d'environnement essentielles
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is required"
    exit 1
fi

if [ -z "$NEO4J_URI" ]; then
    echo "❌ NEO4J_URI is required"
    exit 1
fi

echo "✅ Environment variables validated"

# Fonction de nettoyage
cleanup() {
    echo "🛑 Shutting down services..."
    kill $API_PID $STREAMLIT_PID 2>/dev/null || true
    exit 0
}

# Trap pour nettoyage propre
trap cleanup SIGTERM SIGINT

# Démarrer l'API FastAPI en arrière-plan
echo "🚀 Starting FastAPI server..."
cd /app/KnowledgeGraphRagAPI
python -m uvicorn main:app --host 0.0.0.0 --port ${API_PORT:-8000} &
API_PID=$!

# Attendre que l'API soit prête
echo "⏳ Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:${API_PORT:-8000}/health > /dev/null; then
        echo "✅ API is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ API failed to start"
        exit 1
    fi
    sleep 2
done

# Démarrer Streamlit en arrière-plan
echo "🎨 Starting Streamlit interface..."
cd /app
streamlit run streamlit_rag_simple.py --server.port ${STREAMLIT_SERVER_PORT:-8501} --server.address 0.0.0.0 &
STREAMLIT_PID=$!

echo ""
echo "🎉 GraphRAG Demo Started Successfully!"
echo "📊 API Documentation: http://localhost:${API_PORT:-8000}/docs"
echo "🎨 Streamlit Interface: http://localhost:${STREAMLIT_SERVER_PORT:-8501}"
echo ""
echo "💡 Remember to configure your Neo4j connection in environment variables"
echo ""

# Garder le container actif
wait