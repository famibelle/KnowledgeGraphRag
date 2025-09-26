# 🧠 GraphRAG Knowledge Graph - Docker Image
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="GraphRAG Team"
LABEL description="GraphRAG Knowledge Graph - PoC Demo Application"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV API_PORT=8000

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements en premier (cache Docker)
COPY requirements.txt .
COPY KnowledgeGraphRagAPI/requirements.txt ./requirements-api.txt

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-api.txt

# Créer un utilisateur non-root pour la sécurité
RUN groupadd -r graphrag && useradd -r -g graphrag graphrag

# Copier le code de l'application
COPY KnowledgeGraphRagAPI/ ./KnowledgeGraphRagAPI/
COPY streamlit_rag_simple.py .
COPY streamlit_kg_interface.py .
COPY docker-start.sh .

# Script de démarrage (faire avant changement d'utilisateur)
RUN chmod +x docker-start.sh && \
    chown -R graphrag:graphrag /app

# Changer d'utilisateur
USER graphrag

# Exposer les ports
EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Point d'entrée
CMD ["./docker-start.sh"]