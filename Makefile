# 🧠 GraphRAG Knowledge Graph - Makefile (PoC Demo)

.PHONY: help build run stop clean logs shell

# Variables
IMAGE_NAME := graphrag-knowledge-graph
CONTAINER_NAME := graphrag-demo

help: ## Afficher l'aide
	@echo "🧠 GraphRAG Knowledge Graph - PoC Demo Commands"
	@echo "==============================================="
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ 🚀 Commandes principales

build: ## Construire l'image Docker
	@echo "🔨 Building GraphRAG Docker image..."
	docker-compose build --no-cache

run: ## Démarrer l'application (build + run)
	@echo "🚀 Starting GraphRAG Demo..."
	@echo "⚠️  Assurez-vous d'avoir configuré .env.docker avec vos API keys!"
	docker-compose up --build -d
	@echo ""
	@echo "🎉 GraphRAG Demo started!"
	@echo "🎨 Interface: http://localhost:8501"
	@echo "📊 API Docs: http://localhost:8000/docs"

stop: ## Arrêter l'application
	@echo "🛑 Stopping GraphRAG Demo..."
	docker-compose down

restart: stop run ## Redémarrer l'application

##@ 📊 Monitoring et Debug

logs: ## Voir les logs en temps réel
	docker-compose logs -f

status: ## Vérifier le statut
	docker-compose ps

health: ## Vérifier la santé de l'application
	@echo "🔍 Checking GraphRAG health..."
	@curl -s http://localhost:8000/health | jq . || echo "❌ API not responding"

shell: ## Accéder au shell du container
	docker exec -it $(CONTAINER_NAME) /bin/bash

##@ 🧹 Nettoyage

clean: ## Nettoyer les containers et images
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f
	docker image rm $(IMAGE_NAME):latest || true

clean-all: ## Nettoyage complet (ATTENTION: supprime tout Docker)
	@echo "⚠️  This will remove ALL Docker containers and images!"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ]
	docker system prune -a -f --volumes

##@ 🔧 Utilitaires

env-example: ## Créer un exemple de .env.docker
	@cp .env.docker .env.docker.example
	@echo "📝 Created .env.docker.example"

test-connection: ## Tester la connexion Neo4j et OpenAI
	@echo "🧪 Testing connections..."
	@docker exec $(CONTAINER_NAME) python -c "
import os, requests, openai, neo4j
print('✅ OpenAI Key:', 'CONFIGURED' if os.getenv('OPENAI_API_KEY') else '❌ MISSING')
print('✅ Neo4j URI:', os.getenv('NEO4J_URI', '❌ MISSING'))
try:
    r = requests.get('http://localhost:8000/health', timeout=5)
    print('✅ API Health:', r.json())
except:
    print('❌ API not responding')
"

##@ � Publication Docker

publish-hub: ## Publier sur Docker Hub uniquement
	@echo "📤 Publishing to Docker Hub..."
	@./publish-docker.ps1 -Registry hub

publish-github: ## Publier sur GitHub Container Registry uniquement
	@echo "📦 Publishing to GitHub Container Registry..."
	@./publish-docker.ps1 -Registry github

publish: ## Publier sur Docker Hub ET GitHub Container Registry
	@echo "🚀 Publishing to both registries..."
	@./publish-docker.ps1 -Registry both

tag-release: ## Créer un tag de release (ex: make tag-release VERSION=v1.0.0)
	@echo "🏷️ Creating release tag $(VERSION)..."
	@git tag $(VERSION)
	@git push origin $(VERSION)
	@echo "✅ Release tag $(VERSION) created and pushed"

##@ �📋 Instructions

setup: ## Instructions de configuration initiale
	@echo "📋 GraphRAG Setup Instructions"
	@echo "=============================="
	@echo ""
	@echo "1. 🔐 Configure .env.docker:"
	@echo "   - OPENAI_API_KEY=your_key"
	@echo "   - NEO4J_URI=your_neo4j_uri"
	@echo "   - NEO4J_PASSWORD=your_password"
	@echo ""
	@echo "2. 🚀 Start the application:"
	@echo "   make run"
	@echo ""
	@echo "3. 🌐 Open in browser:"
	@echo "   http://localhost:8501"
	@echo ""
	@echo "4. 📚 Upload documents and start querying!"

quick-start: ## Démarrage rapide avec image publiée
	@echo "🚀 GraphRAG Quick Start with Published Image"
	@echo "==========================================="
	@echo ""
	@echo "1. 📥 Download config template:"
	@echo "   curl -o .env.docker https://raw.githubusercontent.com/famibelle/KnowledgeGraphRag/master/.env.docker"
	@echo ""
	@echo "2. 🔐 Edit .env.docker with your API keys"
	@echo ""
	@echo "3. 🚀 Run with published image:"
	@echo "   docker run -d -p 8000:8000 -p 8501:8501 --env-file .env.docker famibelle/graphrag-knowledge-graph:latest"
	@echo ""
	@echo "4. 🌐 Open: http://localhost:8501"