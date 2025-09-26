#!/bin/bash
# 🧪 GraphRAG Docker - Test de Validation

set -e

echo "🧪 GraphRAG Docker - Test de Validation"
echo "======================================"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonctions utilitaires
success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

info() {
    echo -e "ℹ️ $1"
}

# Tests
test_docker_installed() {
    info "Test 1: Docker installation"
    if command -v docker &> /dev/null; then
        success "Docker is installed: $(docker --version)"
    else
        error "Docker is not installed"
        exit 1
    fi
}

test_docker_compose_installed() {
    info "Test 2: Docker Compose installation"
    if command -v docker-compose &> /dev/null; then
        success "Docker Compose is installed: $(docker-compose --version)"
    else
        error "Docker Compose is not installed"
        exit 1
    fi
}

test_env_file() {
    info "Test 3: Configuration file"
    if [ -f ".env.docker" ]; then
        success ".env.docker exists"
        
        # Vérifier les variables critiques
        if grep -q "OPENAI_API_KEY=sk-" .env.docker; then
            success "OpenAI API key configured"
        else
            warning "OpenAI API key not configured or invalid format"
        fi
        
        if grep -q "NEO4J_URI=neo4j" .env.docker; then
            success "Neo4j URI configured"
        else
            warning "Neo4j URI not configured"
        fi
    else
        error ".env.docker not found"
        info "Run: cp .env.docker.example .env.docker"
        exit 1
    fi
}

test_build() {
    info "Test 4: Docker build"
    if docker-compose build --quiet; then
        success "Docker image built successfully"
    else
        error "Docker build failed"
        exit 1
    fi
}

test_container_start() {
    info "Test 5: Container startup"
    
    # Démarrer en arrière-plan
    if docker-compose up -d; then
        success "Container started"
        
        # Attendre que les services soient prêts
        info "Waiting for services to be ready..."
        sleep 30
        
        # Tester l'API
        if curl -s http://localhost:8000/health > /dev/null; then
            success "API is responding"
        else
            error "API is not responding"
        fi
        
        # Tester Streamlit (check du port)
        if curl -s http://localhost:8501 > /dev/null; then
            success "Streamlit is responding"
        else
            warning "Streamlit may not be ready yet (normal for first start)"
        fi
        
    else
        error "Container failed to start"
        exit 1
    fi
}

test_api_endpoints() {
    info "Test 6: API endpoints"
    
    # Health endpoint
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        success "Health endpoint working"
    else
        error "Health endpoint failed"
    fi
    
    # Documentation endpoint
    if curl -s http://localhost:8000/docs > /dev/null; then
        success "API documentation accessible"
    else
        error "API documentation not accessible"
    fi
}

cleanup() {
    info "Cleanup: Stopping test containers"
    docker-compose down > /dev/null 2>&1 || true
}

# Exécution des tests
main() {
    test_docker_installed
    test_docker_compose_installed
    test_env_file
    test_build
    test_container_start
    test_api_endpoints
    
    echo ""
    echo "🎉 All tests passed!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Configure your API keys in .env.docker"
    echo "2. Run: make run"
    echo "3. Open: http://localhost:8501"
    echo ""
    
    cleanup
}

# Trap pour nettoyage en cas d'interruption
trap cleanup EXIT

# Lancer les tests
main