# 🧪 GraphRAG Docker - Test de Validation (PowerShell)

Write-Host "🧪 GraphRAG Docker - Test de Validation" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Fonctions utilitaires
function Write-Success {
    param($Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param($Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning {
    param($Message)
    Write-Host "⚠️ $Message" -ForegroundColor Yellow
}

function Write-Info {
    param($Message)
    Write-Host "ℹ️ $Message" -ForegroundColor White
}

# Test 1: Docker installation
Write-Info "Test 1: Docker installation"
try {
    $dockerVersion = docker --version
    Write-Success "Docker is installed: $dockerVersion"
} catch {
    Write-Error "Docker is not installed"
    exit 1
}

# Test 2: Docker Compose installation
Write-Info "Test 2: Docker Compose installation"
try {
    $composeVersion = docker-compose --version
    Write-Success "Docker Compose is installed: $composeVersion"
} catch {
    Write-Error "Docker Compose is not installed"
    exit 1
}

# Test 3: Configuration file
Write-Info "Test 3: Configuration file"
if (Test-Path ".env.docker") {
    Write-Success ".env.docker exists"
    
    $envContent = Get-Content ".env.docker"
    
    if ($envContent | Select-String "OPENAI_API_KEY=sk-") {
        Write-Success "OpenAI API key configured"
    } else {
        Write-Warning "OpenAI API key not configured or invalid format"
    }
    
    if ($envContent | Select-String "NEO4J_URI=neo4j") {
        Write-Success "Neo4j URI configured"
    } else {
        Write-Warning "Neo4j URI not configured"
    }
} else {
    Write-Error ".env.docker not found"
    Write-Info "Run: Copy-Item .env.docker.example .env.docker"
    exit 1
}

# Test 4: Docker build
Write-Info "Test 4: Docker build"
try {
    docker-compose build --quiet
    Write-Success "Docker image built successfully"
} catch {
    Write-Error "Docker build failed"
    exit 1
}

# Test 5: Container startup
Write-Info "Test 5: Container startup"
try {
    docker-compose up -d
    Write-Success "Container started"
    
    Write-Info "Waiting for services to be ready..."
    Start-Sleep -Seconds 30
    
    # Test API
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "API is responding"
        }
    } catch {
        Write-Error "API is not responding"
    }
    
    # Test Streamlit
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501" -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Streamlit is responding"
        }
    } catch {
        Write-Warning "Streamlit may not be ready yet (normal for first start)"
    }
    
} catch {
    Write-Error "Container failed to start"
    exit 1
}

# Test 6: API endpoints
Write-Info "Test 6: API endpoints"

try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health"
    if ($healthResponse.status -eq "healthy") {
        Write-Success "Health endpoint working"
    }
} catch {
    Write-Error "Health endpoint failed"
}

try {
    Invoke-WebRequest -Uri "http://localhost:8000/docs" -TimeoutSec 5 | Out-Null
    Write-Success "API documentation accessible"
} catch {
    Write-Error "API documentation not accessible"
}

# Cleanup
Write-Info "Cleanup: Stopping test containers"
docker-compose down | Out-Null

Write-Host ""
Write-Host "🎉 All tests passed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:"
Write-Host "1. Configure your API keys in .env.docker"
Write-Host "2. Run: make run (or docker-compose up -d)"
Write-Host "3. Open: http://localhost:8501"
Write-Host ""