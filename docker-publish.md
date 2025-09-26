# 🚀 Guide de Publication Docker - GraphRAG

## 🐳 Publication sur Docker Hub

### Étape 1: Créer un compte Docker Hub
1. Aller sur https://hub.docker.com/
2. Créer un compte gratuit
3. Noter votre `username`

### Étape 2: Se connecter localement
```powershell
# Se connecter à Docker Hub
docker login

# Entrer username et password
```

### Étape 3: Tag et Push de l'image
```powershell
# Builder l'image avec le bon tag
docker build -t famibelle/graphrag-knowledge-graph:latest .
docker build -t famibelle/graphrag-knowledge-graph:v1.0.0 .

# Push vers Docker Hub
docker push famibelle/graphrag-knowledge-graph:latest
docker push famibelle/graphrag-knowledge-graph:v1.0.0
```

### Étape 4: Utilisation par les autres
```powershell
# Les utilisateurs peuvent maintenant faire:
docker pull famibelle/graphrag-knowledge-graph:latest
docker run -d -p 8000:8000 -p 8501:8501 --env-file .env.docker famibelle/graphrag-knowledge-graph:latest
```

## 🏢 GitHub Container Registry (Recommandé)

### Avantages:
- ✅ Intégré à GitHub
- ✅ Gratuit pour les repos publics
- ✅ Meilleure sécurité
- ✅ CI/CD intégré

### Configuration GitHub Container Registry:

```powershell
# 1. Créer un Personal Access Token
# GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
# Permissions: read:packages, write:packages

# 2. Se connecter au registry
echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u famibelle --password-stdin

# 3. Tag avec le registry GitHub
docker build -t ghcr.io/famibelle/graphrag-knowledge-graph:latest .
docker build -t ghcr.io/famibelle/graphrag-knowledge-graph:v1.0.0 .

# 4. Push vers GitHub
docker push ghcr.io/famibelle/graphrag-knowledge-graph:latest
docker push ghcr.io/famibelle/graphrag-knowledge-graph:v1.0.0
```

## 🤖 Automatisation avec GitHub Actions

### Créer .github/workflows/docker-publish.yml:

```yaml
name: 🐳 Build and Publish Docker Image

on:
  push:
    branches: [ master ]
    tags: [ 'v*.*.*' ]
  pull_request:
    branches: [ master ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: 📥 Checkout repository
      uses: actions/checkout@v4

    - name: 🔐 Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: 🏷️ Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: 🔨 Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
```

## 📋 Mise à jour du README pour Docker

### Ajouter une section "Docker Deployment":

```markdown
## 🐳 Déploiement Docker

### Option 1: GitHub Container Registry (Recommandé)
```bash
# Pull de l'image officielle
docker pull ghcr.io/famibelle/graphrag-knowledge-graph:latest

# Configuration
cp .env.docker.example .env.docker
# Éditer .env.docker avec vos clés

# Démarrage
docker run -d \
  --name graphrag-demo \
  -p 8000:8000 \
  -p 8501:8501 \
  --env-file .env.docker \
  ghcr.io/famibelle/graphrag-knowledge-graph:latest
```

### Option 2: Docker Hub
```bash
docker pull famibelle/graphrag-knowledge-graph:latest
```

### Option 3: Build Local
```bash
git clone https://github.com/famibelle/KnowledgeGraphRag.git
cd KnowledgeGraphRag
make run
```
```

## 🔄 Script de Publication Automatique

### Créer publish-docker.ps1:

```powershell
#!/usr/bin/env pwsh
# 🚀 Script de publication Docker automatique

param(
    [string]$Version = "latest",
    [string]$Registry = "both"  # "hub", "github", ou "both"
)

Write-Host "🐳 GraphRAG Docker Publication" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

# Variables
$ImageName = "graphrag-knowledge-graph"
$DockerHubRepo = "famibelle/$ImageName"
$GitHubRepo = "ghcr.io/famibelle/$ImageName"

# Build de l'image
Write-Host "🔨 Building image..." -ForegroundColor Yellow
docker build -t $ImageName:$Version .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

# Publication sur Docker Hub
if ($Registry -eq "hub" -or $Registry -eq "both") {
    Write-Host "📤 Publishing to Docker Hub..." -ForegroundColor Green
    
    docker tag $ImageName:$Version $DockerHubRepo:$Version
    docker tag $ImageName:$Version $DockerHubRepo:latest
    
    docker push $DockerHubRepo:$Version
    docker push $DockerHubRepo:latest
    
    Write-Host "✅ Published to Docker Hub: $DockerHubRepo:$Version" -ForegroundColor Green
}

# Publication sur GitHub Container Registry
if ($Registry -eq "github" -or $Registry -eq "both") {
    Write-Host "📤 Publishing to GitHub Container Registry..." -ForegroundColor Green
    
    docker tag $ImageName:$Version $GitHubRepo:$Version
    docker tag $ImageName:$Version $GitHubRepo:latest
    
    docker push $GitHubRepo:$Version
    docker push $GitHubRepo:latest
    
    Write-Host "✅ Published to GitHub: $GitHubRepo:$Version" -ForegroundColor Green
}

Write-Host ""
Write-Host "🎉 Publication completed!" -ForegroundColor Green
Write-Host "📋 Images published:" -ForegroundColor White

if ($Registry -eq "hub" -or $Registry -eq "both") {
    Write-Host "   🐳 Docker Hub: docker pull $DockerHubRepo:$Version" -ForegroundColor Cyan
}

if ($Registry -eq "github" -or $Registry -eq "both") {
    Write-Host "   📦 GitHub: docker pull $GitHubRepo:$Version" -ForegroundColor Cyan
}
```

## 🎯 Recommandations

### Pour votre PoC GraphRAG:

1. **GitHub Container Registry** (Recommandé)
   - Intégré à votre repo GitHub
   - Gratuit et sécurisé
   - CI/CD automatique

2. **Docker Hub** (Alternative)
   - Plus visible publiquement
   - Plus facile à découvrir

3. **Les deux** (Optimal)
   - Redondance
   - Maximum de visibilité