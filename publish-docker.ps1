#!/usr/bin/env pwsh
# 🚀 GraphRAG Docker Publication Script

param(
    [string]$Version = "latest",
    [string]$Registry = "both",  # "hub", "github", ou "both"
    [switch]$SkipBuild
)

Write-Host "🐳 GraphRAG Docker Publication" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

# Variables
$ImageName = "graphrag-knowledge-graph"
$DockerHubRepo = "famibelle/$ImageName"
$GitHubRepo = "ghcr.io/famibelle/$ImageName"

# Vérifications préalables
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Build de l'image (sauf si skip)
if (-not $SkipBuild) {
    Write-Host "🔨 Building image version $Version..." -ForegroundColor Yellow
    docker build -t $ImageName`:$Version .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Build successful" -ForegroundColor Green
} else {
    Write-Host "⏭️ Skipping build (using existing image)" -ForegroundColor Yellow
}

# Publication sur Docker Hub
if ($Registry -eq "hub" -or $Registry -eq "both") {
    Write-Host ""
    Write-Host "📤 Publishing to Docker Hub..." -ForegroundColor Green
    
    # Vérifier la connexion Docker Hub
    $loginCheck = docker info 2>&1 | Select-String "Username"
    if (-not $loginCheck) {
        Write-Host "⚠️ Not logged in to Docker Hub. Please run: docker login" -ForegroundColor Yellow
        $continue = Read-Host "Continue anyway? (y/N)"
        if ($continue -ne "y") {
            Write-Host "❌ Aborted" -ForegroundColor Red
            exit 1
        }
    }
    
    try {
        docker tag $ImageName`:$Version $DockerHubRepo`:$Version
        docker tag $ImageName`:$Version $DockerHubRepo`:latest
        
        Write-Host "📤 Pushing $DockerHubRepo`:$Version..." -ForegroundColor Cyan
        docker push $DockerHubRepo`:$Version
        
        Write-Host "📤 Pushing $DockerHubRepo`:latest..." -ForegroundColor Cyan  
        docker push $DockerHubRepo`:latest
        
        Write-Host "✅ Successfully published to Docker Hub" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to publish to Docker Hub: $_" -ForegroundColor Red
    }
}

# Publication sur GitHub Container Registry
if ($Registry -eq "github" -or $Registry -eq "both") {
    Write-Host ""
    Write-Host "📦 Publishing to GitHub Container Registry..." -ForegroundColor Green
    
    # Vérifier la connexion GitHub
    $ghcrLogin = docker info 2>&1 | Select-String "ghcr.io"
    if (-not $ghcrLogin) {
        Write-Host "⚠️ Not logged in to GitHub Container Registry." -ForegroundColor Yellow
        Write-Host "   Please run: echo 'YOUR_GITHUB_TOKEN' | docker login ghcr.io -u famibelle --password-stdin" -ForegroundColor Yellow
        $continue = Read-Host "Continue anyway? (y/N)"
        if ($continue -ne "y") {
            Write-Host "❌ Aborted" -ForegroundColor Red
            exit 1
        }
    }
    
    try {
        docker tag $ImageName`:$Version $GitHubRepo`:$Version
        docker tag $ImageName`:$Version $GitHubRepo`:latest
        
        Write-Host "📦 Pushing $GitHubRepo`:$Version..." -ForegroundColor Cyan
        docker push $GitHubRepo`:$Version
        
        Write-Host "📦 Pushing $GitHubRepo`:latest..." -ForegroundColor Cyan
        docker push $GitHubRepo`:latest
        
        Write-Host "✅ Successfully published to GitHub Container Registry" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to publish to GitHub Container Registry: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🎉 Publication completed!" -ForegroundColor Green
Write-Host "📋 Images published:" -ForegroundColor White

if ($Registry -eq "hub" -or $Registry -eq "both") {
    Write-Host "   🐳 Docker Hub:" -ForegroundColor Cyan
    Write-Host "      docker pull $DockerHubRepo`:$Version" -ForegroundColor White
    Write-Host "      docker pull $DockerHubRepo`:latest" -ForegroundColor White
}

if ($Registry -eq "github" -or $Registry -eq "both") {
    Write-Host "   📦 GitHub Container Registry:" -ForegroundColor Cyan
    Write-Host "      docker pull $GitHubRepo`:$Version" -ForegroundColor White  
    Write-Host "      docker pull $GitHubRepo`:latest" -ForegroundColor White
}

Write-Host ""
Write-Host "💡 Usage examples:" -ForegroundColor Yellow
Write-Host "   # Quick start with published image:" -ForegroundColor Gray
Write-Host "   docker run -d -p 8000:8000 -p 8501:8501 --env-file .env.docker $DockerHubRepo`:latest" -ForegroundColor Gray
Write-Host ""
Write-Host "   # Or with docker-compose:" -ForegroundColor Gray
Write-Host "   # Update docker-compose.yml to use: $DockerHubRepo`:latest" -ForegroundColor Gray