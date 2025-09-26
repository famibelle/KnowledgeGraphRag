# 🪟 GraphRAG Docker - Guide Windows

Guide spécifique pour Windows PowerShell et Docker Desktop.

## 📋 Prérequis Windows

### 1. Docker Desktop
```powershell
# Télécharger et installer Docker Desktop
# https://docs.docker.com/desktop/windows/install/

# Vérifier l'installation
docker --version
docker-compose --version
```

### 2. Configuration PowerShell
```powershell
# Activer l'exécution des scripts PowerShell (Admin requis)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Alternative pour un script spécifique
powershell -ExecutionPolicy Bypass -File .\test-docker.ps1
```

## 🚀 Démarrage Rapide

### 1. Clone et Configuration
```powershell
# Cloner le repository
git clone <your-repo-url>
cd KnowledgeGraphRag

# Copier la configuration
Copy-Item .env.docker.example .env.docker

# Éditer la configuration
notepad .env.docker
```

### 2. Lancement avec Make (si disponible)
```powershell
# Si make est installé (chocolatey, scoop, etc.)
make run

# Sinon, utiliser docker-compose directement
docker-compose up -d
```

### 3. Lancement Manuel
```powershell
# Build et démarrage
docker-compose build
docker-compose up -d

# Vérifier les logs
docker-compose logs -f

# Arrêter
docker-compose down
```

## 🧪 Test et Validation

### Script de Test PowerShell
```powershell
# Exécuter les tests
.\test-docker.ps1

# Ou avec bypass de politique
powershell -ExecutionPolicy Bypass -File .\test-docker.ps1
```

### Vérifications Manuelles
```powershell
# Status des containers
docker-compose ps

# Logs en temps réel
docker-compose logs -f

# Test des endpoints
Invoke-WebRequest http://localhost:8000/health
Invoke-WebRequest http://localhost:8501
```

## 🛠 Commandes Utiles Windows

### Gestion des Containers
```powershell
# Démarrer
docker-compose up -d

# Arrêter
docker-compose down

# Redémarrer
docker-compose restart

# Reconstruire
docker-compose build --no-cache
```

### Debug et Logs
```powershell
# Logs détaillés
docker-compose logs -f --tail=100

# Shell dans le container
docker-compose exec graphrag powershell
# ou
docker-compose exec graphrag bash

# Nettoyer les volumes
docker-compose down -v
docker system prune -f
```

### Monitoring
```powershell
# Status système
docker stats

# Espace disque
docker system df

# Images
docker images
```

## 🚨 Résolution de Problèmes Windows

### Docker Desktop Issues
```powershell
# Redémarrer Docker Desktop
Stop-Service com.docker.service
Start-Service com.docker.service

# Ou via l'interface graphique:
# Docker Desktop → Settings → Reset → Restart Docker Desktop
```

### Problèmes de Permissions
```powershell
# Exécuter PowerShell en tant qu'Admin
Start-Process powershell -Verb RunAs

# Vérifier les groupes utilisateur
net localgroup docker-users
```

### Problèmes de Port
```powershell
# Vérifier les ports utilisés
netstat -an | findstr "8000\|8501"

# Tuer un processus sur un port
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
```

### WSL2 Issues
```powershell
# Redémarrer WSL2
wsl --shutdown
wsl

# Mettre à jour WSL2
wsl --update
```

## 🔧 Configuration Avancée Windows

### Variables d'Environnement
```powershell
# Définir temporairement
$env:OPENAI_API_KEY = "sk-your-key-here"

# Ou éditer le fichier .env.docker directement
notepad .env.docker
```

### Performance
```powershell
# Allouer plus de RAM à Docker (Docker Desktop Settings)
# Settings → Resources → Advanced → Memory: 4GB+

# Utiliser WSL2 backend pour de meilleures performances
# Settings → General → Use WSL 2 based engine
```

## 📱 Accès aux Services

Une fois lancé:

- **Interface Web**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 💡 Tips Windows

1. **Firewall**: Windows Defender peut bloquer les ports, autoriser Docker Desktop
2. **Antivirus**: Exclure le dossier Docker des scans en temps réel
3. **WSL2**: Utiliser WSL2 pour de meilleures performances
4. **RAM**: Allouer au moins 4GB à Docker Desktop
5. **SSD**: Installer sur SSD pour de meilleures performances I/O