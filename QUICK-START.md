# 🚀 Guide de Déploiement Rapide - GraphRAG Docker

## 📦 Images Docker Publiées

Votre système GraphRAG est disponible sous forme d'images Docker pré-construites :

| Registry | Image | Taille | Architectures |
|----------|--------|---------|---------------|
| 🐳 **Docker Hub** | `famibelle/graphrag-knowledge-graph` | ~800MB | amd64, arm64 |
| 📦 **GitHub Container Registry** | `ghcr.io/famibelle/knowledgegraphrag` | ~800MB | amd64, arm64 |

## ⚡ Démarrage Ultra-Rapide (2 minutes)

### Étape 1: Configuration
```bash
# Télécharger le template de configuration
curl -o .env.docker https://raw.githubusercontent.com/famibelle/KnowledgeGraphRag/master/.env.docker

# Éditer avec vos clés (remplacer les valeurs)
# OPENAI_API_KEY=your_openai_key_here
# NEO4J_URI=your_neo4j_uri
# NEO4J_PASSWORD=your_password
```

### Étape 2: Lancement
```bash
# Option A: Docker Run (simple)
docker run -d \
  --name graphrag-demo \
  -p 8000:8000 \
  -p 8501:8501 \
  --env-file .env.docker \
  famibelle/graphrag-knowledge-graph:latest

# Option B: Docker Compose (recommandé)
curl -o docker-compose.production.yml https://raw.githubusercontent.com/famibelle/KnowledgeGraphRag/master/docker-compose.production.yml
docker-compose -f docker-compose.production.yml up -d
```

### Étape 3: Accès
- 🎨 **Interface Web**: http://localhost:8501
- 📊 **API Documentation**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health

## 🔧 Commandes Utiles

### Gestion des Containers
```bash
# Voir les logs
docker logs graphrag-demo -f

# Accéder au shell
docker exec -it graphrag-demo /bin/bash

# Redémarrer
docker restart graphrag-demo

# Arrêter
docker stop graphrag-demo

# Supprimer
docker rm graphrag-demo
```

### Mise à Jour
```bash
# Récupérer la dernière version
docker pull famibelle/graphrag-knowledge-graph:latest

# Redémarrer avec la nouvelle image
docker stop graphrag-demo
docker rm graphrag-demo
docker run -d --name graphrag-demo -p 8000:8000 -p 8501:8501 --env-file .env.docker famibelle/graphrag-knowledge-graph:latest
```

## 🌍 Configurations d'Exemple

### Configuration Neo4j Aura (Cloud)
```env
# .env.docker
OPENAI_API_KEY=sk-your-openai-key
NEO4J_URI=neo4j+s://your-id.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j
```

### Configuration Neo4j Local
```env
# .env.docker
OPENAI_API_KEY=sk-your-openai-key
NEO4J_URI=bolt://host.docker.internal:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-local-password
NEO4J_DATABASE=neo4j
```

## 🔒 Sécurité et Production

### Variables d'Environnement Sensibles
```bash
# Utiliser Docker secrets pour la production
echo "sk-your-openai-key" | docker secret create openai_key -
echo "your-neo4j-password" | docker secret create neo4j_password -

# Référencer dans docker-compose.yml
services:
  graphrag:
    secrets:
      - openai_key
      - neo4j_password
```

### Reverse Proxy (Nginx)
```nginx
# /etc/nginx/sites-available/graphrag
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoring

### Health Checks
```bash
# Vérifier la santé de l'API
curl http://localhost:8000/health

# Réponse attendue:
# {"status": "healthy", "timestamp": "2025-09-26T..."}
```

### Métriques Docker
```bash
# Statistiques en temps réel
docker stats graphrag-demo

# Utilisation des ressources
docker exec graphrag-demo df -h
docker exec graphrag-demo free -m
```

## 🚀 Cas d'Usage

### Demo/PoC
```bash
# Configuration minimale pour demo
docker run -d -p 8501:8501 --env-file .env.docker famibelle/graphrag-knowledge-graph:latest
```

### Développement
```bash
# Avec volumes pour les logs
docker run -d \
  -p 8000:8000 -p 8501:8501 \
  --env-file .env.docker \
  -v ./logs:/app/logs \
  famibelle/graphrag-knowledge-graph:latest
```

### Production
```bash
# Avec restart policy et limites
docker run -d \
  --name graphrag-prod \
  -p 8000:8000 -p 8501:8501 \
  --env-file .env.docker \
  --restart unless-stopped \
  --memory 2g \
  --cpus 1.0 \
  famibelle/graphrag-knowledge-graph:latest
```

## 🆘 Dépannage

### Problèmes Courants
```bash
# Container ne démarre pas
docker logs graphrag-demo

# Port occupé
sudo lsof -i :8501
sudo lsof -i :8000

# Réinitialiser complètement
docker stop graphrag-demo
docker rm graphrag-demo
docker rmi famibelle/graphrag-knowledge-graph:latest
docker pull famibelle/graphrag-knowledge-graph:latest
```

### Support
- 📖 **Documentation**: [README.md](./README.md)
- 🔧 **Troubleshooting**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/famibelle/KnowledgeGraphRag/issues)

---

🎉 **Prêt à explorer vos documents avec GraphRAG !**