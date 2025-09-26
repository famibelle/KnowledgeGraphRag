# 🐳 GraphRAG Knowledge Graph - Docker Deployment

> **Déploiement simplifié avec Docker pour PoC et démonstrations**

## 🚀 Démarrage Rapide

### **1. Configuration**
```bash
# Copier et éditer la configuration
cp .env.docker.example .env.docker
nano .env.docker  # Ajouter vos API keys
```

### **2. Lancement**
```bash
# Option A: Avec Make (recommandé)
make run

# Option B: Avec Docker Compose
docker-compose up --build -d
```

### **3. Accès**
- **Interface** : http://localhost:8501
- **API Docs** : http://localhost:8000/docs
- **Health Check** : http://localhost:8000/health

## 📋 Configuration Requise

### **.env.docker - Variables Essentielles**
```env
# OpenAI (OBLIGATOIRE)
OPENAI_API_KEY=sk-your-openai-key-here

# Neo4j (OBLIGATOIRE)  
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

### **Services Externes Requis**
- ✅ **Neo4j Aura** (cloud) ou instance Neo4j locale
- ✅ **OpenAI API** avec crédits disponibles

## 🛠️ Commandes Utiles

```bash
# Monitoring
make logs          # Voir les logs en temps réel
make status        # Status des containers
make health        # Test santé de l'application

# Debug  
make shell         # Accéder au container
make test-connection  # Tester Neo4j et OpenAI

# Maintenance
make stop          # Arrêter
make restart       # Redémarrer  
make clean         # Nettoyer
```

## 📊 Architecture Docker

```
┌─────────────────────────────────────┐
│          Container GraphRAG         │
├─────────────────────────────────────┤
│  🎨 Streamlit (Port 8501)          │
│  ⚡ FastAPI (Port 8000)            │
│  🐍 Python 3.11                   │
│  📦 Dependencies                   │
└─────────────────────────────────────┘
           │                │
           ▼                ▼
    🤖 OpenAI API    🗄️ Neo4j Database
```

## 🔧 Customisation

### **Variables d'Environnement Avancées**
```env
# Performance
DEFAULT_SIMILARITY_THRESHOLD=0.9
DEFAULT_TOP_K=5
LOG_LEVEL=INFO

# Ports (si conflit)
API_PORT=8000
STREAMLIT_SERVER_PORT=8501
```

### **Volumes Persistants**
```yaml
# Dans docker-compose.yml (optionnel)
volumes:
  - ./logs:/app/logs
  - ./uploads:/app/uploads
```

## 🐛 Troubleshooting

### **Container ne démarre pas**
```bash
# Vérifier les logs
docker-compose logs graphrag

# Vérifier la configuration
cat .env.docker
```

### **API non accessible**
```bash
# Tester la santé
curl http://localhost:8000/health

# Vérifier les ports
docker-compose ps
```

### **Erreurs Neo4j/OpenAI**
```bash
# Tester les connections
make test-connection

# Vérifier les variables
make shell
env | grep -E "(OPENAI|NEO4J)"
```

## 🎯 Cas d'Usage

### **✅ Parfait pour :**
- PoC et démonstrations
- Tests rapides
- Développement isolé
- Présentation client

### **❌ Non recommandé pour :**
- Production à haute charge
- Données sensibles sans chiffrement
- Déploiement multi-nœuds

## 📝 Notes Importantes

- **Sécurité** : Les API keys sont en variables d'environnement
- **Performance** : Optimisé pour demo, pas production  
- **Persistance** : Les données Neo4j sont externes
- **Logs** : Disponibles via `make logs`

## 🚀 Next Steps

1. **Configurez** vos API keys dans `.env.docker`
2. **Lancez** avec `make run`
3. **Uploadez** vos documents via l'interface
4. **Explorez** votre graphe de connaissances !

---

💡 **Astuce** : Utilisez `make help` pour voir toutes les commandes disponibles