# 🔧 Troubleshooting Guide - GraphRAG Knowledge Graph

## 🚨 Problèmes Courants et Solutions

### 1. **Problèmes de Connexion Neo4j**

#### ❌ `Failed to connect to Neo4j`
**Causes possibles :**
- URI Neo4j incorrect dans `.env`
- Credentials erronés
- Instance Neo4j non démarrée
- Firewall/réseau

**Solutions :**
```bash
# Vérifier la connexion
curl -u neo4j:password http://localhost:7474/db/data/

# Tester les paramètres .env
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('URI:', os.getenv('NEO4J_URI'))
print('User:', os.getenv('NEO4J_USERNAME'))
"
```

#### ❌ `Vector index not found`
**Solution :**
```bash
# Réinitialiser la base
curl -X POST "http://localhost:8000/initialize_db"
```

### 2. **Problèmes OpenAI API**

#### ❌ `OpenAI API key invalid`
**Solutions :**
- Vérifier la clé sur https://platform.openai.com/api-keys
- Contrôler les quotas/limites
- Tester avec curl :
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### ❌ `Rate limit exceeded`
**Solutions :**
- Attendre et réessayer
- Upgrader votre plan OpenAI
- Réduire `top_k` dans les requêtes

### 3. **Problèmes d'Installation**

#### ❌ `ModuleNotFoundError`
**Solutions :**
```bash
# Réinstaller les dépendances
pip install -r requirements.txt
pip install -r KnowledgeGraphRagAPI/requirements.txt

# Vérifier l'environnement virtuel
which python
pip list | grep fastapi
```

#### ❌ `Permission denied` sur Linux/Mac
**Solutions :**
```bash
chmod +x setup.sh
chmod +x start.py
```

### 4. **Problèmes de Performance**

#### ❌ **Recherche lente (>5 secondes)**
**Diagnostic :**
```cypher
// Vérifier l'index vectoriel
SHOW INDEXES
```

**Solutions :**
- Recréer l'index vectoriel
- Réduire `top_k`
- Optimiser Neo4j (plus de RAM)

#### ❌ **Ingestion lente**
**Solutions :**
- Réduire la taille des documents
- Paralléliser avec `MAX_WORKERS`
- Utiliser des chunks plus petits

### 5. **Problèmes Interface Streamlit**

#### ❌ `Connection refused`
**Solutions :**
```bash
# Vérifier si l'API tourne
curl http://localhost:8000/health

# Redémarrer Streamlit
streamlit run streamlit_rag_simple.py --server.port 8501
```

#### ❌ **Upload de fichiers échoue**
**Solutions :**
- Vérifier les formats supportés (PDF, MD, DOCX, TXT)
- Contrôler la taille (<50MB recommandé)
- Logs backend pour plus de détails

### 6. **Problèmes de Qualité des Résultats**

#### ❌ **Résultats non pertinents**
**Solutions :**
- Augmenter `similarity_threshold` (0.85-0.95)
- Réduire `top_k` pour moins de bruit
- Reformuler la question
- Vérifier la qualité de l'ingestion

#### ❌ **Pas de résultats trouvés**
**Solutions :**
- Diminuer `similarity_threshold` (0.7-0.8)
- Vérifier que les documents sont ingérés
- Utiliser des mots-clés plus génériques

### 7. **Problèmes de Développement**

#### ❌ **Hot reload ne marche pas**
**Solutions :**
```bash
# FastAPI
uvicorn main:app --reload --log-level debug

# Streamlit
streamlit run app.py --server.fileWatcherType poll
```

#### ❌ **Logs insuffisants**
**Solutions :**
```python
# Dans .env
LOG_LEVEL=DEBUG

# Ou directement
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔍 Outils de Diagnostic

### **1. Test de Santé Complet**
```python
# test_health.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_api():
    try:
        r = requests.get("http://localhost:8000/health")
        print(f"API: {r.status_code} - {r.json()}")
        return True
    except Exception as e:
        print(f"API Error: {e}")
        return False

def test_neo4j():
    try:
        r = requests.get("http://localhost:8000/db_info")
        print(f"Neo4j: {r.status_code} - {r.json()}")
        return True
    except Exception as e:
        print(f"Neo4j Error: {e}")
        return False

def test_openai():
    import openai
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        models = client.models.list()
        print(f"OpenAI: OK - {len(models.data)} models available")
        return True
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 GraphRAG Health Check")
    print("-" * 30)
    
    api_ok = test_api()
    neo4j_ok = test_neo4j()
    openai_ok = test_openai()
    
    if all([api_ok, neo4j_ok, openai_ok]):
        print("✅ Tous les services sont opérationnels!")
    else:
        print("❌ Certains services ont des problèmes")
```

### **2. Test de Performance**
```python
# test_performance.py
import time
import requests

def benchmark_search():
    start = time.time()
    
    response = requests.post(
        "http://localhost:8000/semantic_search_with_context",
        json={
            "question": "test query",
            "top_k": 5,
            "similarity_threshold": 0.8
        }
    )
    
    duration = time.time() - start
    print(f"Search took: {duration:.2f}s")
    return duration < 2.0  # Should be under 2 seconds

if __name__ == "__main__":
    if benchmark_search():
        print("✅ Performance OK")
    else:
        print("❌ Performance issue detected")
```

### **3. Vérification des Données**
```cypher
// Dans Neo4j Browser ou via /cypher endpoint

// Compter les documents et chunks
MATCH (d:Document) RETURN count(d) as documents;
MATCH (c:Chunk) RETURN count(c) as chunks;

// Vérifier les embeddings
MATCH (c:Chunk) 
WHERE c.textEmbedding IS NULL 
RETURN count(c) as chunks_without_embeddings;

// Statistiques des relations
MATCH ()-[r]->() 
RETURN type(r) as relation_type, count(r) as count 
ORDER BY count DESC;

// Vérifier l'index vectoriel
SHOW INDEXES WHERE name = 'GrahRAG';
```

## 📞 Support et Contact

- **GitHub Issues**: [https://github.com/famibelle/KnowledgeGraphRag/issues](https://github.com/famibelle/KnowledgeGraphRag/issues)
- **Documentation**: README.md principal
- **Logs**: Activez `LOG_LEVEL=DEBUG` dans `.env`

## 🔄 Procédure de Reset Complet

Si tout échoue, reset complet :

```bash
# 1. Arrêter tous les services
pkill -f uvicorn
pkill -f streamlit

# 2. Nettoyer Neo4j
# Via Neo4j Browser: MATCH (n) DETACH DELETE n;

# 3. Réinstaller
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Reconfigurer
cp .env.example .env
# Éditer .env avec vos vraies valeurs

# 5. Redémarrer
python start.py
```

---

💡 **Conseil** : En cas de problème persistant, activez les logs debug et consultez les fichiers de log pour plus de détails.