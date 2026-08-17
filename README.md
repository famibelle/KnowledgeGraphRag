# 🧠 GraphRAG Knowledge Graph System

> **Système intelligent de questions-réponses basé sur un graphe de connaissances**
> 
> Combine Neo4j, OpenAI et FastAPI pour créer un système RAG (Retrieval Augmented Generation) avancé avec recherche sémantique vectorielle et navigation contextuelle dans un graphe de connaissances.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-green.svg)](https://neo4j.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-red.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-orange.svg)](https://streamlit.io)

> **⏱️ Pressé ? La démo tient en trois commandes** — voir
> [🎬 Lancer la démo GraphRAG](#-lancer-la-démo-graphrag-ligne-de-commande) :
> ```bash
> .venv/bin/python -m streamlit run demo_streamlit.py --server.port 8502   # interface
> .venv/bin/python demo_build_kg.py                                        # construction du graphe
> .venv/bin/python demo_query.py                                           # vectoriel seul vs graphe
> ```

## 🚀 Fonctionnalités Clés

### 🔍 **Recherche Sémantique Avancée**
- **Embeddings vectoriels** via OpenAI (text-embedding-3-small, 1536 dimensions)
- **Index vectoriel Neo4j natif** pour une recherche sub-seconde
- **Seuils de similarité configurables** (0.7-0.9) pour éviter les résultats non pertinents
- **Support multilingue** optimisé pour le français

### 🕸️ **Graphe de Connaissances Intelligent**
- **Relations automatiques** entre chunks similaires (`RELATES_TO`)
- **Navigation séquentielle** dans les documents (`NEXT_CHUNK`, `PREVIOUS_CHUNK`)
- **Contexte enrichi** avec métadonnées des documents
- **Requêtes Cypher flexibles** pour l'exploration avancée

### 📚 **Ingestion Multi-Format**
- **Formats supportés** : PDF, Markdown, Word, Texte
- **Découpage intelligent** en chunks optimisés
- **Création automatique** du graphe de connaissances
- **Processing parallèle** pour les performances

### 🧠 **Génération LLM Contextuelle**
- **Integration ChatGPT-3.5-turbo** pour les réponses
- **Réponses contextualisées** basées sur les documents
- **Filtrage intelligent** pour éviter les hallucinations
- **Recherche multi-documents** pour des requêtes complexes

---

## 🎬 Lancer la démo GraphRAG (ligne de commande)

> Cette branche embarque une **démo autonome** : on dépose des documents, la pipeline en
> extrait un graphe de connaissances, et le graphe se laisse interroger. Elle est bâtie sur
> [`neo4j-graphrag`](https://neo4j.com/docs/neo4j-graphrag-python/) et s'exécute **sans
> l'API ni l'interface Streamlit du projet** — trois commandes Python suffisent.

### **Étape 0 — Environnement (une seule fois)**

**Python 3.11+ est obligatoire** (`numpy` et `scipy` ne compilent pas en 3.10).

```bash
# Linux / macOS — uv télécharge CPython 3.11 au besoin
uv venv --python 3.11 .venv

# requirements.txt est un pip freeze Windows : pywin32 n'a pas de wheel ailleurs
grep -viE '^pywin32==' requirements.txt > /tmp/requirements.linux.txt
uv pip install --python .venv/bin/python -r /tmp/requirements.linux.txt
```

```powershell
# Windows
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Renseignez `.env` à la racine (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`,
`NEO4J_DATABASE`, `OPENAI_API_KEY`), puis créez **une fois** l'index vectoriel dans Neo4j :

```cypher
CREATE VECTOR INDEX GrahRAG IF NOT EXISTS
FOR (c:Chunk) ON (c.textEmbedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
```

### **Les 3 commandes de la démo**

| # | Commande | Ce qu'elle fait |
|---|----------|-----------------|
| 1️⃣ | `.venv/bin/python -m streamlit run demo_streamlit.py` | **L'interface de démo** — pipeline illustrée, dépôt de documents, construction du graphe en un bouton, visualisation, interrogation |
| 2️⃣ | `.venv/bin/python demo_build_kg.py` | **La pipeline en CLI** — ingère tout le dossier `PDFs/`, consolide les entités, imprime la volumétrie du graphe |
| 3️⃣ | `.venv/bin/python demo_query.py` | **La démonstration de valeur** — 3 questions posées deux fois : RAG vectoriel seul, puis enrichi par le graphe |

#### 1️⃣ Interface de démo (le chemin recommandé en présentation)

```bash
.venv/bin/python -m streamlit run demo_streamlit.py --server.port 8502
```

Ouvre `http://localhost:8502`. Le port 8502 évite la collision avec
`streamlit_rag_simple.py` (l'interface du projet, qui occupe 8501). Quatre pages :
*la pipeline* → *documents* → *le graphe obtenu* → *interroger le graphe*.
Comptez **~30 s par document** pour la construction.

#### 2️⃣ Construction du graphe en ligne de commande

```bash
# Traite TOUS les fichiers de PDFs/ (formats : .pdf, .txt, .md)
.venv/bin/python demo_build_kg.py
```

Sortie : découpage → embeddings → extraction libre des entités (1 appel LLM par chunk) →
consolidation, puis l'inventaire des labels, des relations et le décompte
`chunks / embeddings`. Ce script est le module importé par l'interface : les deux chemins
exécutent exactement le même code.

#### 3️⃣ Interrogation comparée

```bash
.venv/bin/python demo_query.py
```

Pose les 3 questions du script (plafond d'hébergement, seuil d'approbation, procédure
d'achat) à deux systèmes — `VectorRetriever` seul puis `VectorCypherRetriever` qui étend
chaque chunk à ses entités et à leurs voisins — et imprime les deux réponses côte à côte.

> **Sous Windows**, remplacez `.venv/bin/python` par `.venv\Scripts\python.exe` dans les
> trois commandes.

📖 Détail de la pipeline, mesures et limites assumées : **[DEMO-GRAPHRAG.md](./DEMO-GRAPHRAG.md)**

---

## 🐳 Déploiement Docker (Recommandé)

### **Démarrage Ultra-Rapide**

```bash
# 1. Télécharger la configuration
curl -o .env.docker https://raw.githubusercontent.com/famibelle/KnowledgeGraphRag/master/.env.docker

# 2. Éditer avec vos clés API
nano .env.docker  # ou notepad .env.docker sur Windows

# 3. Démarrer avec l'image publiée
docker run -d \
  --name graphrag-demo \
  -p 8000:8000 \
  -p 8501:8501 \
  --env-file .env.docker \
  famibelle/graphrag-knowledge-graph:latest
```

**🎉 C'est tout ! Ouvrez http://localhost:8501**

### **Images Docker Disponibles**

| Registry | Image | Commande |
|----------|--------|----------|
| 🐳 **Docker Hub** | `famibelle/graphrag-knowledge-graph` | `docker pull famibelle/graphrag-knowledge-graph:latest` |
| 📦 **GitHub** | `ghcr.io/famibelle/knowledgegraphrag` | `docker pull ghcr.io/famibelle/knowledgegraphrag:latest` |

### **Options de Déploiement**

#### **Option 1: Docker Run (Simple)**
```bash
docker run -d -p 8000:8000 -p 8501:8501 --env-file .env.docker famibelle/graphrag-knowledge-graph:latest
```

#### **Option 2: Docker Compose (Recommandé)**
```bash
# Avec image publiée
curl -o docker-compose.production.yml https://raw.githubusercontent.com/famibelle/KnowledgeGraphRag/master/docker-compose.production.yml
docker-compose -f docker-compose.production.yml up -d
```

#### **Option 3: Build Local**
```bash
git clone https://github.com/famibelle/KnowledgeGraphRag.git
cd KnowledgeGraphRag
make run
```

## 🏗️ Architecture Technique

### **Stack Technologique**

```mermaid
flowchart TB
    %% User Interface Layer
    subgraph "🎨 Interface Utilisateur"
        UI[Streamlit Frontend<br/>Port 8501]
        BROWSER[🌐 Navigateur Web]
    end
    
    %% API Layer
    subgraph "⚡ API Layer"
        API[FastAPI Backend<br/>Port 8000<br/>ThreadPoolExecutor]
        DOCS[📊 Swagger/ReDoc<br/>Auto-generated]
    end
    
    %% External Services
    subgraph "🤖 Services IA"
        OPENAI_EMB[OpenAI Embeddings<br/>text-embedding-3-small<br/>1536 dimensions]
        OPENAI_LLM[OpenAI LLM<br/>GPT-3.5-turbo<br/>Response Generation]
    end
    
    %% Database Layer
    subgraph "🗄️ Base de Données"
        NEO4J[Neo4j 5.x<br/>Graph + Vector DB]
        VECTOR_IDX[Vector Index<br/>GrahRAG<br/>Cosine Similarity]
        GRAPH_REL[Graph Relations<br/>NEXT/PREV/RELATES_TO]
    end
    
    %% Data Processing
    subgraph "📊 Traitement Documents"
        INGEST[Document Ingestion<br/>PDF/MD/DOCX/TXT]
        CHUNK[Text Chunking<br/>RecursiveCharacterTextSplitter]
        EMBED[Embedding Generation<br/>Parallel Processing]
    end
    
    %% Connections
    BROWSER --> UI
    UI <--> API
    API --> DOCS
    
    API <--> OPENAI_EMB
    API <--> OPENAI_LLM
    API <--> NEO4J
    
    NEO4J --> VECTOR_IDX
    NEO4J --> GRAPH_REL
    
    API --> INGEST
    INGEST --> CHUNK
    CHUNK --> EMBED
    EMBED --> NEO4J
    
    %% Styling
    classDef frontend fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef backend fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef ai fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef database fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class UI,BROWSER frontend
    class API,DOCS backend
    class OPENAI_EMB,OPENAI_LLM ai
    class NEO4J,VECTOR_IDX,GRAPH_REL database
    class INGEST,CHUNK,EMBED processing
```

**Architecture Technique :**
- **Frontend** : Streamlit (Python) - Interface web interactive
- **Backend** : FastAPI (Python async) - API REST haute performance
- **IA Services** : OpenAI (embeddings + LLM) - Traitement sémantique
- **Base de Données** : Neo4j 5.x - Graphe + index vectoriel natif
- **Performance** : ThreadPoolExecutor - Traitement parallèle optimisé

### **Modèle de Données Neo4j**

```mermaid
graph LR
    %% Entités principales
    D1[📄 Document A]
    D2[📄 Document B]
    
    C1[📝 Chunk 1]
    C2[📝 Chunk 2]
    C3[📝 Chunk 3]
    C4[📝 Chunk 4]
    
    %% Relations document → chunks
    D1 --> C1
    D1 --> C2
    D2 --> C3
    D2 --> C4
    
    %% Navigation séquentielle
    C1 -.-> C2
    
    %% Relations sémantiques inter-documents
    C1 -.-> C3
    C2 -.-> C4
    
    %% Styling
    classDef doc fill:#e1f5fe,stroke:#1976d2,stroke-width:2px
    classDef chunk fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class D1,D2 doc
    class C1,C2,C3,C4 chunk
```

**Structure GraphRAG :**
```cypher
(:Document) -[CONTAINS_CHUNK]-> (:Chunk)
(:Chunk) -[NEXT_CHUNK]-> (:Chunk)  
(:Chunk) -[RELATES_TO]-> (:Chunk)
```

**Types de Relations :**
- **Ligne pleine** : CONTAINS_CHUNK (hiérarchique)
- **Ligne pointillée** : NEXT_CHUNK (séquentielle) 
- **Ligne pointillée courbe** : RELATES_TO (sémantique inter-documents)

### **🔍 Requêtes Cypher d'Exploration**

**Pour explorer votre graphe dans [Neo4j Browser](https://console-preview.neo4j.io/tools/query) :**

```cypher
// 📊 Vue d'ensemble du graphe complet
MATCH (d:Document)-[r]-(c:Chunk)
RETURN d, r, c
LIMIT 50;

// 📈 Statistiques générales du Knowledge Graph
MATCH (d:Document) 
WITH count(d) as documents
MATCH (c:Chunk) 
WITH documents, count(c) as chunks
MATCH ()-[r]->() 
RETURN documents, chunks, count(r) as total_relations;

// 📄 Documents avec leurs chunks et métadonnées
MATCH (d:Document)-[:CONTAINS_CHUNK]->(c:Chunk)
RETURN d.filename, d.chunk_count, d.created_at, 
       count(c) as actual_chunks, 
       collect(c.chunk_index)[0..3] as first_chunks
ORDER BY d.created_at DESC;

// 🕸️ Navigation séquentielle dans un document
MATCH (d:Document {filename: 'your-document.pdf'})-[:CONTAINS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
OPTIONAL MATCH (c)-[:PREVIOUS_CHUNK]->(prev:Chunk)
RETURN c.chunk_index, c.text[0..100] + '...' as preview,
       prev.chunk_index as previous, next.chunk_index as next
ORDER BY c.chunk_index;

// 🌐 Relations sémantiques inter-documents
MATCH (c1:Chunk)-[r:RELATES_TO]->(c2:Chunk)
WHERE c1.filename <> c2.filename
RETURN c1.filename, c2.filename, r.similarity,
       c1.text[0..80] + '...' as chunk1_preview,
       c2.text[0..80] + '...' as chunk2_preview
ORDER BY r.similarity DESC
LIMIT 20;

// 📊 Chunks les plus connectés (hubs sémantiques)
MATCH (c:Chunk)-[r:RELATES_TO]-()
WITH c, count(r) as connections
WHERE connections > 2
RETURN c.filename, c.chunk_index, connections,
       c.text[0..100] + '...' as preview
ORDER BY connections DESC
LIMIT 10;

// 🔄 Chemins entre deux documents spécifiques  
MATCH path = shortestPath(
  (d1:Document {filename: 'doc1.pdf'})-[*]-(d2:Document {filename: 'doc2.pdf'})
)
RETURN path, length(path) as path_length;

// 📋 Métadonnées complètes d'un chunk spécifique
MATCH (c:Chunk {filename: 'your-doc.pdf', chunk_index: 0})
OPTIONAL MATCH (c)-[r1:RELATES_TO]->(related:Chunk)
OPTIONAL MATCH (c)-[r2:NEXT_CHUNK]->(next:Chunk)
OPTIONAL MATCH (c)-[r3:PREVIOUS_CHUNK]->(prev:Chunk)
RETURN c, 
       collect(DISTINCT related.filename) as related_docs,
       next.chunk_index as next_chunk,
       prev.chunk_index as prev_chunk;
```

### **🎯 Requêtes Cypher Avancées**

```cypher

// 🔗 Détection de chunks "pont" entre documents
MATCH (c:Chunk)-[:RELATES_TO]-(other:Chunk)
WHERE c.filename <> other.filename
WITH c, collect(DISTINCT other.filename) as connected_docs
WHERE size(connected_docs) > 2
RETURN c.filename, c.chunk_index, connected_docs,
       c.text[0..100] + '...' as bridge_content
ORDER BY size(connected_docs) DESC;

// 🎯 Recherche par proximité sémantique (k-NN manuel)
MATCH (target:Chunk {filename: 'your-doc.pdf', chunk_index: 0})
MATCH (c:Chunk)
WHERE c <> target
WITH c, gds.similarity.cosine(target.textEmbedding, c.textEmbedding) as similarity
ORDER BY similarity DESC
LIMIT 10
RETURN c.filename, c.chunk_index, similarity,
       c.text[0..120] + '...' as similar_content;
```

**�💡 Conseils d'utilisation :**
- **Neo4j Browser** : https://console-preview.neo4j.io/tools/query
- Remplacez `'your-document.pdf'` par vos vrais noms de fichiers  
- Ajustez les `LIMIT` selon la taille de votre corpus
- Utilisez `PROFILE` ou `EXPLAIN` pour analyser les performances
- Les résultats s'affichent en mode graphique interactif
- **GDS (Graph Data Science)** requis pour les algorithmes avancés

### **Workflow GraphRAG**

```mermaid
sequenceDiagram
    participant User as 👤 Utilisateur
    participant UI as 🎨 Streamlit
    participant API as ⚡ FastAPI
    participant Neo4j as 🗄️ Neo4j
    participant OpenAI as 🤖 OpenAI
    
    %% Phase 1: Ingestion de Document
    Note over User,OpenAI: 📄 Phase 1: Ingestion de Document
    User->>+UI: Upload Document (PDF/MD/DOCX)
    UI->>+API: POST /ingest_file
    API->>API: Parse & Chunk Document
    API->>+OpenAI: Generate Embeddings
    OpenAI-->>-API: Vector[1536] per chunk
    API->>+Neo4j: Store Chunks + Embeddings
    Neo4j->>Neo4j: Create NEXT/PREV Relations
    Neo4j-->>-API: Ingestion Complete
    API-->>-UI: Success Response
    UI-->>-User: Document Processed ✅
    
    %% Phase 2: Construction Relations
    Note over User,OpenAI: 🕸️ Phase 2: Construction Relations Sémantiques
    User->>+UI: Build Inter-doc Relations
    UI->>+API: POST /smart_inter_document_relationships
    API->>+Neo4j: Find Similar Chunks (cosine > 0.8)
    Neo4j->>Neo4j: Create RELATES_TO Relations
    Neo4j-->>-API: Relations Created
    API-->>-UI: Graph Enhanced ✅
    UI-->>-User: Knowledge Graph Ready 🕸️
    
    %% Phase 3: Recherche Intelligente  
    Note over User,OpenAI: 🔍 Phase 3: Recherche Contextuelle
    User->>+UI: Ask Question
    UI->>+API: POST /semantic_search_with_context
    API->>+OpenAI: Embed Question
    OpenAI-->>-API: Question Vector[1536]
    API->>+Neo4j: Vector Search + Graph Traversal
    Neo4j->>Neo4j: Find Similar Chunks (threshold > 0.9)
    Neo4j->>Neo4j: Enrich with NEXT/PREV/RELATES_TO
    Neo4j-->>-API: Contextualized Chunks
    API->>+OpenAI: Generate Answer with Context
    OpenAI-->>-API: Intelligent Response
    API-->>-UI: Enhanced Results
    UI-->>-User: Answer + Sources + Context 🧠
```

**Étapes Détaillées :**

1. **📄 Ingestion** : Document → Parsing → Chunking → Embeddings → Neo4j Storage
2. **🕸️ Relations** : Analyse similarité → Création liens sémantiques → Graphe enrichi  
3. **🔍 Recherche** : Question → Vector Search → Filtrage seuil → Enrichissement contexte
4. **✨ Génération** : Contexte étendu → LLM → Réponse intelligente avec sources

## ⚡ Installation Rapide

### **1. Prérequis**
- Python 3.11+
- Neo4j 5.x avec support vectoriel
- Clé API OpenAI
- Git

### **2. Clone & Setup**
```bash
git clone https://github.com/famibelle/KnowledgeGraphRag.git
cd KnowledgeGraphRag

# Créer l'environnement virtuel (Python 3.11+ impératif)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

> ⚠️ **Linux / macOS** : `requirements.txt` est un `pip freeze` réalisé sous Windows et
> épingle `pywin32==311`, qui n'a pas de wheel ailleurs et fait échouer tout l'install.
> Filtrez-le : `grep -viE '^pywin32==' requirements.txt > /tmp/requirements.linux.txt`
> puis installez ce fichier-là.

### **3. Configuration Environnement**
```bash
# Copier le fichier d'exemple
copy .env.example .env

# Éditer .env avec vos paramètres
```

**Contenu `.env` requis :**
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j
```

### **4. Initialisation Neo4j**

L'index vectoriel se crée **manuellement**, une seule fois, depuis le Neo4j Browser
(il n'existe pas d'endpoint d'initialisation) :

```cypher
CREATE VECTOR INDEX GrahRAG IF NOT EXISTS
FOR (c:Chunk) ON (c.textEmbedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
```

Le nom `GrahRAG` (avec sa coquille) est celui codé en dur dans l'API : ne le corrigez pas.

### **5. Lancement des deux processus**

```bash
# API — À LANCER DEPUIS KnowledgeGraphRagAPI/ : main.py charge `../.env`,
# chemin relatif au répertoire courant du processus.
cd KnowledgeGraphRagAPI && ../.venv/bin/python -m uvicorn main:app --reload --port 8000

# Interface (autre terminal, depuis la racine)
.venv/bin/python -m streamlit run streamlit_rag_simple.py --server.port 8501

# Ou les deux d'un coup (contrôles version/.env/dépendances, puis ouvre le navigateur)
.venv/bin/python start.py
```

API sur `http://localhost:8000/docs`, interface sur `http://localhost:8501`.
`GET /health` dit immédiatement si la configuration Neo4j est bonne.

## 📖 Utilisation

### **Interface Web (Recommandée)**
1. Ouvrir `http://localhost:8501` (Streamlit)
2. **📤 Onglet "Gestion Documents"** : Upload vos fichiers
3. **🔍 Onglet "Recherche RAG"** : Poser vos questions
4. **🕸️ Onglet "Graphe"** : Explorer les relations

### **API REST**
Documentation interactive : `http://localhost:8000/docs`

**Endpoints principaux :**
- `POST /ingest_file` - Ingestion de documents
- `POST /semantic_search_with_context` - Recherche avec contexte graphique
- `POST /query` - Recherche avec réponse LLM
- `GET /graph_stats` - Statistiques du graphe

### **Exemple API**
```bash
# Recherche contextuelle
curl -X POST "http://localhost:8000/semantic_search_with_context" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "résultats financiers LuxConnect",
    "top_k": 5,
    "similarity_threshold": 0.9
  }'
```

## 🧩 Architecture Détaillée

### **1. Recherche Sémantique Hybride**
Notre approche unique combine :
- **Recherche vectorielle** : Similarité cosinus sur embeddings 1536D
- **Filtrage par seuil** : Élimination automatique des résultats non pertinents
- **Enrichissement contextuel** : Navigation dans les relations du graphe
- **Métadonnées dynamiques** : Informations sur les documents sources

### **2. Parallel Processing Optimisé**
```python
# Exemple d'implémentation
with ThreadPoolExecutor() as executor:
    result = await loop.run_in_executor(
        executor,
        kg.query,  # Requête Neo4j
        cypher_query,
        parameters
    )
```

**Avantages :**
- ⚡ **22+ opérations parallélisées** dans le code
- 🚀 **Performance sub-seconde** pour la plupart des requêtes
- 🔄 **Traitement asynchrone** des embeddings et requêtes

### **3. Gestion Intelligente des Relations**
```cypher
-- Création automatique de relations sémantiques
MATCH (c1:Chunk), (c2:Chunk)
WHERE c1.filename <> c2.filename 
  AND gds.similarity.cosine(c1.textEmbedding, c2.textEmbedding) > 0.85
CREATE (c1)-[:RELATES_TO {score: similarity}]->(c2)
```


## 🧪 Validation

Il n'y a pas de suite de tests automatisés dans ce dépôt. Les vérifications se font
à la main :

```bash
# 1. La configuration est-elle vivante ?
curl -s http://localhost:8000/health

# 2. Le graphe contient-il ce qu'on croit ?
.venv/bin/python demo_build_kg.py     # imprime labels, relations, chunks/embeddings

# 3. Le graphe apporte-t-il quelque chose au vectoriel seul ?
.venv/bin/python demo_query.py        # les deux réponses, côte à côte
```

> ⚠️ L'extraction d'entités **n'est pas déterministe**, même à `temperature=0` : deux
> exécutions sur le même corpus donnent des volumétries différentes. Ne validez jamais
> une exécution sur un décompte d'entités.

### **Exemples de Requêtes**
- **Mono-document** : "Quels sont les résultats financiers de LuxConnect?"
- **Multi-documents** : "LuxConnect financials et TMD utilisation IA friction"
- **Contextuelle** : "Comment l'innovation technologique impacte les performances?"

## 📊 Métriques de Performance

### **Benchmarks Typiques**
- **Recherche vectorielle** : < 100ms sur 10K+ chunks
- **Génération LLM** : 2-5 secondes selon la complexité
- **Ingestion document** : 30-60 secondes selon la taille
- **Relations inter-documents** : 1-3 minutes selon le corpus

### **Capacités Scalabilité**
- ✅ **Millions de chunks** supportés (index vectoriel Neo4j)
- ✅ **Centaines de documents** simultanés
- ✅ **Requêtes parallèles** sans dégradation

## 🤝 Contribution

### **Structure du Projet**
```
├── KnowledgeGraphRagAPI/     # Backend FastAPI
│   ├── main.py              # API principale
│   └── requirements.txt     # Dépendances backend
├── streamlit_rag_simple.py  # Interface Streamlit du projet (port 8501)
├── demo_streamlit.py        # 🎬 Démo : interface de présentation (port 8502)
├── demo_build_kg.py         # 🎬 Démo : construction du graphe (neo4j-graphrag)
├── demo_query.py            # 🎬 Démo : vectoriel seul vs enrichi par le graphe
├── PDFs/                    # Corpus de démo (dossier lu par demo_build_kg.py)
├── start.py                 # Lance API + interface d'un coup
├── requirements.txt         # Dépendances globales (pip freeze Windows)
├── .env.example            # Template configuration
└── README.md               # Cette documentation
```

### **Documentation**

| Guide | Contenu |
|-------|---------|
| [DEMO-GRAPHRAG.md](./DEMO-GRAPHRAG.md) | 🧪 Démo autonome : graphe de connaissances sur un référentiel documentaire, mesures vectoriel vs graphe |
| [QUICK-START.md](./QUICK-START.md) | 🚀 Déploiement Docker en 2 minutes |
| [DOCKER.md](./DOCKER.md) | 🐳 Détails de la conteneurisation |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | 🔧 Dépannage |
| [WINDOWS.md](./WINDOWS.md) | 🪟 Spécificités Windows |


## 🐛 Dépannage

### **Problèmes Courants**
- **Neo4j connexion** : vérifier `.env` et URL/credentials ; `GET /health` répond
  `unhealthy` avec une erreur `NoneType` quand la base est injoignable au démarrage
- **L'API ne voit pas le `.env`** : elle le charge via `../.env`, chemin **relatif au
  répertoire courant** — lancez uvicorn depuis `KnowledgeGraphRagAPI/`
- **OpenAI API** : valider la clé et les quotas
- **`pip install` échoue sur `pywin32`** (Linux/macOS) : filtrer la ligne, cf. §Installation
- **Recherche qui ne renvoie rien** : l'index vectoriel `GrahRAG` n'existe pas, ou les
  embeddings ne sont pas dans `Chunk.textEmbedding`
- **Démo : `Text2CypherRetriever` en erreur** : le LLM a produit du Cypher refusé par
  Neo4j ; l'interface le signale et propose d'écrire la requête à la main


---

**🚀 Prêt à explorer vos documents avec l'IA ? Commencez dès maintenant !**

*Pour plus d'aide : [Issues GitHub](https://github.com/famibelle/KnowledgeGraphRag/issues)*
