# Knowledge Graph RAG API Documentation

## Vue d'ensemble

Cette API FastAPI offre des fonctionnalités complètes de Knowledge Graph pour un système RAG (Retrieval-Augmented Generation) utilisant Neo4j et OpenAI.

## Architecture

### Base de données
- **Neo4j** avec embeddings vectoriels natifs
- **OpenAI Embeddings** (text-embedding-3-small)
- **Structure de graphe** : Documents → Chunks avec relations

### Structure du Graphe

```
Document (filename, created_at, chunk_count, file_extension, file_size)
    ↓ CONTAINS_CHUNK
Chunk (id, text, filename, chunk_index, textEmbedding)
    ↓ NEXT_CHUNK / PREVIOUS_CHUNK (relations séquentielles)
    ↓ SIMILAR_TO (relations sémantiques)
    ↓ RELATES_TO (relations personnalisées)
```

## Endpoints

### 1. Ingestion de Documents

**POST /ingest_file**
- Upload et traitement de fichiers (PDF, Markdown, TXT, DOCX)
- Création automatique de la structure de graphe
- Génération d'embeddings avec Neo4j natif
- Création de relations séquentielles entre chunks

```python
# Formats supportés
files = {'file': ('document.pdf', file_content, 'application/pdf')}
response = requests.post('http://localhost:8000/ingest_file', files=files)
```

### 2. Recherche et Interrogation

**POST /query**
- Recherche vectorielle avec Neo4j natif
- Génération de réponse LLM basée sur le contexte trouvé

```python
request = {
    "question": "What is a knowledge graph?",
    "top_k": 5
}
```

**POST /semantic_search_with_context**
- Recherche sémantique enrichie avec contexte de graphe
- Inclut les chunks précédents/suivants et similaires

```python
request = {
    "question": "knowledge graph applications",
    "top_k": 3,
    "similarity_threshold": 0.6
}
```

### 3. Gestion des Relations

**POST /create_relationship**
- Création de relations personnalisées entre chunks

```python
request = {
    "source_chunk_id": "doc1-chunk0",
    "target_chunk_id": "doc1-chunk5", 
    "relationship_type": "EXPLAINS",
    "properties": {"confidence": 0.9}
}
```

**POST /discover_semantic_relationships**
- Découverte automatique de relations sémantiques basées sur la similarité

```python
# Crée des relations SIMILAR_TO automatiquement
POST /discover_semantic_relationships?similarity_threshold=0.8
```

### 4. Exploration du Graphe

**GET /graph_structure/{filename}**
- Structure complète du graphe pour un document
- Relations et métadonnées des chunks

**GET /graph_stats**
- Statistiques globales du Knowledge Graph
- Nombre de documents, chunks, relations

**POST /graph_traversal** 
- Parcours avancé du graphe

```python
# Types de parcours disponibles
request = {
    "traversal_type": "document_flow",  # ou "shortest_path", "related_chunks"
    "start_chunk_id": "doc1-chunk0",
    "max_results": 10
}
```

### 5. Requêtes Avancées

**POST /cypher**
- Exécution de requêtes Cypher personnalisées

```python
request = {
    "query": "MATCH (d:Document)-[:CONTAINS_CHUNK]->(c:Chunk) RETURN count(c)",
    "params": {}
}
```

## Types de Relations

### Relations Automatiques
- **CONTAINS_CHUNK** : Document → Chunk
- **NEXT_CHUNK** : Chunk → Chunk suivant
- **PREVIOUS_CHUNK** : Chunk → Chunk précédent
- **SIMILAR_TO** : Chunks sémantiquement similaires (auto-découvertes)

### Relations Personnalisées
- **RELATES_TO** : Relation générique personnalisable
- **EXPLAINS** : Un chunk explique un autre
- **SUPPORTS** : Un chunk supporte un argument d'un autre
- Et toute autre relation définie par l'utilisateur

## Configuration

### Variables d'environnement (.env)
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
OPENAI_API_KEY=sk-...
```

### Index vectoriel requis
```cypher
CREATE VECTOR INDEX vector FOR (c:Chunk) ON (c.textEmbedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
```

## Exemples d'utilisation

### Workflow complet

1. **Ingestion**
```python
import requests

# Upload d'un document
with open('document.pdf', 'rb') as f:
    files = {'file': ('document.pdf', f, 'application/pdf')}
    response = requests.post('http://localhost:8000/ingest_file', files=files)
```

2. **Recherche simple**
```python
# Recherche avec réponse LLM
response = requests.post('http://localhost:8000/query', json={
    "question": "What are the benefits of knowledge graphs?",
    "top_k": 5
})
```

3. **Recherche avec contexte**
```python
# Recherche enrichie avec relations de graphe
response = requests.post('http://localhost:8000/semantic_search_with_context', json={
    "question": "knowledge graph applications",
    "top_k": 3,
    "similarity_threshold": 0.6
})
```

4. **Exploration du graphe**
```python
# Structure d'un document
response = requests.get('http://localhost:8000/graph_structure/document.pdf')

# Statistiques globales
response = requests.get('http://localhost:8000/graph_stats')
```

5. **Relations sémantiques**
```python
# Découverte automatique
response = requests.post('http://localhost:8000/discover_semantic_relationships?similarity_threshold=0.7')

# Création manuelle
response = requests.post('http://localhost:8000/create_relationship', json={
    "source_chunk_id": "doc1-0",
    "target_chunk_id": "doc1-5",
    "relationship_type": "SUPPORTS",
    "properties": {"evidence_type": "statistical"}
})
```

## Démarrage

### 1. Démarrer l'API
```bash
python start_api.py
```

### 2. Tests
```bash
python test_kg_api.py
```

### 3. Documentation interactive
Accéder à : http://localhost:8000/docs

## Avantages du Knowledge Graph

1. **Contexte enrichi** : Les chunks sont liés à leur document parent et aux chunks adjacents
2. **Relations sémantiques** : Découverte automatique de connections conceptuelles
3. **Parcours de graphe** : Navigation intelligente entre concepts reliés
4. **Recherche contextuelle** : Résultats enrichis avec le contexte du graphe
5. **Flexibilité** : Ajout de relations personnalisées selon les besoins

Cette API transforme un simple stockage de chunks en un véritable Knowledge Graph explorable et interrogeable.