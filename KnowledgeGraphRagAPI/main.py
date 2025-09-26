from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import logging
import traceback
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KnowledgeGraphRag API",
    description="""
    ## API de Graphe de Connaissances avec RAG (Retrieval Augmented Generation)
    
    Cette API combine Neo4j et OpenAI pour créer un système intelligent de questions-réponses
    basé sur vos documents. Elle offre des capacités avancées de recherche sémantique 
    et de génération de réponses contextualisées.
    
    ### Fonctionnalités principales
    
    🔍 **Recherche sémantique avancée**
    - Embeddings vectoriels via OpenAI (text-embedding-3-small)
    - Recherche par similarité cosinus avec seuils configurables
    - Support multilingue optimisé pour le français
    
    📚 **Ingestion intelligente de documents**
    - Support PDF, Markdown, Word, Texte
    - Découpage automatique en chunks optimisés
    - Création automatique du graphe de connaissances
    
    🧠 **Génération de réponses LLM** 
    - Integration ChatGPT-3.5-turbo
    - Réponses contextualisées basées sur les documents
    - Filtrage intelligent pour éviter les hallucinations
    
    🕸️ **Graphe de connaissances Neo4j**
    - Relations automatiques entre chunks similaires
    - Navigation séquentielle dans les documents
    - Requêtes Cypher flexibles pour l'exploration
    
    ### Architecture technique
    
    - **Base de données**: Neo4j (graphe vectoriel)
    - **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
    - **LLM**: OpenAI GPT-3.5-turbo
    - **Backend**: FastAPI (Python async)
    - **Recherche**: Similarité cosinus avec index vectoriel
    
    ### Configuration requise
    
    - Neo4j 5.x avec support vectoriel
    - Clé API OpenAI valide
    - Variables d'environnement configurées (.env)
    """,
    version="1.2.0",
    contact={
        "name": "Support KnowledgeGraphRag",
        "url": "https://github.com/famibelle/KnowledgeGraphRag",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    tags_metadata=[
        {
            "name": "Query",
            "description": "Endpoints de recherche sémantique et génération de réponses LLM"
        },
        {
            "name": "RAG", 
            "description": "Retrieval Augmented Generation - Recherche augmentée par génération"
        },
        {
            "name": "LLM",
            "description": "Large Language Model integration (OpenAI GPT)"
        },
        {
            "name": "Ingestion",
            "description": "Upload et traitement de documents"
        },
        {
            "name": "Documents", 
            "description": "Gestion des documents dans le graphe"
        },
        {
            "name": "Knowledge Graph",
            "description": "Operations sur le graphe de connaissances Neo4j"
        },
        {
            "name": "Neo4j",
            "description": "Requêtes directes sur la base de données graphe"
        },
        {
            "name": "Cypher",
            "description": "Langage de requête Neo4j pour exploration avancée"
        },
        {
            "name": "Database",
            "description": "Opérations de base de données et statistiques"
        }
    ]
)

# Gestionnaire d'erreur global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Chargement des variables d'environnement
load_dotenv('../.env', override=True)

# Configuration Neo4j
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

logger.info(f"Connecting to Neo4j at: {NEO4J_URI}")

try:
    kg = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    # Test de connexion
    test_query = "RETURN 1 as test"
    kg.query(test_query)
    logger.info("Neo4j connection successful")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    kg = None

embeddings_service = OpenAIEmbeddings(model="text-embedding-3-small")

class QueryRequest(BaseModel):
    """
    Modèle de requête pour la recherche sémantique avec LLM
    
    Attributes:
        question: Question en langage naturel (français recommandé)
        top_k: Nombre maximum de chunks à retourner (défaut: 5, max recommandé: 10)
        similarity_threshold: Seuil minimum de similarité cosinus (défaut: 0.9)
                             - 0.9-1.0: Très pertinent uniquement
                             - 0.8-0.9: Pertinent avec tolérance
                             - 0.7-0.8: Recherche large
                             - <0.7: Peut inclure du bruit
    """
    question: str = Field(
        ..., 
        description="Question en langage naturel sur le contenu des documents",
        example="Comment fonctionne le mécanisme d'attention dans les transformers ?"
    )
    top_k: int = Field(
        default=5, 
        ge=1, 
        le=20,
        description="Nombre maximum de chunks pertinents à retourner"
    )
    similarity_threshold: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0, 
        description="Seuil minimum de similarité cosinus (0.9 = très strict, 0.7 = plus permissif)"
    )

class SemanticSearchRequest(BaseModel):
    question: str
    top_k: int = 5
    similarity_threshold: float = 0.8

class CypherRequest(BaseModel):
    query: str
    params: Optional[dict] = None


@app.post("/query", 
          summary="Recherche sémantique avec génération de réponse LLM",
          description="Effectue une recherche sémantique vectorielle dans la base Neo4j et génère une réponse intelligente via ChatGPT",
          response_description="Résultats de recherche avec réponse LLM générée",
          tags=["Query", "RAG", "LLM"])
async def query_chunks(request: QueryRequest):
    """
    Recherche sémantique dans Neo4j avec génération de réponse LLM
    
    Cette endpoint combine recherche vectorielle et génération de langage naturel pour fournir
    des réponses contextualisées basées sur le contenu des documents ingérés.
    
    **Fonctionnalités:**
    - Recherche vectorielle avec embedding OpenAI (text-embedding-3-small)
    - Filtrage par seuil de similarité pour éviter les résultats non pertinents
    - Génération de réponse contextuelle via ChatGPT-3.5-turbo
    - Support multilingue (optimisé pour le français)
    
    **Paramètres:**
    - **question**: La question en langage naturel
    - **top_k**: Nombre maximum de chunks à retourner (1-10)
    - **similarity_threshold**: Seuil minimum de similarité cosinus (0.1-1.0)
    
    **Réponse:**
    - **results**: Liste des chunks trouvés avec scores de similarité
    - **llm_answer**: Réponse générée par l'IA basée sur le contexte
    - **similarity_threshold_used**: Seuil effectivement utilisé
    - **total_relevant_chunks**: Nombre de chunks pertinents trouvés
    
    **Codes d'erreur:**
    - 400: Paramètres invalides
    - 500: Erreur Neo4j ou OpenAI
    
    **Exemples d'usage:**
    ```python
    # Requête standard
    {
        "question": "Comment fonctionne l'attention dans les transformers ?",
        "top_k": 5,
        "similarity_threshold": 0.9
    }
    
    # Recherche large
    {
        "question": "Quels sont les avantages de cette technologie ?",
        "top_k": 3,
        "similarity_threshold": 0.7
    }
    ```
    """
    loop = asyncio.get_event_loop()
    
    # Vérification des index en async
    with ThreadPoolExecutor() as executor:
        indexes = await loop.run_in_executor(executor, kg.query, "SHOW INDEXES")
    
    vector_indexes = [idx for idx in indexes if 'vector' in str(idx.get('type', '')).lower()]
    if not vector_indexes:
        return {"error": "No vector index found"}
    
    index_name = vector_indexes[0]['name']
    
    # Utiliser Neo4j natif pour l'embedding de la question
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_endpoint = "https://api.openai.com/v1/embeddings"
    
    vector_search_query = f"""
    WITH genai.vector.encode(
        $question, 
        "OpenAI", 
        {{
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }}) AS question_embedding
    CALL db.index.vector.queryNodes('GrahRAG', $top_k, question_embedding)
    YIELD node, score
    WHERE score >= $similarity_threshold
    RETURN score, node.text AS text, node.filename AS source, node.id AS chunk_id
    ORDER BY score DESC
    """
    
    # Exécution de la recherche vectorielle en async
    with ThreadPoolExecutor() as executor:
        results = await loop.run_in_executor(
            executor,
            kg.query,
            vector_search_query,
            {
                'question': request.question,
                'openAiApiKey': openai_api_key,
                'openAiEndpoint': openai_endpoint,
                'top_k': request.top_k,
                'similarity_threshold': request.similarity_threshold
            }
        )
    
    # Vérifier si nous avons des résultats pertinents
    if not results:
        return {
            "results": [], 
            "llm_answer": f"Désolé, aucun contenu pertinent trouvé pour votre question avec un seuil de similarité de {request.similarity_threshold}. Essayez de reformuler votre question ou de réduire le seuil de similarité.",
            "similarity_threshold_used": request.similarity_threshold,
            "no_results_reason": "Aucun résultat ne dépasse le seuil de similarité minimum"
        }
    
    # Construire le contexte pour le LLM
    context = "\n\n".join([r['text'] for r in results])
    
    # Génération de réponse LLM en async
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = f"Voici le contexte extrait de documents:\n{context}\n\nQuestion utilisateur: {request.question}\n\nRéponds de façon concise et précise."
    
    # Exécution LLM en async
    with ThreadPoolExecutor() as executor:
        llm_response = await loop.run_in_executor(
            executor,
            llm.invoke,
            prompt
        )
    
    return {
        "results": results, 
        "llm_answer": llm_response.content,
        "similarity_threshold_used": request.similarity_threshold,
        "total_relevant_chunks": len(results)
    }

@app.post("/ingest_file",
          summary="Ingestion de document dans le graphe de connaissances",
          description="Traite et ingère un fichier dans Neo4j avec création automatique d'embeddings et relations",
          response_description="Statut de l'ingestion avec statistiques de création",
          tags=["Ingestion", "Documents", "Knowledge Graph"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingestion complète d'un fichier dans le graphe de connaissances
    
    Traite automatiquement un fichier uploadé et créé:
    - Nœuds Document et Chunk dans Neo4j
    - Embeddings vectoriels via OpenAI
    - Relations séquentielles entre chunks (NEXT_CHUNK, PREVIOUS_CHUNK)
    - Relations document-chunk (CONTAINS_CHUNK)
    
    **Formats supportés:**
    - PDF (.pdf)
    - Markdown (.md, .markdown) 
    - Texte (.txt)
    - Word (.docx)
    
    **Traitement automatique:**
    1. Chargement et parsing du document
    2. Découpage intelligent en chunks (800 caractères, overlap 80)
    3. Génération d'embeddings OpenAI (text-embedding-3-small)
    4. Création de la structure graphe dans Neo4j
    5. Relations séquentielles pour navigation
    
    **Réponse:**
    - **status**: "success" si réussi
    - **chunks_created**: Nombre de chunks créés
    - **filename**: Nom du fichier traité
    - **document_created**: Confirmation de création du document
    - **sequential_relations_created**: Nombre de relations séquentielles
    
    **Limitations:**
    - Taille max: dépend de la configuration serveur
    - Timeout: 300 secondes pour les gros fichiers
    - Formats non supportés retournent une erreur 400
    
    **Exemple de réponse:**
    ```json
    {
        "status": "success",
        "chunks_created": 45,
        "filename": "document.pdf",
        "document_created": true,
        "sequential_relations_created": 44
    }
    ```
    """
    try:
        if kg is None:
            raise HTTPException(status_code=500, detail="Neo4j connection not available")
        
        logger.info(f"Starting ingestion of file: {file.filename}")
        
        # Sauvegarder temporairement le fichier
        ext = file.filename.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        filename = file.filename
        
        # Traitement du fichier en thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            # Parallélisation du loading et splitting
            docs = await loop.run_in_executor(executor, _load_and_split_file, tmp_path, ext)
            
            if isinstance(docs, dict) and "error" in docs:
                os.remove(tmp_path)
                return docs
        
        # 1. Créer ou récupérer le nœud Document
        document_query = """
        MERGE (d:Document {filename: $filename})
        ON CREATE SET 
            d.created_at = datetime(),
            d.chunk_count = 0,
            d.file_extension = $extension,
            d.file_size = $file_size
        ON MATCH SET 
            d.updated_at = datetime()
        RETURN d
        """
        
        file_size = len(content)
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                kg.query,
                document_query,
                {
                    "filename": filename,
                    "extension": ext,
                    "file_size": file_size
                }
            )
        
        # 2. Préparer les données pour insertion par batch avec relations
        chunks_data = []
        for idx, chunk in enumerate(docs):
            chunk_text = chunk.page_content
            chunk_id = f"{filename}-{idx}"
            chunks_data.append({
                "id": chunk_id,
                "filename": filename,
                "text": chunk_text,
                "chunk_index": idx
            })
        
        # 3. Insertion par batch avec relations Document->Chunk
        batch_query = """
        MATCH (d:Document {filename: $filename})
        UNWIND $chunks_data AS chunk_data
        CREATE (c:Chunk {
            id: chunk_data.id, 
            filename: chunk_data.filename, 
            text: chunk_data.text,
            chunk_index: chunk_data.chunk_index,
            created_at: datetime()
        })
        CREATE (d)-[:CONTAINS_CHUNK {chunk_index: chunk_data.chunk_index}]->(c)
        WITH d, count(c) as chunk_count
        SET d.chunk_count = chunk_count
        """
        
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor, 
                kg.query, 
                batch_query, 
                {
                    "filename": filename,
                    "chunks_data": chunks_data
                }
            )
        
        # 4. Créer des relations séquentielles entre chunks
        sequential_relations_query = """
        MATCH (d:Document {filename: $filename})-[:CONTAINS_CHUNK]->(c:Chunk)
        WITH c ORDER BY c.chunk_index
        WITH collect(c) as chunks
        UNWIND range(0, size(chunks)-2) as i
        WITH chunks[i] as current_chunk, chunks[i+1] as next_chunk
        CREATE (current_chunk)-[:NEXT_CHUNK]->(next_chunk)
        CREATE (next_chunk)-[:PREVIOUS_CHUNK]->(current_chunk)
        """
        
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                kg.query,
                sequential_relations_query,
                {"filename": filename}
            )
        
        # 5. Générer les embeddings avec Neo4j natif
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_endpoint = "https://api.openai.com/v1/embeddings"
        
        embedding_query = """
            MATCH (chunk:Chunk) 
            WHERE chunk.filename = $filename AND chunk.textEmbedding IS NULL
            WITH chunk, genai.vector.encode(
                chunk.text, 
                "OpenAI", 
                {
                  token: $openAiApiKey,
                  endpoint: $openAiEndpoint
                }) AS vector
            CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
            RETURN count(chunk) as chunksProcessed
            """
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                kg.query,
                embedding_query,
                {
                    "filename": filename,
                    "openAiApiKey": openai_api_key, 
                    "openAiEndpoint": openai_endpoint
                }
            )
        
        os.remove(tmp_path)
        chunks_processed = result[0]['chunksProcessed'] if result else 0
        logger.info(f"Successfully ingested {chunks_processed} chunks from {filename}")
        return {
            "status": "success", 
            "chunks_created": chunks_processed, 
            "filename": filename,
            "document_created": True,
            "sequential_relations_created": max(0, chunks_processed - 1)
        }
    
    except Exception as e:
        logger.error(f"Error during file ingestion: {str(e)}")
        logger.error(traceback.format_exc())
        # Nettoyer le fichier temporaire en cas d'erreur
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")

def _load_and_split_file(tmp_path: str, ext: str):
    """Fonction helper pour le loading et splitting de fichiers"""
    # Choisir le loader adapté
    if ext == "pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext in ["md", "markdown"]:
        loader = UnstructuredMarkdownLoader(tmp_path)
    elif ext in ["txt"]:
        loader = TextLoader(tmp_path)
    elif ext in ["docx"]:
        loader = UnstructuredWordDocumentLoader(tmp_path)
    else:
        return {"error": f"Format non supporté: {ext}"}
    
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    return chunks

@app.get("/graph_structure/{filename}")
async def get_graph_structure(filename: str):
    """Obtenir la structure du graphe pour un document donné"""
    loop = asyncio.get_event_loop()
    
    structure_query = """
    MATCH (d:Document {filename: $filename})
    OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(c:Chunk)
    RETURN 
        d as document,
        collect(DISTINCT c) as chunks,
        count(c) as chunk_count
    """
    
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            kg.query,
            structure_query,
            {"filename": filename}
        )
    
    if not result:
        return {"error": "Document not found"}
    
    document_info = result[0]['document']
    chunks = result[0]['chunks']
    chunk_count = result[0]['chunk_count']
    
    # Récupérer les relations pour chaque chunk
    relationships_query = """
    MATCH (d:Document {filename: $filename})-[:CONTAINS_CHUNK]->(c:Chunk)
    OPTIONAL MATCH (c)-[r]->(related:Chunk)
    RETURN 
        c.id as chunk_id,
        collect({
            type: type(r),
            target: related.id,
            properties: properties(r)
        }) as relationships
    """
    
    with ThreadPoolExecutor() as executor:
        relationships_result = await loop.run_in_executor(
            executor,
            kg.query,
            relationships_query,
            {"filename": filename}
        )
    
    # Créer un dictionnaire des relations par chunk
    relationships_by_chunk = {}
    for rel_data in relationships_result:
        chunk_id = rel_data['chunk_id']
        relationships = [r for r in rel_data['relationships'] if r['type'] is not None]
        relationships_by_chunk[chunk_id] = relationships
    
    # Combiner les chunks avec leurs relations
    chunks_with_relationships = []
    for chunk in chunks:
        chunk_id = chunk.get('id')
        chunk_relationships = relationships_by_chunk.get(chunk_id, [])
        chunks_with_relationships.append({
            'chunk': chunk,
            'relationships': chunk_relationships
        })
    
    return {
        "document": document_info,
        "chunk_count": chunk_count,
        "chunks_with_relationships": chunks_with_relationships
    }

@app.post("/smart_inter_document_relationships")
async def smart_inter_document_relationships(
    similarity_threshold: float = 0.80,
    max_relationships_total: int = 50,
    chunk_sampling_ratio: float = 0.3
):
    """Création intelligente de relations inter-documents avec échantillonnage et optimisations"""
    try:
        if kg is None:
            raise HTTPException(status_code=500, detail="Neo4j connection not available")
        
        loop = asyncio.get_event_loop()
        
        # 1. Analyse de la distribution des chunks par document
        analysis_query = """
        MATCH (d:Document)-[:CONTAINS_CHUNK]->(c:Chunk)
        WHERE c.textEmbedding IS NOT NULL
        RETURN d.filename as Document, 
               count(c) as TotalChunks,
               collect(c.id)[0..10] as SampleChunkIds
        ORDER BY TotalChunks DESC
        """
        
        with ThreadPoolExecutor() as executor:
            analysis_result = await loop.run_in_executor(executor, kg.query, analysis_query)
        
        documents_info = {doc['Document']: doc for doc in analysis_result}
        total_chunks = sum(doc['TotalChunks'] for doc in analysis_result)
        
        logger.info(f"Analyse: {len(documents_info)} documents, {total_chunks} chunks total")
        
        # 2. Stratégie d'échantillonnage intelligent
        relations_created = 0
        max_comparisons = min(5000, total_chunks * 2)  # Limite raisonnable
        
        # 3. Utiliser une approche par batch avec échantillonnage
        smart_relations_query = """
        // Prendre un échantillon représentatif de chunks de chaque document
        MATCH (c1:Chunk) 
        WHERE c1.textEmbedding IS NOT NULL
        WITH c1.filename as doc1, collect(c1) as chunks1
        
        MATCH (c2:Chunk)
        WHERE c2.textEmbedding IS NOT NULL AND c2.filename <> doc1
        WITH doc1, chunks1, c2.filename as doc2, collect(c2) as chunks2
        WHERE doc1 < doc2  // Éviter les doublons
        
        // Échantillonnage intelligent : prendre les chunks les plus représentatifs
        UNWIND chunks1[0..toInteger(size(chunks1) * $sampling_ratio)] as chunk1
        UNWIND chunks2[0..toInteger(size(chunks2) * $sampling_ratio)] as chunk2
        
        // Calculer la similarité
        WITH chunk1, chunk2, 
             gds.similarity.cosine(chunk1.textEmbedding, chunk2.textEmbedding) as similarity
        WHERE similarity > $threshold
        AND NOT EXISTS((chunk1)-[:RELATES_TO]-(chunk2))
        
        // Limiter globalement et prioriser les meilleures similarités
        ORDER BY similarity DESC
        LIMIT $max_total
        
        // Créer les relations bidirectionnelles optimisées
        CREATE (chunk1)-[:RELATES_TO {
            similarity: similarity,
            created_at: datetime(),
            type: 'smart_inter_document',
            source_doc: chunk1.filename,
            target_doc: chunk2.filename,
            method: 'intelligent_sampling'
        }]->(chunk2)
        
        CREATE (chunk2)-[:RELATES_TO {
            similarity: similarity,
            created_at: datetime(),
            type: 'smart_inter_document',
            source_doc: chunk2.filename,
            target_doc: chunk1.filename,
            method: 'intelligent_sampling'
        }]->(chunk1)
        
        RETURN count(*) as relations_created
        """
        
        with ThreadPoolExecutor() as executor:
            smart_result = await loop.run_in_executor(
                executor,
                kg.query,
                smart_relations_query,
                {
                    "threshold": similarity_threshold,
                    "max_total": max_relationships_total,
                    "sampling_ratio": chunk_sampling_ratio
                }
            )
        
        relations_created = smart_result[0]['relations_created'] if smart_result else 0
        
        # 4. Statistiques post-création
        stats_query = """
        MATCH (c1:Chunk)-[r:RELATES_TO]->(c2:Chunk)
        WHERE c1.filename <> c2.filename
        RETURN 
            count(r) as total_inter_doc_relations,
            avg(r.similarity) as avg_similarity,
            max(r.similarity) as max_similarity,
            min(r.similarity) as min_similarity,
            count(DISTINCT c1.filename) as documents_connected
        """
        
        with ThreadPoolExecutor() as executor:
            stats_result = await loop.run_in_executor(executor, kg.query, stats_query)
        
        stats = stats_result[0] if stats_result else {}
        
        return {
            "status": "success",
            "method": "intelligent_sampling",
            "documents_analyzed": list(documents_info.keys()),
            "total_chunks_available": total_chunks,
            "sampling_ratio_used": chunk_sampling_ratio,
            "relations_created_this_run": relations_created,
            "total_inter_document_relations": stats.get('total_inter_doc_relations', 0),
            "similarity_stats": {
                "threshold_used": similarity_threshold,
                "average": round(float(stats.get('avg_similarity', 0)), 3),
                "maximum": round(float(stats.get('max_similarity', 0)), 3),
                "minimum": round(float(stats.get('min_similarity', 0)), 3)
            },
            "documents_connected": stats.get('documents_connected', 0),
            "efficiency_note": f"Processed ~{int(total_chunks * chunk_sampling_ratio * 2)} comparisons instead of {total_chunks * (total_chunks - 1) // 2}"
        }
        
    except Exception as e:
        logger.error(f"Error in smart inter-document relationships: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Smart relationship creation failed: {str(e)}")

@app.get("/graph_stats")
async def get_graph_stats():
    """Statistiques globales du Knowledge Graph"""
    loop = asyncio.get_event_loop()
    
    # Requêtes séparées pour plus de fiabilité
    documents_query = "MATCH (d:Document) RETURN count(d) as documents"
    chunks_query = "MATCH (c:Chunk) RETURN count(c) as chunks"
    relations_query = """
    MATCH ()-[r]->() 
    RETURN count(r) as relationships, collect(DISTINCT type(r)) as types
    """
    
    try:
        with ThreadPoolExecutor() as executor:
            # Exécuter les requêtes en parallèle
            doc_result = await loop.run_in_executor(executor, kg.query, documents_query)
            chunk_result = await loop.run_in_executor(executor, kg.query, chunks_query)
            rel_result = await loop.run_in_executor(executor, kg.query, relations_query)
        
        documents = doc_result[0]['documents'] if doc_result else 0
        chunks = chunk_result[0]['chunks'] if chunk_result else 0
        
        if rel_result:
            relationships = rel_result[0]['relationships']
            rel_types = [rt for rt in rel_result[0]['types'] if rt]
        else:
            relationships = 0
            rel_types = []
        
        return {
            "documents": documents,
            "chunks": chunks, 
            "relationships": relationships,
            "relationship_types": rel_types
        }
    except Exception as e:
        logger.error(f"Error getting graph stats: {str(e)}")
        return {
            "documents": 0,
            "chunks": 0,
            "relationships": 0,
            "relationship_types": []
        }

@app.post("/semantic_search_with_context")
async def semantic_search_with_context(request: SemanticSearchRequest):
    """Recherche sémantique avec contexte du graphe"""
    loop = asyncio.get_event_loop()
    
    # Recherche vectorielle
    vector_search_query = """
    WITH genai.vector.encode(
        $question, 
        "OpenAI", 
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS question_embedding
    CALL db.index.vector.queryNodes('GrahRAG', $top_k, question_embedding)
    YIELD node, score
    WHERE score >= $similarity_threshold
    RETURN score, node.text AS text, node.filename AS source, node.id AS chunk_id
    ORDER BY score DESC
    """
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_endpoint = "https://api.openai.com/v1/embeddings"
    
    with ThreadPoolExecutor() as executor:
        results = await loop.run_in_executor(
            executor,
            kg.query,
            vector_search_query,
            {
                'question': request.question,
                'openAiApiKey': openai_api_key,
                'openAiEndpoint': openai_endpoint,
                'top_k': request.top_k,
                'similarity_threshold': request.similarity_threshold
            }
        )
    
    # Enrichir avec le contexte du graphe
    enriched_chunks = []
    for chunk in results:
        chunk_id = chunk.get("chunk_id")
        
        context_query = """
        MATCH (c:Chunk {id: $chunk_id})
        OPTIONAL MATCH (c)<-[:CONTAINS_CHUNK]-(d:Document)
        OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
        OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
        OPTIONAL MATCH (c)-[:RELATES_TO]-(related:Chunk)
        RETURN 
            c as chunk,
            d as document,
            next as next_chunk,
            prev as previous_chunk,
            collect(DISTINCT related)[0..2] as related_chunks
        """
        
        with ThreadPoolExecutor() as executor:
            context_result = await loop.run_in_executor(
                executor, kg.query, context_query, {"chunk_id": chunk_id}
            )
        
        if context_result:
            context = context_result[0]
            enriched_chunk = {
                **chunk,
                "document_info": context.get("document"),
                "next_chunk": context.get("next_chunk"),
                "previous_chunk": context.get("previous_chunk"),
                "related_chunks": context.get("related_chunks", [])
            }
            enriched_chunks.append(enriched_chunk)
        else:
            enriched_chunks.append(chunk)
    
    # Génération de réponse LLM avec le contexte enrichi
    context = "\n\n".join([chunk.get('text', '') for chunk in enriched_chunks])
    
    if context.strip():  # Seulement si on a du contexte
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = f"Voici le contexte extrait de documents avec contexte graphique:\n{context}\n\nQuestion utilisateur: {request.question}\n\nRéponds de façon concise et précise."
        
        # Génération LLM en async
        with ThreadPoolExecutor() as executor:
            llm_response = await loop.run_in_executor(
                executor,
                llm.invoke,
                prompt
            )
        
        llm_answer = llm_response.content
    else:
        llm_answer = "Désolé, aucun contexte pertinent trouvé pour répondre à votre question."
    
    return {
        "chunks": enriched_chunks,
        "llm_answer": llm_answer,
        "enhanced_with_graph_context": True,
        "total_found": len(enriched_chunks)
    }

@app.get("/db_info")
async def get_db_info():
    """Obtient des informations sur la base de données et les index"""
    try:
        if kg is None:
            raise HTTPException(status_code=500, detail="Neo4j connection not available")
        
        # Index
        indexes = kg.query("SHOW INDEXES")
        
        # Statistiques générales
        stats = kg.query("""
        MATCH (n) 
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY count DESC
        """)
        
        # Vérifier l'index vectoriel
        vector_indexes = [idx for idx in indexes if 'vector' in str(idx.get('type', '')).lower()]
        vector_index_name = vector_indexes[0]['name'] if vector_indexes else None
        
        return {
            "neo4j_connection": "active",
            "indexes": indexes,
            "vector_index": vector_index_name,
            "node_statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get database info: {str(e)}")

@app.post("/cypher",
          summary="Exécution de requêtes Cypher personnalisées",
          description="Interface pour exécuter des requêtes Cypher directes sur Neo4j",
          response_description="Résultats de la requête Cypher",
          tags=["Neo4j", "Cypher", "Database"])
def run_cypher(request: CypherRequest = None):
    """
    Exécute une requête Cypher personnalisée sur la base Neo4j
    
    Interface flexible pour l'exploration et manipulation directe de la base de données.
    Utile pour le debug, l'analyse des données et les opérations avancées.
    
    **Fonctionnalités:**
    - Exécution de requêtes Cypher arbitraires
    - Support des paramètres de requête
    - Requête par défaut si aucune fournie (topologie DB)
    - Gestion d'erreurs avec détails
    
    **Paramètres:**
    - **query**: Requête Cypher à exécuter (optionnel)
    - **params**: Dictionnaire de paramètres pour la requête (optionnel)
    
    **Exemples de requêtes utiles:**
    ```cypher
    // Statistiques générales
    MATCH (n) RETURN labels(n)[0] as Type, count(n) as Count
    
    // Documents et leurs chunks
    MATCH (d:Document)-[:CONTAINS_CHUNK]->(c:Chunk) 
    RETURN d.filename, count(c) as chunks
    
    // Relations de similarité
    MATCH (c1:Chunk)-[r:RELATES_TO]->(c2:Chunk) 
    WHERE c1.filename <> c2.filename 
    RETURN c1.filename, c2.filename, r.similarity 
    ORDER BY r.similarity DESC LIMIT 10
    
    // Recherche de contenu
    MATCH (c:Chunk) 
    WHERE c.text CONTAINS $searchTerm 
    RETURN c.filename, c.text LIMIT 5
    ```
    
    **Sécurité:**
    - Aucune restriction sur les requêtes (utiliser avec précaution)
    - Accès complet à la base de données
    - Recommandé pour développement et debug uniquement
    
    **Comportement par défaut:**
    Si aucune requête n'est fournie, retourne `CALL db.schema.visualization()`
    pour visualiser la structure de la base.
    """
    try:
        if request is None or not request.query:
            # Requête par défaut : topologie de la base
            cypher = "CALL db.schema.visualization()"
            result = kg.query(cypher)
            return {"result": result, "default_query": cypher}
        result = kg.query(request.query, params=request.params or {})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/document/{filename}")
def delete_document(filename: str):
    """Supprime un document et tous ses chunks associés"""
    try:
        # Requête pour supprimer le document et ses chunks
        delete_query = """
        MATCH (d:Document {filename: $filename})
        OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)-[r]-()
        WITH d, collect(DISTINCT c) as chunks, collect(DISTINCT r) as relations
        
        // Supprimer d'abord les relations des chunks
        FOREACH(rel in relations | DELETE rel)
        
        // Supprimer les chunks
        FOREACH(chunk in chunks | DELETE chunk)
        
        // Supprimer le document
        DELETE d
        
        RETURN count(chunks) as deleted_chunks, count(relations) as deleted_relations
        """
        
        result = kg.query(delete_query, params={"filename": filename})
        
        if result:
            deleted_chunks = result[0].get('deleted_chunks', 0)
            deleted_relations = result[0].get('deleted_relations', 0)
            
            return {
                "success": True,
                "message": f"Document '{filename}' supprimé avec succès",
                "deleted_chunks": deleted_chunks,
                "deleted_relations": deleted_relations,
                "filename": filename
            }
        else:
            return {
                "success": False,
                "message": f"Document '{filename}' non trouvé",
                "filename": filename
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la suppression du document {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression: {str(e)}")

@app.get("/")
def root():
    return {"message": "KnowledgeGraphRag API is running"}

@app.get("/health")
def health_check():
    """Route de vérification de santé de l'API"""
    try:
        # Vérifier la connexion Neo4j
        result = kg.query("RETURN 1 as test")
        neo4j_status = "connected" if result else "disconnected"
        
        return {
            "status": "healthy",
            "neo4j": neo4j_status,
            "timestamp": time.time(),
            "message": "API is running and Neo4j is accessible"
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "neo4j": "error",
            "error": str(e),
            "timestamp": time.time(),
            "message": "API is running but Neo4j connection failed"
        }
