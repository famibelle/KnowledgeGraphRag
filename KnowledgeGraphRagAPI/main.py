from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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

app = FastAPI(title="KnowledgeGraphRag API")

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
    question: str
    top_k: int = 5

class IngestRequest(BaseModel):
    filename: str
    chunks: List[str]
    metadata: Optional[dict] = None

class RelationshipRequest(BaseModel):
    source_chunk_id: str
    target_chunk_id: str
    relationship_type: str
    properties: Optional[dict] = None

class GraphTraversalRequest(BaseModel):
    traversal_type: str  # "shortest_path", "related_chunks", "document_flow"
    start_chunk_id: str
    end_chunk_id: Optional[str] = None
    max_results: Optional[int] = 10

class SemanticSearchRequest(BaseModel):
    question: str
    top_k: int = 5
    similarity_threshold: float = 0.7

class CypherRequest(BaseModel):
    query: str
    params: Optional[dict] = None


@app.post("/query")
async def query_chunks(request: QueryRequest):
    """Recherche sémantique dans Neo4j et génération de réponse LLM (async)"""
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
                'top_k': request.top_k
            }
        )
    
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
    
    return {"results": results, "llm_answer": llm_response.content}

@app.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingestion d'un fichier (pdf, md, txt, docx) en chunks dans Neo4j avec structure de graphe (async)"""
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

@app.post("/create_relationship")
async def create_relationship(request: RelationshipRequest):
    """Créer une relation personnalisée entre deux chunks"""
    loop = asyncio.get_event_loop()
    
    relationship_query = f"""
    MATCH (source:Chunk {{id: $source_id}})
    MATCH (target:Chunk {{id: $target_id}})
    CREATE (source)-[r:{request.relationship_type}]->(target)
    SET r += $properties
    RETURN r
    """
    
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            kg.query,
            relationship_query,
            {
                "source_id": request.source_chunk_id,
                "target_id": request.target_chunk_id,
                "properties": request.properties or {}
            }
        )
    
    return {
        "status": "success",
        "relationship_created": len(result) > 0,
        "relationship_type": request.relationship_type
    }

@app.post("/discover_semantic_relationships")
async def discover_semantic_relationships(similarity_threshold: float = 0.85, max_relationships_per_chunk: int = 3):
    """Découvrir automatiquement des relations sémantiques RELATES_TO entre chunks (SIMILAR_TO supprimé)"""
    loop = asyncio.get_event_loop()
    
    # Trouver des chunks avec des relations sémantiques fortes via leurs embeddings
    # Note: SIMILAR_TO relation supprimée - utilisation de RELATES_TO seulement pour éviter le bruit
    similarity_query = """
    MATCH (c1:Chunk)
    WHERE c1.textEmbedding IS NOT NULL
    WITH c1
    LIMIT 30
    MATCH (c2:Chunk)
    WHERE c2.id <> c1.id 
    AND c2.textEmbedding IS NOT NULL
    AND c1.filename <> c2.filename
    AND NOT EXISTS((c1)-[:RELATES_TO]-(c2))
    WITH c1, c2, gds.similarity.cosine(c1.textEmbedding, c2.textEmbedding) AS similarity
    WHERE similarity > $threshold
    ORDER BY c1.id, similarity DESC
    WITH c1, collect({chunk: c2, similarity: similarity})[0..$max_per_chunk] as related_chunks
    UNWIND related_chunks as related_data
    WITH c1, related_data.chunk as c2, related_data.similarity as sim_score
    CREATE (c1)-[:RELATES_TO {similarity: sim_score, created_at: datetime(), type: 'semantic'}]->(c2)
    RETURN count(*) as relationships_created
    """
    
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            kg.query,
            similarity_query,
            {
                "threshold": similarity_threshold,
                "max_per_chunk": max_relationships_per_chunk
            }
        )
    
    relationships_created = result[0]['relationships_created'] if result else 0
    
    return {
        "status": "success",
        "semantic_relationships_created": relationships_created,
        "similarity_threshold": similarity_threshold,
        "max_relationships_per_chunk": max_relationships_per_chunk,
        "note": "SIMILAR_TO relations removed - using RELATES_TO between documents only"
    }

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

@app.post("/cleanup_relationships")
async def cleanup_relationships(max_similar_per_chunk: int = 5):
    """Supprimer toutes les relations SIMILAR_TO et nettoyer les autres relations"""
    try:
        if kg is None:
            raise HTTPException(status_code=500, detail="Neo4j connection not available")
        
        loop = asyncio.get_event_loop()
        
        # Compter les relations SIMILAR_TO avant suppression
        count_similar_before = kg.query("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count")[0]['count']
        
        # Supprimer TOUTES les relations SIMILAR_TO
        delete_similar_query = """
        MATCH ()-[r:SIMILAR_TO]->()
        DELETE r
        RETURN count(*) as deleted_similar
        """
        
        with ThreadPoolExecutor() as executor:
            similar_result = await loop.run_in_executor(
                executor,
                kg.query,
                delete_similar_query
            )
        
        deleted_similar = similar_result[0]['deleted_similar'] if similar_result else 0
        
        # Nettoyer les relations RELATES_TO en excès si nécessaire
        cleanup_relates_query = """
        MATCH (c:Chunk)-[r:RELATES_TO]->(related:Chunk)
        WITH c, r, related
        ORDER BY c.id, coalesce(r.similarity, 0) DESC
        WITH c, collect({rel: r, target: related, similarity: coalesce(r.similarity, 0)}) as relationships
        WHERE size(relationships) > $max_keep
        UNWIND relationships[$max_keep..] as delete_rel
        DELETE delete_rel.rel
        RETURN count(*) as deleted_relates
        """
        
        with ThreadPoolExecutor() as executor:
            relates_result = await loop.run_in_executor(
                executor,
                kg.query,
                cleanup_relates_query,
                {"max_keep": max_similar_per_chunk}
            )
        
        deleted_relates = relates_result[0]['deleted_relates'] if relates_result else 0
        
        return {
            "status": "success",
            "similar_to_before": count_similar_before,
            "similar_to_deleted": deleted_similar,
            "relates_to_cleaned": deleted_relates,
            "total_deleted": deleted_similar + deleted_relates,
            "note": "All SIMILAR_TO relations removed permanently"
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up relationships: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Relationship cleanup failed: {str(e)}")

@app.get("/graph_stats")
async def get_graph_stats():
    """Statistiques globales du Knowledge Graph"""
    loop = asyncio.get_event_loop()
    
    stats_query = """
    MATCH (d:Document) 
    OPTIONAL MATCH (c:Chunk)
    WITH count(DISTINCT d) as total_documents, count(DISTINCT c) as total_chunks
    MATCH ()-[r]->()
    RETURN 
        total_documents,
        total_chunks,
        count(r) as total_relationships,
        collect(DISTINCT type(r)) as relationship_types
    """
    
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, kg.query, stats_query)
    
    if result:
        stats = result[0]
        return {
            "total_documents": stats['total_documents'],
            "total_chunks": stats['total_chunks'], 
            "total_relationships": stats['total_relationships'],
            "relationship_types": [rt for rt in stats['relationship_types'] if rt]
        }
    
    return {"error": "Could not retrieve stats"}

@app.post("/graph_traversal")
async def graph_traversal(request: GraphTraversalRequest):
    """Parcourir le graphe pour trouver des chemins et des connections"""
    loop = asyncio.get_event_loop()
    
    if request.traversal_type == "shortest_path":
        query = """
        MATCH path = shortestPath((start:Chunk {id: $start_id})-[*]-(end:Chunk {id: $end_id}))
        RETURN path, length(path) as path_length
        """
        params = {"start_id": request.start_chunk_id, "end_id": request.end_chunk_id}
        
    elif request.traversal_type == "related_chunks":
        query = """
        MATCH (start:Chunk {id: $start_id})-[r*1..3]-(related:Chunk)
        WHERE start <> related
        RETURN DISTINCT related, 
               [rel in r | {type: type(rel), properties: properties(rel)}] as relationship_path,
               length(r) as distance
        ORDER BY distance
        LIMIT $limit
        """
        params = {
            "start_id": request.start_chunk_id, 
            "limit": request.max_results or 10
        }
        
    elif request.traversal_type == "document_flow":
        query = """
        MATCH (start:Chunk {id: $start_id})
        MATCH (start)-[:NEXT_CHUNK*0..10]->(following:Chunk)
        MATCH (preceding:Chunk)-[:NEXT_CHUNK*0..10]->(start)
        RETURN 
            collect(DISTINCT preceding) as preceding_chunks,
            start as current_chunk,
            collect(DISTINCT following) as following_chunks
        """
        params = {"start_id": request.start_chunk_id}
        
    else:
        return {"error": "Invalid traversal type"}
    
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, kg.query, query, params)
    
    return {
        "traversal_type": request.traversal_type,
        "results": result
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
    
    return {
        "chunks": enriched_chunks,
        "enhanced_with_graph_context": True,
        "total_found": len(enriched_chunks)
    }

@app.post("/remove_similar_to_relations")
async def remove_similar_to_relations():
    """Supprimer toutes les relations SIMILAR_TO de la base de données"""
    try:
        if kg is None:
            raise HTTPException(status_code=500, detail="Neo4j connection not available")
        
        loop = asyncio.get_event_loop()
        
        # Compter les relations SIMILAR_TO avant suppression
        count_query = "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count"
        count_before = kg.query(count_query)[0]['count']
        
        # Supprimer toutes les relations SIMILAR_TO
        delete_query = """
        MATCH ()-[r:SIMILAR_TO]->()
        DELETE r
        RETURN count(*) as deleted
        """
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, kg.query, delete_query)
        
        deleted_count = result[0]['deleted'] if result else 0
        
        return {
            "status": "success",
            "similar_to_relations_before": count_before,
            "similar_to_relations_deleted": deleted_count,
            "message": "All SIMILAR_TO relations have been permanently removed from the knowledge graph"
        }
        
    except Exception as e:
        logger.error(f"Error removing SIMILAR_TO relations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to remove SIMILAR_TO relations: {str(e)}")

@app.post("/initialize_db")
async def initialize_db():
    """Initialise la base de données avec les index nécessaires"""
    try:
        if kg is None:
            raise HTTPException(status_code=500, detail="Neo4j connection not available")
        
        logger.info("Initializing database...")
        
        # Vérifier les index existants
        indexes = kg.query("SHOW INDEXES")
        vector_indexes = [idx for idx in indexes if 'vector' in str(idx.get('type', '')).lower()]
        
        if not vector_indexes:
            # Créer l'index vectoriel s'il n'existe pas
            create_index_query = """
            CREATE VECTOR INDEX GrahRAG FOR (c:Chunk) ON (c.textEmbedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 1536,
             `vector.similarity_function`: 'cosine'
            }}
            """
            kg.query(create_index_query)
            logger.info("Vector index 'GrahRAG' created")
        
        # Retourner les informations sur les index
        updated_indexes = kg.query("SHOW INDEXES")
        return {
            "status": "success",
            "message": "Database initialized",
            "indexes": updated_indexes
        }
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

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

@app.post("/cypher")
def run_cypher(request: CypherRequest = None):
    """Exécute une requête Cypher arbitraire sur Neo4j. Par défaut, retourne la topologie de la base."""
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

@app.get("/graph_visualization")
async def get_graph_visualization_data(
    relation_types: str = "RELATES_TO,NEXT",
    min_similarity: float = 0.0,
    max_nodes: int = 100
):
    """Récupérer les données formatées pour visualisation du graphe inter-documents"""
    try:
        loop = asyncio.get_event_loop()
        
        # Construire la liste des types de relations à inclure
        rel_types = [rt.strip() for rt in relation_types.split(",") if rt.strip()]
        rel_filter = " OR ".join([f"type(r) = '{rt}'" for rt in rel_types])
        
        # Requête pour récupérer les relations inter-documents et les documents connectés
        graph_query = f"""
        // Récupérer les relations inter-documents via chunks
        MATCH (c1:Chunk)-[r]->(c2:Chunk)
        WHERE c1.filename <> c2.filename
        AND ({rel_filter})
        AND coalesce(r.similarity, 0.5) >= $min_similarity
        WITH c1, c2, r, type(r) as rel_type
        ORDER BY coalesce(r.similarity, 0.5) DESC
        LIMIT 200
        
        // Construire la liste des documents connectés
        WITH collect({{
            source: c1.filename,
            target: c2.filename,
            type: rel_type,
            similarity: coalesce(r.similarity, 0.5),
            weight: coalesce(r.similarity * 10, 5)
        }}) as relationships
        
        // Obtenir les documents uniques avec leurs métadonnées
        UNWIND relationships as rel
        WITH relationships, [rel.source, rel.target] as doc_names
        UNWIND doc_names as doc_name
        WITH relationships, collect(DISTINCT doc_name) as unique_docs
        
        MATCH (d:Document)
        WHERE d.filename IN unique_docs
        OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(c:Chunk)
        WITH relationships, d, count(c) as chunk_count
        ORDER BY d.filename
        LIMIT $max_nodes
        
        RETURN relationships,
               collect({{
                   id: d.filename,
                   label: d.filename,
                   type: 'Document',
                   chunk_count: chunk_count,
                   size: chunk_count * 2 + 10
               }}) as documents
        """
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                kg.query,
                graph_query,
                {
                    "min_similarity": min_similarity,
                    "max_nodes": max_nodes
                }
            )
        
        if result:
            relationships = result[0].get('relationships', [])
            documents = result[0].get('documents', [])
            
            return {
                "nodes": documents,
                "edges": relationships,
                "stats": {
                    "total_nodes": len(documents),
                    "total_edges": len(relationships),
                    "relation_types": list(set(rel['type'] for rel in relationships)) if relationships else []
                }
            }
        else:
            return {"nodes": [], "edges": [], "stats": {"total_nodes": 0, "total_edges": 0, "relation_types": []}}
            
    except Exception as e:
        logger.error(f"Error getting graph visualization data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get graph data: {str(e)}")

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
