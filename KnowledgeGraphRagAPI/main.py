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
    similarity_threshold: float = 0.9  # Seuil minimum de similarité

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
    similarity_threshold: float = 0.8

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
async def discover_semantic_relationships(similarity_threshold: float = 0.80, max_relationships_per_chunk: int = 5):
    """Découvrir automatiquement des relations sémantiques RELATES_TO entre chunks - optimisé pour relations inter-documents"""
    loop = asyncio.get_event_loop()
    
    # Version améliorée pour forcer les relations inter-documents
    # Seuil abaissé à 0.75 et limite augmentée pour plus de relations
    similarity_query = """
    // Prendre TOUS les chunks, pas seulement 30
    MATCH (c1:Chunk)
    WHERE c1.textEmbedding IS NOT NULL
    
    // Pour chaque chunk, trouver les chunks similaires d'AUTRES documents
    MATCH (c2:Chunk)
    WHERE c2.id <> c1.id 
    AND c2.textEmbedding IS NOT NULL
    AND c1.filename <> c2.filename  // FORCER les relations inter-documents
    AND NOT EXISTS((c1)-[:RELATES_TO]-(c2))
    
    // Calculer la similarité cosinus
    WITH c1, c2, gds.similarity.cosine(c1.textEmbedding, c2.textEmbedding) AS similarity
    WHERE similarity > $threshold
    
    // Trier par similarité et limiter par chunk source
    ORDER BY c1.id, similarity DESC
    WITH c1, collect({chunk: c2, similarity: similarity})[0..$max_per_chunk] as related_chunks
    
    // Créer les relations bidirectionnelles pour chaque paire
    UNWIND related_chunks as related_data
    WITH c1, related_data.chunk as c2, related_data.similarity as sim_score
    
    // Relation dans les deux sens avec métadonnées enrichies
    CREATE (c1)-[:RELATES_TO {
        similarity: sim_score, 
        created_at: datetime(), 
        type: 'inter_document_semantic',
        source_doc: c1.filename,
        target_doc: c2.filename
    }]->(c2)
    
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

@app.post("/force_inter_document_relationships")
async def force_inter_document_relationships(
    similarity_threshold: float = 0.80, 
    max_relationships_per_document_pair: int = 3,
    batch_size: int = 50
):
    """Force la création de relations inter-documents en comparant tous les documents par paires"""
    try:
        if kg is None:
            raise HTTPException(status_code=500, detail="Neo4j connection not available")
        
        loop = asyncio.get_event_loop()
        
        # D'abord, obtenir la liste des documents
        docs_query = "MATCH (d:Document) RETURN d.filename as filename ORDER BY filename"
        with ThreadPoolExecutor() as executor:
            docs_result = await loop.run_in_executor(executor, kg.query, docs_query)
        
        documents = [doc['filename'] for doc in docs_result]
        logger.info(f"Documents trouvés: {documents}")
        
        total_relations_created = 0
        
        # Comparer chaque paire de documents
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents):
                if i < j:  # Éviter les doublons et l'auto-comparaison
                    logger.info(f"Comparaison {doc1} <-> {doc2}")
                    
                    # Requête spécifique pour cette paire de documents
                    inter_doc_query = """
                    // Chunks du document 1
                    MATCH (c1:Chunk {filename: $doc1})
                    WHERE c1.textEmbedding IS NOT NULL
                    
                    // Chunks du document 2  
                    MATCH (c2:Chunk {filename: $doc2})
                    WHERE c2.textEmbedding IS NOT NULL
                    AND NOT EXISTS((c1)-[:RELATES_TO]-(c2))
                    
                    // Calculer similarité
                    WITH c1, c2, gds.similarity.cosine(c1.textEmbedding, c2.textEmbedding) AS similarity
                    WHERE similarity > $threshold
                    
                    // Prendre les meilleures relations pour cette paire de documents
                    ORDER BY similarity DESC
                    LIMIT $max_relations
                    
                    // Créer les relations bidirectionnelles
                    CREATE (c1)-[:RELATES_TO {
                        similarity: similarity,
                        created_at: datetime(),
                        type: 'forced_inter_document',
                        source_doc: $doc1,
                        target_doc: $doc2,
                        method: 'pairwise_comparison'
                    }]->(c2)
                    
                    CREATE (c2)-[:RELATES_TO {
                        similarity: similarity,
                        created_at: datetime(),
                        type: 'forced_inter_document',
                        source_doc: $doc2,
                        target_doc: $doc1,
                        method: 'pairwise_comparison'
                    }]->(c1)
                    
                    RETURN count(*) as relations_created_for_pair
                    """
                    
                    with ThreadPoolExecutor() as executor:
                        pair_result = await loop.run_in_executor(
                            executor,
                            kg.query,
                            inter_doc_query,
                            {
                                "doc1": doc1,
                                "doc2": doc2,
                                "threshold": similarity_threshold,
                                "max_relations": max_relationships_per_document_pair * 2  # *2 car bidirectionnel
                            }
                        )
                    
                    relations_for_pair = pair_result[0]['relations_created_for_pair'] if pair_result else 0
                    total_relations_created += relations_for_pair
                    logger.info(f"Relations créées pour {doc1} <-> {doc2}: {relations_for_pair}")
        
        return {
            "status": "success",
            "documents_compared": documents,
            "document_pairs_processed": len(documents) * (len(documents) - 1) // 2,
            "total_inter_document_relations_created": total_relations_created,
            "similarity_threshold": similarity_threshold,
            "max_relations_per_pair": max_relationships_per_document_pair,
            "method": "forced_pairwise_document_comparison"
        }
        
    except Exception as e:
        logger.error(f"Error forcing inter-document relationships: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to force inter-document relationships: {str(e)}")

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

@app.post("/hybrid_relationship_strategy")
async def hybrid_relationship_strategy(
    similarity_threshold: float = 0.65,
    top_k_per_chunk: int = 2,
    use_clustering: bool = True
):
    """Stratégie hybride : clustering + top-K + optimisations"""
    try:
        if kg is None:
            raise HTTPException(status_code=500, detail="Neo4j connection not available")
        
        loop = asyncio.get_event_loop()
        
        if use_clustering:
            # Approche par clustering sémantique
            clustering_query = """
            // 1. Identifier les "centres" de chaque document (chunks les plus représentatifs)
            MATCH (d:Document)-[:CONTAINS_CHUNK]->(c:Chunk)
            WHERE c.textEmbedding IS NOT NULL
            WITH d.filename as doc, collect(c) as chunks
            
            // 2. Pour chaque document, prendre quelques chunks centraux
            UNWIND chunks[0..5] as center_chunk  // Top 5 chunks par document
            
            // 3. Comparer ces centres avec les centres des autres documents
            MATCH (other_chunk:Chunk)
            WHERE other_chunk.textEmbedding IS NOT NULL 
            AND other_chunk.filename <> doc
            AND NOT EXISTS((center_chunk)-[:RELATES_TO]-(other_chunk))
            
            // 4. Calcul de similarité optimisé
            WITH center_chunk, other_chunk, 
                 gds.similarity.cosine(center_chunk.textEmbedding, other_chunk.textEmbedding) as sim
            WHERE sim > $threshold
            
            // 5. Top-K par chunk source
            ORDER BY center_chunk.id, sim DESC
            WITH center_chunk, collect({chunk: other_chunk, similarity: sim})[0..$top_k] as top_relations
            
            // 6. Création des relations
            UNWIND top_relations as rel_data
            CREATE (center_chunk)-[:RELATES_TO {
                similarity: rel_data.similarity,
                created_at: datetime(),
                type: 'hybrid_clustering',
                method: 'document_centers'
            }]->(rel_data.chunk)
            
            RETURN count(*) as relations_created
            """
        else:
            # Approche directe optimisée
            clustering_query = """
            MATCH (c1:Chunk)
            WHERE c1.textEmbedding IS NOT NULL
            WITH c1 LIMIT 100  // Limiter le nombre de chunks sources
            
            MATCH (c2:Chunk)
            WHERE c2.textEmbedding IS NOT NULL 
            AND c2.filename <> c1.filename
            AND NOT EXISTS((c1)-[:RELATES_TO]-(c2))
            
            WITH c1, c2, gds.similarity.cosine(c1.textEmbedding, c2.textEmbedding) as sim
            WHERE sim > $threshold
            
            ORDER BY c1.id, sim DESC
            WITH c1, collect({chunk: c2, similarity: sim})[0..$top_k] as top_similar
            
            UNWIND top_similar as similar_data
            CREATE (c1)-[:RELATES_TO {
                similarity: similar_data.similarity,
                created_at: datetime(),
                type: 'hybrid_direct',
                method: 'top_k_similarity'
            }]->(similar_data.chunk)
            
            RETURN count(*) as relations_created
            """
        
        with ThreadPoolExecutor() as executor:
            hybrid_result = await loop.run_in_executor(
                executor,
                kg.query,
                clustering_query,
                {
                    "threshold": similarity_threshold,
                    "top_k": top_k_per_chunk
                }
            )
        
        relations_created = hybrid_result[0]['relations_created'] if hybrid_result else 0
        
        return {
            "status": "success",
            "method": "clustering" if use_clustering else "direct_top_k",
            "relations_created": relations_created,
            "similarity_threshold": similarity_threshold,
            "top_k_per_chunk": top_k_per_chunk,
            "clustering_used": use_clustering
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid relationship strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid strategy failed: {str(e)}")

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
