import streamlit as st
import requests
import json
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from streamlit_agraph import agraph, Node, Edge, Config

# Configuration de la page
st.set_page_config(
    page_title="Knowledge Graph RAG Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration de l'API
API_BASE_URL = "http://localhost:8000"

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .query-section {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .info-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_connection():
    """Vérifier la connexion à l'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_db_info():
    """Récupérer les informations de la base de données"""
    try:
        response = requests.get(f"{API_BASE_URL}/db_info")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_graph_stats():
    """Récupérer les statistiques du graphe"""
    try:
        response = requests.get(f"{API_BASE_URL}/graph_stats")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_files_in_database():
    """Récupérer la liste des fichiers en base de données"""
    try:
        response = requests.post(f"{API_BASE_URL}/cypher", json={
            'query': '''
            MATCH (d:Document) 
            OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(c:Chunk)
            WITH d, count(c) as chunk_count
            RETURN d.filename as filename, 
                   d.file_size as file_size,
                   d.created_at as created_at,
                   chunk_count
            ORDER BY d.created_at DESC
            '''
        })
        if response.status_code == 200:
            return response.json().get('result', [])
        return []
    except:
        return []

def get_graph_visualization_data(relation_types="RELATES_TO,NEXT", min_similarity=0.0, max_nodes=100, include_isolated=False):
    """Récupérer les données pour la visualisation du graphe via requêtes Cypher"""
    try:
        # Construire la liste des types de relations
        rel_types = [rt.strip() for rt in relation_types.split(",") if rt.strip()]
        rel_filter = " OR ".join([f"type(r) = '{rt}'" for rt in rel_types])
        
        # Requête optimisée pour récupérer les relations inter-documents (évite le produit cartésien)
        relations_query = f"""
        MATCH (d1:Document)-[:CONTAINS_CHUNK]->(c1:Chunk)-[r]->(c2:Chunk)<-[:CONTAINS_CHUNK]-(d2:Document)
        WHERE d1.filename <> d2.filename
        AND ({rel_filter})
        AND coalesce(r.similarity, 0.5) >= {min_similarity}
        RETURN DISTINCT d1.filename as source, d2.filename as target, 
               type(r) as type, coalesce(r.similarity, 0.5) as similarity
        ORDER BY similarity DESC
        LIMIT 200
        """
        
        relations_response = requests.post(f"{API_BASE_URL}/cypher", json={'query': relations_query})
        if relations_response.status_code != 200:
            return None
            
        relations_data = relations_response.json().get('result', [])
        
        if not relations_data:
            return {"nodes": [], "edges": [], "stats": {"total_nodes": 0, "total_edges": 0, "relation_types": []}}
        
        # Obtenir les documents uniques connectés
        connected_docs = set()
        for rel in relations_data:
            connected_docs.add(rel['source'])
            connected_docs.add(rel['target'])
        
        # Si include_isolated=True, ajouter TOUS les documents
        if include_isolated:
            all_docs_query = """
            MATCH (d:Document)
            RETURN d.filename as filename
            """
            all_docs_response = requests.post(f"{API_BASE_URL}/cypher", json={'query': all_docs_query})
            if all_docs_response.status_code == 200:
                all_docs_data = all_docs_response.json().get('result', [])
                all_docs = {doc['filename'] for doc in all_docs_data}
                connected_docs.update(all_docs)
        
        # Requête pour obtenir les métadonnées des documents
        doc_names = list(connected_docs)[:max_nodes]
        docs_query = f"""
        MATCH (d:Document)
        WHERE d.filename IN {doc_names}
        OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(c:Chunk)
        RETURN d.filename as filename, count(c) as chunk_count
        ORDER BY d.filename
        """
        
        docs_response = requests.post(f"{API_BASE_URL}/cypher", json={'query': docs_query})
        if docs_response.status_code != 200:
            return None
            
        docs_data = docs_response.json().get('result', [])
        
        # Identifier les documents connectés vs isolés
        connected_in_relations = set()
        for rel in relations_data:
            connected_in_relations.add(rel['source'])
            connected_in_relations.add(rel['target'])
        
        # Formater les données pour la visualisation
        nodes = []
        for doc in docs_data:
            is_isolated = doc['filename'] not in connected_in_relations
            nodes.append({
                "id": doc['filename'],
                "label": doc['filename'],
                "type": "Document",
                "chunk_count": doc['chunk_count'],
                "size": max(15, min(50, doc['chunk_count'] * 2)),
                "isolated": is_isolated,
                "color": "#FFB6C1" if is_isolated else "#87CEEB"  # Rose pour isolés, bleu pour connectés
            })
        
        edges = [
            {
                "source": rel['source'],
                "target": rel['target'],
                "type": rel['type'],
                "similarity": rel['similarity'],
                "weight": max(1, min(8, rel['similarity'] * 10))
            }
            for rel in relations_data
            if rel['source'] in doc_names and rel['target'] in doc_names
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "relation_types": list(set(rel['type'] for rel in edges))
            }
        }
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données de visualisation: {str(e)}")
        return None

def ingest_file(file):
    """Ingérer un fichier via l'API"""
    try:
        files = {'file': (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/ingest_file", files=files, timeout=120)
        return response.status_code, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return 500, str(e)

def query_knowledge_graph(question, top_k=5):
    """Effectuer une requête sur le knowledge graph"""
    try:
        data = {"question": question, "top_k": top_k}
        response = requests.post(f"{API_BASE_URL}/query", json=data, timeout=60)
        return response.status_code, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return 500, str(e)

def semantic_search_with_context(question, top_k=5, similarity_threshold=0.6):
    """Recherche sémantique avec contexte de graphe"""
    try:
        data = {
            "question": question, 
            "top_k": top_k, 
            "similarity_threshold": similarity_threshold
        }
        response = requests.post(f"{API_BASE_URL}/semantic_search_with_context", json=data, timeout=60)
        return response.status_code, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return 500, str(e)

def discover_semantic_relationships(similarity_threshold=0.8, max_per_chunk=5):
    """Découvrir des relations sémantiques"""
    try:
        params = {
            "similarity_threshold": similarity_threshold,
            "max_relationships_per_chunk": max_per_chunk
        }
        response = requests.post(f"{API_BASE_URL}/discover_semantic_relationships", params=params, timeout=120)
        return response.status_code, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return 500, str(e)

# Interface principale
def main():
    # Titre principal
    st.markdown('<h1 class="main-header">🧠 Knowledge Graph RAG Interface</h1>', unsafe_allow_html=True)
    
    # Vérification de la connexion API
    if not check_api_connection():
        st.error("❌ Impossible de se connecter à l'API. Assurez-vous que l'API est démarrée sur http://localhost:8000")
        st.stop()
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("📊 État du Système")
        
        # Informations de la DB
        db_info = get_db_info()
        if db_info:
            st.success(f"✅ Neo4j: {db_info.get('neo4j_connection', 'N/A')}")
            st.info(f"🔍 Index vectoriel: {db_info.get('vector_index', 'N/A')}")
        
        # Statistiques du graphe
        stats = get_graph_stats()
        if stats:
            st.subheader("📈 Statistiques")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get('total_documents', 0))
                st.metric("Relations", f"{stats.get('total_relationships', 0):,}")
            with col2:
                st.metric("Chunks", stats.get('total_chunks', 0))
                rel_types = stats.get('relationship_types', [])
                st.metric("Types Relations", len(rel_types))
        
        st.markdown("---")
        st.subheader("🔧 Actions Rapides")
        if st.button("🧹 Nettoyer Relations"):
            with st.spinner("Nettoyage en cours..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/cleanup_relationships?max_similar_per_chunk=3")
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Supprimé {result.get('relationships_deleted', 0)} relations")
                    else:
                        st.error("Erreur lors du nettoyage")
                except:
                    st.error("Erreur de connexion")
    
    # Interface principale avec onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Ingestion", "🔍 Recherche", "📊 Analyse", "🕸️ Graphe", "⚡ Cypher"])
    
    with tab1:
        st.header("📤 Ingestion de Documents")
        ingestion_interface()
    
    with tab2:
        st.header("🔍 Recherche dans le Knowledge Graph")
        query_interface()
    
    with tab3:
        st.header("📊 Analyse et Statistiques")
        analytics_interface()
    
    with tab4:
        st.header("🕸️ Exploration du Graphe")
        graph_exploration_interface()
    
    with tab5:
        st.header("⚡ Requêtes Cypher")
        cypher_interface()

def cypher_interface():
    """Interface pour exécuter des requêtes Cypher directes"""
    st.markdown("### ⚡ Éditeur Cypher")
    st.markdown("Exécutez des requêtes Cypher personnalisées sur votre graphe de connaissances.")
    
    # Requêtes prédéfinies
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("#### 📋 Requêtes Utiles")
        
        # Sélecteur de document pour analyse détaillée
        st.markdown("##### 🔍 Analyse par Document")
        
        # Récupérer la liste des documents disponibles
        try:
            response = requests.post(f"{API_BASE_URL}/cypher", 
                                   json={'query': 'MATCH (d:Document) RETURN DISTINCT d.filename as filename ORDER BY d.filename'}, 
                                   timeout=10)
            if response.status_code == 200:
                result = response.json()
                if 'result' in result and result['result']:
                    document_names = [doc['filename'] for doc in result['result']]
                    
                    selected_document = st.selectbox(
                        "Choisir un document à analyser:",
                        options=[""] + document_names,
                        help="Sélectionnez un document pour voir ses chunks et embeddings"
                    )
                    
                    if selected_document and st.button("📄 Analyser ce document"):
                        document_query = f'''MATCH (d:Document {{filename: "{selected_document}"}})-[:CONTAINS_CHUNK]->(c:Chunk)
RETURN d.filename as Document,
       c.chunkId as ChunkID,
       c.text[..150] + '...' as TexteAperçu,
       CASE 
         WHEN c.textEmbedding IS NOT NULL 
         THEN size(c.textEmbedding) 
         ELSE 0 
       END as DimensionEmbedding,
       CASE 
         WHEN c.textEmbedding IS NOT NULL 
         THEN 'Oui' 
         ELSE 'Non' 
       END as EmbeddingPresent
ORDER BY c.chunkId'''
                        st.session_state['cypher_query'] = document_query
                        st.rerun()
                        
        except Exception as e:
            st.warning(f"Impossible de récupérer la liste des documents: {str(e)}")
        
        st.markdown("---")
        
        predefined_queries = {
            "📊 Statistiques générales": """MATCH (n) 
RETURN labels(n)[0] as Type, count(*) as Count 
ORDER BY Count DESC""",
            
            "🔍 Embeddings disponibles": """MATCH (c:Chunk) 
WHERE c.textEmbedding IS NOT NULL
RETURN c.filename as Fichier, 
       count(*) as ChunksAvecEmbeddings
ORDER BY ChunksAvecEmbeddings DESC""",
            
            "🔗 Top similarités": """MATCH ()-[r:RELATES_TO]->()
WHERE r.similarity IS NOT NULL
RETURN r.similarity as Similarité
ORDER BY r.similarity DESC 
LIMIT 10""",
            
            "📄 Documents et chunks": """MATCH (d:Document)-[:CONTAINS_CHUNK]->(c:Chunk)
RETURN d.filename as Document, 
       count(c) as NbChunks
ORDER BY NbChunks DESC""",
            
            "🌐 Relations par type": """MATCH ()-[r]->()
RETURN type(r) as TypeRelation, 
       count(r) as Nombre
ORDER BY Nombre DESC""",
            
            "🎯 Chunks avec texte": """MATCH (c:Chunk)
WHERE c.text IS NOT NULL
RETURN c.filename as Fichier,
       c.text[..100] + '...' as Aperçu,
       size(c.textEmbedding) as DimensionEmbedding
LIMIT 5""",
            
            "📋 Document détaillé": """MATCH (d:Document {filename: "nom_du_fichier.pdf"})-[:CONTAINS_CHUNK]->(c:Chunk)
RETURN d.filename as Document,
       c.chunkId as ChunkID,
       c.text[..150] + '...' as TexteAperçu,
       CASE 
         WHEN c.textEmbedding IS NOT NULL 
         THEN size(c.textEmbedding) 
         ELSE 0 
       END as DimensionEmbedding,
       CASE 
         WHEN c.textEmbedding IS NOT NULL 
         THEN 'Oui' 
         ELSE 'Non' 
       END as EmbeddingPresent
ORDER BY c.chunkId"""
        }
        
        selected_query = st.selectbox(
            "Choisir une requête prédéfinie:",
            options=[""] + list(predefined_queries.keys()),
            help="Sélectionnez une requête pour la charger dans l'éditeur"
        )
        
        if st.button("📋 Charger la requête"):
            if selected_query and selected_query in predefined_queries:
                st.session_state['cypher_query'] = predefined_queries[selected_query]
                st.rerun()
    
    with col1:
        # Éditeur de requête
        default_query = st.session_state.get('cypher_query', "MATCH (n) RETURN count(n) as total_nodes LIMIT 10")
        
        cypher_query = st.text_area(
            "Requête Cypher:",
            value=default_query,
            height=200,
            help="Entrez votre requête Cypher. Ex: MATCH (n) RETURN n LIMIT 10"
        )
        
        # Boutons d'action
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            execute_button = st.button("▶️ Exécuter", type="primary")
        
        with col_btn2:
            if st.button("🗑️ Effacer"):
                st.session_state['cypher_query'] = ""
                st.rerun()
        
        with col_btn3:
            if st.button("📋 Exemples"):
                st.session_state['show_examples'] = not st.session_state.get('show_examples', False)
                st.rerun()
    
    # Affichage des exemples
    if st.session_state.get('show_examples', False):
        with st.expander("📚 Exemples de requêtes Cypher", expanded=True):
            st.code("""
// 1. Trouver tous les types de nœuds
MATCH (n) RETURN DISTINCT labels(n) as Types

// 2. Chunks avec embeddings d'un document
MATCH (c:Chunk)
WHERE c.filename CONTAINS "CSR-Report" 
  AND c.textEmbedding IS NOT NULL
RETURN c.text[..80] as Apercu, size(c.textEmbedding) as Dimension

// 3. Relations entre documents différents
MATCH (d1:Document)-[:CONTAINS_CHUNK]->(c1)-[r:RELATES_TO]->(c2)<-[:CONTAINS_CHUNK]-(d2:Document)
WHERE d1 <> d2
RETURN d1.filename, d2.filename, r.similarity
ORDER BY r.similarity DESC LIMIT 5

// 4. Statistiques des embeddings
MATCH (c:Chunk)
WHERE c.textEmbedding IS NOT NULL
RETURN count(*) as TotalEmbeddings, 
       avg(size(c.textEmbedding)) as DimensionMoyenne

// 5. Analyser un document spécifique (remplacer le nom)
MATCH (d:Document)-[:CONTAINS_CHUNK]->(c:Chunk)
WHERE d.filename CONTAINS "CSR-Report"  // Modifier ici
RETURN c.chunkId as Chunk,
       c.text[..100] as Apercu,
       size(c.textEmbedding) as Embedding_Dim,
       c.created_at as DateCreation
ORDER BY c.chunkId

// 6. Relations d'un document vers d'autres documents
MATCH (d1:Document {filename: "nom_document.pdf"})-[:CONTAINS_CHUNK]->(c1)
-[r:RELATES_TO]->(c2)<-[:CONTAINS_CHUNK]-(d2:Document)
WHERE d1 <> d2
RETURN d2.filename as DocumentCible, 
       count(r) as NbRelations,
       avg(r.similarity) as SimilariteMoyenne
ORDER BY NbRelations DESC
            """, language="cypher")
    
    # Exécution de la requête
    if execute_button and cypher_query.strip():
        with st.spinner("Exécution de la requête..."):
            try:
                response = requests.post(f"{API_BASE_URL}/cypher", 
                                       json={'query': cypher_query.strip()}, 
                                       timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'result' in result and result['result']:
                        st.success(f"✅ Requête exécutée avec succès! {len(result['result'])} résultat(s)")
                        
                        # Affichage des résultats sous forme de tableau
                        if result['result']:
                            df = pd.DataFrame(result['result'])
                            st.dataframe(df, width="stretch")
                            
                            # Option de téléchargement
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger CSV",
                                data=csv,
                                file_name=f"cypher_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        # Affichage brut pour debugging
                        with st.expander("🔧 Résultat brut (JSON)", expanded=False):
                            st.json(result)
                            
                    else:
                        st.info("ℹ️ Requête exécutée mais aucun résultat retourné")
                        st.json(result)
                
                else:
                    st.error(f"❌ Erreur HTTP {response.status_code}")
                    try:
                        error_detail = response.json()
                        st.json(error_detail)
                    except:
                        st.text(response.text)
                        
            except requests.exceptions.Timeout:
                st.error("⏱️ Timeout - La requête a pris trop de temps")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

def ingestion_interface():
    """Interface d'ingestion de documents"""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sélectionnez vos documents")
        uploaded_files = st.file_uploader(
            "Choisissez vos fichiers",
            accept_multiple_files=True,
            type=['md', 'txt', 'docx', 'pdf'],
            help="Formats supportés: Markdown (.md), Texte (.txt), Word (.docx), PDF (.pdf)"
        )
    
    with col2:
        st.subheader("Paramètres")
        batch_mode = st.checkbox("Mode batch", value=True, help="Ingérer tous les fichiers d'un coup")
        auto_discover = st.checkbox("Auto-découverte relations", value=False, help="Découvrir automatiquement les relations sémantiques après ingestion")
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} fichier(s) sélectionné(s):**")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size:,} bytes)")
        
        if st.button("🚀 Démarrer l'ingestion", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            results = []
            
            for i, file in enumerate(uploaded_files):
                status_text.write(f"Traitement de {file.name}...")
                progress_bar.progress((i) / len(uploaded_files))
                
                status_code, result = ingest_file(file)
                
                if status_code == 200:
                    results.append({
                        'Fichier': file.name,
                        'Statut': '✅ Succès',
                        'Chunks créés': result.get('chunks_created', 0),
                        'Relations séq.': result.get('sequential_relations_created', 0)
                    })
                    st.success(f"✅ {file.name}: {result.get('chunks_created', 0)} chunks créés")
                else:
                    results.append({
                        'Fichier': file.name,
                        'Statut': '❌ Erreur',
                        'Chunks créés': 0,
                        'Relations séq.': 0
                    })
                    st.error(f"❌ {file.name}: {result}")
            
            progress_bar.progress(1.0)
            status_text.write("Ingestion terminée!")
            
            # Affichage du résumé
            df_results = pd.DataFrame(results)
            st.subheader("📋 Résumé de l'ingestion")
            st.dataframe(df_results, width="stretch")
            
            total_chunks = df_results['Chunks créés'].sum()
            total_relations = df_results['Relations séq.'].sum()
            success_count = len([r for r in results if '✅' in r['Statut']])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Fichiers traités", len(uploaded_files))
            with col2:
                st.metric("Succès", success_count)
            with col3:
                st.metric("Chunks totaux", total_chunks)
            with col4:
                st.metric("Relations créées", total_relations)
            
            # Auto-découverte des relations si activée
            if auto_discover and success_count > 0:
                st.info("🧠 Découverte automatique des relations sémantiques...")
                status_code, discovery_result = discover_semantic_relationships(0.8, 5)
                if status_code == 200:
                    semantic_relations = discovery_result.get('semantic_relationships_created', 0)
                    st.success(f"✅ {semantic_relations} relations sémantiques découvertes!")
                else:
                    st.warning(f"⚠️ Erreur lors de la découverte: {discovery_result}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def query_interface():
    """Interface de recherche"""
    st.markdown('<div class="query-section">', unsafe_allow_html=True)
    
    # Type de recherche
    search_type = st.selectbox(
        "Type de recherche",
        ["🔍 Recherche simple", "🧠 Recherche avec contexte graphe"],
        help="Choisissez le type de recherche souhaité"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_area(
            "Votre question",
            placeholder="Ex: Qu'est-ce qu'un knowledge graph ?",
            height=100
        )
    
    with col2:
        st.subheader("Paramètres")
        top_k = st.slider("Nombre de résultats", 1, 10, 5)
        if search_type == "🧠 Recherche avec contexte graphe":
            similarity_threshold = st.slider("Seuil de similarité", 0.0, 1.0, 0.6, 0.1)
    
    if st.button("🔍 Rechercher", type="primary") and question:
        with st.spinner("Recherche en cours..."):
            if search_type == "🔍 Recherche simple":
                status_code, result = query_knowledge_graph(question, top_k)
            else:
                status_code, result = semantic_search_with_context(question, top_k, similarity_threshold)
        
        if status_code == 200:
            display_search_results(result, search_type)
        else:
            st.error(f"Erreur lors de la recherche: {result}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_search_results(result, search_type):
    """Afficher les résultats de recherche"""
    st.subheader("📋 Résultats de la recherche")
    
    if search_type == "🔍 Recherche simple":
        # Réponse LLM
        if 'llm_answer' in result:
            st.markdown("### 🤖 Réponse générée")
            st.markdown(f'<div class="info-card">{result["llm_answer"]}</div>', unsafe_allow_html=True)
        
        # Sources
        chunks = result.get('results', [])
    else:
        # Recherche avec contexte
        chunks = result.get('chunks', [])
        if result.get('enhanced_with_graph_context'):
            st.info(f"✨ Résultats enrichis avec le contexte du graphe ({result.get('total_found', 0)} chunks trouvés)")
    
    if chunks:
        st.markdown("### 📚 Sources trouvées")
        
        for i, chunk in enumerate(chunks[:10]):  # Limiter à 10 résultats
            with st.expander(f"📄 Source {i+1} - Score: {chunk.get('score', 0):.3f}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Contenu:**")
                    text_content = chunk.get('text', 'N/A')
                    st.text_area("Contenu du chunk", text_content, height=150, disabled=True, key=f"content_{i}", label_visibility="collapsed")
                
                with col2:
                    st.markdown("**Métadonnées:**")
                    st.write(f"📁 **Fichier:** {chunk.get('source', chunk.get('filename', 'N/A'))}")
                    st.write(f"🆔 **ID:** {chunk.get('chunk_id', chunk.get('id', 'N/A'))}")
                    st.write(f"📊 **Score:** {chunk.get('score', 0):.3f}")
                    
                    # Contexte graphe si disponible
                    if 'document_info' in chunk:
                        doc_info = chunk['document_info']
                        st.write(f"📋 **Document parent:** {doc_info.get('filename', 'N/A')}")
                    
                    if chunk.get('next_chunk'):
                        st.write("➡️ **Chunk suivant disponible**")
                    
                    if chunk.get('previous_chunk'):
                        st.write("⬅️ **Chunk précédent disponible**")
                    
                    similar_chunks = chunk.get('similar_chunks', [])
                    if similar_chunks:
                        st.write(f"🔗 **{len(similar_chunks)} chunks similaires**")

def analytics_interface():
    """Interface d'analyse et statistiques"""
    stats = get_graph_stats()
    
    if not stats:
        st.warning("Impossible de récupérer les statistiques")
        return
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{stats.get('total_documents', 0)}</h3>
            <p>Documents</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{stats.get('total_chunks', 0):,}</h3>
            <p>Chunks</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{stats.get('total_relationships', 0):,}</h3>
            <p>Relations</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        rel_types = stats.get('relationship_types', [])
        st.markdown(f'''
        <div class="metric-card">
            <h3>{len(rel_types)}</h3>
            <p>Types Relations</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Liste des fichiers en base
    st.subheader("📁 Fichiers en Base de Données")
    files_in_db = get_files_in_database()
    
    if files_in_db:
        # Créer un DataFrame pour un meilleur affichage
        files_df = pd.DataFrame(files_in_db)
        
        # Formatage des données
        if 'file_size' in files_df.columns:
            files_df['Taille (KB)'] = files_df['file_size'].apply(
                lambda x: f"{x/1024:.1f}" if x and x > 0 else "N/A"
            )
        
        if 'created_at' in files_df.columns:
            files_df['Date de création'] = files_df['created_at'].apply(
                lambda x: x.split('T')[0] if isinstance(x, str) and 'T' in x else str(x)[:10] if x else "N/A"
            )
        
        # Renommer les colonnes pour l'affichage
        display_columns = {}
        if 'filename' in files_df.columns:
            display_columns['filename'] = 'Nom du fichier'
        if 'chunk_count' in files_df.columns:
            display_columns['chunk_count'] = 'Nb chunks'
        if 'Taille (KB)' in files_df.columns:
            display_columns['Taille (KB)'] = 'Taille (KB)'
        if 'Date de création' in files_df.columns:
            display_columns['Date de création'] = 'Date de création'
        
        # Sélectionner et renommer les colonnes à afficher
        columns_to_display = [col for col in display_columns.keys() if col in files_df.columns]
        files_display_df = files_df[columns_to_display].rename(columns=display_columns)
        
        # Afficher le tableau
        st.dataframe(
            files_display_df, 
            width="stretch",
            height=min(400, (len(files_display_df) + 1) * 35 + 50)
        )
        
        # Statistiques rapides des fichiers
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total fichiers", len(files_in_db))
        with col2:
            total_chunks = sum(f.get('chunk_count', 0) for f in files_in_db)
            st.metric("Total chunks", f"{total_chunks:,}")
        with col3:
            avg_chunks = total_chunks / len(files_in_db) if files_in_db else 0
            st.metric("Moy. chunks/fichier", f"{avg_chunks:.1f}")
        with col4:
            total_size = sum(f.get('file_size', 0) for f in files_in_db if f.get('file_size'))
            if total_size > 0:
                st.metric("Taille totale", f"{total_size/1024/1024:.1f} MB")
            else:
                st.metric("Taille totale", "N/A")
    else:
        st.info("Aucun fichier trouvé en base de données. Veuillez d'abord ingérer des documents.")
    
    # Graphiques
    st.subheader("📊 Visualisations")
    
    if rel_types:
        # Graphique des types de relations
        rel_counts = []
        for rel_type in rel_types:
            try:
                response = requests.post(f"{API_BASE_URL}/cypher", json={
                    'query': f'MATCH ()-[r:{rel_type}]->() RETURN count(r) as count'
                })
                if response.status_code == 200:
                    count = response.json()['result'][0]['count']
                    rel_counts.append({'Type': rel_type, 'Count': count})
            except:
                pass
        
        if rel_counts:
            df_relations = pd.DataFrame(rel_counts)
            fig = px.bar(df_relations, x='Type', y='Count', 
                        title="Distribution des Types de Relations",
                        color='Count',
                        color_continuous_scale='viridis')
            # Configuration Plotly
            config = {'displayModeBar': False}
            st.plotly_chart(fig, config=config, width="stretch")
    
    # Informations détaillées
    st.subheader("🔍 Détails du Système")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Types de relations actifs:**")
        
        # Descriptions détaillées des types de relations
        relation_descriptions = {
            'NEXT': '🔗 **NEXT** - Relation séquentielle entre documents dans l\'ordre d\'ingestion (inter-documents)',
            'RELATES_TO': '🧠 **RELATES_TO** - Relation sémantique découverte automatiquement entre chunks de **documents différents**',
            'CONTAINS_CHUNK': '📄 **CONTAINS_CHUNK** - Relation hiérarchique liant un document à ses chunks',
            'NEXT_CHUNK': '➡️ **NEXT_CHUNK** - Relation séquentielle vers le chunk suivant dans le même document (intra-document)',
            'PREVIOUS_CHUNK': '⬅️ **PREVIOUS_CHUNK** - Relation séquentielle vers le chunk précédent dans le même document (intra-document)'
        }
        
        for rel_type in rel_types:
            if rel_type in relation_descriptions:
                st.markdown(relation_descriptions[rel_type])
            else:
                st.write(f"• {rel_type}")
        
        # Si certaines relations standard ne sont pas encore créées
        missing_relations = set(relation_descriptions.keys()) - set(rel_types)
        if missing_relations:
            st.markdown("---")
            st.markdown("**Relations potentielles (non encore créées):**")
            for rel_type in missing_relations:
                st.markdown(f"⚪ {relation_descriptions[rel_type]}")
    
    with col2:
        db_info = get_db_info()
        if db_info:
            st.markdown("**Informations de la base:**")
            st.write(f"• Connexion: {db_info.get('neo4j_connection', 'N/A')}")
            st.write(f"• Index vectoriel: {db_info.get('vector_index', 'N/A')}")
        
        st.markdown("---")
        st.markdown("**Architecture du Knowledge Graph:**")
        st.markdown("""
        - **Documents** 📄 : Fichiers sources ingérés
        - **Chunks** 📝 : Segments de texte avec embeddings vectoriels
        - **Relations** 🔗 : Connexions sémantiques et structurelles
        - **Index vectoriel** 🧮 : Recherche par similarité sémantique
        """)

def graph_exploration_interface():
    """Interface d'exploration du graphe"""
    st.subheader("🕸️ Exploration du Knowledge Graph")
    
    # Onglets pour différents types d'exploration
    tab1, tab2 = st.tabs(["📊 Visualisation Interactive", "🔍 Structure de Document"])
    
    with tab1:
        st.markdown("### 🌐 Visualisation des Relations Inter-Documents")
        
        # Contrôles de filtrage
        col1, col2, col3 = st.columns(3)
        
        with col1:
            relation_types = st.multiselect(
                "Types de relations",
                options=["RELATES_TO", "NEXT", "NEXT_CHUNK"],
                default=["RELATES_TO"],
                help="Sélectionnez les types de relations à afficher"
            )
        
        with col2:
            min_similarity = st.slider(
                "Seuil de similarité minimum",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Afficher seulement les relations avec une similarité ≥ ce seuil"
            )
        
        with col3:
            max_nodes = st.slider(
                "Nombre max de documents",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Limiter le nombre de documents affichés"
            )
        
        # Option pour inclure les documents isolés
        include_isolated = st.checkbox(
            "🏝️ Inclure les documents sans relations",
            value=False,
            help="Afficher aussi les documents qui n'ont aucune relation avec d'autres (nœuds isolés)"
        )
        
        if st.button("🚀 Générer la visualisation", type="primary"):
            if relation_types:
                rel_types_str = ",".join(relation_types)
                
                with st.spinner("Récupération des données du graphe..."):
                    graph_data = get_graph_visualization_data(rel_types_str, min_similarity, max_nodes, include_isolated)
                
                if graph_data and graph_data.get('nodes'):
                    display_interactive_graph(graph_data)
                else:
                    st.warning("Aucune relation trouvée avec les critères sélectionnés. Essayez de réduire le seuil de similarité.")
            else:
                st.warning("Veuillez sélectionner au moins un type de relation.")
    
    with tab2:
        st.markdown("### 📄 Explorer la Structure d'un Document")
        
        # Sélection de document
        try:
            response = requests.post(f"{API_BASE_URL}/cypher", json={
                'query': 'MATCH (d:Document) RETURN d.filename as filename ORDER BY filename'
            })
            
            if response.status_code == 200:
                documents = [doc['filename'] for doc in response.json()['result']]
                
                if documents:
                    selected_doc = st.selectbox("Sélectionnez un document à explorer:", documents)
                    
                    if st.button("🔍 Explorer la structure"):
                        with st.spinner("Récupération de la structure..."):
                            response = requests.get(f"{API_BASE_URL}/graph_structure/{selected_doc}")
                            
                            if response.status_code == 200:
                                structure = response.json()
                                display_document_structure(structure)
                            else:
                                st.error(f"Erreur: {response.text}")
                else:
                    st.info("Aucun document trouvé. Veuillez d'abord ingérer des documents.")
            else:
                st.error("Impossible de récupérer la liste des documents")
        except Exception as e:
            st.error(f"Erreur de connexion: {str(e)}")

def display_interactive_graph(graph_data):
    """Afficher le graphe interactif avec streamlit-agraph"""
    nodes_data = graph_data.get('nodes', [])
    edges_data = graph_data.get('edges', [])
    stats = graph_data.get('stats', {})
    
    # Calculer le nombre de documents isolés
    isolated_count = sum(1 for node in nodes_data if node.get('isolated', False))
    
    # Afficher les statistiques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Documents total", stats.get('total_nodes', 0))
    with col2:
        st.metric("🔗 Relations inter-documents", stats.get('total_edges', 0))
    with col3:
        connected_count = stats.get('total_nodes', 0) - isolated_count
        st.metric("🌐 Documents connectés", connected_count)
    with col4:
        st.metric("🏝️ Documents isolés", isolated_count)
        
    # Types de relations (en dessous)
    if stats.get('relation_types'):
        types_str = ", ".join(stats.get('relation_types', []))
        st.caption(f"🏷️ **Types de relations:** {types_str}")
    
    # Légende des couleurs
    st.markdown("""
    **Légende:** 
    🔵 Documents connectés (bleu) • 
    🌸 Documents isolés (rose)
    """)
    
    if not nodes_data:
        st.warning("Aucun nœud à afficher. Les documents n'ont peut-être pas de relations avec les critères sélectionnés.")
        return
    
    # Création des nœuds pour streamlit-agraph
    nodes = []
    for node in nodes_data:
        # Utiliser la couleur définie dans les données du nœud, sinon couleur par défaut
        color = node.get('color', "#1f77b4" if node['type'] == 'Document' else "#ff7f0e")
        size = max(15, min(50, node.get('chunk_count', 1) * 2))
        
        status = " (🏝️ isolé)" if node.get('isolated', False) else ""
        nodes.append(Node(
            id=node['id'],
            label=node['label'][:20] + "..." if len(node['label']) > 20 else node['label'],
            size=size,
            color=color,
            title=f"📄 {node['label']}{status}\n📊 {node.get('chunk_count', 0)} chunks"
        ))
    
    # Création des arêtes pour streamlit-agraph
    edges = []
    for edge in edges_data:
        color = "#2ca02c" if edge['type'] == 'RELATES_TO' else "#d62728"
        width = max(1, min(8, edge.get('similarity', 0.5) * 10))
        
        edges.append(Edge(
            source=edge['source'],
            target=edge['target'],
            color=color,
            width=width,
            title=f"{edge['type']}\nSimilarité: {edge.get('similarity', 0):.3f}"
        ))
    
    # Configuration de l'affichage
    config = Config(
        width=800,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False
    )
    
    # Affichage du graphe
    st.subheader("🌐 Graphe des Relations Inter-Documents")
    
    # Légende
    with st.expander("📖 Légende de la visualisation"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Nœuds (Documents):**")
            st.markdown("🔵 **Bleu** : Documents")
            st.markdown("📏 **Taille** : Proportionnelle au nombre de chunks")
        with col2:
            st.markdown("**Arêtes (Relations):**")
            st.markdown("🟢 **Vert** : Relations RELATES_TO (sémantiques)")
            st.markdown("🔴 **Rouge** : Relations NEXT (séquentielles)")
            st.markdown("📏 **Épaisseur** : Proportionnelle à la similarité")
    
    # Affichage du graphe interactif
    return_value = agraph(nodes=nodes, edges=edges, config=config)
    
    # Affichage des détails si un nœud est sélectionné
    if return_value:
        st.subheader("📋 Détails de la sélection")
        st.json(return_value)

def display_document_structure(structure):
    """Afficher la structure d'un document"""
    doc_info = structure.get('document', {})
    chunks_data = structure.get('chunks_with_relationships', [])
    
    st.success(f"📄 Document: {doc_info.get('filename', 'N/A')}")
    st.info(f"📊 {structure.get('chunk_count', 0)} chunks avec {len(chunks_data)} ayant des relations")
    
    if chunks_data:
        # Résumé des relations
        total_relations = sum(len(chunk['relationships']) for chunk in chunks_data)
        relation_types = set()
        for chunk in chunks_data:
            for rel in chunk['relationships']:
                if rel.get('type'):
                    relation_types.add(rel['type'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Relations totales", total_relations)
        with col2:
            st.metric("Types de relations", len(relation_types))
        with col3:
            avg_relations = total_relations / len(chunks_data) if chunks_data else 0
            st.metric("Moy. relations/chunk", f"{avg_relations:.1f}")
        
        # Affichage détaillé des chunks
        st.subheader("📋 Détails des chunks")
        
        for i, chunk_data in enumerate(chunks_data[:10]):  # Limiter à 10
            chunk = chunk_data.get('chunk', {})
            relationships = chunk_data.get('relationships', [])
            
            with st.expander(f"Chunk {i+1} - {len(relationships)} relations"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    chunk_text = chunk.get('text', 'N/A')
                    if len(chunk_text) > 300:
                        chunk_text = chunk_text[:300] + "..."
                    st.text_area("Contenu:", chunk_text, height=100, disabled=True, key=f"explore_chunk_{i}")
                
                with col2:
                    st.write(f"**ID:** {chunk.get('id', 'N/A')}")
                    st.write(f"**Relations:** {len(relationships)}")
                    
                    # Types de relations pour ce chunk
                    rel_types_chunk = {}
                    for rel in relationships:
                        rel_type = rel.get('type', 'Unknown')
                        rel_types_chunk[rel_type] = rel_types_chunk.get(rel_type, 0) + 1
                    
                    for rel_type, count in rel_types_chunk.items():
                        st.write(f"• {rel_type}: {count}")

if __name__ == "__main__":
    main()