import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="RAG Knowledge Graph Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration de l'API
API_BASE_URL = "http://localhost:8000"

# CSS simple pour une interface claire
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
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

def ingest_file(file):
    """Ingérer un fichier via l'API"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/ingest_file", files=files, timeout=300)
        return response.status_code, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return 500, str(e)

def query_knowledge_graph(question, top_k=5, similarity_threshold=0.9):
    """Effectuer une requête sur le knowledge graph"""
    try:
        response = requests.post(f"{API_BASE_URL}/query", json={
            "question": question, 
            "top_k": top_k,
            "similarity_threshold": similarity_threshold
        })
        return response.status_code, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return 500, str(e)

def semantic_search_with_context(question, top_k=5, similarity_threshold=0.8):
    """Recherche sémantique avec contexte de graphe"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/semantic_search_with_context",
            json={
                "question": question,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            }
        )
        return response.status_code, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return 500, str(e)

def get_documents_list():
    """Récupérer la liste des documents via requête Cypher"""
    try:
        # Formatage de la date directement dans Cypher pour éviter les objets DateTime complexes
        cypher_query = """
        MATCH (d:Document) 
        RETURN d.filename as filename, 
               d.chunk_count as chunks, 
               toString(d.created_at) as created_str,
               d.created_at as created_raw
        ORDER BY d.created_at DESC
        """
        response = requests.post(f"{API_BASE_URL}/cypher", json={"query": cypher_query})
        if response.status_code == 200:
            result = response.json()
            documents = result.get('result', [])
            
            # Post-traitement pour formater les dates proprement
            for doc in documents:
                created_raw = doc.get('created_raw')
                if created_raw:
                    try:
                        # Si c'est un objet DateTime Neo4j, extraire la date
                        if isinstance(created_raw, dict) and '_DateTime__date' in created_raw:
                            date_info = created_raw['_DateTime__date']
                            if isinstance(date_info, dict):
                                year = date_info.get('_Date__year', 2025)
                                month = date_info.get('_Date__month', 1)
                                day = date_info.get('_Date__day', 1)
                                doc['created'] = f"{day:02d}/{month:02d}/{year}"
                            else:
                                doc['created'] = doc.get('created_str', 'N/A')
                        else:
                            # Utiliser la version string si disponible
                            doc['created'] = doc.get('created_str', 'N/A')
                    except:
                        doc['created'] = doc.get('created_str', 'N/A')
                else:
                    doc['created'] = 'N/A'
            
            return documents
        return []
    except Exception as e:
        st.error(f"Erreur lors de la récupération des documents : {str(e)}")
        return []

def delete_document(filename):
    """Supprimer un document via l'API"""
    try:
        response = requests.delete(f"{API_BASE_URL}/document/{filename}")
        return response.status_code, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return 500, str(e)

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🧠 RAG Knowledge Graph</h1>', unsafe_allow_html=True)
    st.markdown("**Système de Recherche Augmentée par Génération avec Graphe de Connaissances**")
    
    # Sidebar avec informations système
    with st.sidebar:
        st.header("📊 Statut du Système")
        
        # Vérification de la connexion API
        api_status = check_api_connection()
        if api_status:
            st.success("✅ API connectée")
        else:
            st.error("❌ API non accessible")
            st.stop()
        
        # Informations de la base de données
        db_info = get_db_info()
        if db_info:
            st.success("✅ Base de données connectée")
            # Vérifier différents champs possibles pour le nom de la base
            database_name = db_info.get('database') or db_info.get('neo4j_connection') or 'Neo4j'
            if database_name == 'active':
                database_name = 'Neo4j'
            st.info(f"🏗️ Database: {database_name}")
            
            # Afficher le statut de connexion Neo4j
            if db_info.get('neo4j_connection') == 'active':
                st.success("🔗 Neo4j: Connecté")
            else:
                st.warning("⚠️ Neo4j: Déconnecté")
        else:
            st.warning("⚠️ Informations DB non disponibles")
        
        # Statistiques du graphe
        stats = get_graph_stats()
        if stats:
            st.markdown("#### 📈 Statistiques")
            # Compatibilité avec les anciennes et nouvelles versions de l'API
            documents = stats.get('documents', stats.get('total_documents', 0))
            chunks = stats.get('chunks', stats.get('total_chunks', 0))
            relationships = stats.get('relationships', stats.get('total_relationships', 0))
            
            st.metric("Documents", documents)
            st.metric("Chunks", chunks)
            st.metric("Relations", relationships)
    
    # Interface principale RAG + Knowledge Graph
    tab1, tab2, tab3 = st.tabs(["📚 Documents", "🧠 Knowledge Graph", "🔍 RAG Query"])
    
    with tab1:
        documents_interface()
    
    with tab2:
        st.header("🧠 Knowledge Graph")
        knowledge_graph_interface()
    
    with tab3:
        st.header("🔍 Recherche RAG")
        rag_query_interface()

def documents_interface():
    """Interface simplifiée pour la gestion des documents"""
    
    # Upload de fichiers
    st.subheader("📤 Upload de Documents")
    uploaded_files = st.file_uploader(
        "Choisissez vos fichiers",
        accept_multiple_files=True,
        type=['md', 'txt', 'docx', 'pdf'],
        help="Formats supportés: Markdown (.md), Texte (.txt), Word (.docx), PDF (.pdf)"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} fichier(s) sélectionné(s):**")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size:,} bytes)")
        
        if st.button("🚀 Ingérer les Documents", type="primary"):
            progress_bar = st.progress(0)
            results = []
            
            for i, file in enumerate(uploaded_files):
                st.write(f"Traitement de {file.name}...")
                progress_bar.progress(i / len(uploaded_files))
                
                status_code, result = ingest_file(file)
                
                if status_code == 200:
                    results.append({
                        'Fichier': file.name,
                        'Statut': '✅ Succès',
                        'Chunks créés': result.get('chunks_created', 0)
                    })
                    st.success(f"✅ {file.name}: {result.get('chunks_created', 0)} chunks créés")
                else:
                    results.append({
                        'Fichier': file.name,
                        'Statut': '❌ Erreur',
                        'Chunks créés': 0
                    })
                    st.error(f"❌ {file.name}: {result}")
            
            progress_bar.progress(1.0)
            st.success("✅ Ingestion terminée!")
            
            # Résumé
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                total_chunks = df_results['Chunks créés'].sum()
                success_count = len([r for r in results if '✅' in r['Statut']])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fichiers traités", len(uploaded_files))
                with col2:
                    st.metric("Succès", success_count)
                with col3:
                    st.metric("Chunks totaux", total_chunks)
                
                # Auto-refresh après upload réussi
                if success_count > 0:
                    st.info("🔄 La liste des documents va se mettre à jour...")
                    st.rerun()
    
    st.markdown("---")
    
    # Liste des documents existants
    st.subheader("📋 Documents en Base")
    documents = get_documents_list()
    
    if documents:
        st.markdown(f"**{len(documents)} document(s) dans la base :**")
        
        # Affichage des documents avec boutons de suppression
        for doc in documents:
            filename = doc.get('filename', 'N/A')
            chunks = doc.get('chunks', 0)
            created = doc.get('created', 'N/A')
            
            col1, col2, col3, col4 = st.columns([3, 1, 2, 1])
            
            with col1:
                st.write(f"📄 **{filename}**")
            with col2:
                st.write(f"{chunks} chunks")
            with col3:
                if created and created != 'N/A':
                    st.write(f"📅 {created}")
                else:
                    st.write("📅 N/A")
            with col4:
                if st.button("🗑️", key=f"delete_{filename}", help=f"Supprimer {filename}"):
                    # Confirmation de suppression
                    if st.session_state.get(f"confirm_delete_{filename}", False):
                        with st.spinner(f"Suppression de {filename}..."):
                            status_code, result = delete_document(filename)
                            if status_code == 200:
                                st.success(f"✅ {filename} supprimé avec succès!")
                                st.session_state[f"confirm_delete_{filename}"] = False
                                st.rerun()
                            else:
                                st.error(f"❌ Erreur lors de la suppression : {result}")
                    else:
                        st.session_state[f"confirm_delete_{filename}"] = True
                        st.warning(f"⚠️ Cliquez à nouveau pour confirmer la suppression de {filename}")
                        st.rerun()
        
        # Bouton pour actualiser la liste
        if st.button("🔄 Actualiser la liste", key="refresh_docs"):
            st.rerun()
            
    else:
        st.info("💡 Aucun document trouvé. Uploadez vos premiers documents ci-dessus.")

def knowledge_graph_interface():
    """Interface consolidée pour le Knowledge Graph"""
    st.markdown("Créez des relations entre vos documents et explorez le graphe avec des requêtes Cypher.")
    
    # Section Création de Relations
    st.subheader("🔗 Création de Relations Inter-Documents")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        smart_threshold = st.slider(
            "Seuil de similarité", 
            0.1, 0.9, 0.80, 0.05, 
            help="Plus le seuil est élevé, plus les relations créées seront pertinentes"
        )
        max_total_relations = st.number_input(
            "Relations maximum", 
            10, 200, 50, 
            help="Nombre maximum de relations inter-documents à créer"
        )
    
    with col2:
        st.write("")
        if st.button("🚀 Créer Relations", type="primary", use_container_width=True):
            with st.spinner("🧠 Création des relations..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/smart_inter_document_relationships",
                        json={
                            "similarity_threshold": smart_threshold,
                            "max_relationships_total": max_total_relations,
                            "chunk_sampling_ratio": 0.3
                        },
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Relations créées avec succès !")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Nouvelles Relations", result.get("relations_created_this_run", 0))
                        with col2:
                            st.metric("Total Inter-docs", result.get("total_inter_document_relations", 0))
                        with col3:
                            st.metric("Documents Liés", result.get("documents_connected", 0))
                    else:
                        st.error(f"❌ Erreur : {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
    
    # Section Requêtes Cypher
    st.markdown("---")
    st.subheader("⚡ Requêtes Cypher")
    
    # Requêtes prédéfinies
    st.markdown("**Requêtes utiles :**")
    predefined_queries = [
        ("Voir tous les documents", "MATCH (d:Document) RETURN d.filename, d.chunk_count LIMIT 10"),
        ("Relations entre chunks", "MATCH (c1:Chunk)-[r:RELATES_TO]-(c2:Chunk) WHERE c1.filename <> c2.filename RETURN c1.filename, c2.filename, round(COALESCE(r.similarity, r.confidence)*1000)/1000 as score, substring(c1.text, 0, 120) + '...' as extrait_chunk1, substring(c2.text, 0, 120) + '...' as extrait_chunk2 ORDER BY score DESC LIMIT 10"),
        ("Relations entre documents", "MATCH (d1:Document)-[:CONTAINS_CHUNK]-(c1:Chunk)-[r:RELATES_TO]-(c2:Chunk)-[:CONTAINS_CHUNK]-(d2:Document) WHERE d1 <> d2 RETURN d1.filename, d2.filename, count(r) as nb_relations, round(avg(COALESCE(r.similarity, r.confidence))*1000)/1000 as score_moyen ORDER BY nb_relations DESC LIMIT 10"),
        ("Statistiques du graphe", "MATCH (n) RETURN labels(n)[0] as Type, count(*) as Count"),
        ("Top relations par similarité", "MATCH (c1:Chunk)-[r:RELATES_TO]-(c2:Chunk) WHERE c1.filename <> c2.filename AND r.similarity IS NOT NULL RETURN c1.filename, c2.filename, r.similarity ORDER BY r.similarity DESC LIMIT 10"),
        ("Types de relations", "MATCH ()-[r:RELATES_TO]-() RETURN r.method, r.type, count(*) as count ORDER BY count DESC")
    ]
    
    selected_query = st.selectbox("Choisir une requête prédéfinie:", 
                                 [""] + [q[0] for q in predefined_queries])
    
    if selected_query:
        query = next((q[1] for q in predefined_queries if q[0] == selected_query), "")
    else:
        query = ""
    
    query = st.text_area(
        "Requête Cypher:",
        value=query,
        height=100,
        placeholder="MATCH (n) RETURN n LIMIT 10",
        help="Exécutez des requêtes Cypher pour explorer votre graphe"
    )
    
    if st.button("Exécuter", key="cypher_exec"):
        if query.strip():
            try:
                response = requests.post(f"{API_BASE_URL}/cypher", json={'query': query})
                if response.status_code == 200:
                    result = response.json()
                    if result.get('result'):
                        df = pd.DataFrame(result['result'])
                        st.dataframe(df, use_container_width=True)
                        st.success(f"✅ {len(result['result'])} résultats trouvés")
                    else:
                        st.info("Aucun résultat trouvé")
                else:
                    st.error(f"❌ Erreur: {response.text}")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

def rag_query_interface():
    """Interface simplifiée pour les requêtes RAG"""
    question = st.text_area(
        "Votre question:",
        height=100,
        placeholder="Que voulez-vous savoir sur vos documents ?",
        help="Posez une question en français sur le contenu de vos documents"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Nombre de résultats", 1, 10, 5)
    with col2:
        similarity_threshold = st.slider("Seuil de similarité", 0.1, 1.0, 0.9, 0.05, 
                                        help="Plus élevé = résultats plus pertinents")
    use_context = True  # Toujours activé par défaut
    
    if st.button("🔍 Rechercher", type="primary"):
        if question.strip():
            with st.spinner("🧠 Recherche en cours..."):
                try:
                    # Utiliser le nouveau paramètre de seuil de similarité
                    status_code, result = query_knowledge_graph(question, top_k, similarity_threshold)
                    if use_context:
                        search_type = "RAG avec contexte graphe (via /query)"
                    else:
                        search_type = "RAG simple"
                    
                    if status_code == 200:
                        threshold_used = result.get('similarity_threshold_used', similarity_threshold)
                        total_chunks = result.get('total_relevant_chunks', 0)
                        st.subheader(f"📋 Résultats ({search_type})")
                        st.caption(f"Seuil de similarité: {threshold_used} | Chunks pertinents: {total_chunks}")
                        
                        # Affichage de la réponse générée (si disponible)
                        if 'llm_answer' in result:
                            st.markdown("### 🤖 Réponse Générée")
                            st.markdown(result['llm_answer'])
                        elif 'answer' in result:
                            st.markdown("### 🤖 Réponse Générée")
                            st.markdown(result['answer'])
                        
                        # Affichage des chunks trouvés (format /semantic_search_with_context)
                        if 'chunks' in result and result['chunks']:
                            st.markdown("### 📚 Sources Trouvées")
                            for i, chunk in enumerate(result['chunks'], 1):
                                score = chunk.get('score', 0)
                                source = chunk.get('source', 'N/A')
                                text = chunk.get('text', '')
                                
                                with st.expander(f"Source {i}: {source} (Score: {score:.3f})"):
                                    st.text(text)
                        
                        # Affichage des résultats (format /query)
                        elif 'results' in result and result['results']:
                            st.markdown("### 📚 Sources Trouvées")
                            for i, chunk in enumerate(result['results'], 1):
                                score = chunk.get('score', 0)
                                source = chunk.get('source', 'N/A')
                                text = chunk.get('text', '')
                                
                                with st.expander(f"Source {i}: {source} (Score: {score:.3f})"):
                                    st.text(text)
                        
                        # Affichage des sources (format alternatif)
                        elif 'sources' in result and result['sources']:
                            st.markdown("### 📚 Sources")
                            for i, source in enumerate(result['sources'], 1):
                                with st.expander(f"Source {i}: {source.get('document_name', 'N/A')} (Score: {source.get('similarity', 'N/A'):.3f})"):
                                    st.text(source.get('content', ''))
                        
                        # Si aucun format reconnu, affichage brut
                        else:
                            st.markdown("### 📋 Résultats Bruts")
                            st.json(result)
                        
                        # Contexte du graphe si disponible
                        if 'graph_context' in result:
                            st.markdown("### 🕸️ Contexte du Graphe")
                            st.json(result['graph_context'])
                    else:
                        st.error(f"❌ Erreur : {result}")
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
        else:
            st.warning("⚠️ Veuillez saisir une question")

if __name__ == "__main__":
    main()