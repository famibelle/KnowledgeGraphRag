-- ============================================
-- REQUÊTES CYPHER UTILES POUR LES EMBEDDINGS
-- ============================================

-- 1. 📊 EXPLORER LA STRUCTURE DES EMBEDDINGS
-- Voir les propriétés disponibles
MATCH (c:Chunk) 
RETURN keys(c) as properties 
LIMIT 1;

-- Compter les chunks avec embeddings
MATCH (c:Chunk)
WHERE c.textEmbedding IS NOT NULL
RETURN COUNT(c) as chunks_with_embeddings, 
       size(c.textEmbedding) as embedding_dimension
LIMIT 1;

-- 2. 🔍 RÉCUPÉRER DES EMBEDDINGS SPÉCIFIQUES
-- Embeddings par document
MATCH (c:Chunk)
WHERE c.filename CONTAINS "CSR-Report"
RETURN c.text[..100] as text_preview,
       c.textEmbedding[..5] as embedding_sample,
       size(c.textEmbedding) as dimension;

-- Embeddings avec métadonnées
MATCH (c:Chunk)
WHERE c.textEmbedding IS NOT NULL
RETURN c.id as chunk_id,
       c.text[..80] as text_preview, 
       c.filename as source_file,
       c.textEmbedding as full_embedding
LIMIT 3;

-- 3. 📏 ANALYSER LES EMBEDDINGS
-- Statistiques des embeddings (approximatives via échantillonnage)
MATCH (c:Chunk)
WHERE c.textEmbedding IS NOT NULL
WITH c.textEmbedding as emb
RETURN 
    COUNT(*) as total_embeddings,
    emb[0] as first_dimension_sample,
    emb[100] as middle_dimension_sample,
    size(emb) as dimension
LIMIT 5;

-- 4. 🔗 SIMILARITÉS CALCULÉES
-- Top similarités entre chunks
MATCH (c1:Chunk)-[r:RELATES_TO]-(c2:Chunk)
WHERE r.similarity IS NOT NULL
RETURN c1.text[..50] as text1,
       c2.text[..50] as text2,
       r.similarity as calculated_similarity,
       c1.filename as file1,
       c2.filename as file2
ORDER BY r.similarity DESC
LIMIT 10;

-- 5. 🎯 RECHERCHE PAR SIMILARITÉ MANUELLE
-- (Remplacez $queryEmbedding par un vrai vecteur)
/*
MATCH (c:Chunk)
WHERE c.textEmbedding IS NOT NULL
WITH c, gds.similarity.cosine(c.textEmbedding, $queryEmbedding) as similarity
WHERE similarity > 0.7
RETURN c.text, similarity, c.filename
ORDER BY similarity DESC
LIMIT 10;
*/

-- 6. 📊 DISTRIBUTION DES SIMILARITÉS
-- Voir la distribution des relations RELATES_TO par similarité
MATCH ()-[r:RELATES_TO]->()
WHERE r.similarity IS NOT NULL
WITH r.similarity as sim
RETURN 
    CASE 
        WHEN sim >= 0.9 THEN "0.9-1.0 (Très haute)"
        WHEN sim >= 0.8 THEN "0.8-0.9 (Haute)"  
        WHEN sim >= 0.7 THEN "0.7-0.8 (Moyenne)"
        WHEN sim >= 0.6 THEN "0.6-0.7 (Faible)"
        ELSE "< 0.6 (Très faible)"
    END as similarity_range,
    COUNT(*) as count
ORDER BY similarity_range DESC;

-- 7. 🔍 EMBEDDINGS PAR TYPE DE CONTENU
-- Analyser les embeddings selon le type de contenu
MATCH (c:Chunk)
WHERE c.textEmbedding IS NOT NULL
WITH c,
     CASE 
         WHEN c.text CONTAINS "intelligence" THEN "AI_Related"
         WHEN c.text CONTAINS "data" THEN "Data_Related" 
         WHEN c.text CONTAINS "graph" THEN "Graph_Related"
         ELSE "Other"
     END as content_type
RETURN content_type, 
       COUNT(*) as chunk_count,
       AVG(size(c.textEmbedding)) as avg_embedding_size
ORDER BY chunk_count DESC;

-- 8. 📈 ÉVOLUTION DES EMBEDDINGS PAR DOCUMENT
-- Voir comment les embeddings sont distribués par fichier
MATCH (c:Chunk)
WHERE c.textEmbedding IS NOT NULL
RETURN c.filename as document,
       COUNT(*) as chunks_with_embeddings,
       MIN(c.chunk_index) as first_chunk,
       MAX(c.chunk_index) as last_chunk
ORDER BY chunks_with_embeddings DESC;