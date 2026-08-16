# 🕸️ GraphRAG Builder — des documents vers un graphe interrogeable

Démo autonome : l'utilisateur dépose des documents, la pipeline en extrait un graphe de
connaissances, et le graphe se laisse interroger en langage naturel.

Construite sur [`neo4j-graphrag`](https://neo4j.com/docs/neo4j-graphrag-python/), la
librairie officielle. Contrairement au reste du projet — qui calcule les embeddings
*dans* Neo4j via `genai.vector.encode` — cette démo les calcule côté Python. Elle écrit
dans la même base et réutilise le même index vectoriel.

## ⚡ Lancement

```bash
.venv/bin/python -m streamlit run demo_streamlit.py     # interface
.venv/bin/python demo_build_kg.py                       # ou en ligne de commande
```

Prérequis : le `.env` de la racine renseigné, et l'index vectoriel créé sur la base :

```cypher
CREATE VECTOR INDEX GrahRAG IF NOT EXISTS
FOR (c:Chunk) ON (c.textEmbedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
```

## 🔄 La pipeline, en six étapes

| Étape | Ce qui se passe |
|---|---|
| **1 · Documents** | Dépôt de fichiers (`.pdf`, `.txt`, `.md`), listés avec leur état d'ingestion. Suppression du disque **et** du graphe. |
| **2 · Schéma d'extraction** | Le LLM propose les types d'entités et de relations à partir d'un échantillon du corpus. **Le schéma est éditable en JSON.** |
| **3 · Construction** | Découpage, embeddings, extraction d'entités. Ingestion incrémentale ou reconstruction complète. |
| **4 · Graphe** | Volumétrie, vues interactives (`neo4j-viz`), voisinage d'une entité. |
| **5 · Entités extraites** | L'inventaire de ce que le LLM a produit, avec le chunk source de chaque entité. |
| **6 · Interroger** | Question en langage naturel → Cypher (`Text2CypherRetriever`) → résultat → réponse. La requête générée est affichée. |

## 🎯 Le point de conception

**Le schéma d'extraction est le principal levier de qualité.** En extraction libre, le
modèle produit `Société` / `Entreprise` / `Organisation` pour la même chose et le graphe
devient inexploitable — aucun chemin ne relie plus rien.

L'étape 2 rend ce choix explicite : le schéma est proposé, puis **relu et corrigé** avant
construction. C'est le geste qui sépare une démo qui marche d'une démo qui impressionne.

Une propriété `name` est ajoutée d'office à chaque type d'entité : le résolveur de la
librairie ne compare que celle-là.

## ⚠️ Limites, mesurées et assumées

**L'extraction n'est pas déterministe.** À `temperature=0`, deux exécutions sur le même
corpus donnent des volumétries différentes (58 puis 61 entités sur un corpus de test).
Les valeurs métier restent stables. Ne validez jamais une exécution sur un décompte.

**La proposition de schéma est instable** — et sort tantôt en français, tantôt en
anglais selon l'échantillon tiré. D'où l'édition manuelle.

**La négation n'est pas gérée.** « Ce seuil est supprimé » produit quand même une entité
pour ce seuil.

**Le résolveur d'entités ne compare que `name`.** Il fusionne les mentions d'une même
entité — utile — mais écrase aussi des nœuds distincts partageant un libellé générique.
Sur un corpus de test, les quatre lignes d'un tableau de plafonds ont fusionné en une
seule parce qu'elles s'appelaient toutes « Plafond par nuit ».

**`Text2CypherRetriever` échoue parfois** : le LLM produit du Cypher que Neo4j refuse.
L'interface le signale et propose d'écrire la requête à la main.

## 🔧 Pièges de configuration `neo4j-graphrag`

Quatre défauts de la librairie qui **échouent en silence** — aucune erreur levée, juste
un résultat faux :

| Réglage | Défaut | Valeur nécessaire | Symptôme sinon |
|---|---|---|---|
| `chunk_embedding_property` | `embedding` | `textEmbedding` | l'index `GrahRAG` reste vide, la recherche renvoie 0 résultat |
| `text_splitter` | `FixedSizeSplitter(4000)` | `RecursiveCharacterTextSplitter(1000, 100)` | un chunk ≈ un document, et la coupe au caractère près tranche les tableaux |
| propriété identifiante | — | doit s'appeler **`name`** | le résolveur tourne sans rien fusionner |
| modèle d'embedding | `genai.vector.encode` utilise **ada-002** | préciser `text-embedding-3-small` | 1536 dimensions des deux côtés, donc aucune erreur — mais recherche entre deux espaces vectoriels : score max 0.509 au lieu de 0.768 |

Autre divergence : `SimpleKGPipeline` écrit `(:Chunk)-[:FROM_DOCUMENT]->(:Document)`,
soit l'inverse du `CONTAINS_CHUNK` du reste du projet, avec `path` au lieu de
`filename`. `demo_build_kg.py` applique donc une projection de compatibilité en fin
d'ingestion (constante `COMPAT`), sans quoi l'API et l'interface Streamlit du projet ne
voient rien.

Les formats non-PDF n'ont pas de loader dans la librairie : le texte est passé
directement, mais aucun nœud `Document` n'est alors créé. La constante `RATTACHER` le
crée et y raccroche les chunks laissés orphelins.

## 📦 Fichiers

| Fichier | Rôle |
|---|---|
| `demo_build_kg.py` | La pipeline : extraction de texte, proposition de schéma, ingestion, retrait, purge |
| `demo_streamlit.py` | L'interface en six étapes |
| `demo_query.py` | Comparaison RAG vectoriel / enrichi par le graphe, en ligne de commande |

---

📖 **Documentation** : [README.md](./README.md) · 🐳 [DOCKER.md](./DOCKER.md) · 🔧 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
