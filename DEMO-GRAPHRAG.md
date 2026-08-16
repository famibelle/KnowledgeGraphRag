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
| **1 · Upload** | Dépôt de PDF (aussi `.txt`, `.md`). Un bouton enchaîne toute la construction. |
| **2 · Chunking** | `RecursiveCharacterTextSplitter`, 1000 caractères, 100 de recouvrement. |
| **3 · Embeddings** | `text-embedding-3-small`, 1536 dimensions, dans `Chunk.textEmbedding`. |
| **4 · Extraction des entités** | Un appel LLM **par chunk**, en extraction **libre** : aucun schéma imposé. |
| **5 · Consolidation des entités** | Fusion exacte sur nom normalisé → harmonisation des types → fusion approchée. |
| **6 · Affichage du graphe** | Vues interactives `neo4j-viz`, inventaire des entités avec leur provenance. |

Chaque chunk garde un lien `FROM_DOCUMENT` vers son document, chaque entité un lien
`FROM_CHUNK` vers le chunk qui l'a produite.

## 🎯 Le point de conception : consolider après, pas contraindre avant

Les documents n'étant pas connus a priori, aucun schéma ne peut être écrit à l'avance.
L'extraction est donc **libre** — le modèle nomme et type ce qu'il trouve — et le
nettoyage se fait après coup, sur les entités réellement produites.

Le problème que cela pose est mesurable : en extraction libre, le modèle invente ses
types au fil des chunks. Mesuré sur ce corpus, **41 à 47 types distincts pour 12 chunks**,
y compris à l'intérieur d'un seul document, avec les collisions habituelles
(`organisation` / `Organisation` / `organisme`).

La consolidation répond en trois étapes, dont **deux sont déterministes** :

| Étape | Méthode | LLM |
|---|---|---|
| **a · Fusion exacte** | Regroupement sur le nom normalisé — minuscules, sans accent, sans ponctuation, sans article. Fusion par `apoc.refactor.mergeNodes`. | non |
| **b · Harmonisation des types** | Un seul appel qui associe chaque étiquette observée à une étiquette canonique. | oui |
| **c · Fusion approchée** | `rapidfuzz`, similarité de noms ≥ 92, à étiquette égale. | non |

L'étape a fusionne « la Direction des Achats » et « DIRECTION DES ACHATS » — et, comme
elle opère toutes étiquettes confondues, elle règle du même coup les doublons de type
portant sur une même entité.

## ⚠️ Limites, mesurées et assumées

**L'extraction n'est pas déterministe.** À `temperature=0`, deux exécutions sur le même
corpus donnent des volumétries différentes (58 puis 61 entités sur un corpus de test).
Les valeurs métier restent stables. Ne validez jamais une exécution sur un décompte.

**La négation n'est pas gérée.** « Ce seuil est supprimé » produit quand même une entité
pour ce seuil.

**Le résolveur de la librairie est désactivé.** Il ne compare que `name` à étiquette
égale, ce qui ne suffit pas quand l'extraction libre produit aussi des étiquettes
divergentes. La consolidation le remplace.

**La fusion peut être trop agressive.** Deux entités distinctes portant un libellé
générique identique seront réunies. Sur un corpus de test, les quatre lignes d'un tableau
de plafonds ont fusionné parce qu'elles s'appelaient toutes « Plafond par nuit ». Le
seuil de similarité de l'étape c est réglable.

**`Text2CypherRetriever` échoue parfois** : le LLM produit du Cypher que Neo4j refuse.
L'interface le signale et propose d'écrire la requête à la main.

## 🔧 Pièges de configuration `neo4j-graphrag`

Quatre défauts de la librairie qui **échouent en silence** — aucune erreur levée, juste
un résultat faux :

| Réglage | Défaut | Valeur nécessaire | Symptôme sinon |
|---|---|---|---|
| `chunk_embedding_property` | `embedding` | `textEmbedding` | l'index `GrahRAG` reste vide, la recherche renvoie 0 résultat |
| `text_splitter` | `FixedSizeSplitter(4000)` | `RecursiveCharacterTextSplitter(1000, 100)` | un chunk ≈ un document, et la coupe au caractère près tranche les tableaux |
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
| `demo_build_kg.py` | La pipeline : extraction de texte, ingestion, consolidation, retrait, purge |
| `demo_streamlit.py` | L'interface : pipeline, upload et construction, graphe, interrogation |
| `demo_query.py` | Comparaison RAG vectoriel / enrichi par le graphe, en ligne de commande |

---

📖 **Documentation** : [README.md](./README.md) · 🐳 [DOCKER.md](./DOCKER.md) · 🔧 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
