# 🧪 Démo GraphRAG — Référentiel documentaire d'entreprise

Démo autonome de construction d'un knowledge graph à partir de PDF de procédures internes,
avec comparaison **RAG vectoriel seul** vs **RAG enrichi par le graphe**.

Contrairement au reste du projet (qui calcule les embeddings *dans* Neo4j via
`genai.vector.encode`), cette démo s'appuie sur la librairie officielle
[`neo4j-graphrag`](https://neo4j.com/docs/neo4j-graphrag-python/) et calcule les
embeddings côté Python. Elle écrit dans la même base et réutilise le même index vectoriel.

## 🎯 Ce que la démo montre

| | Question type | Résultat mesuré |
|---|---|---|
| 🔍 **Consultation** | « Quel est le plafond d'hébergement pour Paris ? » | Le RAG vectoriel seul suffit — 3/3 réponses correctes |
| 🕸️ **Agrégation / filtre** | « Tous les seuils qui impliquent la Direction Financière » | Le vectoriel échoue (2 faux positifs sur 3), le graphe répond juste |

Conclusion : **le graphe ne sert pas à mieux retrouver, il sert à répondre à des questions
qu'une recherche par similarité ne sait pas poser.** Un `top_k` ne peut pas balayer un
référentiel entier ni filtrer sur un critère structurel.

## 📚 Corpus

Six documents fictifs de procédure interne dans `PDFs/`. La démo applique l'hypothèse
« la GED ne sert que la version en vigueur » **littéralement**, par un filtre sur le champ
`Statut` de l'en-tête :

| Fichier | Statut | Ingéré |
|---|---|---|
| `01_DIR-FIN-004_..._v1.2_2023.pdf` | ABROGÉE | ❌ |
| `02_DIR-FIN-004_..._v2.0_2026.pdf` | EN VIGUEUR | ✅ |
| `03_POL-ACH-001_Politique_Achats_et_Seuils.pdf` | EN VIGUEUR | ✅ |
| `04_GUI-CON-002_Guide_Utilisateur_Concur.pdf` | EN VIGUEUR | ✅ |
| `05_FAQ_Procurement_Finance_Intranet.pdf` | *(absent)* | ❌ |
| `06_NOTE-2026-017_Note_de_service_Transition.pdf` | *(absent)* | ❌ |

> ⚠️ Le filtre n'est pas cosmétique. Le document 01 et la FAQ 05 contiennent des valeurs
> périmées (forfait repas de 35 EUR, délai de 60 jours, justificatif au-delà de 25 EUR)
> qui contredisent la directive en vigueur. Sans filtre, la FAQ — rédigée sous forme de
> questions — remonte **en tête** de la recherche vectorielle et le système répond faux.

## ⚡ Lancement

### Prérequis

- Le `.env` de la racine renseigné (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `OPENAI_API_KEY`)
- L'index vectoriel créé sur la base :

```cypher
CREATE VECTOR INDEX GrahRAG IF NOT EXISTS
FOR (c:Chunk) ON (c.textEmbedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
```

### Exécution

```bash
# 1. Vider le graphe (chaque reconstruction repart de zéro, sinon les ingestions se superposent)
.venv/bin/python -c "import os,neo4j;from dotenv import load_dotenv;load_dotenv();\
d=neo4j.GraphDatabase.driver(os.environ['NEO4J_URI'],auth=(os.environ['NEO4J_USERNAME'],os.environ['NEO4J_PASSWORD']));\
d.session().run('MATCH (n) DETACH DELETE n');d.close();print('graphe vidé')"

# 2. Construire le graphe (~2 min, ~20 appels LLM)
.venv/bin/python demo_build_kg.py

# 3. Interroger — les 3 questions en double affichage vectoriel / graphe
.venv/bin/python demo_query.py
```

> Les deux scripts se lancent **depuis la racine** (ils lisent `.env` via `load_dotenv()`),
> contrairement à l'API qui doit être lancée depuis `KnowledgeGraphRagAPI/`.

## 🕸️ Graphe produit

```
(:Document)<-[:FROM_DOCUMENT]-(:Chunk)-[:NEXT_CHUNK]->(:Chunk)
(:Chunk)<-[:FROM_CHUNK]-(:Role|Seuil|Zone|Outil|DocumentRef)
(:Role)-[:APPROUVE]->(:Seuil)-[:S_APPLIQUE_A]->(:Zone)
(:Seuil)-[:DECLENCHE]->(:Role)
(:Role)-[:UTILISE]->(:Outil)
```

| Élément | Volume |
|---|---|
| Documents / Chunks | 3 / 17 *(déterministe)* |
| Entités typées | ~55–60 (Seuil, Role, Zone, DocumentRef, Outil) |
| Relations métier | ~45–55 |

> ℹ️ Le découpage et les embeddings sont déterministes, **l'extraction ne l'est pas** :
> malgré `temperature=0`, deux exécutions donnent des volumétries différentes (58 vs 61
> entités sur deux runs consécutifs). Les valeurs métier, elles, restent stables — les
> 7 plafonds sont corrects à chaque run. Ne comptez pas sur des chiffres exacts pour
> valider une exécution ; validez sur le contenu.

Le type `Seuil` porte les valeurs (`montant`, `unite`, `consequence`) : c'est lui qui rend
les chiffres interrogeables. Exemple de ce que le graphe désambiguïse — la ligne PDF
`Luxembourg 25 EUR 40 EUR 65 EUR`, illisible hors de son en-tête de colonne, devient :

```
Plafond repas Luxembourg          65 EUR  ─S_APPLIQUE_A→ Luxembourg
Plafond repas Union européenne    75 EUR  ─S_APPLIQUE_A→ Union européenne
Plafond repas Hors Union eur.     90 EUR  ─S_APPLIQUE_A→ Hors Union européenne
```

## ⚠️ Pièges de configuration `neo4j-graphrag`

Quatre défauts de la librairie qui **échouent en silence** sur ce projet — aucune erreur
levée, juste un résultat faux :

| Réglage | Défaut | Valeur nécessaire | Symptôme si laissé au défaut |
|---|---|---|---|
| `chunk_embedding_property` | `embedding` | `textEmbedding` | l'index `GrahRAG` reste vide, la recherche renvoie 0 résultat |
| `text_splitter` | `FixedSizeSplitter(4000)` | `RecursiveCharacterTextSplitter(1000, 100)` | un chunk ≈ un document ; et la coupe au caractère près tranche les tableaux de plafonds |
| propriété identifiante | — | doit s'appeler **`name`** | le résolveur d'entités ne compare que `name` : nommez-la `nom`, il tourne sans rien fusionner |
| `name` d'un `Seuil` | — | doit être **discriminant** | les 4 plafonds d'hébergement fusionnent à 160 EUR car ils partagent le libellé « Plafond par nuit » |

Le dernier est le plus vicieux : la résolution d'entités, une fois qu'elle *fonctionne*,
écrase les lignes d'un tableau qui partagent le même intitulé de colonne. Correctif appliqué
dans le schéma — la description du type impose d'inclure la zone dans le `name`. Après
correction : 7 plafonds sur 7 corrects.

Autre divergence à connaître : la relation chunk↔document est écrite
`(:Chunk)-[:FROM_DOCUMENT]->(:Document)`, soit le **sens inverse** du
`(:Document)-[:CONTAINS_CHUNK]->(:Chunk)` du reste du projet. Les endpoints
`/graph_stats` et `/graph_structure` de l'API ne voient donc pas ce graphe.

## 📊 Exemple de contraste

**Question** — « Liste tous les seuils du référentiel qui déclenchent une intervention
de la Direction Financière. »

*RAG vectoriel* : renvoie le tableau des seuils d'achat et le recopie sans filtrer sur le
critère — 2 des 3 seuils cités n'impliquent pas la Direction Financière.

*Requête sur le graphe* : le bon ensemble, réparti sur deux documents.

```cypher
MATCH (r:Role)-[rel]-(x:Seuil)
WHERE toLower(r.name) CONTAINS 'financi'
RETURN DISTINCT x.name, x.montant, x.unite, type(rel)
ORDER BY x.montant
```

| Seuil | Montant | Source |
|---|---|---|
| Durée de vol (classe affaires) | 10 heures | DIR-FIN-004 §3.2 |
| Coût prévisionnel de mission | 5 000 EUR | DIR-FIN-004 §2 |
| Marché 60 000 – 143 000 EUR | 143 000 EUR | POL-ACH-001 §3 |

Les deux premières lignes viennent de sections que la recherche vectorielle n'a jamais
rapportées.

## 🔧 Limites connues

- L'extraction de la chaîne d'approbation est correcte à **4/5** — le palier
  1 500–15 000 EUR se voit attribuer le Comité de Direction au lieu du responsable budgétaire.
- Le type `DocumentRef` capte encore du bruit (« directive », « version 1.2 ») malgré
  l'instruction de schéma.
- Une négation n'est pas gérée : « le seuil de 25 EUR est supprimé » produit un `Seuil` à 25 EUR.
- Les ligatures PDF (`ﬁ`, `ﬀ`, `ﬃ`) ne sont pas normalisées. Sans conséquence pour une
  recherche vectorielle, mais toute requête en texte exact (`CONTAINS 'justificatif'`)
  échoue. Correctif : `unicodedata.normalize('NFKC', text)` à l'ingestion.

---

📖 **Documentation** : [README.md](./README.md) · 🐳 [DOCKER.md](./DOCKER.md) · 🔧 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
