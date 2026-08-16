"""Pipeline GraphRAG générique : des documents déposés par l'utilisateur vers un graphe.

Étapes : extraction du texte → proposition d'un schéma d'extraction → découpage,
embeddings et extraction d'entités via `neo4j-graphrag` → projection de compatibilité.

En ligne de commande, ingère tout le dossier `PDFs/` avec un schéma inféré :
    .venv/bin/python demo_build_kg.py
"""
import asyncio
import json
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import neo4j
import pypdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphConfig
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    SchemaFromTextExtractor,
)
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

load_dotenv()

DOSSIER = Path("PDFs")
FORMATS = {".pdf", ".txt", ".md"}
MODELE_LLM = "gpt-4o-mini"
MODELE_EMBEDDING = "text-embedding-3-small"  # 1536 dims, comme l'index GrahRAG


# --------------------------------------------------------------------------- #
# Connexion
# --------------------------------------------------------------------------- #
def connexion() -> neo4j.Driver:
    return neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )


def base() -> str:
    return os.environ.get("NEO4J_DATABASE", "neo4j")


def llm(json_mode: bool = True) -> OpenAILLM:
    params = {"temperature": 0}
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    return OpenAILLM(model_name=MODELE_LLM, model_params=params)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def documents(dossier: Path = DOSSIER) -> list[Path]:
    """Tous les fichiers de format supporté présents dans le dossier."""
    if not dossier.exists():
        return []
    return sorted(p for p in dossier.iterdir() if p.suffix.lower() in FORMATS)


def extraire_texte(chemin: Path) -> str:
    if chemin.suffix.lower() == ".pdf":
        return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(str(chemin)).pages)
    return chemin.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Schéma d'extraction
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Dérivation automatique du schéma, en trois passes
# --------------------------------------------------------------------------- #
# Passe 1 : extraction LIBRE sur un échantillon de chunks — on observe ce que le
#           modèle produit réellement, au lieu de le deviner depuis du texte brut.
# Passe 2 : consolidation des types bruts en un jeu canonique, par un seul appel.
# Passe 3 : extraction sur TOUS les chunks, contrainte par ce schéma (voir ingerer).
#
# Sans la passe 2, l'extraction libre produit un type par entité ou presque :
# mesuré sur 12 chunks d'un corpus réel, 55 types distincts dont 48 vus une seule
# fois, avec les collisions habituelles (organisation / Organisation / organisme).

DECOUVERTE = """Extrais les entités et les relations de ce texte.
Choisis toi-même les types qui te paraissent pertinents, sans liste imposée.

Réponds en JSON strict :
{"entites": [{"nom": "...", "type": "..."}],
 "relations": [{"source": "...", "type": "...", "cible": "..."}]}

TEXTE :
"""

CONSOLIDATION = """Voici les types d'entités et de relations qu'un modèle a produits
en analysant un corpus, avec leur nombre d'occurrences. Ils sont redondants et
incohérents : mêmes concepts sous des libellés différents, variantes de casse,
types trop spécifiques vus une seule fois.

Consolide-les en un schéma canonique de {n_types} types d'entités au maximum et
{n_rel} types de relations au maximum. Règles impératives :
- fusionne agressivement : synonymes, variantes de casse, hyperonymes. Deux types
  qui désigneraient les mêmes objets dans un graphe DOIVENT être fusionnés
  (Organisation et Institution, Evenement et Action, Document et Texte…).
- écarte les types trop spécifiques pour structurer un graphe
- libellés d'entités : CamelCase, en {langue}, **sans accent ni espace** (Evenement,
  et non Événement)
- libellés de relations : MAJUSCULES_AVEC_UNDERSCORES, **sans accent** (FAIT_PARTIE_DE)
- pour chaque type, propose 0 à 3 propriétés utiles. N'utilise JAMAIS `name`, `nom`
  ni `titre` : une propriété `name` est ajoutée d'office à chaque type.
- propose des patterns (source, RELATION, cible) cohérents avec les types retenus

TYPES D'ENTITÉS OBSERVÉS :
{types}

TYPES DE RELATIONS OBSERVÉS :
{relations}

Réponds en JSON strict :
{{"node_types": {{"Type": ["propriete1", "propriete2"]}},
  "relationship_types": ["RELATION_A", "RELATION_B"],
  "patterns": [["TypeSource", "RELATION_A", "TypeCible"]]}}
"""


def decouper(fichiers: list[Path]) -> list[tuple[str, str]]:
    """Découpe les documents avec le même découpeur que la pipeline d'ingestion,
    pour que la découverte porte sur les chunks réellement extraits ensuite."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return [
        (morceau, f.name)
        for f in fichiers
        for morceau in splitter.split_text(extraire_texte(f))
    ]


def tirer(chunks: list[tuple[str, str]], n: int, graine: int = 0) -> list[tuple[str, str]]:
    """Échantillon réparti : proportionnel au poids de chaque document, et tiré
    dans tout le document plutôt qu'en tête."""
    par_doc: dict[str, list] = {}
    for texte, doc in chunks:
        par_doc.setdefault(doc, []).append(texte)
    total = len(chunks) or 1
    tirage = []
    alea = random.Random(graine)
    for doc, textes in par_doc.items():
        k = max(1, round(n * len(textes) / total))
        tirage += [(t, doc) for t in alea.sample(textes, min(k, len(textes)))]
    return tirage[:n]


def _json(reponse: str) -> dict:
    try:
        return json.loads(reponse[reponse.find("{") : reponse.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {}


async def decouvrir_types(
    chunks: list[tuple[str, str]], n: int = 20, parallele: int = 5, trace=print
) -> tuple[Counter, Counter]:
    """Passe 1 — extraction libre sur un échantillon, en parallèle."""
    tirage = tirer(chunks, n)
    trace(f"découverte : {len(tirage)} chunks échantillonnés sur {len(chunks)}")
    modele = llm()
    verrou = asyncio.Semaphore(parallele)

    async def un(texte: str) -> dict:
        async with verrou:
            return _json((await modele.ainvoke(DECOUVERTE + texte[:900])).content)

    resultats = await asyncio.gather(*(un(t) for t, _ in tirage), return_exceptions=True)

    types, relations = Counter(), Counter()
    for r in resultats:
        if not isinstance(r, dict):
            continue
        types.update(e["type"] for e in r.get("entites", []) if e.get("type"))
        relations.update(x["type"] for x in r.get("relations", []) if x.get("type"))
    trace(f"découverte : {len(types)} types d'entités, {len(relations)} de relations")
    return types, relations


def normaliser_schema(schema: dict) -> dict:
    """Applique en code ce que le prompt demande — un prompt n'est pas une garantie.

    Les accents dans un label Neo4j obligent à des backticks partout et déroutent
    la traduction texte→Cypher ; une propriété `nom` ferait doublon avec le `name`
    ajouté d'office, sur lequel repose la résolution d'entités.
    """
    sans_accent = lambda s: "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
    interdites = {"name", "nom", "titre", "title", "label"}

    noeuds, renommage = {}, {}
    for label, props in (schema.get("node_types") or {}).items():
        propre = re.sub(r"[^A-Za-z0-9]", "", sans_accent(str(label)))
        if not propre:
            continue
        renommage[label] = propre
        # Les noms de propriétés aussi : un accent impose des backticks en Cypher
        # et déroute la traduction texte→Cypher.
        noeuds[propre] = [
            re.sub(r"[^A-Za-z0-9_]", "", sans_accent(str(p)))
            for p in (props or [])
            if str(p).lower() not in interdites
        ][:3]
        noeuds[propre] = [p for p in noeuds[propre] if p]

    relations, ren_rel = [], {}
    for r in schema.get("relationship_types") or []:
        propre = re.sub(r"[^A-Z0-9_]", "", sans_accent(str(r)).upper().replace(" ", "_"))
        if propre and propre not in relations:
            relations.append(propre)
        ren_rel[r] = propre

    patterns = []
    for p in schema.get("patterns") or []:
        if len(p) != 3:
            continue
        s, r, c = renommage.get(p[0]), ren_rel.get(p[1]), renommage.get(p[2])
        if s in noeuds and c in noeuds and r in relations:
            patterns.append([s, r, c])

    return {"node_types": noeuds, "relationship_types": relations, "patterns": patterns}


async def consolider(
    types: Counter, relations: Counter, n_types: int = 10, n_rel: int = 8,
    langue: str = "français", trace=print,
) -> dict:
    """Passe 2 — fusionne les types bruts en un schéma canonique, en un appel."""
    if not types:
        return {"node_types": {}, "relationship_types": [], "patterns": []}
    fmt = lambda c: "\n".join(f"- {k} ({v})" for k, v in c.most_common(80))
    reponse = await llm().ainvoke(
        CONSOLIDATION.format(
            n_types=n_types, n_rel=n_rel, langue=langue,
            types=fmt(types), relations=fmt(relations) or "- (aucune)",
        )
    )
    schema = normaliser_schema(_json(reponse.content))
    trace(
        f"consolidation : {len(types)} types bruts -> {len(schema['node_types'])} canoniques"
    )
    return schema


async def deriver_schema(
    fichiers: list[Path], n: int = 20, langue: str = "français", trace=print
) -> dict:
    """Passes 1 et 2 enchaînées : des documents vers un schéma prêt à contraindre."""
    chunks = decouper(fichiers)
    types, relations = await decouvrir_types(chunks, n=n, trace=trace)
    return await consolider(types, relations, langue=langue, trace=trace)


def echantillonner(texte: str, budget: int, fenetres: int = 3) -> str:
    """Prélève plusieurs fenêtres réparties dans le document.

    Prendre uniquement le début est trompeur : sur un texte long, les premiers
    milliers de caractères sont la page de titre et le sommaire, jamais le fond.
    """
    if len(texte) <= budget:
        return texte
    taille = budget // fenetres
    positions = [int(len(texte) * p) for p in (0.05, 0.45, 0.80)][:fenetres]
    return "\n[…]\n".join(texte[p : p + taille] for p in positions)


async def proposer_schema(fichiers: list[Path], echantillon: int = 12000) -> GraphSchema:
    """Fait proposer par le LLM les types d'entités et de relations du corpus.

    C'est le point de qualité de toute la pipeline : en extraction libre, le modèle
    produit des types incohérents d'un chunk à l'autre. Le schéma proposé ici est
    destiné à être relu et corrigé avant construction.

    Le budget est réparti au prorata de la taille des documents, pour qu'un texte
    long ne soit pas réduit au même extrait qu'une note de deux pages.
    """
    textes = {f: extraire_texte(f) for f in fichiers}
    total = sum(len(t) for t in textes.values()) or 1
    morceaux = []
    for f, t in textes.items():
        part = max(int(echantillon * len(t) / total), echantillon // (4 * len(fichiers)))
        morceaux.append(f"--- {f.name} ---\n{echantillonner(t, part)}")
    return await SchemaFromTextExtractor(llm=llm()).run(text="\n\n".join(morceaux)[:echantillon * 2])


def schema_en_dict(schema: GraphSchema) -> dict:
    """Forme éditable : {"Type": [propriétés]} et liste de relations."""
    return {
        "node_types": {
            n.label: [p.name for p in n.properties] for n in schema.node_types
        },
        "relationship_types": [r.label for r in schema.relationship_types],
        "patterns": [list(p) for p in schema.patterns],
    }


def dict_en_schema(d: dict) -> dict:
    """Reconstruit le dict attendu par SimpleKGPipeline depuis la forme éditable."""
    return {
        "node_types": [
            {
                "label": label,
                "properties": [{"name": "name", "type": "STRING"}]
                + [
                    {"name": p, "type": "STRING"}
                    for p in props
                    if p != "name"
                ],
            }
            for label, props in d["node_types"].items()
        ],
        "relationship_types": [{"label": r} for r in d["relationship_types"]],
        "patterns": [tuple(p) for p in d.get("patterns", []) if len(p) == 3],
    }


# --------------------------------------------------------------------------- #
# Construction du graphe
# --------------------------------------------------------------------------- #
# SimpleKGPipeline écrit (:Chunk)-[:FROM_DOCUMENT]->(:Document) avec `path` et `index`.
# L'API et l'interface Streamlit du projet sont indexées sur `filename` et sur
# (:Document)-[:CONTAINS_CHUNK]->(:Chunk). Sans cette projection, le graphe est
# invisible pour elles : toutes les requêtes renvoient du vide, sans erreur.
COMPAT = """
MATCH (d:Document)
SET d.filename = coalesce(d.filename, last(split(d.path, '/')))
WITH d
MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d)
SET c.filename = d.filename,
    c.chunk_index = c.index,
    c.id = d.filename + '-' + toString(c.index)
MERGE (d)-[:CONTAINS_CHUNK {chunk_index: c.index}]->(c)
WITH d, count(c) AS n
SET d.chunk_count = n
RETURN sum(n) AS chunks
"""

# Pour les formats non-PDF, la librairie n'a pas de loader : on passe le texte
# directement, mais aucun nœud Document n'est alors créé. On le crée nous-mêmes
# et on y rattache les chunks laissés orphelins par ce passage.
RATTACHER = """
MATCH (c:Chunk) WHERE NOT (c)-[:FROM_DOCUMENT]->(:Document)
WITH collect(c) AS orphelins
MERGE (d:Document {filename: $filename})
SET d.path = $path
WITH d, orphelins UNWIND orphelins AS c
MERGE (c)-[:FROM_DOCUMENT]->(d)
RETURN count(c) AS rattachés
"""


def construire_pipeline(
    driver: neo4j.Driver, schema: dict, depuis_pdf: bool = True
) -> SimpleKGPipeline:
    return SimpleKGPipeline(
        llm=llm(),
        driver=driver,
        embedder=OpenAIEmbeddings(model=MODELE_EMBEDDING),
        schema=schema,
        from_pdf=depuis_pdf,
        # Le défaut (FixedSizeSplitter, 4000 car.) découpe au caractère près, en
        # plein milieu des tableaux. Le découpeur récursif respecte les sauts de ligne.
        text_splitter=LangChainTextSplitterAdapter(
            RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        ),
        # L'index vectoriel `GrahRAG` porte sur Chunk.textEmbedding ; le défaut de
        # la librairie est `embedding`, l'index resterait vide.
        lexical_graph_config=LexicalGraphConfig(chunk_embedding_property="textEmbedding"),
        # Le résolveur ne compare que la propriété `name`. Il fusionne les mentions
        # d'une même entité — utile ici, dangereux si deux nœuds distincts partagent
        # un libellé générique (deux lignes d'un même tableau, par exemple).
        perform_entity_resolution=True,
        neo4j_database=base(),
    )


async def ingerer(
    fichiers: list[Path], driver: neo4j.Driver, schema: dict, trace=print
) -> int:
    """Ingère des documents dans le graphe existant. Incrémental : n'efface rien."""
    pdf = construire_pipeline(driver, schema, depuis_pdf=True)
    txt = None
    for f in fichiers:
        trace(f"ingestion : {f.name}")
        if f.suffix.lower() == ".pdf":
            await pdf.run_async(file_path=str(f))
        else:
            txt = txt or construire_pipeline(driver, schema, depuis_pdf=False)
            await txt.run_async(text=extraire_texte(f))
            with driver.session(database=base()) as s:
                s.run(RATTACHER, filename=f.name, path=str(f))
    with driver.session(database=base()) as s:
        return s.run(COMPAT).single()["chunks"] or 0


# --------------------------------------------------------------------------- #
# Retrait
# --------------------------------------------------------------------------- #
def retirer(driver: neo4j.Driver, filename: str) -> dict:
    """Retire un document : le document, ses chunks, et les entités qui n'ont plus
    aucun chunk source.

    Une entité citée par plusieurs documents survit : la résolution d'entités les a
    fusionnées, elle garde donc un lien vers les chunks des autres documents.
    """
    with driver.session(database=base()) as s:
        avant = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        s.run(
            "MATCH (d:Document {filename: $f}) "
            "OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d) "
            "DETACH DELETE d, c",
            f=filename,
        )
        orphelines = s.run(
            "MATCH (e:__Entity__) WHERE NOT (e)-[:FROM_CHUNK]->() "
            "WITH e, count(*) AS _ DETACH DELETE e RETURN count(*) AS n"
        ).single()["n"]
        apres = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    return {"supprimés": avant - apres, "entités orphelines": orphelines}


def vider(driver: neo4j.Driver) -> int:
    with driver.session(database=base()) as s:
        n = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        s.run("MATCH (n) DETACH DELETE n")
    return n


# --------------------------------------------------------------------------- #
# Ligne de commande
# --------------------------------------------------------------------------- #
async def main() -> None:
    fichiers = documents()
    if not fichiers:
        print(f"Aucun document dans {DOSSIER}/ (formats : {', '.join(sorted(FORMATS))})")
        return
    print(f"{len(fichiers)} document(s) :", ", ".join(f.name for f in fichiers))

    print("\nDérivation du schéma (passes 1 et 2)…")
    d = await deriver_schema(fichiers, trace=lambda m: print("  " + m))
    print("  entités  :", ", ".join(d["node_types"]))
    print("  relations:", ", ".join(d["relationship_types"]))

    driver = connexion()
    print()
    n = await ingerer(fichiers, driver, dict_en_schema(d))
    print(f"\n{n} chunks projetés (filename + CONTAINS_CHUNK)")

    with driver.session(database=base()) as s:
        print("\n--- graphe obtenu ---")
        for r in s.run(
            "MATCH (n:__Entity__) UNWIND labels(n) AS l WITH l "
            "WHERE NOT l STARTS WITH '__' RETURN l, count(*) AS n ORDER BY n DESC"
        ):
            print(f"  {r['l']:20s} {r['n']}")
        for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC"):
            print(f"  [{r['t']}] {r['n']}")
        r = s.run("MATCH (c:Chunk) RETURN count(c) AS t, count(c.textEmbedding) AS e").single()
        print(f"\n  chunks {r['t']} / embeddings {r['e']}")

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())
