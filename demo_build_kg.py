"""Pipeline GraphRAG générique : des documents déposés par l'utilisateur vers un graphe.

Étapes : extraction du texte → proposition d'un schéma d'extraction → découpage,
embeddings et extraction d'entités via `neo4j-graphrag` → projection de compatibilité.

En ligne de commande, ingère tout le dossier `PDFs/` avec un schéma inféré :
    .venv/bin/python demo_build_kg.py
"""
import asyncio
import os
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

    print("\nProposition d'un schéma d'extraction…")
    propose = await proposer_schema(fichiers)
    d = schema_en_dict(propose)
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
