"""Démo étude de cas — construction du knowledge graph à partir des PDF.

Hypothèse assumée : la GED ne fournit que la version en vigueur d'un document.
On l'applique littéralement — filtre sur le champ `Statut` de l'en-tête.

Usage : .venv/bin/python demo_build_kg.py
"""
import asyncio
import os
import re
from pathlib import Path

import neo4j
import pypdf
from dotenv import load_dotenv
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

load_dotenv()

PDF_DIR = Path("PDFs")

# --- Schéma contraint : ce que le LLM a le droit d'extraire --------------------
# NB : la propriété identifiante DOIT s'appeler `name` — le résolveur d'entités
# par défaut compare cette propriété et rien d'autre. Avec `nom` ou `code`,
# il tourne sans rien fusionner et sans lever d'erreur.
SCHEMA = {
    "node_types": [
        {"label": "Role", "description": "fonction ou instance qui décide, approuve ou est responsable",
         "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "Outil", "description": "application ou système informatique",
         "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "Zone", "description": "zone géographique d'application d'une règle",
         "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "DocumentRef",
         "description": "renvoi à un autre document interne, identifié par un CODE de la forme "
                        "XXX-YYY-NNN (DIR-FIN-004, POL-ACH-001, PRO-FIN-011, GUI-CON-002). "
                        "Ne jamais créer de DocumentRef pour un mot générique comme "
                        "'directive', 'ordre de mission' ou 'dérogation'.",
         "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "Seuil",
         "description": "montant, durée ou plafond déclenchant une règle ou une procédure. "
                        "`name` DOIT être unique et discriminant : y inclure la zone ou le cas "
                        "d'application, ex. 'Plafond hébergement Reste de l'Europe', "
                        "'Plafond repas Union européenne'. Ne jamais utiliser un intitulé de "
                        "colonne seul comme 'Plafond journalier' : chaque ligne d'un tableau "
                        "est un Seuil distinct.",
         "properties": [{"name": "name", "type": "STRING"},
                        {"name": "montant", "type": "FLOAT"},
                        {"name": "unite", "type": "STRING"},
                        {"name": "consequence", "type": "STRING"}]},
    ],
    "relationship_types": [
        {"label": "APPROUVE", "description": "le rôle approuve la dépense, la mission ou l'achat"},
        {"label": "S_APPLIQUE_A", "description": "le seuil s'applique à cette zone ou à ce cas"},
        {"label": "REFERENCE", "description": "renvoi vers un autre document"},
        {"label": "DECLENCHE", "description": "le franchissement du seuil déclenche cette procédure ou approbation"},
        {"label": "UTILISE", "description": "la procédure passe par cet outil"},
    ],
    "patterns": [
        ("Seuil", "DECLENCHE", "Role"),
        ("Seuil", "S_APPLIQUE_A", "Zone"),
        ("Role", "APPROUVE", "Seuil"),
        ("Role", "UTILISE", "Outil"),
        ("DocumentRef", "REFERENCE", "DocumentRef"),
    ],
}


def statut(pdf: Path) -> str | None:
    """Lit le champ `Statut` de l'en-tête. None si absent."""
    head = pypdf.PdfReader(str(pdf)).pages[0].extract_text()[:600]
    m = re.search(r"Statut\s*:\s*(.+)", head)
    return m.group(1).strip() if m else None


def corpus() -> list[Path]:
    """Applique l'hypothèse GED : on ne garde que ce qui est explicitement en vigueur."""
    retenus = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        s = statut(pdf)
        garde = s is not None and "EN VIGUEUR" in s.upper()
        print(f"  {'GARDE ' if garde else 'ECARTE'}  {pdf.name[:52]:52s} statut={s or '(absent)'}")
        if garde:
            retenus.append(pdf)
    return retenus


# SimpleKGPipeline écrit (:Chunk)-[:FROM_DOCUMENT]->(:Document) avec `path` et `index`.
# L'API et l'interface Streamlit du projet, elles, sont indexées sur `filename` et sur
# (:Document)-[:CONTAINS_CHUNK]->(:Chunk). Sans cette projection, la démo est invisible
# dans l'UI : toutes les requêtes renvoient du vide, sans erreur.
COMPAT = """
MATCH (d:Document)
SET d.filename = last(split(d.path, '/'))
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


def connexion() -> neo4j.Driver:
    return neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )


def base() -> str:
    return os.environ.get("NEO4J_DATABASE", "neo4j")


def construire_pipeline(driver: neo4j.Driver) -> SimpleKGPipeline:
    return SimpleKGPipeline(
        llm=OpenAILLM(
            model_name="gpt-4o-mini",
            model_params={"temperature": 0, "response_format": {"type": "json_object"}},
        ),
        driver=driver,
        embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
        schema=SCHEMA,
        from_pdf=True,
        # Le défaut (FixedSizeSplitter, 4000 car.) a deux défauts : un chunk vaut
        # presque un document, et la coupe se fait au caractère près, en plein
        # milieu des tableaux de plafonds. Le découpeur récursif respecte les
        # sauts de ligne : mesuré, il garde les tableaux entiers sur ce corpus.
        text_splitter=LangChainTextSplitterAdapter(
            RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        ),
        # l'index vectoriel existant `GrahRAG` porte sur Chunk.textEmbedding,
        # le défaut de la librairie est `embedding` -> l'index resterait vide
        lexical_graph_config=LexicalGraphConfig(chunk_embedding_property="textEmbedding"),
        perform_entity_resolution=True,
        neo4j_database=base(),
    )


async def ingerer(fichiers: list[Path], driver: neo4j.Driver, trace=print) -> int:
    """Ingère des PDF dans le graphe existant : chunks, embeddings, entités.

    N'efface rien — l'ajout est incrémental. La projection de compatibilité est
    rejouée à la fin ; elle est idempotente (SET + MERGE).
    """
    pipeline = construire_pipeline(driver)
    for pdf in fichiers:
        trace(f"ingestion : {pdf.name}")
        await pipeline.run_async(file_path=str(pdf))
    with driver.session(database=base()) as s:
        return s.run(COMPAT).single()["chunks"] or 0


def retirer(driver: neo4j.Driver, filename: str) -> dict:
    """Retire un document du graphe : le document, ses chunks, et les entités qui
    n'ont plus aucun chunk source.

    Une entité citée par plusieurs documents survit — la résolution d'entités les a
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
    """Vide entièrement le graphe. Utilisé avant une reconstruction complète."""
    with driver.session(database=base()) as s:
        n = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        s.run("MATCH (n) DETACH DELETE n")
    return n


async def main() -> None:
    print("Filtrage du corpus (hypothèse : seule la version en vigueur est ingérée)")
    fichiers = corpus()
    print(f"\n{len(fichiers)} document(s) retenu(s)\n")

    driver = connexion()
    n = await ingerer(fichiers, driver)
    print(f"\ncompatibilité API/Streamlit : {n} chunks projetés (filename + CONTAINS_CHUNK)")

    with driver.session(database=base()) as s:
        print("\n--- graphe obtenu ---")
        for r in s.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"):
            print(f"  {r['label']:16s} {r['n']}")
        for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC"):
            print(f"  [{r['t']}] {r['n']}")
        r = s.run("MATCH (c:Chunk) RETURN count(c) AS t, count(c.textEmbedding) AS e").single()
        print(f"\n  chunks {r['t']} / embeddings {r['e']}")

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())
