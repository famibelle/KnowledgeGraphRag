"""Démo étude de cas — interrogation : RAG vectoriel seul vs. enrichi par le graphe.

Usage : .venv/bin/python demo_query.py
"""
import os

import neo4j
from dotenv import load_dotenv
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever, VectorRetriever

load_dotenv()

INDEX = "GrahRAG"

# Le chunk récupéré, plus les entités qui y sont rattachées et leurs voisins :
# c'est là que le graphe ajoute ce que la similarité seule ne donne pas.
EXPANSION = """
WITH node AS chunk, score
OPTIONAL MATCH (e:__Entity__)-[:FROM_CHUNK]->(chunk)
OPTIONAL MATCH (e)-[r]-(v:__Entity__)
WITH chunk, score,
     collect(DISTINCT e.name) AS entites,
     collect(DISTINCT
       CASE WHEN v IS NOT NULL
       THEN e.name + ' -[' + type(r) + ']-> ' + v.name
            + CASE WHEN v.montant IS NOT NULL
              THEN ' (' + toString(v.montant) + ' ' + coalesce(v.unite,'') + ')' ELSE '' END
       END) AS liens
RETURN chunk.text AS text,
       'ENTITÉS: ' + apoc.text.join([x IN entites WHERE x IS NOT NULL], ', ') AS info,
       'GRAPHE: ' + apoc.text.join([x IN liens WHERE x IS NOT NULL][..25], ' | ') AS graphe,
       score
"""

QUESTIONS = [
    "Quel est le plafond de remboursement pour l'hébergement lors d'une mission à Paris ?",
    "Qui doit approuver une mission à New York dont le coût prévisionnel est de 6 000 EUR ?",
    "J'organise un séminaire pour 12 personnes avec 20 000 EUR de déplacements groupés. "
    "Quelle procédure d'achat dois-je suivre ?",
]


def main() -> None:
    driver = neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})

    nu = GraphRAG(retriever=VectorRetriever(driver, INDEX, embedder), llm=llm)
    graphe = GraphRAG(
        retriever=VectorCypherRetriever(driver, INDEX, EXPANSION, embedder), llm=llm
    )

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'=' * 78}\nQ{i}. {q}\n{'=' * 78}")
        for nom, rag in (("VECTORIEL SEUL", nu), ("+ GRAPHE     ", graphe)):
            r = rag.search(q, retriever_config={"top_k": 4})
            print(f"\n[{nom}] {r.answer.strip()}")

    driver.close()


if __name__ == "__main__":
    main()
