"""Pipeline GraphRAG : des PDF déposés par l'utilisateur vers un graphe.

Découpage → embeddings → extraction libre des entités par chunk → consolidation
des entités → graphe. Chaque chunk garde un lien vers son document, chaque entité
vers le chunk dont elle provient.

En ligne de commande, traite tout le dossier `PDFs/` :
    .venv/bin/python demo_build_kg.py
"""
import asyncio
import json
import os
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
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from rapidfuzz import fuzz
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
    """Pilote borné dans le temps.

    Sans ces bornes, une connexion du pool devenue morte — mise en veille de la
    machine, bascule de réseau, instance Aura suspendue — bloque la requête
    suivante pendant des minutes au lieu de lever une erreur. L'interface reste
    alors figée sur la première requête venue, sans message.
    """
    return neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        connection_timeout=10,  # établissement de la connexion
        connection_acquisition_timeout=15,  # attente d'une connexion du pool
        max_transaction_retry_time=10,
        liveness_check_timeout=0,  # teste toute connexion réutilisée du pool
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
# Consolidation des entités
# --------------------------------------------------------------------------- #
# L'extraction est libre : le modèle nomme et type ce qu'il trouve, sans liste
# imposée. Le nettoyage se fait donc APRÈS, sur les entités produites, en trois
# étapes dont deux sont déterministes.

HARMONISATION = """Voici les types d'entités produits par une extraction libre sur un
corpus, avec leur nombre d'occurrences. Ils sont redondants : synonymes, variantes de
casse, granularités mélangées.

Associe chaque type à un type canonique, en visant {cible} types au total.
Règles : libellés en CamelCase sans accent ni espace ; fusionne les synonymes et les
variantes de casse ; ne laisse aucun type sans correspondance.

TYPES OBSERVÉS :
{types}

Réponds en JSON strict, en associant CHAQUE type observé à son canonique :
{{"correspondances": {{"type observé": "TypeCanonique"}}}}
"""


def _json(reponse: str) -> dict:
    try:
        return json.loads(reponse[reponse.find("{") : reponse.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {}


def cle_nom(nom: str) -> str:
    """Clé de regroupement : minuscules, sans accent, sans ponctuation, sans article."""
    s = unicodedata.normalize("NFD", str(nom).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"^(le |la |les |l'|un |une |des |the |a )", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# En extraction libre, le modèle choisit aussi ses noms de propriétés : sur un corpus
# de test, 169 entités portaient `name`, 128 `title`, et certaines aucun identifiant.
# Sans cette étape, la moitié des entités échappe à la consolidation et s'affiche sans
# étiquette dans le graphe.
CANDIDATS_NOM = ("name", "title", "titre", "nom", "label", "libelle", "id")


def nommer_entites(driver: neo4j.Driver, trace=print) -> dict:
    """Étape 0 — garantit que chaque entité porte un `name` exploitable.

    Reprend le premier identifiant disponible ; à défaut, la plus longue valeur
    textuelle. Les entités sans aucune identité sont supprimées : elles ne
    désignent rien et ne peuvent ni être fusionnées ni être affichées.
    """
    inutiles = {"", "unknown", "n/a", "none", "null", "inconnu"}
    renommees = supprimees = 0
    with driver.session(database=base()) as s:
        lignes = s.run(
            "MATCH (e:__Entity__) WHERE e.name IS NULL OR trim(e.name) = '' "
            "RETURN elementId(e) AS id, properties(e) AS p"
        ).data()
        for l in lignes:
            props = l["p"] or {}
            valeurs = [
                str(props[k]) for k in CANDIDATS_NOM
                if isinstance(props.get(k), str) and str(props[k]).strip().lower() not in inutiles
            ]
            if not valeurs:  # à défaut, la plus longue chaîne utile
                valeurs = sorted(
                    (v for v in props.values()
                     if isinstance(v, str) and v.strip().lower() not in inutiles),
                    key=len, reverse=True,
                )
            if valeurs:
                s.run("MATCH (e) WHERE elementId(e) = $id SET e.name = $n",
                      id=l["id"], n=valeurs[0][:120])
                renommees += 1
            else:
                s.run("MATCH (e) WHERE elementId(e) = $id DETACH DELETE e", id=l["id"])
                supprimees += 1
    trace(f"0 · nommage : {renommees} entité(s) renommée(s), {supprimees} sans identité supprimée(s)")
    return {"renommees": renommees, "supprimees": supprimees}


def _fusionner(session, groupes: list[list[str]]) -> int:
    """Fusionne des groupes de nœuds via APOC, en gardant le nom le plus complet."""
    fusions = 0
    for ids in groupes:
        session.run(
            "MATCH (e) WHERE elementId(e) IN $ids "
            "WITH e ORDER BY size(coalesce(e.name, '')) DESC "
            "WITH collect(e) AS noeuds "
            "CALL apoc.refactor.mergeNodes(noeuds, "
            "  {properties: 'discard', mergeRels: true}) YIELD node "
            "RETURN node",
            ids=ids,
        ).consume()
        fusions += len(ids) - 1
    return fusions


def consolider_entites(
    driver: neo4j.Driver, seuil_flou: int = 92, cible_types: int = 10, trace=print
) -> dict:
    """Trois étapes, dont une seule fait appel au LLM.

    A — fusion exacte sur le nom normalisé, toutes étiquettes confondues : c'est ce
        qui réunit « la Direction des Achats » et « DIRECTION DES ACHATS », et du
        même coup les doublons de type portant sur la même entité.
    B — harmonisation des étiquettes en un jeu canonique (un appel LLM).
    C — fusion approchée entre entités de même étiquette, sur similarité de nom.
    """
    stats = {"exactes": 0, "types_avant": 0, "types_apres": 0, "approchees": 0}
    stats.update(nommer_entites(driver, trace=trace))

    with driver.session(database=base()) as s:
        # --- A : fusion exacte ------------------------------------------------ #
        lignes = s.run(
            "MATCH (e:__Entity__) WHERE e.name IS NOT NULL "
            "RETURN elementId(e) AS id, e.name AS nom"
        ).data()
        groupes: dict[str, list[str]] = {}
        for l in lignes:
            groupes.setdefault(cle_nom(l["nom"]), []).append(l["id"])
        doublons = [ids for ids in groupes.values() if len(ids) > 1]
        stats["exactes"] = _fusionner(s, doublons)
        trace(f"A · fusion exacte : {stats['exactes']} entité(s) fusionnée(s)")

    # --- B : harmonisation des étiquettes ------------------------------------- #
    with driver.session(database=base()) as s:
        etiquettes = Counter(
            {
                r["l"]: r["n"]
                for r in s.run(
                    "MATCH (n:__Entity__) UNWIND labels(n) AS l "
                    "WITH l WHERE NOT l STARTS WITH '__' "
                    "RETURN l AS l, count(*) AS n ORDER BY n DESC"
                )
            }
        )
    stats["types_avant"] = len(etiquettes)
    if len(etiquettes) > cible_types:
        reponse = llm().invoke(
            HARMONISATION.format(
                cible=cible_types,
                types="\n".join(f"- {k} ({v})" for k, v in etiquettes.most_common(120)),
            )
        )
        corr = _json(reponse.content).get("correspondances", {})
        propre = lambda s_: re.sub(
            r"[^A-Za-z0-9]", "",
            "".join(c for c in unicodedata.normalize("NFD", str(s_))
                    if unicodedata.category(c) != "Mn"),
        )
        with driver.session(database=base()) as s:
            for ancien, neuf in corr.items():
                neuf = propre(neuf)
                if not neuf or ancien not in etiquettes or neuf == ancien:
                    continue
                s.run(
                    f"MATCH (e:`{ancien}`) REMOVE e:`{ancien}` SET e:`{neuf}`"
                ).consume()

    with driver.session(database=base()) as s:
        stats["types_apres"] = s.run(
            "MATCH (n:__Entity__) UNWIND labels(n) AS l "
            "WITH l WHERE NOT l STARTS WITH '__' RETURN count(DISTINCT l) AS n"
        ).single()["n"]
    trace(f"B · étiquettes : {stats['types_avant']} -> {stats['types_apres']}")

    # --- C : fusion approchée, à étiquette égale ------------------------------ #
    with driver.session(database=base()) as s:
        lignes = s.run(
            "MATCH (e:__Entity__) WHERE e.name IS NOT NULL "
            "RETURN elementId(e) AS id, e.name AS nom, "
            "[l IN labels(e) WHERE NOT l STARTS WITH '__'][0] AS type"
        ).data()
        par_type: dict[str, list] = {}
        for l in lignes:
            par_type.setdefault(l["type"], []).append(l)

        groupes = []
        for entites in par_type.values():
            restants = list(entites)
            while restants:
                pivot = restants.pop(0)
                proches = [
                    e for e in restants
                    if fuzz.token_sort_ratio(pivot["nom"], e["nom"]) >= seuil_flou
                ]
                if proches:
                    groupes.append([pivot["id"]] + [e["id"] for e in proches])
                    restants = [e for e in restants if e not in proches]
        stats["approchees"] = _fusionner(s, groupes)
    trace(f"C · fusion approchée : {stats['approchees']} entité(s) fusionnée(s)")
    return stats


# --------------------------------------------------------------------------- #
# Construction du graphe
# --------------------------------------------------------------------------- #
# SimpleKGPipeline écrit (:Chunk)-[:FROM_DOCUMENT]->(:SourceDocument) avec `path`.
# L'API et l'interface Streamlit du projet sont indexées sur `filename` et sur
# (:Document)-[:CONTAINS_CHUNK]->(:Chunk). Sans projection, le graphe est
# invisible pour elles : toutes les requêtes renvoient du vide, sans erreur.
COMPAT = """
MATCH (d:SourceDocument)
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
MATCH (c:Chunk) WHERE NOT (c)-[:FROM_DOCUMENT]->(:SourceDocument)
WITH collect(c) AS orphelins
MERGE (d:SourceDocument {filename: $filename})
SET d.path = $path
WITH d, orphelins UNWIND orphelins AS c
MERGE (c)-[:FROM_DOCUMENT]->(d)
RETURN count(c) AS rattachés
"""


def construire_pipeline(driver: neo4j.Driver, depuis_pdf: bool = True) -> SimpleKGPipeline:
    return SimpleKGPipeline(
        llm=llm(),
        driver=driver,
        embedder=OpenAIEmbeddings(model=MODELE_EMBEDDING),
        # Extraction LIBRE : aucun schéma imposé, le modèle nomme et type ce qu'il
        # trouve. Le nettoyage a lieu après, dans consolider_entites().
        from_pdf=depuis_pdf,
        # Le défaut (FixedSizeSplitter, 4000 car.) découpe au caractère près, en
        # plein milieu des tableaux. Le découpeur récursif respecte les sauts de ligne.
        text_splitter=LangChainTextSplitterAdapter(
            RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        ),
        # L'index vectoriel `GrahRAG` porte sur Chunk.textEmbedding ; le défaut de
        # la librairie est `embedding`, l'index resterait vide.
        # `Document` est un type que le LLM produit spontanément en extraction libre :
        # ses entités écrasaient alors le label réservé aux fichiers, sans `__Entity__`,
        # donc invisibles à la consolidation et sans provenance. Un label improbable
        # supprime la collision par construction.
        lexical_graph_config=LexicalGraphConfig(
            chunk_embedding_property="textEmbedding",
            document_node_label="SourceDocument",
        ),
        # Le résolveur de la librairie ne compare que `name`, à étiquette égale.
        # Trop faible ici, puisque l'extraction libre produit aussi des étiquettes
        # divergentes : la consolidation est faite après, par consolider_entites().
        perform_entity_resolution=False,
        neo4j_database=base(),
    )


async def ingerer(fichiers: list[Path], driver: neo4j.Driver, trace=print) -> int:
    """Découpage, embeddings et extraction des entités. Incrémental : n'efface rien."""
    pdf = construire_pipeline(driver, depuis_pdf=True)
    txt = None
    for f in fichiers:
        trace(f"ingestion : {f.name}")
        if f.suffix.lower() == ".pdf":
            await pdf.run_async(file_path=str(f))
        else:
            txt = txt or construire_pipeline(driver, depuis_pdf=False)
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
            "MATCH (d:SourceDocument {filename: $f}) "
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

    driver = connexion()

    print("\nDécoupage, embeddings et extraction des entités…")
    n = await ingerer(fichiers, driver, trace=lambda m: print("  " + m))
    print(f"  {n} chunks")

    print("\nConsolidation des entités…")
    stats = consolider_entites(driver, trace=lambda m: print("  " + m))
    print(f"  {stats}")

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
