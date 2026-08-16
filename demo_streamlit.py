"""Interface de démo GraphRAG — support d'étude de cas.

Autonome : parle directement à Neo4j et à OpenAI, sans passer par l'API du projet.
Prérequis : le graphe construit par `demo_build_kg.py`.

Usage : .venv/bin/python -m streamlit run demo_streamlit.py
"""
import ast
import os
import re
from pathlib import Path

import neo4j
import pandas as pd
import pypdf
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j_viz.neo4j import from_neo4j
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever, VectorRetriever

from demo_query import EXPANSION, INDEX

load_dotenv()
PDF_DIR = Path("PDFs")

st.set_page_config(page_title="Démo GraphRAG", page_icon="🕸️", layout="wide")


# --------------------------------------------------------------------------- #
# Ressources partagées
# --------------------------------------------------------------------------- #
@st.cache_resource
def driver() -> neo4j.Driver:
    return neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )


@st.cache_resource
def rags():
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
    return (
        GraphRAG(retriever=VectorRetriever(driver(), INDEX, emb), llm=llm),
        GraphRAG(retriever=VectorCypherRetriever(driver(), INDEX, EXPANSION, emb), llm=llm),
    )


def q(cypher: str, **params) -> list[dict]:
    with driver().session() as s:
        return s.run(cypher, **params).data()


def graphe(cypher: str, hauteur: int = 560, **params) -> None:
    """Rendu interactif via neo4j-viz (NVL, le moteur de Neo4j Bloom).

    Le HTML produit embarque tout le bundle JS : ~6 Mo, mais aucun appel réseau —
    la visualisation fonctionne même sans connexion pendant une présentation.
    """
    with driver().session() as s:
        g = s.run(cypher, **params).graph()
    if not g.nodes:
        st.caption("Aucun résultat pour cette requête.")
        return
    vg = from_neo4j(g)
    # L'ordre compte : colorer par type tant que la légende porte encore les labels,
    # puis remplacer la légende par le nom métier.
    vg.color_nodes(field="caption")
    vg.set_node_captions(property="name")
    # Les Chunk et Document n'ont pas de `name` : sans repli ils s'affichent nus.
    for n in vg.nodes:
        if not n.caption:
            p = n.properties or {}
            n.caption = (
                str(p.get("filename", ""))[:28]
                or f"chunk {p.get('chunk_index', '?')}"
            )
    components.html(vg.render(height=f"{hauteur}px").data, height=hauteur + 10)
    st.caption(f"{len(vg.nodes)} nœuds · {len(vg.relationships)} relations — molette pour zoomer, clic pour déplacer")


def lisible(contenu: str) -> str:
    """Extrait le texte du chunk. Les deux retrievers renvoient des reprs différentes :
    un dict Python pour VectorRetriever, un Record Neo4j pour VectorCypherRetriever."""
    contenu = str(contenu)
    try:  # VectorRetriever : repr de dict
        d = ast.literal_eval(contenu)
        if isinstance(d, dict) and "text" in d:
            return " ".join(str(d["text"]).split())
    except (ValueError, SyntaxError):
        pass
    m = re.search(r'text=(["\'])(.*?)\1(?:\s|,|>)', contenu, re.S)  # Record Neo4j
    return " ".join((m.group(2) if m else contenu).replace("\\n", " ").split())


@st.cache_data(ttl=300)
def corpus_statuts() -> pd.DataFrame:
    lignes = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        head = pypdf.PdfReader(str(pdf)).pages[0].extract_text()[:600]
        m = re.search(r"Statut\s*:\s*(.+)", head)
        statut = m.group(1).strip() if m else None
        lignes.append(
            {
                "Document": pdf.name,
                "Statut déclaré": statut or "— absent —",
                "Ingéré": "✅" if statut and "EN VIGUEUR" in statut.upper() else "❌",
            }
        )
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------- #
# Barre latérale
# --------------------------------------------------------------------------- #
st.sidebar.title("🕸️ Démo GraphRAG")
st.sidebar.caption("Référentiel documentaire d'entreprise")

try:
    stats = q(
        "MATCH (d:Document) WITH count(d) AS d "
        "MATCH (c:Chunk) WITH d, count(c) AS c "
        "MATCH (e:__Entity__) RETURN d, c, count(e) AS e"
    )[0]
    st.sidebar.success(
        f"Neo4j connecté\n\n{stats['d']} documents · {stats['c']} chunks · {stats['e']} entités"
    )
except Exception as exc:  # graphe absent ou base injoignable
    st.sidebar.error(f"Neo4j : {exc}")
    st.sidebar.info("Lancez d'abord `python demo_build_kg.py`")

PAGE = st.sidebar.radio(
    "Étapes",
    [
        "1 · Corpus & filtre",
        "2 · Graphe construit",
        "3 · Entités extraites",
        "4 · Vectoriel vs Graphe",
        "5 · Le contraste",
        "6 · Exploration Cypher",
    ],
)

# --------------------------------------------------------------------------- #
# 1 · Corpus
# --------------------------------------------------------------------------- #
if PAGE.startswith("1"):
    st.title("Corpus et filtre d'ingestion")
    st.markdown(
        "Six documents de procédure interne. L'hypothèse « la GED ne sert que la version "
        "en vigueur » est appliquée **littéralement**, par un filtre sur le champ `Statut` "
        "de l'en-tête — pas par une convention implicite."
    )
    st.dataframe(corpus_statuts(), width="stretch", hide_index=True)

    st.subheader("Pourquoi le filtre n'est pas cosmétique")
    st.markdown(
        "Les documents écartés contiennent des valeurs **périmées qui contredisent** la "
        "directive en vigueur. La FAQ est le cas dangereux : rédigée sous forme de "
        "questions, elle remonte en tête d'une recherche vectorielle."
    )
    st.dataframe(
        pd.DataFrame(
            [
                ["Repas Luxembourg", "35 € forfait", "65 €/j au réel", "35 € forfait ❌"],
                ["Seuil justificatif", "> 25 €", "toute dépense", "> 25 € ❌"],
                ["Délai note de frais", "60 j", "30 j", "60 j ❌"],
                ["Préavis ordre de mission", "5 j ouvrables", "10 j ouvrables", "5 j ❌"],
                ["Classe affaires", "vol > 8 h", "vol > 10 h", "> 8 h ❌"],
                ["Kilométrique", "0,30 €/km", "0,38 €/km", "0,30 € ❌"],
            ],
            columns=["Règle", "v1.2 (abrogée)", "v2.0 (en vigueur)", "FAQ (écartée)"],
        ),
        width="stretch",
        hide_index=True,
    )
    st.info(
        "Six des neuf réponses de la FAQ sont fausses. Aucun réglage de seuil de "
        "similarité ne corrige cela : le texte périmé n'est pas moins pertinent "
        "sémantiquement, il est **plus** pertinent. Il est seulement faux."
    )

# --------------------------------------------------------------------------- #
# 2 · Graphe
# --------------------------------------------------------------------------- #
elif PAGE.startswith("2"):
    st.title("Le graphe construit")

    types = q(
        "MATCH (n:__Entity__) UNWIND labels(n) AS l WITH l WHERE NOT l STARTS WITH '__' "
        "RETURN l AS Type, count(*) AS Nombre ORDER BY Nombre DESC"
    )
    rels = q("MATCH ()-[r]->() RETURN type(r) AS Relation, count(*) AS Nombre ORDER BY Nombre DESC")

    c1, c2 = st.columns(2)
    c1.subheader("Entités typées")
    c1.dataframe(pd.DataFrame(types), width="stretch", hide_index=True)
    c2.subheader("Relations")
    c2.dataframe(pd.DataFrame(rels), width="stretch", hide_index=True)

    st.subheader("Schéma")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; bgcolor="transparent";
          node [shape=box style="rounded,filled" fontname="Helvetica" color="#888"];
          Document [fillcolor="#e3f2fd"]; Chunk [fillcolor="#e3f2fd"];
          Role [fillcolor="#fff3e0"]; Seuil [fillcolor="#fce4ec"];
          Zone [fillcolor="#f1f8e9"]; Outil [fillcolor="#ede7f6"];
          DocumentRef [fillcolor="#fffde7"];
          Chunk -> Document [label="FROM_DOCUMENT"];
          Chunk -> Chunk [label="NEXT_CHUNK"];
          Role -> Chunk [label="FROM_CHUNK" style=dashed color="#bbb"];
          Seuil -> Chunk [label="FROM_CHUNK" style=dashed color="#bbb"];
          Role -> Seuil [label="APPROUVE"];
          Seuil -> Zone [label="S_APPLIQUE_A"];
          Seuil -> Role [label="DECLENCHE"];
          Role -> Outil [label="UTILISE"];
          DocumentRef -> DocumentRef [label="REFERENCE"];
        }
        """
    )

    st.subheader("Ce que le graphe désambiguïse")
    st.markdown("Le tableau des plafonds, tel que `pypdf` l'extrait du PDF :")
    st.code(
        "Zone            Déjeuner  Dîner  Plafond journalier\n"
        "Luxembourg       25 EUR   40 EUR      65 EUR\n"
        "Union européenne 30 EUR   45 EUR      75 EUR",
        language="text",
    )
    st.markdown(
        "Hors de son en-tête de colonne, la ligne `Luxembourg 25 EUR 40 EUR 65 EUR` est "
        "illisible. Dans le graphe, chaque valeur porte sa dimension :"
    )
    plafonds = q(
        "MATCH (x:Seuil)-[:S_APPLIQUE_A]->(z:Zone) WHERE toLower(x.name) CONTAINS 'plafond' "
        "AND x.montant IS NOT NULL "
        "RETURN x.name AS Seuil, x.montant AS Montant, x.unite AS Unité, z.name AS Zone "
        "ORDER BY Seuil"
    )
    st.dataframe(pd.DataFrame(plafonds), width="stretch", hide_index=True)

    st.subheader("Le graphe, en vrai")
    st.caption("Rendu par `neo4j-viz`, le binding Python de NVL — le moteur de Neo4j Bloom.")

    vues = {
        "Tout le graphe métier": (
            "MATCH p=(a:__Entity__)-[r]->(b:__Entity__) RETURN p LIMIT 120", 620),
        "Seuils et leurs zones": (
            "MATCH p=(s:Seuil)-[:S_APPLIQUE_A]->(z:Zone) RETURN p LIMIT 40", 520),
        "Qui approuve quoi": (
            "MATCH p=(r:Role)-[:APPROUVE|DECLENCHE]-(s:Seuil) RETURN p LIMIT 60", 560),
        "Couche lexicale (documents et chunks)": (
            "MATCH p=(c:Chunk)-[:FROM_DOCUMENT]->(d:Document) RETURN p LIMIT 60", 560),
    }
    vue = st.radio("Vue", list(vues), horizontal=True)
    cypher, hauteur = vues[vue]
    with st.expander("Requête"):
        st.code(cypher, language="cypher")
    graphe(cypher, hauteur)

    st.subheader("Voisinage d'un rôle")
    roles = [r["n"] for r in q("MATCH (n:Role) RETURN DISTINCT n.name AS n ORDER BY n")]
    if roles:
        choix = st.selectbox("Rôle", roles)
        graphe(
            "MATCH p=(r:Role {name:$n})-[]-(v:__Entity__) RETURN p LIMIT 30",
            480,
            n=choix,
        )

# --------------------------------------------------------------------------- #
# 3 · Entités extraites
# --------------------------------------------------------------------------- #
elif PAGE.startswith("3"):
    st.title("Ce que le LLM a extrait")
    st.markdown(
        "58 entités, produites par `gpt-4o-mini` sous **schéma contraint** : les types et "
        "leurs propriétés sont imposés, le modèle ne peut pas en inventer d'autres. "
        "Chaque entité garde le lien vers le chunk dont elle provient — la traçabilité "
        "est une propriété du graphe, pas une promesse."
    )

    entites = q(
        """
        MATCH (e:__Entity__)
        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c:Chunk)-[:FROM_DOCUMENT]->(doc:Document)
        RETURN [l IN labels(e) WHERE NOT l STARTS WITH '__'][0] AS Type,
               e.name AS Nom, e.montant AS Montant, e.unite AS Unité,
               e.consequence AS Conséquence,
               collect(DISTINCT doc.filename)[0] AS Source,
               collect(DISTINCT substring(c.text, 0, 240))[0] AS Extrait
        ORDER BY Type, Nom
        """
    )
    df = pd.DataFrame(entites)

    types = sorted(df["Type"].dropna().unique())
    choix = st.multiselect("Filtrer par type", types, default=types)
    vue = df[df["Type"].isin(choix)]

    st.dataframe(
        vue.drop(columns=["Extrait"]), width="stretch", hide_index=True, height=420
    )
    st.caption(f"{len(vue)} entité(s) sur {len(df)}")

    st.subheader("Provenance")
    if not vue.empty:
        nom = st.selectbox("Entité", vue["Nom"].tolist())
        ligne = vue[vue["Nom"] == nom].iloc[0]
        c1, c2 = st.columns([1, 2])
        c1.metric("Type", ligne["Type"])
        if pd.notna(ligne["Montant"]):
            c1.metric("Valeur", f"{ligne['Montant']:g} {ligne['Unité'] or ''}")
        c2.caption(f"**Source :** {ligne['Source']}")
        c2.info(ligne["Extrait"] or "—")

    st.subheader("Ce qui ne marche pas")
    st.markdown(
        "Le tableau ci-dessus contient des erreurs réelles, gardées telles quelles :\n\n"
        "- **`DocumentRef`** capte du bruit — `Directive`, `directive`, `version 1.2` ne "
        "sont pas des documents, malgré l'instruction de schéma qui l'interdit.\n"
        "- **La négation n'est pas gérée** : la directive dit que le seuil de 25 EUR *est "
        "supprimé*, l'extracteur en fait un seuil de 25 EUR.\n"
        "- **La chaîne d'approbation est juste à 4/5** : le palier 1 500–15 000 EUR se voit "
        "attribuer le Comité de Direction au lieu du responsable budgétaire.\n\n"
        "Et l'extraction n'est **pas déterministe** : à `temperature=0`, deux exécutions "
        "donnent 58 et 61 entités. Les valeurs métier, elles, restent stables."
    )

# --------------------------------------------------------------------------- #
# 4 · Comparaison
# --------------------------------------------------------------------------- #
elif PAGE.startswith("4"):
    st.title("Vectoriel seul vs enrichi par le graphe")

    presets = {
        "Plafond d'hébergement": "Quel est le plafond de remboursement pour l'hébergement "
        "lors d'une mission à Paris ?",
        "Circuit d'approbation": "Qui doit approuver une mission à New York dont le coût "
        "prévisionnel est de 6 000 EUR ?",
        "Séminaire groupé": "J'organise un séminaire pour 12 personnes avec 20 000 EUR de "
        "déplacements groupés. Quelle procédure d'achat dois-je suivre ?",
    }
    p = st.selectbox("Question type", list(presets))
    question = st.text_area("Question", presets[p], height=80)

    if st.button("Interroger", type="primary"):
        nu, gr = rags()
        c1, c2 = st.columns(2)
        for col, titre, rag in ((c1, "🔍 Vectoriel seul", nu), (c2, "🕸️ + Graphe", gr)):
            with col:
                st.subheader(titre)
                with st.spinner("…"):
                    res = rag.search(question, retriever_config={"top_k": 4},
                                     return_context=True)
                st.markdown(res.answer)
                with st.expander(f"Contexte récupéré ({len(res.retriever_result.items)} chunks)"):
                    for n, item in enumerate(res.retriever_result.items, 1):
                        st.caption(f"**{n}.** {lisible(item.content)[:350]}…")

    st.info(
        "Sur ce type de question — une consultation ponctuelle — le vectoriel seul suffit. "
        "Le graphe affine la réponse mais ne la corrige pas. Voir l'étape 4 pour le cas "
        "où l'écart devient structurel."
    )

# --------------------------------------------------------------------------- #
# 5 · Le contraste
# --------------------------------------------------------------------------- #
elif PAGE.startswith("5"):
    st.title("Là où l'écart devient structurel")
    QUESTION = (
        "Liste TOUS les seuils du référentiel qui déclenchent une intervention de la "
        "Direction Financière, avec leur montant."
    )
    st.markdown(f"> **{QUESTION}**")
    st.markdown(
        "C'est une question d'**agrégation** : elle demande de balayer le référentiel et "
        "de filtrer sur un critère. Un `top_k` ne peut structurellement pas le faire."
    )

    if st.button("Comparer", type="primary"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔍 RAG vectoriel")
            nu, _ = rags()
            with st.spinner("…"):
                st.markdown(nu.search(QUESTION, retriever_config={"top_k": 4}).answer)
            st.error(
                "Le modèle récupère le tableau des seuils d'achat, lexicalement proche de "
                "la question, et le recopie sans filtrer sur le critère demandé."
            )
        with c2:
            st.subheader("🕸️ Requête sur le graphe")
            cypher = (
                "MATCH (r:Role)-[rel]-(x:Seuil)\n"
                "WHERE toLower(r.name) CONTAINS 'financi'\n"
                "RETURN DISTINCT x.name AS Seuil, x.montant AS Montant, x.unite AS Unité\n"
                "ORDER BY x.montant"
            )
            st.code(cypher, language="cypher")
            st.dataframe(pd.DataFrame(q(cypher)), width="stretch", hide_index=True)
            st.success("Ensemble correct, réparti sur deux documents distincts.")

# --------------------------------------------------------------------------- #
# 6 · Cypher
# --------------------------------------------------------------------------- #
else:
    st.title("Exploration Cypher")
    exemples = {
        "Seuils et leur zone": "MATCH (s:Seuil)-[:S_APPLIQUE_A]->(z:Zone)\n"
        "RETURN s.name AS seuil, s.montant AS montant, z.name AS zone ORDER BY montant",
        "Qui approuve quoi": "MATCH (r:Role)-[:APPROUVE]->(s:Seuil)\n"
        "RETURN r.name AS role, s.name AS seuil, s.montant AS montant ORDER BY role",
        "Chunks par document": "MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d:Document)\n"
        "RETURN d.filename AS document, count(c) AS chunks ORDER BY document",
        "Renvois entre documents": "MATCH (a:DocumentRef)-[:REFERENCE]->(b:DocumentRef)\n"
        "RETURN a.name AS source, b.name AS cible",
    }
    choix = st.selectbox("Exemple", list(exemples))
    cypher = st.text_area("Requête", exemples[choix], height=110)
    if st.button("Exécuter", type="primary"):
        try:
            rows = q(cypher)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(f"{len(rows)} ligne(s)")
        except Exception as exc:
            st.error(str(exc))
