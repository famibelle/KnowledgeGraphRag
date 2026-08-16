"""Interface de démo GraphRAG — support d'étude de cas.

Autonome : parle directement à Neo4j et à OpenAI, sans passer par l'API du projet.
Prérequis : le graphe construit par `demo_build_kg.py`.

Usage : .venv/bin/python -m streamlit run demo_streamlit.py
"""
import asyncio
import os
import re
from pathlib import Path

import neo4j
import pandas as pd
import pypdf
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM
from neo4j_viz.neo4j import from_neo4j

import demo_build_kg as kg

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
        "4 · Interroger le graphe",
        "5 · Exploration Cypher",
        "6 · Gérer le corpus",
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
        f"{q('MATCH (e:__Entity__) RETURN count(e) AS n')[0]['n']} entités, produites par "
        "`gpt-4o-mini` sous **schéma contraint** : les types et "
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
# 4 · Interroger le graphe
# --------------------------------------------------------------------------- #
elif PAGE.startswith("4"):
    st.title("Interroger le graphe")
    st.markdown(
        "Des questions métier traduites en une requête sur le graphe. Pas de recherche "
        "vectorielle ici : la récupération est **déterministe et exhibable** — la requête "
        "est affichée, le résultat s'en déduit, et un auditeur peut contester la requête."
    )

    QUESTIONS = {
        "Quels seuils impliquent la Direction Financière ?": (
            """MATCH (r:Role)-[rel]-(s:Seuil)
WHERE toLower(r.name) CONTAINS 'financi'
RETURN DISTINCT s.name AS Seuil, s.montant AS Montant, s.unite AS Unité
ORDER BY s.montant""",
            "Question d'**agrégation** : elle demande de balayer le référentiel et de "
            "filtrer sur un critère. Le résultat croise deux documents distincts.",
        ),
        "Quels plafonds s'appliquent selon la zone ?": (
            """MATCH (s:Seuil)-[:S_APPLIQUE_A]->(z:Zone)
WHERE s.montant IS NOT NULL
RETURN s.name AS Seuil, s.montant AS Montant, s.unite AS Unité, z.name AS Zone
ORDER BY Seuil""",
            "Chaque valeur porte sa dimension. Dans le PDF, la ligne "
            "`Luxembourg 25 EUR 40 EUR 65 EUR` est illisible hors de son en-tête.",
        ),
        "Qui approuve quoi, et à partir de quel montant ?": (
            """MATCH (r:Role)-[:APPROUVE|DECLENCHE]-(s:Seuil)
RETURN r.name AS Rôle, s.name AS Seuil, s.montant AS Montant, s.consequence AS Conséquence
ORDER BY s.montant, Rôle""",
            "La chaîne d'approbation, reconstituée depuis deux documents. "
            "⚠️ Elle est juste à 4/5 — voir les limites en page 3.",
        ),
        "Quels documents se renvoient les uns aux autres ?": (
            """MATCH (a:DocumentRef)-[:REFERENCE]->(b:DocumentRef)
RETURN a.name AS Source, b.name AS Cible""",
            "Le graphe de renvois croisés, extrait sans effort supplémentaire.",
        ),
        "Quels outils interviennent dans les procédures ?": (
            """MATCH (r:Role)-[:UTILISE]->(o:Outil)
RETURN o.name AS Outil, collect(DISTINCT r.name) AS Rôles""",
            "Concur revient partout : c'est l'outil central du référentiel.",
        ),
    }

    choix = st.selectbox("Question", list(QUESTIONS))
    cypher, commentaire = QUESTIONS[choix]
    st.caption(commentaire)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Requête**")
        st.code(cypher, language="cypher")
    with c2:
        st.markdown("**Résultat**")
        try:
            rows = q(cypher)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=300)
            st.caption(f"{len(rows)} ligne(s)")
        except Exception as exc:
            rows = []
            st.error(str(exc))

    if rows and st.button("Formuler la réponse", type="primary"):
        with st.spinner("…"):
            llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
            reponse = llm.invoke(
                f"Réponds en français, brièvement, à partir de ces seules données "
                f"extraites d'un référentiel de procédures internes.\n\n"
                f"Question : {choix}\n\nDonnées :\n{rows}"
            ).content
        st.success(reponse)
        st.caption(
            "La génération est ancrée sur le résultat de la requête, pas sur des chunks "
            "récupérés par similarité : chaque chiffre cité est traçable à une ligne."
        )

    st.divider()
    st.subheader("Visualiser ce résultat")
    graphe_par_question = {
        "Quels seuils impliquent la Direction Financière ?":
            "MATCH p=(r:Role)-[]-(s:Seuil) WHERE toLower(r.name) CONTAINS 'financi' RETURN p",
        "Quels plafonds s'appliquent selon la zone ?":
            "MATCH p=(s:Seuil)-[:S_APPLIQUE_A]->(z:Zone) RETURN p LIMIT 40",
        "Qui approuve quoi, et à partir de quel montant ?":
            "MATCH p=(r:Role)-[:APPROUVE|DECLENCHE]-(s:Seuil) RETURN p LIMIT 60",
        "Quels documents se renvoient les uns aux autres ?":
            "MATCH p=(a:DocumentRef)-[:REFERENCE]->(b:DocumentRef) RETURN p",
        "Quels outils interviennent dans les procédures ?":
            "MATCH p=(r:Role)-[:UTILISE]->(o:Outil) RETURN p",
    }
    graphe(graphe_par_question[choix], 480)

# --------------------------------------------------------------------------- #
# 5 · Cypher libre
# --------------------------------------------------------------------------- #
elif PAGE.startswith("5"):
    st.title("Exploration Cypher")
    exemples = {
        "Seuils et leur zone": "MATCH (s:Seuil)-[:S_APPLIQUE_A]->(z:Zone)\n"
        "RETURN s.name AS seuil, s.montant AS montant, z.name AS zone ORDER BY montant",
        "Entités par document": "MATCH (e:__Entity__)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(d:Document)\n"
        "RETURN d.filename AS document, count(DISTINCT e) AS entites ORDER BY entites DESC",
        "Chunks par document": "MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d:Document)\n"
        "RETURN d.filename AS document, count(c) AS chunks ORDER BY document",
        "Entités sans relation métier": "MATCH (e:__Entity__)\n"
        "WHERE NOT (e)-[:APPROUVE|S_APPLIQUE_A|DECLENCHE|UTILISE|REFERENCE]-()\n"
        "RETURN labels(e)[0] AS type, e.name AS nom ORDER BY type, nom",
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

# --------------------------------------------------------------------------- #
# 6 · Gérer le corpus
# --------------------------------------------------------------------------- #
else:
    st.title("Gérer le corpus")
    st.caption(
        "Ajouter ou retirer un PDF déclenche l'extraction des entités et met à jour "
        "le graphe. L'ajout est incrémental ; le retrait supprime le document, ses "
        "chunks, et les entités qui n'ont plus aucune source."
    )

    def rafraichir() -> None:
        st.cache_data.clear()
        st.rerun()

    # --- documents présents ------------------------------------------------- #
    st.subheader("Documents dans le graphe")
    docs = q(
        """
        MATCH (d:Document)
        OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d)
        OPTIONAL MATCH (e:__Entity__)-[:FROM_CHUNK]->(c)
        RETURN d.filename AS Document, count(DISTINCT c) AS Chunks,
               count(DISTINCT e) AS Entités
        ORDER BY Document
        """
    )
    if not docs:
        st.info("Le graphe est vide. Utilisez la reconstruction complète ci-dessous.")
    for d in docs:
        c1, c2, c3, c4 = st.columns([6, 1, 1, 2])
        c1.write(f"📄 {d['Document']}")
        c2.metric("chunks", d["Chunks"], label_visibility="collapsed")
        c3.metric("entités", d["Entités"], label_visibility="collapsed")
        cle = f"del_{d['Document']}"
        if c4.button("🗑️ Retirer", key=cle):
            st.session_state[f"confirm_{cle}"] = True
        if st.session_state.get(f"confirm_{cle}"):
            st.warning(f"Retirer **{d['Document']}** du graphe ?")
            a, b = st.columns(2)
            if a.button("Confirmer", key=f"ok_{cle}", type="primary"):
                with st.spinner("Suppression…"):
                    res = kg.retirer(driver(), d["Document"])
                st.session_state[f"confirm_{cle}"] = False
                st.success(
                    f"{res['supprimés']} nœuds supprimés, dont "
                    f"{res['entités orphelines']} entité(s) devenue(s) orpheline(s)."
                )
                rafraichir()
            if b.button("Annuler", key=f"no_{cle}"):
                st.session_state[f"confirm_{cle}"] = False
                st.rerun()

    st.divider()

    # --- ajout -------------------------------------------------------------- #
    st.subheader("Ajouter des PDF")
    envois = st.file_uploader("Fichiers PDF", type=["pdf"], accept_multiple_files=True)
    if envois:
        apercu = []
        for f in envois:
            chemin = PDF_DIR / f.name
            apercu.append(
                {
                    "Fichier": f.name,
                    "Taille": f"{len(f.getvalue()) / 1024:.0f} Ko",
                    "Déjà présent": "⚠️ oui" if chemin.exists() else "non",
                }
            )
        st.dataframe(pd.DataFrame(apercu), width="stretch", hide_index=True)

        if st.button("Ingérer", type="primary"):
            chemins = []
            for f in envois:
                chemin = PDF_DIR / f.name
                chemin.write_bytes(f.getvalue())
                chemins.append(chemin)
            journal = st.empty()
            with st.spinner("Extraction des entités… (~30 s par document)"):
                n = asyncio.run(
                    kg.ingerer(chemins, driver(), trace=lambda m: journal.caption(m))
                )
            st.success(f"{len(chemins)} document(s) ingéré(s) · {n} chunks au total.")
            rafraichir()

    st.divider()

    # --- reconstruction ----------------------------------------------------- #
    st.subheader("Reconstruire tout le graphe")
    st.caption(
        "Vide le graphe puis réingère les PDF du dossier `PDFs/` dont l'en-tête porte "
        "`Statut : EN VIGUEUR` — le filtre décrit en page 1."
    )
    retenus = kg.corpus() if PDF_DIR.exists() else []
    st.write(f"**{len(retenus)}** document(s) seraient retenus : "
             + ", ".join(f"`{p.name}`" for p in retenus))

    if st.button("Tout reconstruire"):
        st.session_state["confirm_rebuild"] = True
    if st.session_state.get("confirm_rebuild"):
        st.warning("Le graphe actuel sera entièrement effacé.")
        a, b = st.columns(2)
        if a.button("Confirmer la reconstruction", type="primary"):
            st.session_state["confirm_rebuild"] = False
            journal = st.empty()
            with st.spinner("Reconstruction…"):
                efface = kg.vider(driver())
                n = asyncio.run(
                    kg.ingerer(retenus, driver(), trace=lambda m: journal.caption(m))
                )
            st.success(f"{efface} nœuds effacés · {len(retenus)} document(s) réingérés · {n} chunks.")
            rafraichir()
        if b.button("Annuler"):
            st.session_state["confirm_rebuild"] = False
            st.rerun()
