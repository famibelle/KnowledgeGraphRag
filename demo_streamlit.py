"""Constructeur de GraphRAG — déposez des documents, obtenez un graphe interrogeable.

Autonome : parle directement à Neo4j et à OpenAI, sans passer par l'API du projet.

Usage : .venv/bin/python -m streamlit run demo_streamlit.py
"""
import asyncio
import json

import neo4j
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_viz.neo4j import from_neo4j
from streamlit_mermaid import st_mermaid

import demo_build_kg as kg

# Le diagramme de la pipeline, affiché en préambule. Mermaid est embarqué dans le
# paquet streamlit-mermaid : aucun appel réseau, il s'affiche hors connexion.
PIPELINE = """
flowchart TD
    A["1 · Upload de PDF"] --> B["2 · Chunking<br/>1000 caracteres / 100 de recouvrement"]
    B --> C["3 · Embeddings<br/>text-embedding-3-small"]
    B --> D["4 · Extraction des entites<br/>un appel LLM par chunk, extraction LIBRE"]

    D --> E1["5a · Fusion exacte<br/>nom normalise : minuscules, sans accent, sans article"]
    E1 --> E2["5b · Harmonisation des types<br/>un appel LLM"]
    E2 --> E3["5c · Fusion approchee<br/>similarite de noms, rapidfuzz"]

    C --> G[("Chunk<br/>+ textEmbedding")]
    E3 --> H[("Entites<br/>+ relations")]
    A --> I[("Document")]
    H -.->|FROM_CHUNK| G
    G -.->|FROM_DOCUMENT| I

    H --> J["6 · Affichage du graphe<br/>neo4j-viz"]

    classDef llm fill:#fff3e0,stroke:#e69138,color:#333
    classDef store fill:#e3f2fd,stroke:#4a86c8,color:#333
    classDef code fill:#e8f5e9,stroke:#5a9c5a,color:#333
    class D,E2 llm
    class G,H,I store
    class E1,E3 code
"""

st.set_page_config(page_title="GraphRAG Builder", page_icon="🕸️", layout="wide")


# --------------------------------------------------------------------------- #
# Ressources
# --------------------------------------------------------------------------- #
@st.cache_resource
def driver() -> neo4j.Driver:
    return kg.connexion()


def q(cypher: str, **params) -> list[dict]:
    with driver().session(database=kg.base()) as s:
        return s.run(cypher, **params).data()


def rafraichir() -> None:
    st.cache_data.clear()
    st.rerun()


def graphe(cypher: str, hauteur: int = 560, **params) -> None:
    """Rendu interactif via neo4j-viz (NVL, le moteur de Neo4j Bloom).

    Le HTML embarque tout le bundle JS (~6 Mo) et ne fait aucun appel réseau :
    la visualisation survit à une coupure de connexion.
    """
    with driver().session(database=kg.base()) as s:
        g = s.run(cypher, **params).graph()
    if not g.nodes:
        st.caption("Aucun résultat pour cette requête.")
        return
    vg = from_neo4j(g)
    # L'ordre compte : colorer par type tant que la légende porte encore les labels,
    # puis remplacer la légende par le nom métier.
    vg.color_nodes(field="caption")
    vg.set_node_captions(property="name")
    for n in vg.nodes:  # Chunk et Document n'ont pas de `name`
        if not n.caption:
            p = n.properties or {}
            n.caption = str(p.get("filename", ""))[:28] or f"chunk {p.get('chunk_index', '?')}"
    components.html(vg.render(height=f"{hauteur}px").data, height=hauteur + 10)
    st.caption(
        f"{len(vg.nodes)} nœuds · {len(vg.relationships)} relations — "
        "molette pour zoomer, clic pour déplacer"
    )


@st.cache_data(ttl=60)
def etat_graphe() -> dict:
    try:
        return q(
            "OPTIONAL MATCH (d:SourceDocument) WITH count(d) AS docs "
            "OPTIONAL MATCH (c:Chunk) WITH docs, count(c) AS chunks "
            "OPTIONAL MATCH (e:__Entity__) RETURN docs, chunks, count(e) AS entites"
        )[0]
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Barre latérale
# --------------------------------------------------------------------------- #
st.sidebar.title("🕸️ GraphRAG Builder")
st.sidebar.caption("Des documents → un graphe de connaissances interrogeable")

etat = etat_graphe()
if etat:
    st.sidebar.success(
        f"Neo4j connecté\n\n{etat['docs']} documents · {etat['chunks']} chunks · "
        f"{etat['entites']} entités"
    )
else:
    st.sidebar.error("Neo4j injoignable — vérifiez le `.env`")

PAGE = st.sidebar.radio(
    "Étapes",
    [
        "0 · La pipeline",
        "1 · Déposer & construire",
        "2 · Le graphe",
        "3 · Interroger",
    ],
)

if st.session_state.get("consolidation"):
    c = st.session_state["consolidation"]
    st.sidebar.info(
        f"Consolidation : {c['exactes'] + c['approchees']} entités fusionnées, "
        f"{c['types_avant']} → {c['types_apres']} types"
    )

# --------------------------------------------------------------------------- #
# 0 · La pipeline
# --------------------------------------------------------------------------- #
if PAGE.startswith("0"):
    st.title("La pipeline, de bout en bout")
    st.markdown(
        "Des PDF vers un graphe de connaissances, **sans intervention humaine et sans "
        "connaissance préalable des documents**. Les appels au LLM sont en orange, les "
        "étapes déterministes en vert."
    )
    st_mermaid(PIPELINE, height="860px")

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "#### Extraction libre, nettoyage après\n"
        "Aucun schéma imposé : le modèle nomme et type ce qu'il trouve. Mesuré ici, il "
        "produit **41 à 47 types distincts pour 12 chunks**, même à l'intérieur d'un seul "
        "document. La consolidation les ramène ensuite à une dizaine."
    )
    c2.markdown(
        "#### La provenance\n"
        "Chaque entité garde un lien `FROM_CHUNK` vers le passage qui l'a produite, et "
        "chaque chunk un lien `FROM_DOCUMENT` vers son fichier. Toute affirmation du "
        "graphe remonte à une phrase du texte."
    )
    c3.markdown(
        "#### La récupération\n"
        "La question devient du Cypher, **affiché à l'écran**. La réponse est formulée "
        "à partir des seules lignes retournées. Un auditeur peut contester la requête, "
        "pas seulement la réponse."
    )

    st.divider()
    st.caption(
        "Deux couches : la couche **lexicale** (Document → Chunk) porte le texte et les "
        "embeddings ; la couche **métier** (entités typées) porte le sens. C'est la "
        "seconde qui répond aux questions d'agrégation, hors de portée d'une recherche "
        "par similarité."
    )

# --------------------------------------------------------------------------- #
# 1 · Déposer et construire
# --------------------------------------------------------------------------- #
elif PAGE.startswith("1"):
    st.title("Documents")
    st.caption(
        f"Formats acceptés : {', '.join(sorted(kg.FORMATS))}. "
        f"Les fichiers sont déposés dans `{kg.DOSSIER}/`."
    )

    envois = st.file_uploader(
        "Déposer des documents",
        type=[f.lstrip(".") for f in kg.FORMATS],
        accept_multiple_files=True,
    )

    if envois:
        c1, c2 = st.columns([3, 2])
        if c1.button("📥 Déposer et construire le graphe", type="primary"):
            kg.DOSSIER.mkdir(exist_ok=True)
            cibles = []
            for f in envois:
                chemin = kg.DOSSIER / f.name
                chemin.write_bytes(f.getvalue())
                cibles.append(chemin)

            etapes = st.status("Construction du graphe", expanded=True)
            journal = etapes.empty()
            try:
                etapes.write("**Découpage, embeddings et extraction des entités**")
                n = asyncio.run(
                    kg.ingerer(cibles, driver(), trace=lambda m: journal.caption(m))
                )
                avant = q("MATCH (e:__Entity__) RETURN count(e) AS n")[0]["n"]
                etapes.write(f"↳ {n} chunks · **{avant} entités** extraites")

                etapes.write("**Consolidation des entités**")
                stats = kg.consolider_entites(
                    driver(), trace=lambda m: journal.caption(m)
                )
                st.session_state["consolidation"] = stats
                apres = q("MATCH (e:__Entity__) RETURN count(e) AS n")[0]["n"]
                etapes.write(
                    f"↳ {avant} → **{apres} entités** "
                    f"({stats['exactes']} fusions exactes, {stats['approchees']} approchées) · "
                    f"{stats['types_avant']} → **{stats['types_apres']} types**"
                )
                etapes.update(label="Graphe construit", state="complete", expanded=False)
            except Exception as exc:
                etapes.update(label="Échec de la construction", state="error")
                st.error(str(exc))
                st.stop()

            st.success(
                f"{len(cibles)} document(s) · {n} chunks · {apres} entités · "
                f"{stats['types_apres']} types. Voir l'étape 2 pour le graphe."
            )
            rafraichir()

        if c2.button("Déposer seulement"):
            kg.DOSSIER.mkdir(exist_ok=True)
            for f in envois:
                (kg.DOSSIER / f.name).write_bytes(f.getvalue())
            st.success(f"{len(envois)} fichier(s) enregistré(s), sans construction.")
            rafraichir()
        st.caption(
            "« Déposer et construire » enchaîne tout : découpage, embeddings, "
            "extraction des entités et consolidation. Comptez ~30 s par document."
        )

    st.subheader("Corpus")
    fichiers = kg.documents()
    if not fichiers:
        st.info("Aucun document. Déposez des fichiers ci-dessus.")
    else:
        dans_graphe = {r["f"] for r in q("MATCH (d:SourceDocument) RETURN d.filename AS f")}
        for f in fichiers:
            c1, c2, c3, c4 = st.columns([6, 2, 2, 2])
            c1.write(f"📄 {f.name}")
            c2.caption(f"{f.stat().st_size / 1024:.0f} Ko")
            c3.caption("dans le graphe ✅" if f.name in dans_graphe else "non ingéré")
            if c4.button("Supprimer", key=f"rm_{f.name}"):
                st.session_state[f"cf_{f.name}"] = True
            if st.session_state.get(f"cf_{f.name}"):
                st.warning(f"Supprimer **{f.name}** du disque et du graphe ?")
                a, b = st.columns(2)
                if a.button("Confirmer", key=f"ok_{f.name}", type="primary"):
                    res = kg.retirer(driver(), f.name)
                    f.unlink(missing_ok=True)
                    st.session_state[f"cf_{f.name}"] = False
                    st.success(
                        f"Fichier supprimé · {res['supprimés']} nœuds retirés, dont "
                        f"{res['entités orphelines']} entité(s) orpheline(s)."
                    )
                    rafraichir()
                if b.button("Annuler", key=f"no_{f.name}"):
                    st.session_state[f"cf_{f.name}"] = False
                    st.rerun()

# --------------------------------------------------------------------------- #
elif PAGE.startswith("2"):
    st.title("Le graphe obtenu")
    if not etat.get("entites"):
        st.warning("Graphe vide — construisez-le d'abord (étape 1).")
        st.stop()

    types = q(
        "MATCH (n:__Entity__) UNWIND labels(n) AS l WITH l WHERE NOT l STARTS WITH '__' "
        "RETURN l AS Type, count(*) AS Nombre ORDER BY Nombre DESC"
    )
    rels = q("MATCH ()-[r]->() RETURN type(r) AS Relation, count(*) AS Nombre ORDER BY Nombre DESC")
    c1, c2 = st.columns(2)
    c1.subheader("Entités")
    c1.dataframe(pd.DataFrame(types), width="stretch", hide_index=True)
    c2.subheader("Relations")
    c2.dataframe(pd.DataFrame(rels), width="stretch", hide_index=True)

    st.subheader("Vue interactive")
    vues = {
        "Graphe métier": ("MATCH p=(a:__Entity__)-[r]->(b:__Entity__) RETURN p LIMIT 150", 620),
        "Couche lexicale": ("MATCH p=(c:Chunk)-[:FROM_DOCUMENT]->(d:SourceDocument) RETURN p LIMIT 80", 560),
        "Tout": ("MATCH p=()-[r]->() RETURN p LIMIT 300", 640),
    }
    vue = st.radio("Vue", list(vues), horizontal=True)
    cypher, hauteur = vues[vue]
    with st.expander("Requête"):
        st.code(cypher, language="cypher")
    graphe(cypher, hauteur)

    if types:
        st.subheader("Voisinage d'une entité")
        t = st.selectbox("Type", [x["Type"] for x in types])
        noms = [r["n"] for r in q(
            f"MATCH (n:`{t}`) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS n ORDER BY n LIMIT 300"
        )]
        if noms:
            nom = st.selectbox("Entité", noms)
            graphe(
                f"MATCH p=(n:`{t}` {{name: $nom}})-[]-(v:__Entity__) RETURN p LIMIT 40",
                480, nom=nom,
            )

    st.divider()
    st.header("Ce que le LLM a extrait")
    st.markdown(
        f"**{etat['entites']}** entités, sous schéma contraint. Chacune garde le lien "
        "vers le chunk dont elle provient : la traçabilité est une propriété du graphe, "
        "pas une promesse."
    )
    df = pd.DataFrame(
        q(
            """
            MATCH (e:__Entity__)
            OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c:Chunk)-[:FROM_DOCUMENT]->(doc:SourceDocument)
            RETURN [l IN labels(e) WHERE NOT l STARTS WITH '__'][0] AS Type,
                   e.name AS Nom,
                   collect(DISTINCT doc.filename)[0] AS Source,
                   collect(DISTINCT substring(c.text, 0, 240))[0] AS Extrait
            ORDER BY Type, Nom
            """
        )
    )
    types = sorted(df["Type"].dropna().unique())
    choix = st.multiselect("Filtrer par type", types, default=types)
    vue = df[df["Type"].isin(choix)]
    st.dataframe(vue.drop(columns=["Extrait"]), width="stretch", hide_index=True, height=400)
    st.caption(f"{len(vue)} entité(s) sur {len(df)}")

    if not vue.empty:
        st.subheader("Provenance")
        nom = st.selectbox("Entité", vue["Nom"].dropna().tolist())
        ligne = vue[vue["Nom"] == nom].iloc[0]
        st.caption(f"**{ligne['Type']}** — source : {ligne['Source']}")
        st.info(ligne["Extrait"] or "—")
        props = q("MATCH (e:__Entity__ {name:$n}) RETURN properties(e) AS p LIMIT 1", n=nom)
        if props:
            st.json({k: v for k, v in props[0]["p"].items() if k != "id"})

    st.subheader("Limites à garder en tête")
    st.markdown(
        "- L'extraction n'est **pas déterministe** : à `temperature=0`, deux exécutions "
        "sur le même corpus ne donnent pas le même nombre d'entités.\n"
        "- La **négation n'est pas gérée** : « ce seuil est supprimé » produit quand même "
        "une entité pour ce seuil.\n"
        "- Le résolveur d'entités ne compare que `name` : il fusionne des nœuds distincts "
        "qui partagent un libellé générique."
    )

# --------------------------------------------------------------------------- #
else:
    st.title("Interroger le graphe")
    if not etat.get("entites"):
        st.warning("Graphe vide — construisez-le d'abord (étape 1).")
        st.stop()

    st.markdown(
        "La question est traduite en Cypher par le LLM, exécutée sur le graphe, et la "
        "réponse est formulée à partir des seules lignes retournées. **La requête est "
        "affichée** : la récupération est vérifiable, pas devinée."
    )

    question = st.text_input(
        "Question", placeholder="Combien de documents contient le graphe ?"
    )
    if question and st.button("Interroger", type="primary"):
        retriever = Text2CypherRetriever(
            driver=driver(), llm=kg.llm(json_mode=False), neo4j_database=kg.base()
        )
        try:
            with st.spinner("Traduction en Cypher…"):
                res = retriever.search(query_text=question)
        except Exception as exc:
            st.error(
                "Le LLM a produit du Cypher que Neo4j a refusé. C'est la limite connue "
                "de la traduction automatique : reformulez plus simplement, ou écrivez "
                "la requête vous-même ci-dessous."
            )
            st.caption(str(exc)[:400])
            st.stop()

        cypher = (res.metadata or {}).get("cypher", "")
        if cypher:
            st.code(cypher, language="cypher")
        if res.items:
            st.dataframe(
                pd.DataFrame([str(i.content) for i in res.items], columns=["Résultat"]),
                width="stretch", hide_index=True,
            )
            with st.spinner("Formulation…"):
                reponse = kg.llm(json_mode=False).invoke(
                    "Réponds en français, brièvement, à partir de ces seules données.\n\n"
                    f"Question : {question}\n\nDonnées :\n"
                    + "\n".join(str(i.content) for i in res.items[:40])
                ).content
            st.success(reponse)
        else:
            st.warning("La requête n'a retourné aucune ligne.")

    st.divider()
    st.subheader("Requête Cypher directe")
    cypher = st.text_area(
        "Cypher",
        "MATCH (e:__Entity__)-[r]-(v:__Entity__)\n"
        "RETURN e.name AS entite, count(r) AS liens\n"
        "ORDER BY liens DESC LIMIT 15",
        height=110,
    )
    if st.button("Exécuter"):
        try:
            rows = q(cypher)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(f"{len(rows)} ligne(s)")
        except Exception as exc:
            st.error(str(exc))
