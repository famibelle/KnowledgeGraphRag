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
    A["Documents deposes<br/>pdf · txt · md"] --> B["Extraction du texte"]
    B --> F["Decoupage en chunks<br/>1000 car. / 100 de recouvrement"]

    F --> S1["PASSE 1 — Decouverte<br/>extraction LIBRE sur ~20 chunks<br/>echantillonnes dans tous les documents"]
    S1 --> S1b["Types bruts observes<br/>~70 types, la plupart vus une fois"]
    S1b --> S2["PASSE 2 — Consolidation<br/>un seul appel : fusion des synonymes"]
    S2 --> S2b["Normalisation en code<br/>sans accent · relations en MAJUSCULES"]
    S2b --> E["Schema canonique<br/>~10 types + relations"]

    F --> G["Embeddings<br/>text-embedding-3-small"]
    F --> H["PASSE 3 — Extraction<br/>TOUS les chunks, un appel chacun"]
    E -.->|contraint| H
    H --> I["Resolution d'entites<br/>fusion sur la propriete name"]

    G --> J[("Chunk<br/>+ textEmbedding")]
    I --> K[("Entites typees<br/>+ relations")]
    B --> L[("Document")]
    K -.->|FROM_CHUNK| J
    J -.->|FROM_DOCUMENT| L

    K --> M["Text2Cypher<br/>question en langage naturel"]
    M --> N["Cypher genere<br/>affiche a l'ecran"]
    N --> O["Reponse ancree<br/>sur les lignes retournees"]

    classDef llm fill:#fff3e0,stroke:#e69138,color:#333
    classDef store fill:#e3f2fd,stroke:#4a86c8,color:#333
    classDef code fill:#e8f5e9,stroke:#5a9c5a,color:#333
    class S1,S2,H,M llm
    class J,K,L store
    class S2b code
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
            "OPTIONAL MATCH (d:Document) WITH count(d) AS docs "
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
        "1 · Documents",
        "2 · Schéma d'extraction",
        "3 · Construction",
        "4 · Graphe",
        "5 · Entités extraites",
        "6 · Interroger",
    ],
)

if "schema" in st.session_state:
    st.sidebar.info(
        f"Schéma défini : {len(st.session_state['schema']['node_types'])} types "
        f"d'entités, {len(st.session_state['schema']['relationship_types'])} relations"
    )

# --------------------------------------------------------------------------- #
# 0 · La pipeline
# --------------------------------------------------------------------------- #
if PAGE.startswith("0"):
    st.title("La pipeline, de bout en bout")
    st.markdown(
        "Des documents inconnus a priori vers un graphe de connaissances interrogeable, "
        "**sans intervention humaine**. Les appels au LLM sont en orange, la "
        "normalisation déterministe en vert."
    )
    st_mermaid(PIPELINE, height="860px")

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "#### Pourquoi trois passes\n"
        "En extraction libre, le modèle invente un type par entité ou presque. Mesuré "
        "sur un corpus réel : **55 types distincts sur 12 chunks, dont 48 vus une seule "
        "fois**. La passe 2 les fusionne en une dizaine de types canoniques — sans elle, "
        "le graphe n'a aucune connectivité."
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
        "Les deux couches du graphe : la couche **lexicale** (Document → Chunk) porte le "
        "texte et les embeddings ; la couche **métier** (entités typées et leurs "
        "relations) porte le sens. C'est la seconde qui permet de répondre à des "
        "questions d'agrégation, qu'une recherche par similarité ne sait pas poser."
    )

# --------------------------------------------------------------------------- #
# 1 · Documents
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
    if envois and st.button("Enregistrer", type="primary"):
        kg.DOSSIER.mkdir(exist_ok=True)
        for f in envois:
            (kg.DOSSIER / f.name).write_bytes(f.getvalue())
        st.success(f"{len(envois)} fichier(s) enregistré(s).")
        rafraichir()

    st.subheader("Corpus")
    fichiers = kg.documents()
    if not fichiers:
        st.info("Aucun document. Déposez des fichiers ci-dessus.")
    else:
        dans_graphe = {r["f"] for r in q("MATCH (d:Document) RETURN d.filename AS f")}
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
# 2 · Schéma
# --------------------------------------------------------------------------- #
elif PAGE.startswith("2"):
    st.title("Schéma d'extraction, dérivé automatiquement")
    st.markdown(
        "Les documents n'étant pas connus a priori, le schéma est **dérivé du corpus "
        "lui-même**, en deux passes. Aucune saisie n'est nécessaire ; l'édition reste "
        "possible mais facultative."
    )

    fichiers = kg.documents()
    if not fichiers:
        st.warning("Déposez d'abord des documents (étape 1).")
        st.stop()

    c1, c2 = st.columns([2, 1])
    n = c2.slider("Chunks échantillonnés", 10, 60, 20, step=5,
                  help="Passe 1 : un appel LLM par chunk échantillonné.")
    if c1.button("Dériver le schéma", type="primary"):
        journal = st.empty()
        with st.spinner("Passe 1 : extraction libre · Passe 2 : consolidation…"):
            chunks = kg.decouper(fichiers)
            bruts, rel_brutes = asyncio.run(
                kg.decouvrir_types(chunks, n=n, trace=lambda m: journal.caption(m))
            )
            schema = asyncio.run(
                kg.consolider(bruts, rel_brutes, trace=lambda m: journal.caption(m))
            )
        st.session_state["schema"] = schema
        st.session_state["bruts"] = bruts.most_common()
        st.rerun()

    if st.session_state.get("bruts"):
        bruts = st.session_state["bruts"]
        uniques = sum(1 for _, v in bruts if v == 1)
        a, b, c = st.columns(3)
        a.metric("Types bruts découverts", len(bruts))
        b.metric("Vus une seule fois", f"{uniques} / {len(bruts)}")
        c.metric("Types canoniques retenus", len(st.session_state["schema"]["node_types"]))
        with st.expander("Les types bruts, avant consolidation"):
            st.caption(
                "C'est ce que produit une extraction libre : un type par entité ou "
                "presque, avec les collisions habituelles de casse et de synonymie. "
                "Sans la passe de consolidation, le graphe n'aurait aucune connectivité."
            )
            st.dataframe(
                pd.DataFrame(bruts, columns=["Type brut", "Occurrences"]),
                width="stretch", hide_index=True, height=260,
            )

    if "schema" not in st.session_state:
        st.info("Aucun schéma. Lancez la dérivation ci-dessus.")
        st.session_state["schema"] = {
            "node_types": {}, "relationship_types": [], "patterns": []
        }

    s = st.session_state["schema"]
    st.caption(
        "Édition facultative. Les libellés sont normalisés en code après consolidation : "
        "sans accent, relations en majuscules, propriétés `name`/`nom` retirées — un "
        "prompt ne suffit pas à le garantir."
    )
    texte = st.text_area(
        "Schéma (JSON éditable)",
        json.dumps(s, ensure_ascii=False, indent=2),
        height=380,
    )
    if st.button("Valider le schéma"):
        try:
            nouveau = json.loads(texte)
            assert "node_types" in nouveau and "relationship_types" in nouveau
            st.session_state["schema"] = nouveau
            st.success(
                f"{len(nouveau['node_types'])} types d'entités, "
                f"{len(nouveau['relationship_types'])} relations."
            )
        except Exception as exc:
            st.error(f"JSON invalide : {exc}")

    if s["node_types"]:
        st.subheader("Aperçu")
        st.dataframe(
            pd.DataFrame(
                [{"Type": k, "Propriétés": ", ".join(v) or "—"} for k, v in s["node_types"].items()]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Une propriété `name` est ajoutée d'office à chaque type : le résolveur "
            "d'entités de la librairie ne compare que celle-là."
        )

# --------------------------------------------------------------------------- #
# 3 · Construction
# --------------------------------------------------------------------------- #
elif PAGE.startswith("3"):
    st.title("Construction du graphe")

    fichiers = kg.documents()
    if not fichiers:
        st.warning("Déposez d'abord des documents (étape 1).")
        st.stop()
    auto = not st.session_state.get("schema", {}).get("node_types")
    if auto:
        st.info(
            "Aucun schéma défini : il sera **dérivé automatiquement** du corpus avant "
            "l'extraction (passes 1 et 2). Vous pouvez aussi le régler à l'étape 2."
        )

    dans_graphe = {r["f"] for r in q("MATCH (d:Document) RETURN d.filename AS f")}
    nouveaux = [f for f in fichiers if f.name not in dans_graphe]

    st.markdown(
        f"**{len(fichiers)}** document(s) dans le corpus, dont **{len(nouveaux)}** "
        "pas encore ingéré(s)."
    )
    st.caption("Comptez ~30 s et une dizaine d'appels LLM par document.")

    choix = st.multiselect(
        "Documents à ingérer", [f.name for f in fichiers],
        default=[f.name for f in nouveaux],
    )
    cibles = [f for f in fichiers if f.name in choix]

    def schema_pret(journal) -> dict:
        """Dérive le schéma si aucun n'est défini, puis le met en forme pipeline."""
        if not st.session_state.get("schema", {}).get("node_types"):
            chunks = kg.decouper(cibles)
            bruts, rel = asyncio.run(
                kg.decouvrir_types(chunks, n=20, trace=lambda m: journal.caption(m))
            )
            st.session_state["schema"] = asyncio.run(
                kg.consolider(bruts, rel, trace=lambda m: journal.caption(m))
            )
            st.session_state["bruts"] = bruts.most_common()
        return kg.dict_en_schema(st.session_state["schema"])

    c1, c2 = st.columns(2)
    if c1.button("Construire" if auto else "Ingérer (incrémental)",
                 type="primary", disabled=not cibles):
        journal = st.empty()
        with st.spinner("Dérivation du schéma puis extraction des entités…"):
            schema = schema_pret(journal)
            n = asyncio.run(
                kg.ingerer(cibles, driver(), schema, trace=lambda m: journal.caption(m))
            )
        st.success(
            f"{len(cibles)} document(s) ingéré(s) · {n} chunks · "
            f"{len(st.session_state['schema']['node_types'])} types d'entités."
        )
        rafraichir()

    if c2.button("Tout reconstruire", disabled=not cibles):
        st.session_state["confirm_rebuild"] = True
    if st.session_state.get("confirm_rebuild"):
        st.warning("Le graphe actuel sera entièrement effacé.")
        a, b = st.columns(2)
        if a.button("Confirmer", type="primary"):
            st.session_state["confirm_rebuild"] = False
            journal = st.empty()
            with st.spinner("Reconstruction…"):
                schema = schema_pret(journal)
                efface = kg.vider(driver())
                n = asyncio.run(
                    kg.ingerer(cibles, driver(), schema, trace=lambda m: journal.caption(m))
                )
            st.success(f"{efface} nœuds effacés · {len(cibles)} document(s) · {n} chunks.")
            rafraichir()
        if b.button("Annuler"):
            st.session_state["confirm_rebuild"] = False
            st.rerun()

    if etat.get("chunks"):
        st.divider()
        st.subheader("État")
        v = q("MATCH (c:Chunk) RETURN count(c) AS total, count(c.textEmbedding) AS avec")[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", etat["docs"])
        c2.metric("Chunks", v["total"])
        c3.metric("Embeddings", v["avec"], delta=v["avec"] - v["total"] or None)
        if v["avec"] < v["total"]:
            st.error(
                f"{v['total'] - v['avec']} chunk(s) sans embedding : ils comptent dans "
                "les statistiques mais restent invisibles à la recherche vectorielle."
            )

# --------------------------------------------------------------------------- #
# 4 · Graphe
# --------------------------------------------------------------------------- #
elif PAGE.startswith("4"):
    st.title("Le graphe")
    if not etat.get("entites"):
        st.warning("Graphe vide — construisez-le d'abord (étape 3).")
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
        "Couche lexicale": ("MATCH p=(c:Chunk)-[:FROM_DOCUMENT]->(d:Document) RETURN p LIMIT 80", 560),
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

# --------------------------------------------------------------------------- #
# 5 · Entités extraites
# --------------------------------------------------------------------------- #
elif PAGE.startswith("5"):
    st.title("Ce que le LLM a extrait")
    if not etat.get("entites"):
        st.warning("Graphe vide — construisez-le d'abord (étape 3).")
        st.stop()

    st.markdown(
        f"**{etat['entites']}** entités, sous schéma contraint. Chacune garde le lien "
        "vers le chunk dont elle provient : la traçabilité est une propriété du graphe, "
        "pas une promesse."
    )
    df = pd.DataFrame(
        q(
            """
            MATCH (e:__Entity__)
            OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c:Chunk)-[:FROM_DOCUMENT]->(doc:Document)
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
# 6 · Interroger
# --------------------------------------------------------------------------- #
else:
    st.title("Interroger le graphe")
    if not etat.get("entites"):
        st.warning("Graphe vide — construisez-le d'abord (étape 3).")
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
