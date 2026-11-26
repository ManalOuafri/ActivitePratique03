"""
Interface Streamlit pour tester le RAG multimodal.

Lancement :
    .venv\\Scripts\\streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

import rag_core

st.set_page_config(page_title="RAG multimodal", page_icon="📄", layout="wide")
st.title("RAG multimodal — Attention Is All You Need")
st.caption(
    f"Recherche : CLIP {rag_core.MODEL_NAME} + ChromaDB · "
    f"Génération : {rag_core.MODELE_GENERATION} (Ollama, local) · "
    f"Appareil : {rag_core.DEVICE}"
)


@st.cache_resource(show_spinner="Chargement de CLIP et de ChromaDB…")
def initialiser():
    return rag_core._charger()


initialiser()

with st.sidebar:
    st.header("Paramètres")
    choix_type = st.radio("Type de sources", ["tous", "texte", "image"], horizontal=True)
    k = st.slider("Nombre de sources (k)", 1, 12, 6)
    max_images = st.slider("Images transmises au LLM", 0, 3, 1)
    st.info("Le LLM tourne en local : une réponse peut prendre plusieurs minutes.")

question = st.text_area(
    "Question",
    placeholder="Ex. : Why does the Transformer use multi-head attention?",
)
image_televersee = st.file_uploader(
    "Image de requête (facultatif) : la recherche se fait alors à partir de l'image",
    type=["png", "jpg", "jpeg", "webp"],
)

if image_televersee:
    st.image(image_televersee, caption="Image de requête", width=350)

if st.button("Interroger le RAG", type="primary"):
    if not question.strip():
        st.warning("Saisissez une question.")
        st.stop()

    chemin_image = None
    if image_televersee:
        suffixe = Path(image_televersee.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffixe) as fichier:
            fichier.write(image_televersee.getvalue())
            chemin_image = fichier.name

    with st.spinner("Recherche des sources puis génération de la réponse…"):
        try:
            sortie = rag_core.rag(
                question=question,
                k=k,
                type_resultat=None if choix_type == "tous" else choix_type,
                image_requete=chemin_image,
                max_images=max_images,
            )
        except Exception as erreur:
            st.error(f"Erreur : {erreur}")
            st.stop()
        finally:
            if chemin_image:
                Path(chemin_image).unlink(missing_ok=True)

    st.subheader("Réponse")
    st.markdown(sortie["reponse"])

    for message in sortie["avertissements"]:
        st.warning(message)

    st.subheader("Sources retrouvées")
    transmises = {s["id"] for s in sortie["sources_transmises"]}
    for source in sortie["sources_recuperees"]:
        page = source["page"] if source["page"] != -1 else "inconnue"
        etat = "transmise au LLM" if source["id"] in transmises else "non transmise"
        titre = (
            f"[{source['reference']}] {source['type']} · page {page} · "
            f"similarité {source['similarite']:.3f} · {etat}"
        )
        with st.expander(titre):
            if source["type"] == "image" and Path(source["source"]).is_file():
                st.image(Image.open(source["source"]), width=500)
            else:
                st.write(source["contenu"])
