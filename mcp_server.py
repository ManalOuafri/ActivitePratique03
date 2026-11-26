"""
Serveur MCP exposant le RAG multimodal comme outil.

Transport stdio : le serveur est lancé par le client (agent_mcp.py).
Test manuel possible avec :  .venv\\Scripts\\mcp dev mcp_server.py
"""

import json
from typing import Literal

from mcp.server.fastmcp import FastMCP

import rag_core

mcp = FastMCP("rag-multimodal")


@mcp.tool()
def consulter_pdf(
    question: str,
    type_resultat: Literal["tous", "texte", "image"] = "tous",
    k: int = 6,
) -> str:
    """
    Consulter les PDF indexés pour répondre à une question documentaire.

    Utiliser cet outil pour rechercher des explications, chiffres,
    méthodes, résultats ou figures présents dans les PDF.

    type_resultat :
    - "tous" : rechercher parmi les textes et les images ;
    - "texte" : rechercher uniquement des passages textuels ;
    - "image" : rechercher uniquement des figures ou images.

    Renvoie un JSON avec la réponse documentaire, les sources
    (pdf, page, type) et les éventuels avertissements.
    """
    sortie = rag_core.rag(
        question=question,
        k=max(1, min(k, 12)),
        type_resultat=None if type_resultat == "tous" else type_resultat,
        max_images=1,
    )
    sources = [
        {
            "reference": s["reference"],
            "id": s["id"],
            "pdf": s["pdf"],
            "page": s["page"],
            "type": s["type"],
        }
        for s in sortie["sources_transmises"]
    ]
    return json.dumps(
        {
            "question": sortie["question"],
            "reponse_rag": sortie["reponse"],
            "sources_transmises": sources,
            "avertissements": sortie["avertissements"],
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    rag_core._charger()  # charger CLIP avant la première requête
    mcp.run(transport="stdio")
