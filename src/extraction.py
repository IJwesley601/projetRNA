"""
extraction.py
Bloc 1 du pipeline : Extraction PDF -> texte propre -> passages segmentés.

Utilise PyMuPDF (fitz), qui est plus rapide et plus robuste que pdfplumber
sur des PDF avec mise en page complexe (colonnes, cours en slides exportées, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF

from .utils import CONFIG, clean_text, split_into_passages


@dataclass
class Passage:
    """Un passage de texte extrait du document, avec sa provenance.
    Garder la traçabilité (page d'origine) est utile pour la démo :
    on peut indiquer à l'utilisateur d'où vient chaque question."""

    id: str
    text: str
    page: int


def extract_text_by_page(pdf_path: str) -> list[tuple[int, str]]:
    """Extrait le texte brut de chaque page d'un PDF.

    Retourne une liste de tuples (numéro_de_page, texte_brut).
    Lève une erreur explicite si le PDF n'a pas de couche de texte
    exploitable (cas d'un PDF scanné) : ce projet ne fait pas d'OCR.
    """
    doc = fitz.open(pdf_path)
    pages: list[tuple[int, str]] = []

    for page_number, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        if raw.strip():
            pages.append((page_number, raw))

    doc.close()

    if not pages:
        raise ValueError(
            "Aucun texte exploitable n'a été trouvé dans ce PDF. "
            "S'il s'agit d'un document scanné (image), une étape d'OCR "
            "serait nécessaire avant ce pipeline."
        )

    return pages


def extract_passages(pdf_path: str) -> list[Passage]:
    """Pipeline complet d'extraction : PDF -> texte nettoyé par page -> passages.

    C'est la fonction principale à appeler depuis l'app Streamlit ou main.py.
    """
    pages = extract_text_by_page(pdf_path)

    passages: list[Passage] = []
    counter = 0

    for page_number, raw_text in pages:
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        for chunk in split_into_passages(cleaned, CONFIG.min_passage_chars, CONFIG.max_passage_chars):
            counter += 1
            passages.append(Passage(id=f"p{counter:04d}", text=chunk, page=page_number))

    return passages


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.extraction <chemin_vers_pdf>")
        sys.exit(1)

    result = extract_passages(sys.argv[1])
    print(f"{len(result)} passage(s) extrait(s).\n")
    for p in result[:3]:
        print(f"--- {p.id} (page {p.page}) ---")
        print(p.text[:300], "...\n")