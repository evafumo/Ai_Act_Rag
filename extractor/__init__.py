# Package: pdf_extractor
# Contiene moduli per l'estrazione del testo e delle tabelle dai PDF.

from .pdf_extractor import (
    extract_pdf_for_rag,
    save_to_json,
    save_to_txt,
)

__all__ = [
    "extract_pdf_for_rag",
    "save_to_json",
    "save_to_txt",
]

