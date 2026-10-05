
from __future__ import annotations

import numpy as np


_ocr_engine = None


def get_ocr_engine():
    """
    Restituisce (creandola se serve) l'istanza di RapidOCR.
    Il modello viene caricato una sola volta (singleton) per non
    sforare il budget di RAM con inizializzazioni ripetute.

    OUTPUT:
        RapidOCR
    """
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def html_table_to_markdown(html: str) -> str:
    """
    Converte una tabella HTML in Markdown. 

    INPUT:
        html (str): tabella in formato HTML

    OUTPUT:
        str: tabella in formato Markdown
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * n_cols) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, separator] + body)


def debug_dump_raw_result(img: np.ndarray) -> None:
    """
    Utility diagnostica: stampa l'output grezzo di RapidOCR per
    UN'immagine di test. Utile per vedere testo/box/score prima di
    fidarti di extract_page_content, o per debug rapido.

    INPUT:
        img (np.ndarray): immagine BGR di prova

    OUTPUT:
        None (stampa a schermo)
    """
    engine = get_ocr_engine()
    result = engine(img)
    print("testi riconosciuti:", result.txts)
    print("punteggi di confidenza:", result.scores)
    print("box (coordinate):", result.boxes)


def extract_page_content(img: np.ndarray, confidence_threshold: float = 0.6) -> dict:
    """
    Estrae il testo da un'immagine di pagina con RapidOCR.

    Le tabelle non sono ancora gestite in questa fase
    INPUT:
        img (np.ndarray): immagine BGR della pagina (pulita da image_cleaning)
        confidence_threshold (float): sotto questa soglia il blocco di testo
                                       viene segnalato come "low_confidence"

    OUTPUT:
        dict: {
            "text_blocks": list[str],
            "tables_md": list[str],   # sempre [] per ora
            "low_confidence": bool
        }
    """
    engine = get_ocr_engine()
    result = engine(img)

    text_blocks = []
    low_confidence = False

    txts = result.txts or []
    scores = result.scores if getattr(result, "scores", None) else [1.0] * len(txts)

    for txt, score in zip(txts, scores):
        if not txt or not str(txt).strip():
            continue
        if score < confidence_threshold:
            low_confidence = True
        text_blocks.append(str(txt).strip())

    return {
        "text_blocks": text_blocks,
        "tables_md": [],
        "low_confidence": low_confidence,
    }
