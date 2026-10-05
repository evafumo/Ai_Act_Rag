"""
Orchestrazione: PDF/immagine -> preprocessing -> OCR+tabelle -> JSON
compatibile con parse_json_to_blocks del modulo di chunking.

Versione "prima fase": ogni pagina viene sempre passata attraverso
la pipeline di pulizia immagine + OCR/PP-Structure, indipendentemente
dal fatto che il PDF abbia o meno testo nativo selezionabile.
Il controllo automatico native-vs-scan (router) verrà aggiunto in
una fase successiva.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from pdf_to_images import pdf_to_images, load_single_image
from image_cleaning import clean_pipeline
from ocr_table_engine import extract_page_content


def process_document(input_path: str, verbose: bool = True) -> list[dict]:
    """
    Elabora un PDF scannerizzato o una singola immagine, pagina per pagina.

    INPUT:
        input_path (str): percorso a PDF, JPEG o PNG
        verbose (bool): stampa log

    OUTPUT:
        list[dict]: una entry per pagina, formato:
            {"page": int, "text": "<paragrafi>\\n\\n<!-- TABELLE -->\\n\\n<tabelle md>"}
    """
    path = Path(input_path)
    if path.suffix.lower() == ".pdf":
        images = pdf_to_images(str(path))
    else:
        images = [load_single_image(str(path))]

    entries = []
    n = len(images)

    for i, img in enumerate(images, start=1):
        if verbose:
            print(f"  [ocr] pagina {i}/{n}")

        cleaned = clean_pipeline(img, binarize=False)  # PP-Structure preferisce grayscale, non binario
        result = extract_page_content(cleaned)

        if result["low_confidence"] and verbose:
            print(f"    ⚠ confidence bassa pagina {i}, controllare manualmente")

        text_part = "\n\n".join(result["text_blocks"])
        tables_part = "\n\n".join(result["tables_md"])

        if tables_part:
            page_text = f"{text_part}\n\n<!-- TABELLE -->\n\n{tables_part}"
        else:
            page_text = text_part

        entries.append({"page": i, "text": page_text})

    return entries


def save_json(entries: list[dict], out_path: str):
    """
    Salva le entry estratte nel formato JSON atteso da parse_json_to_blocks.

    INPUT:
        entries (list[dict])
        out_path (str)
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [save] {len(entries)} pagine, {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing OCR/tabelle per RAG")
    parser.add_argument("input", help="Percorso a PDF, JPEG o PNG scannerizzato")
    parser.add_argument("--output", default="output/extracted.json")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    entries = process_document(args.input, verbose=not args.quiet)
    save_json(entries, args.output)
