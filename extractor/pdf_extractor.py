"""
Estrazione testo da PDF per RAG — versione leggera (senza VLM).
Salva anche JSON per pipeline con rag_chunker.
"""

import fitz          # PyMuPDF
import pdfplumber
import json
from pathlib import Path


def extract_native_text(page: fitz.Page) -> str:
    """
    Estrae il testo nativo da una pagina PDF usando PyMuPDF.

    INPUT:
        page (fitz.Page): oggetto pagina PyMuPDF
    OUTPUT: 
        str: testo estratto e ripulito
    NOTE: 
        Funziona solo se il PDF contiene testo digitale (non immagini).
    """
    return page.get_text("text").strip()


def extract_tables_pdfplumber(pdf_path: str, page_index: int) -> str:
    """
    Estrae tabelle da una pagina PDF usando pdfplumber e le converte in Markdown.

    INPUT:
        pdf_path (str): percorso al file PDF
        page_index (int): indice della pagina (0-based)
    OUTPUT:
        str: tabelle in formato Markdown, concatenate
    NOTE:
        Se non ci sono tabelle, restituisce stringa vuota.
    """
    tables_md = []
    with pdfplumber.open(pdf_path) as pdf:
        plumb_page = pdf.pages[page_index]
        tables = plumb_page.extract_tables()

        for table in tables:
            if not table:
                continue
            lines = []
            for row_idx, row in enumerate(table):
                cleaned = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
                lines.append("| " + " | ".join(cleaned) + " |")
                if row_idx == 0:
                    lines.append("| " + " | ".join(["---"] * len(cleaned)) + " |")
            tables_md.append("\n".join(lines))

    return "\n\n".join(tables_md)


def extract_pdf_for_rag(pdf_path: str, verbose: bool = True) -> list[dict]:
    """
    Estrae testo e tabelle da ogni pagina del PDF, unendoli in un dizionario

    INPUT:
        pdf_path (str): percorso al PDF
        verbose (bool): se True stampa log di avanzamento

    OUTPUT:
        list[dict]: lista di dizionari, uno per pagina:
            {
                "page": numero pagina,
                "text": testo completo (testo + tabelle),
                "has_tables": bool,
                "source": percorso PDF
            }

    NOTE:
        - Usa PyMuPDF per testo
        - Usa pdfplumber per tabelle
        - Combina tutto in un unico campo "text"
    """
    pdf_path = str(pdf_path)
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    results = []

    for i in range(n_pages):
        if verbose:
            print(f"[INFO] Pagina {i + 1}/{n_pages}...")

        page = doc[i]
        native = extract_native_text(page)

        if not native:
            print(f"[WARN] Pagina {i + 1} sembra scansionata o vuota — testo nativo assente.")

        try:
            tables_md = extract_tables_pdfplumber(pdf_path, i)
        except Exception as e:
            print(f"[WARN] pdfplumber fallito a pagina {i + 1}: {e}")
            tables_md = ""

        parts = []
        if native:
            parts.append(native)
        if tables_md:
            parts.append("<!-- TABELLE -->\n" + tables_md)

        text = "\n\n".join(parts)

        results.append({
            "page": i + 1,
            "text": text,
            "has_tables": bool(tables_md),
            "source": pdf_path,
        })

    doc.close()
    return results


def save_to_txt(pages: list[dict], output_path: str) -> Path:
    """
    Salva il contenuto estratto dal PDF in un file .txt

    INPUT:
        pages (list[dict]): output di extract_pdf_for_rag
        output_path (str): percorso file .txt

    OUTPUT:
        Path — percorso del file salvato

    NOTE:
        Ogni pagina viene separata da un header "=== Pagina X ==="
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for p in pages:
        lines.append(f"=== Pagina {p['page']} ===")
        lines.append(p["text"])
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_to_json(pages: list[dict], output_path: str) -> Path:
    """
    Salva il contenuto estratto dal PDF in un file JSON 

    INPUT:
        pages (list[dict]): output di extract_pdf_for_rag
        output_path (str): percorso file .json

    OUTPUT:
        Path: percorso del file salvato

    NOTE:
        Il chunker usa il formato JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    return output_path


"""
Test delle funzioni 

ARGOMENTI :
    pdf (str) — percorso PDF
    -o / --output (str) — percorso file txt di output
    --quiet — disattiva log

OUTPUT:
    - File TXT
    - File JSON
    - Log riepilogativo
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Estrai testo da PDF per RAG (PyMuPDF + pdfplumber, salva TXT + JSON)."
    )
    parser.add_argument("pdf", help="Percorso al file PDF")
    parser.add_argument(
        "-o", "--output", default=None,
        help="File .txt di output (default: stesso nome del PDF)"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_path = args.output or str(Path(args.pdf).with_suffix(".txt"))

    pages = extract_pdf_for_rag(
        pdf_path=args.pdf,
        verbose=not args.quiet,
    )

    out_txt = save_to_txt(pages, output_path)
    out_json = save_to_json(pages, Path(output_path).with_suffix(".json"))

    print(f"\n File TXT salvato: {out_txt}")
    print(f"File JSON salvato: {out_json}")
    print(f"Pagine totali: {len(pages)}")
    print(f"Pagine con tabelle: {sum(1 for p in pages if p['has_tables'])}")

