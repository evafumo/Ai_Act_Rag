from __future__ import annotations

import re
import json
import argparse
import pdfplumber
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
#  STRUTTURE DATI
# ============================================================

@dataclass
class Block:
    """
    Unità elementare estratta dal documento:
            - un paragrafo di testo
            - una tabella convertita in Markdown

    ATTRIBUTI:
        content (str): contenuto del blocco
        kind (str): "text" oppure "table"
        page (int): numero di pagina da cui proviene
    """
    content: str
    kind: str    # "text" | "table"
    page: int


@dataclass
class Chunk:
    """
    RRappresenta un chunk finale pronto per l'indicizzazione in un vector store (FAISS, Chroma, ecc.).

    ATTRIBUTI:
        text (str): contenuto del chunk
        chunk_id (int): identificatore univoco
        strategy (str): strategia di chunking usata
        overlap (int|None): overlap usato (solo per fixed/sliding/hybrid)
        source_pages (list[int]): pagine da cui proviene il chunk
        contains_table (bool): True se contiene una tabella
    """
    text: str
    chunk_id: int
    strategy: str
    overlap: int | None = None
    source_pages: list[int] = field(default_factory=list)
    contains_table: bool = False

    def to_dict(self) -> dict:
        """
        OUTPUT:
            Dizionario serializzabile in JSONL.
        """
        return {
            "chunk_id": self.chunk_id,
            "strategy": self.strategy,
            "overlap": self.overlap,
            "source_pages": self.source_pages,
            "contains_table": self.contains_table,
            "chars": len(self.text),
            "text": self.text,
        }


def _table_to_markdown(table: list[list]) -> str:
    """
    Converte una tabella pdfplumber in Markdown.

    INPUT:
        table (list[list]): matrice di celle

    OUTPUT:
        str: tabella in formato Markdown
    """
    if not table or not table[0]:
        return ""
    rows = [[str(c).replace("\n", " ").strip() if c is not None else "" for c in row]
            for row in table]
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator] + body_rows)


def parse_pdf_to_blocks(pdf_path: str, verbose: bool = True) -> list[Block]:
    """
    Estrae dal PDF una lista ordinata di blocchi (testo + tabelle).

    INPUT:
        pdf_path (str): percorso al PDF
        verbose (bool): stampa log

    OUTPUT:
        list[Block]: blocchi ordinati per pagina
    """
    blocks: list[Block] = []

    with pdfplumber.open(pdf_path) as pdf:
        n = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            if verbose:
                print(f"  [parse] pagina {page_num}/{n}")

            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            text_page = page
            for bbox in table_bboxes:
                text_page = text_page.outside_bbox(bbox)
            raw_text = text_page.extract_text() or ""

            for para in re.split(r"\n{2,}", raw_text):
                para = para.strip()
                if para:
                    blocks.append(Block(content=para, kind="text", page=page_num))

            for t in tables:
                md = _table_to_markdown(t.extract())
                if md:
                    blocks.append(Block(content=md, kind="table", page=page_num))

    return blocks


def parse_json_to_blocks(json_path: str, verbose: bool = True) -> list[Block]:
    """
    Converte un file JSON generato dal PDF extractor in una lista di blocks.

    INPUT:
        json_path (str): percorso al JSON

    OUTPUT:
        list[Block]
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    blocks = []

    for entry in data:
        page = entry["page"]
        text = entry["text"]

        parts = text.split("<!-- TABELLE -->")

        # Testo
        if parts[0].strip():
            for para in parts[0].split("\n\n"):
                para = para.strip()
                if para:
                    blocks.append(Block(content=para, kind="text", page=page))

        # Tabelle
        if len(parts) > 1:
            tables_md = parts[1].strip()
            if tables_md:
                blocks.append(Block(content=tables_md, kind="table", page=page))

    if verbose:
        print(f"[parse-json] Estratti {len(blocks)} blocchi da JSON")

    return blocks


def save_blocks_json(blocks: list[Block], path: str) -> None:
    """
    Salva i blocchi estratti in un file JSON

    INPUT:
        blocks (list[Block])
        path (str): percorso output
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    serializable = [
        {
            "block_id": i,
            "page": b.page,
            "kind": b.kind,
            "chars": len(b.content),
            "content": b.content
        }
        for i, b in enumerate(blocks)
    ]

    out.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save-blocks] Salvati {len(blocks)} blocchi in {out}")


# ===========================
#       STRATEGIES
# ===========================

def _char_split(text: str, size: int, overlap: int) -> list[str]:
    """
    Divide una stringa in segmenti di lunghezza fissa con overlap.

    INPUT:
        text (str)
        size (int)
        overlap (int)

    OUTPUT:
        list[str]: segmenti di testo
    """
    step = max(1, size - overlap)
    return [text[i: i + size] for i in range(0, len(text), step) if text[i: i + size].strip()]


# ===========================
#  FIXED
# ===========================

def fixed_chunking(blocks, chunk_size=512, overlap=64):
    """
    Chunking a finestra fissa basato sui caratteri.

    INPUT:
        blocks (list[Block])
        chunk_size (int)
        overlap (int)

    OUTPUT:
        list[Chunk]
    """
    chunks = []
    buf, buf_pages, has_table = "", [], False
    cid = 0

    def flush(text, pages, table):
        nonlocal cid
        if not text.strip():
            return
        for seg in _char_split(text, chunk_size, overlap):
            chunks.append(Chunk(
                text=seg, chunk_id=cid, strategy="fixed",
                overlap=overlap, source_pages=sorted(set(pages)),
                contains_table=table,
            ))
            cid += 1

    for block in blocks:
        if block.kind == "table":
            flush(buf, buf_pages, has_table)
            buf, buf_pages, has_table = "", [], False
            chunks.append(Chunk(
                text=block.content, chunk_id=cid, strategy="fixed",
                overlap=overlap, source_pages=[block.page], contains_table=True,
            ))
            cid += 1
        else:
            candidate = (buf + "\n\n" + block.content).strip()
            if len(candidate) > chunk_size and buf.strip():
                flush(buf, buf_pages, has_table)
                tail = buf[-overlap:] if overlap else ""
                buf = (tail + "\n\n" + block.content).strip()
                buf_pages = [block.page]
                has_table = False
            else:
                buf = candidate
                buf_pages.append(block.page)

    flush(buf, buf_pages, has_table)
    return chunks


# ===========================
#  SLIDING WINDOW
# ===========================

def _load_spacy():
    """
    Carica un modello spaCy (IT, fallback EN).

    OUTPUT:
        nlp (spacy.Language)
    """
    import spacy
    for model in ("it_core_news_sm", "en_core_web_sm"):
        try:
            return spacy.load(model)
        except OSError:
            continue
    raise RuntimeError("Installa un modello spaCy (it_core_news_sm o en_core_web_sm)")


def sliding_window_chunking(blocks, sentences_per_chunk=6, overlap_sentences=2):
    """
    Chunking basato sulle frasi (sliding window).

    INPUT:
        blocks (list[Block])
        sentences_per_chunk (int)
        overlap_sentences (int)

    OUTPUT:
        list[Chunk]
    """
    nlp = _load_spacy()
    chunks = []
    cid = 0
    text_run = []

    def flush_run(run):
        """
        Converte un gruppo di blocchi consecutivi in chunk basati sulle frasi.
        """
        nonlocal cid
        if not run:
            return
        full_text = " ".join(b.content for b in run)
        doc = nlp(full_text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        pages = [b.page for b in run]

        i = 0
        while i < len(sentences):
            window = sentences[i: i + sentences_per_chunk]
            chunks.append(Chunk(
                text=" ".join(window),
                chunk_id=cid,
                strategy="sliding",
                overlap=overlap_sentences,
                source_pages=sorted(set(pages)),
                contains_table=False,
            ))
            cid += 1
            i += sentences_per_chunk - overlap_sentences

    for block in blocks:
        if block.kind == "table":
            flush_run(text_run)
            text_run = []
            chunks.append(Chunk(
                text=block.content, chunk_id=cid, strategy="sliding",
                overlap=overlap_sentences, source_pages=[block.page], contains_table=True,
            ))
            cid += 1
        else:
            text_run.append(block)

    flush_run(text_run)
    return chunks


# ===========================
#  HYBRID
# ===========================

def hybrid_chunking(blocks, chunk_size=800, overlap=100, table_context_chars=200):
    """
    Chunking ibrido:
            - testo: fixed
            - tabelle: chunk autonomo con contesto precedente

    INPUT:
        blocks (list[Block])
        chunk_size (int)
        overlap (int)
        table_context_chars (int)

    OUTPUT:
        list[Chunk]
    """
    chunks = []
    buf, buf_pages = "", []
    cid = 0

    def flush_text(text, pages):
        """
        Convertire il buffer in chunk fixed.
        """
        nonlocal cid
        if not text.strip():
            return
        for seg in _char_split(text, chunk_size, overlap):
            chunks.append(Chunk(
                text=seg,
                chunk_id=cid,
                strategy="hybrid",
                overlap=overlap,
                source_pages=sorted(set(pages)),
                contains_table=False,
            ))
            cid += 1

    for block in blocks:
        if block.kind == "table":
            context = buf[-table_context_chars:].strip() if buf else ""
            table_text = (f"{context}\n\n{block.content}" if context else block.content)
            flush_text(buf, buf_pages)
            buf, buf_pages = "", []

            chunks.append(Chunk(
                text=table_text,
                chunk_id=cid,
                strategy="hybrid",
                overlap=overlap,
                source_pages=[block.page],
                contains_table=True,
            ))
            cid += 1
        else:
            candidate = (buf + "\n\n" + block.content).strip()
            if len(candidate) > chunk_size and buf.strip():
                flush_text(buf, buf_pages)
                tail = buf[-overlap:] if overlap else ""
                buf = (tail + "\n\n" + block.content).strip()
                buf_pages = [block.page]
            else:
                buf = candidate
                buf_pages.append(block.page)

    flush_text(buf, buf_pages)
    return chunks


# ============================================================
#  SALVATAGGIO CHUNK
# ============================================================

def save_chunks_txt(chunks, path):
    """
    Salva i chunkS in formato .txt

    INPUT:
        chunks (list[Chunk])
        path (str)
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for c in chunks:
        lines.append("=" * 60)
        lines.append(
            f"CHUNK {c.chunk_id:04d} | strategy={c.strategy} | "
            f"overlap={c.overlap} | pages={c.source_pages} | "
            f"table={c.contains_table} | chars={len(c.text)}"
        )
        lines.append("=" * 60)
        lines.append(c.text)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [save] {len(chunks)} chunk → {out}")


def save_chunks_jsonl(chunks, path):
    """
    Salva i chunk in formato JSONL (uno per riga)

    INPUT:
        chunks (list[Chunk])
        path (str)
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"  [save] {len(chunks)} chunk → {out}")


# ==============================
#       ENTRY POINT
# ==============================

STRATEGIES = ("fixed", "sliding", "hybrid", "all")

if __name__ == "__main__":
    """
    Test dello script da terminale 
        Gestisce:
            - parsing degli argomenti CLI
            - caricamento PDF o JSON
            - esecuzione della strategia di chunking scelta
            - salvataggio dei chunk in TXT e JSONL

    INPUT:
        Argomenti CLI (parser.add_argument)
        Flags:
            pdf (str): percorso al file PDF o JSON
            --strategy (str): strategia di chunking
            --chunk-size (int): dimensione chunk per fixed/hybrid
            --overlap (int): overlap per fixed/hybrid
            --sent-per-chunk (int): frasi per chunk sliding
            --sent-overlap (int): overlap frasi sliding
            --sem-threshold (float): soglia similarità semantica
            --sem-max-chars (int): limite caratteri chunk semantico
            --sem-model (str): modello sentence-transformers
            --output-dir (str): cartella output
            --quiet (bool): silenzia log
            --save-blocks (bool): salva blocchi prima del chunking

    OUTPUT:
        File salvati in output-dir:
            - chunks_<strategy>.txt
            - chunks_<strategy>.jsonl
            - blocks.json (opzionale) 
    """

    parser = argparse.ArgumentParser(
        description="Pipeline di chunking per RAG — PDF o JSON"
    )
    parser.add_argument("pdf", help="Percorso al file PDF o JSON")
    parser.add_argument("--strategy", choices=STRATEGIES, default="all")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--sent-per-chunk", type=int, default=6)
    parser.add_argument("--sent-overlap", type=int, default=2)
    parser.add_argument("--sem-threshold", type=float, default=0.75)
    parser.add_argument("--sem-max-chars", type=int, default=1500)
    parser.add_argument("--sem-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-blocks", action="store_true",
                        help="Salva i blocchi estratti prima del chunking")
    args = parser.parse_args()

    input_path = Path(args.pdf)
    verbose = not args.quiet
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Supporto JSON
    if input_path.suffix.lower() == ".json":
        print(f"\n[1/2] Parsing JSON: {input_path}")
        blocks = parse_json_to_blocks(str(input_path), verbose=verbose)
    else:
        print(f"\n[1/2] Parsing PDF: {input_path}")
        blocks = parse_pdf_to_blocks(str(input_path), verbose=verbose)

    # Salvataggio blocchi se richiesto
    if args.save_blocks:
        save_blocks_json(blocks, out_dir / "blocks.json")

    print(f"\n[2/2] Chunking (strategia: {args.strategy})")

    run_strategies = ["fixed", "sliding", "hybrid", "semantic"] if args.strategy == "all" else [args.strategy]

    for strat in run_strategies:
        print(f"\n  → {strat.upper()}")

        if strat == "fixed":
            chunks = fixed_chunking(blocks, chunk_size=args.chunk_size, overlap=args.overlap)

        elif strat == "sliding":
            chunks = sliding_window_chunking(
                blocks,
                sentences_per_chunk=args.sent_per_chunk,
                overlap_sentences=args.sent_overlap,
            )

        elif strat == "hybrid":
            chunks = hybrid_chunking(
                blocks,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )

        save_chunks_txt(chunks, out_dir / f"chunks_{strat}.txt")
        save_chunks_jsonl(chunks, out_dir / f"chunks_{strat}.jsonl")
