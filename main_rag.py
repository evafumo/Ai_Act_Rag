# ============================================================
# main_rag.py
# Pipeline completa:
# 1) PDF → JSON (pdf_extractor)
# 2) JSON → CHUNKING (rag_chunker_json_ready_v2)
# 3) Costruzione retrievers
# 4) Retrieval + opzionale evaluation
# 5) Answer generation (Qwen 0.5B)
# ============================================================

import argparse
from pathlib import Path
import json

# --- Import dei tuoi moduli ---
from extractor import extract_pdf_for_rag, save_to_json
#ricontrolla nome funzioni 
from chunker import (
    parse_json_to_blocks,
    fixed_chunking,
    sliding_window_chunking,
    hybrid_chunking,
)
from retrievers import build_retriever_combo, unified_retrieval
from eval import load_ground_truth, evaluate_retriever
from llm import generate_answer
from langchain_core.documents import Document


# ============================================================
#  Costruzione Document[] da chunks JSONL
# ============================================================

def load_chunks_as_documents(jsonl_path: str):
    """
    Carica i chunk salvati in formato JSONL e converte
    in oggetti Document di LangChain, pronti per FAISS/BM25.

    INPUT:
        jsonl_path (str):
            Percorso al file .jsonl contenente i chunk.

    OUTPUT:
        List[Document]:
            Lista di Document con:
                - page_content = testo del chunk
                - metadata = chunk_id, strategy, source_pages, contains_table, chars

    NOTE:
        - Ogni riga del JSONL rappresenta un chunk.
        - Questo è il formato richiesto dai retriever LangChain.
    """
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj["text"],
                    metadata={
                        "chunk_id": obj["chunk_id"],
                        "strategy": obj["strategy"],
                        "source_pages": obj["source_pages"],
                        "contains_table": obj["contains_table"],
                        "chars": obj["chars"],
                    },
                )
            )
    return docs


def auto_select_ground_truth(pdf_path: str, chunking: str) -> str | None:
    """
    Seleziona e il file di ground truth corretto in base al nome 
    del PDF e alla strategia di chunking.

    INPUT:
        pdf_path (str):
            Percorso al PDF originale.
        chunking (str):
            Strategia di chunking usata (fixed, sliding, hybrid, semantic).

    OUTPUT:
        str | None:
            Percorso al file JSONL del ground truth, se trovato.
            None se non esiste un GT compatibile.

    NOTE:
        - Usa convenzioni di naming: GT_B_Adult_Fixed.jsonl, ecc.
    """
    name = Path(pdf_path).stem.lower()

    dataset_map = {
        "adult": "GT_B_Adult",
        "compas": "GT_B_COMPAS",
        "german": "GT_B_German",
    }

    chunk_map = {
        "fixed": "Fixed",
        "sliding": "Sliding",
        "hybrid": "Hybrid",
        "semantic": "Semantic",
    }

    dataset_key = None
    for key in dataset_map:
        if key in name:
            dataset_key = dataset_map[key]
            break

    if dataset_key is None:
        return None

    if chunking not in chunk_map:
        return None

    suffix = chunk_map[chunking]
    gt_file = Path("datasets/B_Adults/GT") / f"{dataset_key}_{suffix}.jsonl"

    if Path(gt_file).exists():
        return gt_file

    print(f"[WARN] Ground truth non trovato: {gt_file}")
    return None


# ============================================================
#  MAIN PIPELINE
# ============================================================

def main():
    """
    Esegue l’intera pipeline RAG:
        - Estrazione PDF
        - Chunking
        - Costruzione retriever
        - Evaluation IR (opzionale)
        - Answer generation

    INPUT:
        Argomenti da linea di comando:
            pdf (str): PDF di input
            --chunking: strategia di chunking
            --retriever: tipo di retriever
            --evaluate: flag per eseguire evaluation
            --question: domanda da porre al sistema RAG

    OUTPUT:
        File generati:
            - <pdf>.json
            - chunks_<strategy>.jsonl
            - eval_<pdf>.json (se evaluation attiva)
        Output a console:
            - log della pipeline
            - risposta dell’LLM (se question attiva)
    """
    parser = argparse.ArgumentParser(description="Pipeline completa RAG")
    parser.add_argument("pdf", help="PDF di input")
    parser.add_argument("--chunking", choices=["fixed", "sliding", "hybrid"],
                        default="hybrid")
    parser.add_argument("--retriever", choices=[
        "single_bge", "single_e5", "single_bgem3",
        "dense_bge_e5", "dense_bgem3_e5",
        "bm25_bge", "bm25_e5"
    ], default="single_bge")
    parser.add_argument("--evaluate", action="store_true",
                        help="Esegue evaluation automatica se disponibile")
    parser.add_argument("--question", type=str, default=None,
                        help="Domanda da fare al sistema RAG")
    args = parser.parse_args()

    pdf_path = args.pdf
    pdf_name = Path(pdf_path).stem
    # cartella output
    dataset_name = Path(pdf_path).stem
    output_dir = Path("output") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)


    print("\n[1/4] Estrazione PDF - JSON")
    pages = extract_pdf_for_rag(pdf_path, verbose=True)
    json_path = output_dir / f"{pdf_name}.json"
    save_to_json(pages, json_path)
    print(f"  → Salvato JSON: {json_path}")

    print("\n[2/4] Chunking")
    blocks = parse_json_to_blocks(json_path)

    if args.chunking == "fixed":
        chunks = fixed_chunking(blocks)
    elif args.chunking == "sliding":
        chunks = sliding_window_chunking(blocks)
    elif args.chunking == "hybrid":
        chunks = hybrid_chunking(blocks)

    out_chunks = output_dir /  f"chunks_{args.chunking}.jsonl"
    with open(out_chunks, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    print(f"   Salvati {len(chunks)} chunk in {out_chunks}")

    print("\n[3/4] Costruzione retriever")
    docs = load_chunks_as_documents(out_chunks)
    retrievers = build_retriever_combo(args.retriever, docs)
    print(f"  Retriever selezionato: {args.retriever}")

    # ============================================================
    #  Evaluation IR
    # ============================================================
    if args.evaluate:
        print("\n[4/4] Evaluation IR")

        gt_path = auto_select_ground_truth(pdf_path, args.chunking)
        if gt_path is None:
            print("  [WARN] Nessun ground truth riconosciuto per questo PDF.")
        else:
            print(f"   Ground truth selezionato: {gt_path}")
            gt = load_ground_truth(gt_path)

            def retrieve_fn(q):
                return unified_retrieval(retrievers, q, k=20)

            evaluate_retriever(gt, retrieve_fn, json_path= output_dir / f"eval_{pdf_name}.json")
            print(f"  Evaluation salvata in eval_{pdf_name}.json")

    # ============================================================
    #  Answer generation 
    # ============================================================
    if args.question:
        print("\n[5/5] Answer generation")
        retrieved = unified_retrieval(retrievers, args.question, k=5)
        context = "\n\n".join([d.page_content for d in retrieved])
        answer = generate_answer(args.question, context)
        print("\n=== RISPOSTA ===")
        print(answer)
        (output_dir / "answer.txt").write_text(answer, encoding="utf-8")


if __name__ == "__main__":
    main()

