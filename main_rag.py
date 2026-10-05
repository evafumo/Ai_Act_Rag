# ============================================================
# main_rag_new.py — versione CORRETTA (chiave = (chunk_id, source))
#   + confronto con baseline random (--compare-baseline)
# ============================================================

import argparse
import contextlib
import io
import statistics
from pathlib import Path
import json

from extractor import extract_pdf_for_rag, save_to_json
from chunker import (
    parse_json_to_blocks,
    fixed_chunking,
    sliding_window_chunking,
    hybrid_chunking,
)
from retrievers.retriever import build_retriever_combo, unified_retrieval
from retrievers.multy_queries import (
    multi_query_retrieval_rerank_each,
    multi_query_retrieval_rerank_once,
)
from decomposition.decompose_query import decompose_question

from eval import (
    load_ground_truth_subq,
    load_ground_truth,
    evaluate_retriever_subq,
    evaluate_retriever,
    append_to_combined_csv
)
from eval.baseline_dummy import build_random_baseline, expected_random_metrics

from llm import generate_answer
from langchain_core.documents import Document


# ============================================================
#  Costruzione Document[] da chunks JSONL
# ============================================================

def load_chunks_as_documents(jsonl_path: str):
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = str(obj["chunk_id"])
            src = str(obj["source"])
            docs.append(
                Document(
                    page_content=obj["text"],
                    metadata={
                        "chunk_id": cid,
                        "source": src,
                        "key": (cid, src),
                        "strategy": obj["strategy"],
                        "source_pages": obj["source_pages"],
                        "contains_table": obj["contains_table"],
                        "chars": obj["chars"],
                    },
                )
            )
    return docs


# ============================================================
#  FILTRO GAP GT GRANULARE
# ============================================================

def _filter_valid_gt_entries(gt: list) -> list:
    valid = []
    for entry in gt:
        if not entry.get("relevant_ids", []):
            print(f"  [SKIP] {entry.get('question_id')} senza relevant_ids")
            continue
        valid.append(entry)
    return valid


def auto_select_ground_truth_subq(pdf_path: str, chunking: str) -> str | None:

    pdf_path = Path(pdf_path).resolve()

    # Cartella GT = stessa cartella del PDF
    gt_dir = pdf_path.parent

    # Cerca file che finiscono con il suffisso giusto
    suffix_map = {
        "fixed": "Fixed_mq.jsonl",
        "sliding": "Sliding_mq.jsonl",
        "hybrid": "Hybrid_mq.jsonl",
    }

    if chunking not in suffix_map:
        print(f"[WARN] Chunking non riconosciuto: {chunking}")
        return None

    suffix = suffix_map[chunking]

    # Cerca file che contengono "_mq" e il chunking corretto
    candidates = list(gt_dir.glob(f"*{suffix}"))

    if not candidates:
        print(f"[WARN] Nessun GT subquestion trovato in {gt_dir} con suffisso {suffix}")
        return None

    # Prendi il primo (ce n'è sempre uno)
    return str(candidates[0])


def auto_select_ground_truth(pdf_path: str, chunking: str) -> str | None:
    """
    Selettore automatico del GT AGGREGATO (domanda intera), speculare a
    auto_select_ground_truth_subq ma senza il suffisso '_mq'. Prima mancava
    del tutto: il ramo aggregato la chiamava senza che fosse mai definita,
    con un NameError non appena la evaluation partiva senza --gt-path.
    """

    pdf_path = Path(pdf_path).resolve()
    gt_dir = pdf_path.parent

    suffix_map = {
        "fixed": "Fixed.jsonl",
        "sliding": "Sliding.jsonl",
        "hybrid": "Hybrid.jsonl",
    }

    if chunking not in suffix_map:
        print(f"[WARN] Chunking non riconosciuto: {chunking}")
        return None

    suffix = suffix_map[chunking]

    # Esclude esplicitamente i file granulari (*_mq.jsonl), anche se il
    # pattern non dovrebbe già intersecarli (suffisso diverso).
    candidates = [
        p for p in gt_dir.glob(f"*{suffix}")
        if not p.name.endswith("_mq.jsonl")
    ]

    if not candidates:
        print(f"[WARN] Nessun GT aggregato trovato in {gt_dir} con suffisso {suffix}")
        if gt_dir.exists():
            print(f"       File presenti: {[p.name for p in gt_dir.iterdir()]}")
        return None

    if len(candidates) > 1:
        print(f"[WARN] Più GT aggregati trovati, uso {candidates[0].name}: "
              f"{[c.name for c in candidates]}")

    return str(candidates[0])


# ============================================================
#  BASELINE RANDOM SU PIÙ SEED (media + deviazione standard)
# ============================================================

def evaluate_baseline_multi_seed(gt: list, docs: list, eval_fn, seeds: list) -> dict:
    """
    Esegue eval_fn(gt, random_retrieve_fn) una volta per ogni seed e aggrega
    i risultati con media e deviazione standard. json_path=None: non scrive
    un file per ogni singolo seed (produrrebbe N file poco utili), e l'output
    a schermo di ogni run viene silenziato per non intasare la console con
    N ripetizioni della stessa tabella.

    OUTPUT: {
        "MAP": media, "MAP_std": dev.std,
        "macro_metrics": {"@k": {"precision":.., "recall":.., "ndcg":..}},
        "macro_metrics_std": {"@k": {"precision":.., "recall":.., "ndcg":..}},
        "MAP_per_seed": [...],
    }
    """
    maps = []
    k_values = None
    per_k = {}

    for seed in seeds:
        random_retrieve_fn = build_random_baseline(docs, seed=seed)
        with contextlib.redirect_stdout(io.StringIO()):
            result = eval_fn(gt, random_retrieve_fn, json_path=None)

        maps.append(result["summary"]["MAP"])

        if k_values is None:
            k_values = [k for k in result["summary"]["macro_metrics"]]
            per_k = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}

        for k in k_values:
            m = result["summary"]["macro_metrics"][k]
            per_k[k]["precision"].append(m["precision"])
            per_k[k]["recall"].append(m["recall"])
            per_k[k]["ndcg"].append(m["ndcg"])

    stdev = statistics.pstdev if len(maps) > 1 else (lambda xs: 0.0)

    macro_mean = {
        k: {stat: statistics.mean(vals) for stat, vals in per_k[k].items()}
        for k in per_k
    }
    macro_std = {
        k: {stat: stdev(vals) for stat, vals in per_k[k].items()}
        for k in per_k
    }

    return {
        "MAP": statistics.mean(maps),
        "MAP_std": stdev(maps),
        "MAP_per_seed": maps,
        "macro_metrics": macro_mean,
        "macro_metrics_std": macro_std,
    }


# ============================================================
#  CONFRONTO CON BASELINE RANDOM
# ============================================================

def _print_baseline_comparison(real_summary: dict, baseline_summary: dict,
                                expected: dict, k_values=(5, 10, 20)):
    """
    Stampa fianco a fianco: retriever reale / baseline random misurata /
    baseline random attesa (teorica). Le prime due vengono dagli stessi
    calcoli di evaluate_retriever[_subq] sullo stesso GT; la terza è un
    controllo di coerenza indipendente (distribuzione ipergeometrica).
    """
    print("\n=== CONFRONTO CON BASELINE RANDOM ===")
    print(f"MAP reale:   {real_summary['MAP']:.4f}")

    if "MAP_std" in baseline_summary:
        n_seeds = len(baseline_summary.get("MAP_per_seed", []))
        lo, hi = baseline_summary['MAP'] - baseline_summary['MAP_std'], \
                 baseline_summary['MAP'] + baseline_summary['MAP_std']
        print(f"MAP random:  {baseline_summary['MAP']:.4f}  "
              f"(std={baseline_summary['MAP_std']:.4f}, {n_seeds} seed, "
              f"range 1-sigma [{lo:.4f}, {hi:.4f}])")
    else:
        print(f"MAP random:  {baseline_summary['MAP']:.4f}")

    delta = real_summary['MAP'] - baseline_summary['MAP']
    if baseline_summary['MAP'] > 0:
        ratio = real_summary['MAP'] / baseline_summary['MAP']
        print(f"Delta: {delta:+.4f}  ({ratio:.1f}x la baseline)")
    else:
        print(f"Delta: {delta:+.4f}")

    print(f"\n{'k':>4} | {'P reale':>8} {'P random':>9} {'P attesa':>9} | "
          f"{'R reale':>8} {'R random':>9} {'R attesa':>9}")
    print("-" * 66)
    for k in k_values:
        rm = real_summary["macro_metrics"][f"@{k}"]
        bm = baseline_summary["macro_metrics"][f"@{k}"]
        em = expected.get(k, {"precision": float("nan"), "recall": float("nan")})
        print(f"{k:>4} | {rm['precision']:>8.4f} {bm['precision']:>9.4f} {em['precision']:>9.4f} | "
              f"{rm['recall']:>8.4f} {bm['recall']:>9.4f} {em['recall']:>9.4f}")

    if delta <= 0:
        print("\n[WARN] Il retriever reale non batte la baseline random: "
              "il task/GT è sospetto (troppo facile, o il retriever non sfrutta la query).")


# ============================================================
#  MAIN PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Pipeline completa RAG")
    parser.add_argument("pdf", help="PDF di input oppure file JSON/JSONL già chunked")

    parser.add_argument("--chunking", choices=["fixed", "sliding", "hybrid"],
                        default="hybrid")

    parser.add_argument("--retriever", choices=[
        "single_bge", "single_e5", "single_bgem3",
        "dense_bge_e5", "dense_bgem3_e5",
        "bm25_bge", "bm25_e5"
    ], default="single_bge")

    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--gt-path", type=str, default=None)
    parser.add_argument("--gt-subq", action="store_true")
    parser.add_argument("--gt-subq-path", type=str, default=None)

    parser.add_argument("--decompose", action="store_true")
    parser.add_argument("--multiquery-mode", choices=["rerank_each", "rerank_once"],
                        default="rerank_once")

    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--use-chunked", action="store_true")

    parser.add_argument("--only-parent", type=str, default=None,
                        help="Valuta solo le subquestion della domanda madre specificata (es. D4)")

    parser.add_argument("--compare-baseline", action="store_true",
                        help="Valuta anche una baseline random sullo stesso GT e stampa il confronto")
    parser.add_argument("--baseline-seed", type=int, default=42,
                        help="Seed di partenza della baseline random (default: 42)")
    parser.add_argument("--baseline-runs", type=int, default=1,
                        help="Numero di seed su cui mediare la baseline random "
                             "(default: 1, nessuna media). Consigliato 10+ con --only-parent.")

    args = parser.parse_args()

    pdf_path = args.pdf
    pdf_name = Path(pdf_path).stem
    dataset_name = pdf_name

    output_dir = Path("output") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(pdf_path)

    # ============================================================
    # 1) Estrazione o uso chunk già pronti
    # ============================================================

    if args.use_chunked:
        if input_path.suffix.lower() not in [".json", ".jsonl"]:
            print("[ERROR] --use-chunked richiede file JSON/JSONL")
            return
        print("\n[1/5] Uso file chunked:", input_path)
        out_chunks = input_path

    else:
        print("\n[1/5] Estrazione PDF → JSON")
        pages = extract_pdf_for_rag(pdf_path, verbose=True)
        json_path = output_dir / f"{pdf_name}.json"
        save_to_json(pages, json_path)

        print("\n[2/5] Chunking")
        blocks = parse_json_to_blocks(json_path)

        if args.chunking == "fixed":
            chunks = fixed_chunking(blocks, dataset_name=dataset_name)
        elif args.chunking == "sliding":
            chunks = sliding_window_chunking(blocks, dataset_name=dataset_name)
        else:
            chunks = hybrid_chunking(blocks, dataset_name=dataset_name)

        out_chunks = output_dir / f"chunks_{args.chunking}.jsonl"
        with open(out_chunks, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    # ============================================================
    # 3) Costruzione retriever
    # ============================================================

    print("\n[3/5] Costruzione retriever")
    docs = load_chunks_as_documents(out_chunks)
    retrievers = build_retriever_combo(args.retriever, docs)

    # ============================================================
    # 4) Funzione di retrieval coerente
    # ============================================================

    def retrieve_fn(q: str):
        if not args.decompose:
            return unified_retrieval(retrievers, q, k=20)

        sub_questions = decompose_question(q)

        if len(sub_questions) <= 1:
            return unified_retrieval(retrievers, q, k=20)

        if args.multiquery_mode == "rerank_each":
            docs_multi = multi_query_retrieval_rerank_each(
                retrievers, q, sub_questions, k_per_subquery=20, k_final=20
            )
        else:
            docs_multi = multi_query_retrieval_rerank_once(
                retrievers, q, sub_questions,
                k_per_subquery=30, k_final=20
            )

        seen = set()
        merged = []
        for d in docs_multi:
            key = (str(d.metadata["chunk_id"]), str(d.metadata["source"]))
            if key not in seen:
                seen.add(key)
                merged.append(d)

        return merged

    # ============================================================
    # 5) Evaluation IR
    # ============================================================

    if args.evaluate:
        print("\n[4/5] Evaluation IR")

        if args.gt_subq_path:
            gt_path = Path(args.gt_subq_path)
            gt = load_ground_truth_subq(gt_path)
            eval_fn = evaluate_retriever_subq
            gt_type = "GRANULARE"

        elif args.gt_subq:
            gt_path = auto_select_ground_truth_subq(pdf_path, args.chunking)
            gt = load_ground_truth_subq(gt_path)
            eval_fn = evaluate_retriever_subq
            gt_type = "GRANULARE"

        else:
            gt_path = args.gt_path or auto_select_ground_truth(pdf_path, args.chunking)
            gt = load_ground_truth(gt_path)
            eval_fn = evaluate_retriever
            gt_type = "AGGREGATO"

        print(f"   GT selezionato ({gt_type}): {gt_path}")

        if gt_type == "GRANULARE":
            gt = _filter_valid_gt_entries(gt)

        if args.only_parent and gt_type == "GRANULARE":
            print(f"[INFO] Valutazione limitata alla domanda madre: {args.only_parent}")
            gt = [q for q in gt if q.get("parent_id") == args.only_parent]

            if not gt:
                print(f"[ERROR] Nessuna subquestion trovata per parent_id = {args.only_parent}")
                return

        if not gt:
            print("[ERROR] Il ground truth risultante è vuoto.")
            return

        eval_suffix = args.retriever + "_" + args.chunking
        if args.decompose:
            eval_suffix += f"_decomposed_{args.multiquery_mode}"
        if gt_type == "GRANULARE":
            eval_suffix += "_subq"

        eval_json_path = output_dir / f"eval_{pdf_name}_{eval_suffix}.json"

        # --- Valutazione del retriever reale (catturo il risultato per il confronto) ---
        real_result = eval_fn(gt, retrieve_fn, json_path=eval_json_path)

        append_to_combined_csv(
            json_path=eval_json_path,
            model=args.retriever,
            dataset=pdf_name,
            chunking=args.chunking,
            combined_csv="output/eval_results_combined.csv",
        )

        # --- Confronto con baseline random, stesso GT, stesso corpus ---
        if args.compare_baseline:

            if args.baseline_runs > 1:
                seeds = [args.baseline_seed + i for i in range(args.baseline_runs)]
                print(f"\n[4b/5] Valutazione baseline random su {len(seeds)} seed "
                      f"(stesso GT, media)")

                baseline_summary = evaluate_baseline_multi_seed(gt, docs, eval_fn, seeds)

                baseline_json_path = (output_dir /
                    f"eval_{pdf_name}_{eval_suffix}_baseline_random_avg{len(seeds)}.json")
                with open(baseline_json_path, "w", encoding="utf-8") as f:
                    json.dump({"seeds": seeds, "summary": baseline_summary},
                               f, indent=2, ensure_ascii=False)
                print(f"   MAP per seed: {[round(m, 4) for m in baseline_summary['MAP_per_seed']]}")

                append_to_combined_csv(
                    json_path=baseline_json_path,
                    model=f"{args.retriever}_random_baseline_avg{len(seeds)}",
                    dataset=pdf_name,
                    chunking=args.chunking,
                    combined_csv="output/eval_results_combined.csv",
                )

            else:
                print("\n[4b/5] Valutazione baseline random (stesso GT, seed singolo)")

                random_retrieve_fn = build_random_baseline(docs, seed=args.baseline_seed)

                baseline_json_path = output_dir / f"eval_{pdf_name}_{eval_suffix}_baseline_random.json"
                baseline_summary = eval_fn(gt, random_retrieve_fn, json_path=baseline_json_path)["summary"]

                append_to_combined_csv(
                    json_path=baseline_json_path,
                    model=f"{args.retriever}_random_baseline",
                    dataset=pdf_name,
                    chunking=args.chunking,
                    combined_csv="output/eval_results_combined.csv",
                )

            n_corpus = len(docs)
            n_relevant_avg = sum(len(q["relevant_ids"]) for q in gt) / len(gt)
            expected = expected_random_metrics(
                n_corpus=n_corpus,
                n_relevant=max(round(n_relevant_avg), 1),
            )

            print(f"\nCorpus: {n_corpus} chunk totali. "
                  f"Rilevanti medi per domanda: {n_relevant_avg:.1f}")

            _print_baseline_comparison(
                real_result["summary"],
                baseline_summary,
                expected,
            )

    # ============================================================
    # 6) Answer generation
    # ============================================================

    if args.question:
        print("\n[5/5] Answer generation")

        sub_questions = [args.question]
        if args.decompose:
            sub_questions = decompose_question(args.question)

        retrieved = retrieve_fn(args.question)

        MAX_CHUNKS = 15
        retrieved = retrieved[:MAX_CHUNKS]

        context = "\n\n".join([d.page_content for d in retrieved])

        query_for_llm = args.question

        if not context.strip():
            answer = "Il contesto non contiene informazioni utili per rispondere alla domanda."
        else:
            answer = generate_answer(query_for_llm, context)

        print("\n=== RISPOSTA ===")
        print(answer)

        suffix = "_decomposed" if args.decompose and len(sub_questions) > 1 else ""
        (output_dir / f"answer{suffix}.txt").write_text(answer, encoding="utf-8")


if __name__ == "__main__":
    main()
