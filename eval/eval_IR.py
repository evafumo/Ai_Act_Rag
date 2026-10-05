from typing import List, Dict, Callable
import json
import math

# ============================================================
#   MODULO UNICO DI VALUTAZIONE IR
#   (unione di eval_IR.py + eval_IR_new.py)
# ============================================================


# ==========================
# LOADER DEL GROUND TRUTH (AGGREGATO, per domanda intera)
# ==========================

def load_ground_truth(path: str) -> List[Dict]:
    """
    Converte il ground truth nel formato richiesto da evaluate_retriever().

    OUTPUT:
        [
            {
                "query": "...",
                "relevant_ids": [(chunk_id, source), ...]
            }
        ]
    """
    gt = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            query = obj["question_text"]

            # tuple (chunk_id, source)
            relevant_ids = [
                (str(c["chunk_id"]), c["source"])
                for c in obj["relevant_chunks"]
            ]

            gt.append({
                "query": query,
                "relevant_ids": relevant_ids
            })

    return gt


# ==========================
# LOADER DEL GROUND TRUTH GRANULARE (per sotto-domanda)
# ==========================

def load_ground_truth_subq(path: str) -> List[Dict]:
    """
    Carica il GT granulare e normalizza le chiavi in forma (chunk_id, source),
    cioè una coppia, NON una stringa concatenata.
    """

    def make_key(cid, src):
        return (str(cid), str(src))

    gt = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            subq_id = obj.get("subquestion_id", "?")

            # Normalizzazione coerente con retriever e RRF
            relevant_ids = [
                make_key(c["chunk_id"], c["source"])
                for c in obj.get("relevant_chunks", [])
            ]

            if not relevant_ids:
                print(f"  [SKIP] {subq_id} non ha relevant_chunks "
                      f"(coverage='{obj.get('coverage', 'n/a')}') — esclusa.")
                skipped += 1
                continue

            gt.append({
                "question_id": subq_id,
                "parent_id": obj.get("question_id"),
                "query": obj["subquestion_text"],
                "relevant_ids": relevant_ids,
            })

    if skipped:
        print(f"  [INFO] {skipped} sotto-domande escluse per assenza di relevant_chunks.")

    return gt


# ========================
#    METRICHE IR
#    (condivise da entrambe le valutazioni)
# ========================

def precision_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    rel = sum(1 for item in top_k if item in relevant)
    return rel / k if k > 0 else 0.0


def recall_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    rel_found = sum(1 for item in top_k if item in relevant)
    return rel_found / len(relevant) if relevant else 0.0


def average_precision(retrieved, relevant):
    if not relevant:
        return 0.0

    precisions = []
    rel_found = 0

    for i, item in enumerate(retrieved):
        if item in relevant:
            rel_found += 1
            precisions.append(rel_found / (i + 1))

    return sum(precisions) / len(relevant)


def dcg_at_k(retrieved, relevant, k):
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved, relevant, k):
    dcg = dcg_at_k(retrieved, relevant, k)
    ideal_ranking = relevant[:k]
    idcg = dcg_at_k(ideal_ranking, relevant, k)
    return dcg / idcg if idcg > 0 else 0.0


# ==============================================
#   VALUTAZIONE SU UN INSIEME DI QUERIES (AGGREGATA)
# ==============================================

def evaluate_retriever(
    queries: List[Dict],
    retrieve_fn: Callable[[str], List],
    k_values: List[int] = [5, 10, 20],
    json_path: str = None,
) -> Dict:
    """
    Valuta un retriever usando matching AND tra (chunk_id, source).
    """
    all_ap = []
    metrics_per_k = {k: {"recall": [], "precision": [], "ndcg": []} for k in k_values}

    per_query_results = []

    for q in queries:
        query_text = q["query"]
        relevant = q["relevant_ids"]  # lista di tuple (id, source)

        # Recupero dal retriever: tuple (id, source)
        docs = retrieve_fn(query_text)
        retrieved = [
            (str(d.metadata.get("chunk_id", "")), d.metadata.get("source", ""))
            for d in docs
        ]

        ap = average_precision(retrieved, relevant)
        all_ap.append(ap)

        query_entry = {
            "query": query_text,
            "relevant_ids": relevant,
            "retrieved_ids": retrieved,
            "AP": ap,
            "metrics": {}
        }

        for k in k_values:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            n = ndcg_at_k(retrieved, relevant, k)

            metrics_per_k[k]["precision"].append(p)
            metrics_per_k[k]["recall"].append(r)
            metrics_per_k[k]["ndcg"].append(n)

            query_entry["metrics"][f"@{k}"] = {
                "precision": p,
                "recall": r,
                "ndcg": n,
                "retrieved_ids_at_k": retrieved[:k]
            }

        per_query_results.append(query_entry)

    map_score = sum(all_ap) / len(all_ap) if all_ap else 0.0

    summary = {
        "MAP": map_score,
        "macro_metrics": {}
    }

    print("\n=== RISULTATI DI VALUTAZIONE ===")
    print(f"MAP: {map_score:.4f}")

    for k in k_values:
        avg_p = sum(metrics_per_k[k]["precision"]) / len(metrics_per_k[k]["precision"])
        avg_r = sum(metrics_per_k[k]["recall"]) / len(metrics_per_k[k]["recall"])
        avg_n = sum(metrics_per_k[k]["ndcg"]) / len(metrics_per_k[k]["ndcg"])

        summary["macro_metrics"][f"@{k}"] = {
            "precision": avg_p,
            "recall": avg_r,
            "ndcg": avg_n
        }

        print(f"\n-- k = {k} --")
        print(f"Precision@{k}: {avg_p:.4f}")
        print(f"Recall@{k}:    {avg_r:.4f}")
        print(f"nDCG@{k}:       {avg_n:.4f}")

    final_output = {
        "summary": summary,
        "per_query": per_query_results
    }

    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"\nJSON salvato in: {json_path}")

    return final_output


# ==============================================
#   VALUTAZIONE GRANULARE (per sotto-domanda)
# ==============================================

def evaluate_retriever_subq(
    queries: List[Dict],
    retrieve_fn: Callable[[str], List],
    k_values: List[int] = [5, 10, 20],
    json_path: str = None,
) -> Dict:
    """
    Valutazione coerente con chiave unica (chunk_id, source).
    """

    def make_key(cid, src):
        return (str(cid), str(src))

    all_ap = []
    metrics_per_k = {k: {"recall": [], "precision": [], "ndcg": []} for k in k_values}
    per_query_results = []

    parent_ap: Dict[str, list] = {}
    parent_metrics_per_k: Dict[str, dict] = {}

    for q in queries:
        query_text = q["query"]
        relevant_keys = q["relevant_ids"]
        qid = q.get("question_id", query_text)
        pid = q.get("parent_id", "unknown")

        docs = retrieve_fn(query_text)

        # Normalizzazione coerente con RRF e retriever
        retrieved_keys = []
        for d in docs:
            cid = str(d.metadata["chunk_id"])
            src = str(d.metadata["source"])
            retrieved_keys.append(make_key(cid, src))

        # --- AP ---
        ap = average_precision(retrieved_keys, relevant_keys)
        all_ap.append(ap)
        parent_ap.setdefault(pid, []).append(ap)
        parent_metrics_per_k.setdefault(
            pid, {k: {"recall": [], "precision": [], "ndcg": []} for k in k_values}
        )

        query_entry = {
            "question_id": qid,
            "parent_id": pid,
            "query": query_text,
            "relevant_ids": relevant_keys,
            "retrieved_ids": retrieved_keys,
            "AP": ap,
            "metrics": {}
        }

        # --- metriche @k ---
        for k in k_values:
            p = precision_at_k(retrieved_keys, relevant_keys, k)
            r = recall_at_k(retrieved_keys, relevant_keys, k)
            n = ndcg_at_k(retrieved_keys, relevant_keys, k)

            metrics_per_k[k]["precision"].append(p)
            metrics_per_k[k]["recall"].append(r)
            metrics_per_k[k]["ndcg"].append(n)

            parent_metrics_per_k[pid][k]["precision"].append(p)
            parent_metrics_per_k[pid][k]["recall"].append(r)
            parent_metrics_per_k[pid][k]["ndcg"].append(n)

            query_entry["metrics"][f"@{k}"] = {
                "precision": p,
                "recall": r,
                "ndcg": n,
                "retrieved_ids_at_k": retrieved_keys[:k]
            }

        per_query_results.append(query_entry)

    # --- MAP globale ---
    map_score = sum(all_ap) / len(all_ap) if all_ap else 0.0

    summary = {"MAP": map_score, "macro_metrics": {}}
    for k in k_values:
        avg_p = sum(metrics_per_k[k]["precision"]) / len(metrics_per_k[k]["precision"])
        avg_r = sum(metrics_per_k[k]["recall"]) / len(metrics_per_k[k]["recall"])
        avg_n = sum(metrics_per_k[k]["ndcg"]) / len(metrics_per_k[k]["ndcg"])
        summary["macro_metrics"][f"@{k}"] = {"precision": avg_p, "recall": avg_r, "ndcg": avg_n}

    # --- MAP per domanda madre ---
    by_parent = {}
    for pid, aps in parent_ap.items():
        p_map = sum(aps) / len(aps) if aps else 0.0
        p_summary = {"MAP": p_map, "macro_metrics": {}}
        for k in k_values:
            m = parent_metrics_per_k[pid][k]
            p_summary["macro_metrics"][f"@{k}"] = {
                "precision": sum(m["precision"]) / len(m["precision"]),
                "recall": sum(m["recall"]) / len(m["recall"]),
                "ndcg": sum(m["ndcg"]) / len(m["ndcg"]),
            }
        by_parent[pid] = p_summary

    print("\n=== RISULTATI DI VALUTAZIONE (GRANULARE) ===")
    print(f"MAP globale: {map_score:.4f}")

    for k in k_values:
        m = summary["macro_metrics"][f"@{k}"]
        print(f"\n-- k = {k} --")
        print(f"Precision@{k}: {m['precision']:.4f}")
        print(f"Recall@{k}:    {m['recall']:.4f}")
        print(f"nDCG@{k}:       {m['ndcg']:.4f}")

    print("\n--- MAP aggregata per domanda madre ---")
    for pid, p_summary in by_parent.items():
        print(f"  {pid}: MAP={p_summary['MAP']:.4f}")

    final_output = {
        "summary": summary,
        "by_parent": by_parent,
        "per_query": per_query_results
    }

    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"\nJSON salvato in: {json_path}")

    return final_output
