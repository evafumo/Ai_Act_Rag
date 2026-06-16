from typing import List, Dict, Callable
import json
import math    

# ==========================
# LOADER DEL GROUND TRUTH 
# ==========================

def load_ground_truth(path: str) -> List[Dict]:
    """
    Converte il ground truth nel formato richiesto da evaluate_retriever().

    INPUT:
        path (str):
            Percorso al file JSONL del ground truth.

    OUTPUT:
        List[Dict]:
            [
                {
                    "query": "...",
                    "relevant_ids": ["6", "7"]
                },
                ...
            ]
    """
    gt = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            query = obj["question_text"]
            relevant_ids = [str(c["chunk_id"]) for c in obj["relevant_chunks"]]

            gt.append({
                "query": query,
                "relevant_ids": relevant_ids
            })

    return gt


# ========================
#    METRICHE IR
# ========================

def precision_at_k(retrieved_ids, relevant_ids, k):
    """
    Calcola la precision@k.

    INPUT:
        retrieved_ids (List[str])
        relevant_ids (List[str])
        k (int)

    OUTPUT:
        float:
            Precision@k
            
    """
    top_k = retrieved_ids[:k]
    rel = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return rel / k if k > 0 else 0.0


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    rel_found = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return rel_found / len(relevant_ids) if relevant_ids else 0.0


def average_precision(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0

    precisions = []
    rel_found = 0

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            rel_found += 1
            precisions.append(rel_found / (i + 1))

    return sum(precisions) / len(relevant_ids)


def dcg_at_k(retrieved_ids, relevant_ids, k):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    ideal_ranking = relevant_ids[:k]
    idcg = dcg_at_k(ideal_ranking, relevant_ids, k)
    return dcg / idcg if idcg > 0 else 0.0


# ==============================================
#   VALUTAZIONE SU UN INSIEME DI QUERIES
# ==============================================

def evaluate_retriever(
    queries: List[Dict],
    retrieve_fn: Callable[[str], List],
    k_values: List[int] = [5, 10, 20],
    json_path: str = None,
) -> Dict:
    """
    Valuta un retriever su un insieme di query
    usando le metriche IR definite in precedenza.

    INPUT:
        queries (List[Dict]):
            Ground truth convertito.
        retrieve_fn (Callable):
            Funzione che esegue il retrieval.
        k_values (List[int]):
            Valori di k per precision/recall/nDCG.
        json_path (str|None):
            Se fornito, salva i risultati in JSON.

    OUTPUT:
        Dict:
            {
                "summary": {...},
                "per_query": [...]
            }

    NOTE:
        - Calcola MAP, precision@k, recall@k, nDCG@k.
        - retrieve_fn deve restituire Document con metadata["chunk_id"].
    """
    all_ap = []
    metrics_per_k = {k: {"recall": [], "precision": [], "ndcg": []} for k in k_values}

    # Risultati dettagliati per ogni query
    per_query_results = []

    for q in queries:
        query_text = q["query"]
        relevant_ids = [str(r) for r in q["relevant_ids"]]

        docs = retrieve_fn(query_text)
        retrieved_ids = [str(d.metadata.get("chunk_id", "")) for d in docs]

        ap = average_precision(retrieved_ids, relevant_ids)
        all_ap.append(ap)

        # Struttura JSON per questa query
        query_entry = {
            "query": query_text,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "AP": ap,
            "metrics": {}
        }

        for k in k_values:
            p = precision_at_k(retrieved_ids, relevant_ids, k)
            r = recall_at_k(retrieved_ids, relevant_ids, k)
            n = ndcg_at_k(retrieved_ids, relevant_ids, k)

            metrics_per_k[k]["precision"].append(p)
            metrics_per_k[k]["recall"].append(r)
            metrics_per_k[k]["ndcg"].append(n)

            query_entry["metrics"][f"@{k}"] = {
                "precision": p,
                "recall": r,
                "ndcg": n,
                "retrieved_ids_at_k": retrieved_ids[:k]
            }

        per_query_results.append(query_entry)

    # Riepilogo globale
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

    # Struttura finale JSON
    final_output = {
        "summary": summary,
        "per_query": per_query_results
    }

    # Salvataggio JSON opzionale
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"\nJSON salvato in: {json_path}")

    return final_output

