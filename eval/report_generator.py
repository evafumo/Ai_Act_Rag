import json
from pathlib import Path

def generate_full_report(
    gt_agg,
    gt_subq,
    eval_agg_baseline,
    eval_agg_decomposed,
    eval_subq,
    output_path
):
    """
    Genera un report completo che confronta:
        - baseline vs decomposizione (GT aggregato)
        - coverage granulare (GT subquestion)
        - coverage per domanda madre
        - coverage per sotto-domanda
        - errori di retrieval (missed chunks, false positives)
    """

    report = {}

    # ============================================================
    # 1) Baseline vs Decomposizione (GT aggregato)
    # ============================================================

    report["aggregated_comparison"] = {
        "baseline_MAP": eval_agg_baseline["summary"]["MAP"],
        "decomposed_MAP": eval_agg_decomposed["summary"]["MAP"],
        "delta_MAP": eval_agg_decomposed["summary"]["MAP"] - eval_agg_baseline["summary"]["MAP"],
        "baseline_macro": eval_agg_baseline["summary"]["macro_metrics"],
        "decomposed_macro": eval_agg_decomposed["summary"]["macro_metrics"],
    }

    # ============================================================
    # 2) Coverage per domanda madre (GT aggregato)
    # ============================================================

    parent_coverage = {}
    for entry in gt_agg:
        qid = entry["question_id"]
        relevant = {(str(c["chunk_id"]), str(c["source"])) for c in entry["relevant_chunks"]}

        retrieved_baseline = {
            (cid, src)
            for cid, src in eval_agg_baseline["per_query"][qid]["retrieved_ids"]
        }

        retrieved_decomposed = {
            (cid, src)
            for cid, src in eval_agg_decomposed["per_query"][qid]["retrieved_ids"]
        }

        parent_coverage[qid] = {
            "relevant_chunks": list(relevant),
            "baseline_retrieved": list(retrieved_baseline),
            "decomposed_retrieved": list(retrieved_decomposed),
            "baseline_missing": list(relevant - retrieved_baseline),
            "decomposed_missing": list(relevant - retrieved_decomposed),
            "baseline_false_positive": list(retrieved_baseline - relevant),
            "decomposed_false_positive": list(retrieved_decomposed - relevant),
        }

    report["parent_coverage"] = parent_coverage

    # ============================================================
    # 3) Coverage granulare (GT subquestion)
    # ============================================================

    granular_coverage = {}

    for entry in gt_subq:
        qid = entry["question_id"]
        relevant = set(entry["relevant_ids"])

        # trova entry corrispondente in eval_subq
        eval_entry = next(e for e in eval_subq["per_query"] if e["question_id"] == qid)
        retrieved = set(eval_entry["retrieved_ids"])

        granular_coverage[qid] = {
            "query": entry["query"],
            "relevant_chunks": list(relevant),
            "retrieved_chunks": list(retrieved),
            "missing": list(relevant - retrieved),
            "false_positive": list(retrieved - relevant),
            "AP": eval_entry["AP"],
            "metrics": eval_entry["metrics"],
        }

    report["granular_coverage"] = granular_coverage

    # ============================================================
    # 4) Coverage per domanda madre (aggregazione subquestion)
    # ============================================================

    parent_group = {}
    for entry in gt_subq:
        pid = entry["parent_id"]
        parent_group.setdefault(pid, []).append(entry["question_id"])

    parent_granular = {}

    for pid, subqs in parent_group.items():
        parent_granular[pid] = {
            "subquestions": subqs,
            "missing_chunks": [],
            "false_positive_chunks": [],
            "AP_mean": 0.0,
        }

        aps = []
        missing_all = set()
        fp_all = set()

        for qid in subqs:
            cov = granular_coverage[qid]
            aps.append(cov["AP"])
            missing_all |= set(cov["missing"])
            fp_all |= set(cov["false_positive"])

        parent_granular[pid]["AP_mean"] = sum(aps) / len(aps)
        parent_granular[pid]["missing_chunks"] = list(missing_all)
        parent_granular[pid]["false_positive_chunks"] = list(fp_all)

    report["parent_granular_summary"] = parent_granular

    # ============================================================
    # 5) Salvataggio report
    # ============================================================

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[REPORT] Salvato report completo in {output_path}")

    return report

