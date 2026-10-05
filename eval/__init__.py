# ============================================================
# evaluation package initializer
# ============================================================

# --- IR aggregato ---
from .eval_IR import (
    load_ground_truth,
    precision_at_k,
    recall_at_k,
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    evaluate_retriever,
    load_ground_truth_subq,
    evaluate_retriever_subq
)


# --- Dummy System ---
from .baseline_dummy import  (
    build_random_baseline,
    expected_random_metrics
)

# --- CSV / export ---
from .eval_IR_plots import (
    append_to_combined_csv
)

__all__ = [
    # IR aggregato
    "load_ground_truth",
    "precision_at_k",
    "recall_at_k",
    "average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_retriever",

    # IR granulare
    "load_ground_truth_subq",
    "evaluate_retriever_subq",

    # CSV
    "append_to_combined_csv",
]

