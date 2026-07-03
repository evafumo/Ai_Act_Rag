# Package: evaluation
# Esporta le funzioni pubbliche per la valutazione IR.

from .eval_IR import (
    load_ground_truth,
    precision_at_k,
    recall_at_k,
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    evaluate_retriever
)

from .eval_IR_plots import (
    append_to_combined_csv
)
__all__ = [
    "load_ground_truth",
    "precision_at_k",
    "recall_at_k",
    "average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_retriever",
    "append_to_combined_csv"
]

