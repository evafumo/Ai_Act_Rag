# Package: evaluation
# Esporta le funzioni pubbliche per la valutazione IR.

from .eval_IR import (
    load_ground_truth,
    precision_at_k,
    recall_at_k,
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    evaluate_retriever,
)

__all__ = [
    "load_ground_truth",
    "precision_at_k",
    "recall_at_k",
    "average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_retriever",
]

