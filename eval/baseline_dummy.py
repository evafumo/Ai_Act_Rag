# baseline_dummy.py
# ============================================================
#  BASELINE "DUMMY": retrieval casuale, ignora la query
#  Serve come PAVIMENTO per le metriche: se il retriever reale non batte
#  nettamente questi numeri, il task/GT sono sospetti.
# ============================================================

import random
from typing import List, Callable
from langchain_core.documents import Document


def build_random_baseline(all_docs: List[Document], seed: int = 42) -> Callable[..., List[Document]]:
    """
    retrieve_fn(query, k=20) -> k documenti scelti a caso dal corpus,
    senza reinserimento. Deterministico per (seed, query), così due run
    danno lo stesso risultato ma query diverse non ottengono lo stesso campione.
    """
    pool = list(all_docs)

    def retrieve_fn(query: str, k: int = 20) -> List[Document]:
        rng = random.Random(f"{seed}:{query}")
        return rng.sample(pool, min(k, len(pool)))

    return retrieve_fn


def expected_random_metrics(n_corpus: int, n_relevant: int, k_values=(5, 10, 20)) -> dict:
    """
    Valori attesi per un campionamento uniforme SENZA reinserimento
    (distribuzione ipergeometrica):
        E[precision@k] = n_relevant / n_corpus            (costante su k)
        E[recall@k]    = k * n_relevant / n_corpus / n_relevant = k / n_corpus

    Usali come sanity check: i numeri misurati con build_random_baseline
    devono oscillare intorno a questi, non sistematicamente sopra o sotto.
    """
    out = {}
    for k in k_values:
        k_eff = min(k, n_corpus)
        precision = n_relevant / n_corpus if n_corpus else 0.0
        recall = min(k_eff / n_corpus, 1.0) if n_corpus else 0.0
        out[k] = {"precision": precision, "recall": recall}
    return out
