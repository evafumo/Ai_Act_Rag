from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

_bge_reranker = None


def get_bge_reranker():
    """
    Carica e restituisce un modello CrossEncoder per il reranking.
    Il modello viene caricato una sola volta e riutilizzato per tutte le chiamate successive.

    INPUT:
        Nessun input.

    OUTPUT:
        CrossEncoder:
            Istanza del modello "ms-marco-MiniLM-L-6-v2" già caricata in RAM.

    NOTE:
        - Evita ricaricamenti del modello.
        - Usa la variabile globale _bge_reranker come cache.
    """
    global _bge_reranker
    if _bge_reranker is None:
        _bge_reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu"
        )

    return _bge_reranker


def rerank_with_bge(query: str, docs: List[Document], top_n: int = 10):
    """
    Riordina con una lista di documenti usando un CrossEncoder.
    Il modello valuta la rilevanza query–documento leggendo entrambi contemporaneamente.

    INPUT:
        query (str):
            La domanda dell’utente.
        docs (List[Document]):
            Documenti restituiti dal retriever (BM25, FAISS, ecc.).
        top_n (int):
            Numero massimo di documenti da restituire dopo il reranking.

    OUTPUT:
        List[Document]:
            Lista dei documenti ordinati per rilevanza decrescente,
            tagliata ai primi top_n.

    NOTE:
        - pairs = [[query, doc]] è il formato richiesto dal modello.
        - model.predict() restituisce uno score per ogni coppia.
    """
    if not docs:
        return []

    model = get_bge_reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]

