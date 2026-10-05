from typing import List, Tuple
from langchain_core.documents import Document

from retrievers.retriever import unified_retrieval, retrieve_raw
from rerankers.reranker import rerank_with_bge


def _doc_key(doc: Document):
    return (
        str(doc.metadata.get("chunk_id")),
        str(doc.metadata.get("source")),
    )


def multi_query_retrieval_rerank_each(
    retrievers: Tuple,
    original_query: str,
    sub_questions: List[str],
    k_per_subquery: int = 10,
    k_final: int = 20,
    rrf_k: int = 40,
    debug: bool = False,
) -> List[Document]:

    """
    Multi-query retrieval con reranking per ogni sottodomanda.

    INPUT:
        retrievers (Tuple)
        original_query (str)
        sub_questions (list[str])
        k_per_subquery (int)
        k_final (int)
        rrf_k (int)
        debug (bool)

    OUTPUT:
        list[Document]

    NOTE:
        Per ogni sottodomanda viene eseguito retrieval con eventuale
        fusione RRF tra retriever e reranking locale. Le classifiche
        ottenute vengono poi fuse tramite RRF senza effettuare un
        reranking finale.
    """

    per_subquery_ranked = []

    for sub_q in sub_questions:
        docs = unified_retrieval(
            retrievers,
            sub_q,
            k=k_per_subquery,
            rrf_k=rrf_k,
            debug=debug,
        )

        per_subquery_ranked.append(docs)

    rrf_scores = {}
    doc_lookup = {}

    for docs in per_subquery_ranked:
        for rank, doc in enumerate(docs):

            key = _doc_key(doc)

            doc_lookup[key] = doc

            rrf_scores[key] = (
                rrf_scores.get(key, 0.0)
                + 1.0 / (rrf_k + rank + 1)
            )

    fused_sorted = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    fused_docs = [
        doc_lookup[key]
        for key, _ in fused_sorted
    ]

    return fused_docs[:k_final]


def multi_query_retrieval_rerank_once(
    retrievers: Tuple,
    original_query: str,
    sub_questions: List[str],
    k_per_subquery: int = 15,
    k_final: int = 20,
    rrf_k: int = 40,
    debug: bool = False,
) -> List[Document]:
    """
    Multi-query retrieval con un unico reranking finale.

    INPUT:
        retrievers (Tuple)
        original_query (str)
        sub_questions (list[str])
        k_per_subquery (int)
        k_final (int)
        rrf_k (int)
        debug (bool)

    OUTPUT:
        list[Document]

    NOTE:
        I documenti recuperati dalle diverse sottodomande vengono
        unificati Successivamente viene eseguito
        un unico reranking rispetto alla query originale.
    """

    seen = set()
    merged_docs: List[Document] = []

    for sub_q in sub_questions:

        docs = retrieve_raw(
            retrievers,
            sub_q,
            k=k_per_subquery,
            rrf_k=rrf_k,
            debug=debug,
        )

        for doc in docs:

            key = _doc_key(doc)

            if key not in seen:
                seen.add(key)
                merged_docs.append(doc)

    final_docs = rerank_with_bge(
        original_query,
        merged_docs,
        top_n=k_final,
    )

    return final_docs
