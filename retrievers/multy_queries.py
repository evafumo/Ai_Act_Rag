from typing import List, Tuple
from langchain_core.documents import Document

from retrievers.retriever import retrieve_raw, rrf_fuse_rankings, doc_key
from rerankers.reranker import rerank_with_bge


def multi_query_retrieval_rerank_each(
    retrievers: Tuple,
    original_query: str,
    sub_questions: List[str],
    k_per_subquery: int = 10,
    k_final: int = 20,
    rrf_k: int = 40,
    debug: bool = False,
    k_pool: int | None = None,
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
        k_pool (int | None):
            Candidati recuperati per sottodomanda PRIMA del reranking
            (default: 2 * k_per_subquery). Il cross-encoder li riordina
            e ne tiene i migliori k_per_subquery.

    OUTPUT:
        list[Document]

    NOTE:
        Per ogni sottodomanda: retrieval (con eventuale RRF tra retriever)
        di un pool piu' ampio di k_per_subquery, poi reranking con la
        sottodomanda e taglio a k_per_subquery. In questo modo il reranker
        puo' recuperare documenti che il retrieval aveva messo oltre il
        k_per_subquery-esimo posto. Le classifiche delle sottodomande
        vengono poi fuse con RRF, senza reranking finale.
        Nota: il pool e' limitato dal k dei singoli retriever
        (20 per i single, 40 per le combo).
    """

    if k_pool is None:
        k_pool = 2 * k_per_subquery
    k_pool = max(k_pool, k_per_subquery)

    per_subquery_ranked = []

    for sub_q in sub_questions:
        # 1) pool ampio senza reranking
        pool = retrieve_raw(
            retrievers,
            sub_q,
            k=k_pool,
            rrf_k=rrf_k,
            debug=debug,
        )

        # 2) reranking sulla sottodomanda, taglio a k_per_subquery
        docs = rerank_with_bge(
            sub_q,
            pool,
            top_n=k_per_subquery,
        )

        per_subquery_ranked.append(docs)

    # Fusione RRF tra le classifiche delle sottodomande
    # (chiave (chunk_id, source): sicura con piu' PDF)
    fused_docs, _, _ = rrf_fuse_rankings(
        per_subquery_ranked, k=rrf_k, debug=debug, tag="MQ-RRF"
    )

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

            key = doc_key(doc)

            if key not in seen:
                seen.add(key)
                merged_docs.append(doc)

    final_docs = rerank_with_bge(
        original_query,
        merged_docs,
        top_n=k_final,
    )

    return final_docs
