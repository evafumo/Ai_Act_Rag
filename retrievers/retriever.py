# retrievers/retriever_factory.py
# ============================================================
#  COSTRUZIONE DELLE COMBO DI RETRIEVER + RRF + DEBUG + SINGLE
# ============================================================

from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Import del reranker
from rerankers.reranker import rerank_with_bge


# ============================================================
#  RRF - Reciprocal Rank Fusion (coerente con (chunk_id, source))
# ============================================================

def doc_key(doc: Document):
    """
    Chiave univoca di un chunk: (chunk_id, source).
    Necessaria con piu' PDF nello stesso indice, perche' lo stesso
    chunk_id puo' comparire in documenti diversi.
    Coerente con GT e valutazione.
    """
    return (str(doc.metadata["chunk_id"]), str(doc.metadata["source"]))


def rrf_fuse_rankings(rankings: List[List[Document]], k: int = 60, debug: bool = False, tag: str = "RRF"):
    """
    RRF generale su N classifiche (liste di Document gia' ordinate).
    Usata sia per fondere i retriever (N=2) sia per fondere le
    sotto-domande del multi-query (N=numero di sub_questions).

    OUTPUT:
        fused_docs (List[Document]), scores (dict), docs_map (dict)
        con chiavi (chunk_id, source).
    """
    scores = {}
    docs_map = {}

    for list_idx, results in enumerate(rankings):
        for rank, doc in enumerate(results):
            key = doc_key(doc)
            docs_map.setdefault(key, doc)
            increment = 1 / (k + rank + 1)
            scores[key] = scores.get(key, 0) + increment

            if debug:
                print(f"[{tag}][{list_idx}] rank={rank:3d}  +{increment:.6f}  key={key}")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused_docs = [docs_map[key] for key, _ in ranked]

    return fused_docs, scores, docs_map


def rrf_fusion(resultsA, resultsB, k=60, debug=False):
    """
    RRF tra due retriever (wrapper di rrf_fuse_rankings).
    Chiave unica (chunk_id, source).
    """
    return rrf_fuse_rankings([resultsA, resultsB], k=k, debug=debug, tag="RRF")


# ============================================================
#  DEBUG TABLE (coerente con (chunk_id, source))
# ============================================================

def print_rrf_debug_table(resultsA, resultsB, scores, docs_map, max_len=80):

    rankA = {doc_key(doc): i for i, doc in enumerate(resultsA)}
    rankB = {doc_key(doc): i for i, doc in enumerate(resultsB)}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n==================== RRF DEBUG TABLE ====================")
    print(f"{'RRF Rank':>8} | {'Rank A':>6} | {'Rank B':>6} | {'Score':>10} | {'Content':<40} | Metadata")
    print("----------------------------------------------------------")

    for i, (key, score) in enumerate(ranked):
        doc = docs_map[key]

        rA = rankA.get(key, "-")
        rB = rankB.get(key, "-")

        content = doc.page_content.replace("\n", " ")
        if len(content) > max_len:
            content = content[:max_len] + "..."

        meta = ", ".join(f"{k}={v}" for k, v in doc.metadata.items())

        print(f"{i:8d} | {str(rA):>6} | {str(rB):>6} | {score:10.6f} | {content:<40} | {meta}")

    print("==========================================================\n")


# ============================================================
#  EXPORT RRF DEBUG JSON (coerente con (chunk_id, source))
# ============================================================

def export_rrf_debug_json(resultsA, resultsB, scores, docs_map, max_len=200):

    rankA = {doc_key(doc): i for i, doc in enumerate(resultsA)}
    rankB = {doc_key(doc): i for i, doc in enumerate(resultsB)}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    export_list = []

    for global_rank, (key, score) in enumerate(ranked):
        doc = docs_map[key]

        content = doc.page_content.replace("\n", " ")
        if len(content) > max_len:
            content = content[:max_len] + "..."

        export_list.append({
            "global_rank": global_rank,
            "rank_in_A": rankA.get(key, None),
            "rank_in_B": rankB.get(key, None),
            "rrf_score": score,
            "content_preview": content,
            "metadata": doc.metadata,
            "key": key
        })

    return {
        "rrf_results": export_list,
        "total_documents": len(export_list)
    }


# ============================================================
#  SINGLE RETRIEVERS
# ============================================================

def build_single_bge(chunks: List[Document]):
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vs = FAISS.from_documents(chunks, emb)
    retr = vs.as_retriever(search_kwargs={"k": 20})
    return (retr,)


def build_single_e5(chunks: List[Document]):
    emb = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
    vs = FAISS.from_documents(chunks, emb)
    retr = vs.as_retriever(search_kwargs={"k": 20})
    return (retr,)


def build_single_bgem3(chunks: List[Document]):
    emb = HuggingFaceEmbeddings(model_name="TaylorAI/bge-micro-v2")
    vs = FAISS.from_documents(chunks, emb)
    retr = vs.as_retriever(search_kwargs={"k": 20})
    return (retr,)


def build_single_bm25(chunks: List[Document]):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 20
    return (bm25,)


# ============================================================
#  DUAL RETRIEVER COMBO (RRF)
# ============================================================

def build_dense_bge_e5(chunks: List[Document]) -> Tuple:
    embA = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    embB = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

    vsA = FAISS.from_documents(chunks, embA)
    retrA = vsA.as_retriever(search_kwargs={"k": 40})

    vsB = FAISS.from_documents(chunks, embB)
    retrB = vsB.as_retriever(search_kwargs={"k": 40})

    return retrA, retrB


def build_dense_bgem3_e5(chunks: List[Document]) -> Tuple:
    embA = HuggingFaceEmbeddings(model_name="TaylorAI/bge-micro-v2")
    embB = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

    vsA = FAISS.from_documents(chunks, embA)
    retrA = vsA.as_retriever(search_kwargs={"k": 40})

    vsB = FAISS.from_documents(chunks, embB)
    retrB = vsB.as_retriever(search_kwargs={"k": 40})

    return retrA, retrB


def build_bm25_bge(chunks: List[Document]) -> Tuple:
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 40

    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vs = FAISS.from_documents(chunks, emb)
    dense = vs.as_retriever(search_kwargs={"k": 40})

    return bm25, dense


def build_bm25_e5(chunks: List[Document]) -> Tuple:
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 40

    emb = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
    vs = FAISS.from_documents(chunks, emb)
    dense = vs.as_retriever(search_kwargs={"k": 40})

    return bm25, dense


# ============================================================
#  FACTORY
# ============================================================

def build_retriever_combo(name: str, chunks: List[Document]) -> Tuple:

    if name == "single_bge":
        return build_single_bge(chunks)
    elif name == "single_e5":
        return build_single_e5(chunks)
    elif name == "single_bgem3":
        return build_single_bgem3(chunks)
    elif name == "single_bm25":
        return build_single_bm25(chunks)

    elif name == "dense_bge_e5":
        return build_dense_bge_e5(chunks)
    elif name == "dense_bgem3_e5":
        return build_dense_bgem3_e5(chunks)
    elif name == "bm25_bge":
        return build_bm25_bge(chunks)
    elif name == "bm25_e5":
        return build_bm25_e5(chunks)

    else:
        raise ValueError(f"Combo retriever sconosciuta: {name}")


# ============================================================
#  UNIFIED RETRIEVAL + RERANKING
# ============================================================

def unified_retrieval(
    retrievers: Tuple,
    query: str,
    k: int = 10,
    rrf_k: int = 40,
    debug: bool = False,
    return_debug: bool = False
):

    if len(retrievers) == 1:
        retr = retrievers[0]
        docs = retr.get_relevant_documents(query)
        docs = docs[:k]
        reranked = rerank_with_bge(query, docs, top_n=k)

        if return_debug:
            return {
                "mode": "single_reranked",
                "documents": reranked,
                "scores": None,
                "docs_map": None
            }
        return reranked

    elif len(retrievers) == 2:
        retrA, retrB = retrievers

        resultsA = retrA.get_relevant_documents(query)
        resultsB = retrB.get_relevant_documents(query)

        fused_docs, scores, docs_map = rrf_fusion(
            resultsA, resultsB, k=rrf_k, debug=debug
        )

        fused_docs = fused_docs[:k]

        reranked = rerank_with_bge(query, fused_docs, top_n=k)

        if return_debug:
            return {
                "mode": "rrf_reranked",
                "documents": reranked,
                "scores": scores,
                "docs_map": docs_map,
                "resultsA": resultsA,
                "resultsB": resultsB
            }

        return reranked

    else:
        raise ValueError(
            f"unified_retrieval supporta solo 1 o 2 retriever, ricevuti: {len(retrievers)}"
        )


# ============================================================
#  ONLY RETRIEVAL PER SUBQUESTIONS
# ============================================================        

def retrieve_raw(
    retrievers: Tuple,
    query: str,
    k: int = 20,
    rrf_k: int = 40,
    debug: bool = False,
):

    if len(retrievers) == 1:
        retr = retrievers[0]
        docs = retr.get_relevant_documents(query)
        return docs[:k]

    elif len(retrievers) == 2:
        retrA, retrB = retrievers
        resultsA = retrA.get_relevant_documents(query)
        resultsB = retrB.get_relevant_documents(query)
        fused_docs, _, _ = rrf_fusion(resultsA, resultsB, k=rrf_k, debug=debug)
        return fused_docs[:k]

    else:
        raise ValueError(
            f"retrieve_raw supporta solo 1 o 2 retriever, ricevuti: {len(retrievers)}"
        )

