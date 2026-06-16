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
#  RRF - Reciprocal Rank Fusion (veloce con chunk_id)
# ============================================================

def rrf_fusion(resultsA, resultsB, k=60, debug=False):
    """
    Applica la Reciprocal Rank Fusion (RRF) a due liste di documenti.
    Combina due ranking diversi in un unico ranking.

    INPUT:
        resultsA (List[Document]):
            Risultati del primo retriever.
        resultsB (List[Document]):
            Risultati del secondo retriever.
        k (int):
            Costante di normalizzazione RRF.
        debug (bool):
            Se True stampa i contributi di ogni documento.

    OUTPUT:
        fused_docs (List[Document]):
            Lista di documenti ordinati per score RRF.
        scores (dict):
            Mappa chunk_id - punteggio RRF.
        docs_map (dict):
            Mappa chunk_id - documento originale.

    NOTE:
        - RRF è robusto perché non dipende dai valori assoluti degli score.
        - Usa solo il rank relativo.
    """

    scores = {}
    docs_map = {}

 
    def make_key(doc: Document):
        return doc.metadata.get("chunk_id")

    # --- lista A ---
    for rank, doc in enumerate(resultsA):
        key = make_key(doc)
        docs_map[key] = doc
        increment = 1 / (k + rank + 1)
        scores[key] = scores.get(key, 0) + increment

        if debug:
            print(f"[RRF][A] rank={rank:3d}  +{increment:.6f}  chunk_id={key}")

    # --- lista B ---
    for rank, doc in enumerate(resultsB):
        key = make_key(doc)
        docs_map[key] = doc
        increment = 1 / (k + rank + 1)
        scores[key] = scores.get(key, 0) + increment

        if debug:
            print(f"[RRF][B] rank={rank:3d}  +{increment:.6f}  chunk_id={key}")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused_docs = [docs_map[key] for key, _ in ranked]

    return fused_docs, scores, docs_map


# ============================================================
#  DEBUG TABLE
# ============================================================

def print_rrf_debug_table(resultsA, resultsB, scores, docs_map, max_len=80):
    """
   
    Stampare una tabella che mostra:
            - rank in A
            - rank in B
            - score RRF
            - anteprima contenuto
            - metadata

    INPUT:
        resultsA, resultsB:
            Liste di Document.
        scores (dict):
            Punteggi RRF.
        docs_map (dict):
            chunk_id → Document.
        max_len (int):
            Lunghezza massima dell'anteprima contenuto.

    OUTPUT:
        Nessun output di ritorno.
        Stampa su stdout.

    """

    def make_key(doc):
        return doc.metadata.get("chunk_id")

    rankA = {make_key(doc): i for i, doc in enumerate(resultsA)}
    rankB = {make_key(doc): i for i, doc in enumerate(resultsB)}

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



def export_rrf_debug_json(resultsA, resultsB, scores, docs_map, max_len=200) -> Dict[str, Any]:
    """
        Esporta i risultati RRF in formato JSON

    INPUT:
        resultsA, resultsB:
            Liste di Document.
        scores (dict):
            Punteggi RRF.
        docs_map (dict):
            chunk_id → Document.
        max_len (int):
            Lunghezza massima anteprima contenuto.

    OUTPUT:
        dict:
            {
                "rrf_results": [...],
                "total_documents": int
            }
    """

    def make_key(doc):
        return doc.metadata.get("chunk_id")

    rankA = {make_key(doc): i for i, doc in enumerate(resultsA)}
    rankB = {make_key(doc): i for i, doc in enumerate(resultsB)}

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
            "metadata": doc.metadata
        })

    return {
        "rrf_results": export_list,
        "total_documents": len(export_list)
    }


# ============================================================
#  SINGLE RETRIEVERS
# ============================================================

def build_single_bge(chunks: List[Document]):
    """
    Costruisce un retriever con BGE-small.

    INPUT:
        chunks (List[Document]):
            Documenti da indicizzare.

    OUTPUT:
        (retriever,):
            Tupla contenente un solo retriever FAISS.
    """

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
    """
    Costruisce un retriever BM25 singolo.

    INPUT:
        chunks (List[Document]):
            Documenti da indicizzare.

    OUTPUT:
        (retriever,):
            Tupla contenente un solo retriever BM25.
    """
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 20
    return (bm25,)


# ============================================================
#  DUAL RETRIEVER COMBO (RRF) 
# ============================================================

def build_dense_bge_e5(chunks: List[Document]) -> Tuple:
    """
    Costruisce una coppia di retriever densi (BGE + E5) da usare con RRF.

    INPUT:
        chunks (List[Document])

    OUTPUT:
        (retrA, retrB):
            Due retriever FAISS.
    """

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
    """
    Factory che costruisce automaticamente la combinazione
    di retriever richiesta dal nome.

    INPUT:
        name (str):
            Nome della combo (single_bge, dense_bge_e5, ecc.)
        chunks (List[Document]):
            Documenti da indicizzare.

    OUTPUT:
        Tuple:
            Uno o due retriever a seconda della combo.

    NOTE:
        - Se il nome non è riconosciuto -> ValueError.
    """

    # --- SINGLE ---
    if name == "single_bge":
        return build_single_bge(chunks)
    elif name == "single_e5":
        return build_single_e5(chunks)
    elif name == "single_bgem3":
        return build_single_bgem3(chunks)
    elif name == "single_bm25":
        return build_single_bm25(chunks)

    # --- DUAL (RRF) ---
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
    """
    Esegue il retrieval:
            - 1 retriever - retrieval + reranking
            - 2 retriever - RRF fusion + reranking
    INPUT:
        retrievers (Tuple):
            Uno o due retriever.
        query (str):
            Domanda dell’utente.
        k (int):
            Numero di documenti finali.
        rrf_k (int):
            Costante RRF.
        debug (bool):
            Stampa debug RRF.
        return_debug (bool):
            Restituisce struttura completa per analisi.

    OUTPUT:
        List[Document] oppure dict (se return_debug=True)

    NOTE:
        - Il reranking finale usa sempre il CrossEncoder.
        - Supporta solo 1 o 2 retriever.
    """

    # Caso singolo retriever
    if len(retrievers) == 1:
        retr = retrievers[0]
        docs = retr.get_relevant_documents(query)
        docs = docs[:k]

        # RERANKING ANCHE PER I SINGOLI
        reranked = rerank_with_bge(query, docs, top_n=k)

        if return_debug:
            return {
                "mode": "single_reranked",
                "documents": reranked,
                "scores": None,
                "docs_map": None
            }
        return reranked

    # Caso RRF
    elif len(retrievers) == 2:
        retrA, retrB = retrievers

        resultsA = retrA.get_relevant_documents(query)
        resultsB = retrB.get_relevant_documents(query)

        fused_docs, scores, docs_map = rrf_fusion(
            resultsA, resultsB, k=rrf_k, debug=debug
        )

        fused_docs = fused_docs[:k]

        # RERANKING DOPO RRF
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

