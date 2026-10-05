# Package: retrievers
# Contiene retriever densi, BM25 e funzioni di fusione.

# Package: retrievers
# Esporta solo le funzioni pubbliche utili alla pipeline RAG.

from .retriever import (
    build_retriever_combo,
    unified_retrieval,
    rrf_fusion,
    print_rrf_debug_table,
    export_rrf_debug_json,
)

from .multy_queries import (
    multi_query_retrieval_rerank_each,
    multi_query_retrieval_rerank_once
)

__all__ = [
    "build_retriever_combo",
    "unified_retrieval",
    "rrf_fusion",
    "print_rrf_debug_table",
    "export_rrf_debug_json",
    "multi_query_retrieval_rerank_each",
    "multi_query_retrieval_rerank_once"
]

