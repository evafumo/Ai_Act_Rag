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

__all__ = [
    "build_retriever_combo",
    "unified_retrieval",
    "rrf_fusion",
    "print_rrf_debug_table",
    "export_rrf_debug_json",
]

