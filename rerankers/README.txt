# rerankers — Reranking con CrossEncoder

Questo modulo applica un CrossEncoder per riordinare i documenti recuperati.

------------------------------------------------------------
FUNZIONI
------------------------------------------------------------

get_bge_reranker()
    Carica il modello MS MARCO MiniLM con caching.

rerank_with_bge(query, docs, top_n)
    Ricalcola la rilevanza query-documento e riordina i risultati.

------------------------------------------------------------
NOTE
------------------------------------------------------------

- Il CrossEncoder legge query e documento insieme.
- Migliora la qualità del retrieval rispetto ai soli retriever densi.

