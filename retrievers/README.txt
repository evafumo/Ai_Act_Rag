RETRIEVER FACTORY — DESCRIZIONE DEL MODULO
==========================================

Questo modulo implementa la logica completa per costruire retriever singoli,
retriever combinati e per applicare la Reciprocal Rank Fusion (RRF).

Il file fornisce:
- costruzione retriever singoli (dense e sparse)
- costruzione retriever doppi (combo dense+dense o sparse+dense)
- fusione RRF
- funzioni di debug
- unified retrieval con reranking CrossEncoder


1. RETRIEVER SINGOLI
---------------------

Il modulo implementa retriever basati su:

- BGE-small
- E5-small
- BGEM3-micro
- BM25 

Ogni retriever singolo restituisce una tupla contenente un solo retriever.


2. RETRIEVER COMBINATI (RRF)
-----------------------------

Il modulo permette di combinare due retriever tramite Reciprocal Rank Fusion.

Combinazioni supportate:
- BGE + E5
- BGEM3 + E5
- BM25 + BGE
- BM25 + E5

La funzione rrf_fusion() combina i ranking dei due modelli in un unico ranking
robusto, basato solo sulla posizione relativa dei documenti.


3. RRF FUSION
--------------

La funzione rrf_fusion():
- prende due liste di Document
- assegna un punteggio RRF a ogni chunk_id
- produce un ranking unico
- restituisce:
    - lista documenti fusi
    - punteggi RRF
    - mappa chunk_id - documento

Sono disponibili anche:
- print_rrf_debug_table() per debug leggibile
- export_rrf_debug_json() per esportazione JSON


4. FACTORY DEI RETRIEVER
-------------------------

La funzione build_retriever_combo(name, chunks) costruisce automaticamente
il retriever richiesto in base al nome passato dalla CLI.

Nomi supportati:
- single_bge
- single_e5
- single_bgem3
- single_bm25
- dense_bge_e5
- dense_bgem3_e5
- bm25_bge
- bm25_e5

Se il nome non è riconosciuto - ValueError.


5. UNIFIED RETRIEVAL + RERANKING
---------------------------------

La funzione unified_retrieval() gestisce due casi:

Caso 1 — retriever singolo:
    - retrieval
    - reranking con CrossEncoder

Caso 2 — due retriever:
    - retrieval A
    - retrieval B
    - fusione RRF
    - reranking finale

Supporta anche modalità debug.


6. DIPENDENZE PRINCIPALI
-------------------------

- FAISS per indicizzazione densa
- BM25Retriever per retrieval sparso
- HuggingFaceEmbeddings per embedding
- CrossEncoder per reranking finale


7. NOTE
--------

- Il reranking viene applicato sempre, anche ai retriever singoli.
- RRF usa solo il rank, non gli score.
- I chunk devono avere metadata["chunk_id"].


------------------------------------------------------------
MULTI-QUERY RETRIEVAL (multi_queries.py)
------------------------------------------------------------

Questo modulo implementa due strategie di Multi-Query Retrieval
basate sulla decomposizione della query originale in più
sottodomande.

Le due strategie differiscono principalmente per il momento
in cui viene effettuato il reranking.

Funzioni disponibili:

  _doc_key()
  multi_query_retrieval_rerank_once()
  multi_query_retrieval_rerank_each()

------------------------------------------------------------
IDENTIFICAZIONE DOCUMENTI
------------------------------------------------------------

_doc_key(doc)

Genera una chiave univoca per l'identificazione del documento.

La chiave è composta da:

  (chunk_id, source)

e viene utilizzata per:

  - deduplicazione
  - fusione RRF
  - aggregazione dei ranking

Due documenti sono considerati identici solo se coincidono
sia chunk_id che source.

------------------------------------------------------------
MULTI-QUERY: RERANK ONCE
------------------------------------------------------------

multi_query_retrieval_rerank_once()

Implementa una strategia di retrieval con un unico reranking
finale.

Pipeline:

  per ogni sottodomanda:

      retrieval grezzo

  successivamente:

      merge dei risultati
      deduplicazione
      reranking finale
      selezione top-k

Il retrieval viene eseguito mediante:

  retrieve_raw()

Il reranking finale viene eseguito mediante:

  rerank_with_bge()

utilizzando la query originale.

Output:

  list[Document]

contenente al massimo k_final documenti.

Vantaggi:

  - un solo reranking
  - costo computazionale ridotto
  - maggiore coerenza con la query originale

------------------------------------------------------------
MULTI-QUERY: RERANK EACH
------------------------------------------------------------

multi_query_retrieval_rerank_each()

Implementa una strategia di retrieval con reranking
indipendente per ogni sottodomanda.

Pipeline:

  per ogni sottodomanda:

      retrieval
      eventuale fusione RRF
      reranking locale

  successivamente:

      fusione RRF delle classifiche
      selezione top-k

La funzione utilizza:

  unified_retrieval()

che gestisce:

  - retrieval singolo
  - retrieval multiplo
  - fusione RRF
  - reranking

Per ogni sottodomanda viene quindi ottenuta una classifica
già rerankata.

Le classifiche generate vengono fuse mediante Reciprocal
Rank Fusion:

    score += 1 / (rrf_k + rank + 1)

Non viene effettuato alcun reranking finale.

Output:

  list[Document]

contenente al massimo k_final documenti.

Vantaggi:

  - maggiore diversità dei risultati
  - migliore copertura informativa
  - valorizzazione delle singole sottodomande

