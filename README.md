# RAG Pipeline

Questo progetto implementa una pipeline completa di Retrieval-Augmented Generation (RAG) per documenti PDF. Il sistema estrae testo e tabelle, genera chunk ottimizzati, costruisce retriever densi e sparsi, applica reranking e infine genera risposte tramite un modello LLM locale (Qwen 0.5B GGUF).

La pipeline è modulare e ogni componente è organizzato in una cartella dedicata.

------------------------------------------------------------
STRUTTURA DEL PROGETTO
------------------------------------------------------------

Ai/
│
├── extractor/        Estrazione testo e tabelle dai PDF
├── chunker/          Strategie di chunking (fixed, sliding, hybrid)
├── retrievers/       Retriever densi, sparsi e combinati
├── rerankers/        Reranking con CrossEncoder
├── eval/             Metriche IR ed evaluation
├── llm/              Modello Qwen 0.5B via llama.cpp
├── datasets/         PDF e ground truth
└── main_rag.py       Pipeline completa

------------------------------------------------------------
1. ESTRAZIONE PDF → JSON
------------------------------------------------------------

La cartella extractor contiene il modulo che:
- Estrae testo digitale con PyMuPDF
- Estrae tabelle con pdfplumber
- Converte le tabelle in Markdown
- Combina testo + tabelle in un unico campo "text"
- Salva tutto in formato JSON compatibile con il chunker

------------------------------------------------------------
2. CHUNKING
------------------------------------------------------------

La cartella chunker implementa tre strategie:

- Fixed: chunk a lunghezza fissa con overlap
- Sliding Window: basato sulle frasi (spaCy)
- Hybrid: testo fixed + tabelle come chunk autonomi con contesto

Il risultato è un file JSONL con un chunk per riga.

------------------------------------------------------------
3. RETRIEVER
------------------------------------------------------------

La cartella retrievers contiene:
- Retriever densi (BGE, E5, BGEM3)
- Retriever sparsi (BM25)
- Retriever combinati (RRF)

La funzione unified_retrieval permette di unire più retriever.

------------------------------------------------------------
4. RERANKING
------------------------------------------------------------

La cartella rerankers contiene un CrossEncoder (MS MARCO MiniLM) che ricalcola la rilevanza query-documento e riordina i risultati.

------------------------------------------------------------
5. EVALUATION IR
------------------------------------------------------------

La cartella eval implementa:
- Precision@k
- Recall@k
- nDCG@k
- MAP

La funzione evaluate_retriever produce:
- riepilogo globale
- risultati per query
- file JSON opzionale

------------------------------------------------------------
6. GENERAZIONE RISPOSTA (LLM)
------------------------------------------------------------

La cartella llm contiene:
- Caricamento lazy del modello Qwen 0.5B GGUF via llama.cpp
- Funzione generate_answer che usa SOLO il contesto recuperato

------------------------------------------------------------
7. PIPELINE COMPLETA
------------------------------------------------------------

main_rag.py esegue:

1. Estrazione PDF - JSON
2. Chunking
3. Costruzione retriever
4. Evaluation IR (opzionale)
5. Answer generation 

Esempio:

python3 main_rag.py datasets/B_Adults/GT/B_Adults.pdf \
    --chunking hybrid \
    --retriever single_bge \
    --evaluate \
    --question "What data gaps and shortcomings are there in the data?"

------------------------------------------------------------
                 RAG PIPELINE — ARCHITETTURA
------------------------------------------------------------

                +-----------------------------+
                |         PDF Input           |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |      1. PDF EXTRACTOR       |
                |  - PyMuPDF: testo           |
                |  - pdfplumber: tabelle      |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |     JSON (pagine grezze)    |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |           2. CHUNKER        |
                |  - Fixed                    |
                |  - Sliding Window           |
                |  - Hybrid                   |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |        JSONL (chunks)       |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |        3. RETRIEVERS        |
                |  - Dense (BGE, E5, BGEM3)   |
                |  - Sparse (BM25)            |
                |  - RRF Fusion               |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |        4. RERANKER          |
                |    CrossEncoder (MiniLM)    |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |  Top-k Documenti Rilevanti  |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |        5. LLM (Qwen)        |
                |  - llama.cpp GGUF           |
                |  - Usa SOLO il contesto     |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |       RISPOSTA FINALE       |
                +-----------------------------+



------------------------------------------------------------
NOTE SULLA DOCUMENTAZIONE
------------------------------------------------------------

Questo README fornisce una panoramica completa.
Ogni cartella contiene un README dedicato che spiega teoria e implementazione.

