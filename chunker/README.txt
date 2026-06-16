CHUNKER — SISTEMA DI CHUNKING PER RAG
=====================================

Questo modulo implementa l’intero sistema di chunking utilizzato nella pipeline RAG.
Il suo scopo è trasformare un documento PDF (o un JSON estratto) in una sequenza di
chunk coerenti, informativi e adatti all’indicizzazione in un retriever (FAISS, BM25, ecc.).

Il file contiene:
- parsing PDF - blocchi (testo + tabelle)
- parsing JSON - blocchi
- tre strategie di chunking: FIXED, SLIDING WINDOW, HYBRID
- funzioni di salvataggio in TXT e JSONL
- CLI per eseguire il chunking da terminale


1. STRUTTURE DATI
------------------

Il modulo definisce due dataclass fondamentali:

Block:
    Rappresenta un’unità elementare estratta dal documento:
        - un paragrafo di testo
        - una tabella convertita in Markdown
    Attributi:
        content (str)
        kind ("text" | "table")
        page (int)

Chunk:
    Rappresenta un chunk finale pronto per l’indicizzazione.
    Attributi:
        text (str)
        chunk_id (int)
        strategy (str)
        overlap (int | None)
        source_pages (list[int])
        contains_table (bool)
        chars (int) - calcolato automaticamente


2. PARSING PDF - BLOCKS
------------------------

La funzione parse_pdf_to_blocks():
- apre il PDF con pdfplumber
- identifica le tabelle tramite page.find_tables()
- rimuove le tabelle dal testo per evitare duplicazioni
- divide il testo in paragrafi usando doppio newline
- converte ogni tabella in Markdown
- restituisce una lista ordinata di Block

Ogni Block contiene:
    - contenuto
    - tipo (text/table)
    - numero pagina


3. PARSING JSON - BLOCKS
-------------------------

La funzione parse_json_to_blocks():
- legge il JSON generato dal PDF extractor
- separa testo e tabelle tramite il marker <!-- TABELLE -->
- divide il testo in paragrafi
- crea Block coerenti con il formato PDF


4. STRATEGIE DI CHUNKING
-------------------------

Il modulo implementa tre strategie principali.

------------------------------------------------------------
A) FIXED CHUNKING (chunking a finestra fissa)
------------------------------------------------------------

Funzione: fixed_chunking()

Caratteristiche:
- divide il testo in segmenti di lunghezza fissa (default 512 caratteri)
- usa overlap (default 64 caratteri) per mantenere continuità semantica
- le tabelle diventano chunk autonomi
- il buffer accumula testo finché non supera chunk_size

Vantaggi:
- semplice
- robusto
- ideale per modelli densi (BGE, E5)

Limiti:
- non rispetta i confini semantici
- può spezzare frasi o concetti

------------------------------------------------------------
B) SLIDING WINDOW (basato sulle frasi)
------------------------------------------------------------

Funzione: sliding_window_chunking()

Caratteristiche:
- usa spaCy per segmentare il testo in frasi
- crea chunk composti da N frasi (default 6)
- overlap basato su frasi (default 2)
- le tabelle restano chunk autonomi

Vantaggi:
- chunk più coerenti semanticamente
- ideale per documenti narrativi o legali

Limiti:
- più lento (richiede spaCy)
- chunk di lunghezza variabile

------------------------------------------------------------
C) HYBRID CHUNKING (testo fixed + tabelle con contesto)
------------------------------------------------------------

Funzione: hybrid_chunking()

Caratteristiche:
- il testo viene chunkato come fixed (chunk_size 800, overlap 100)
- le tabelle vengono arricchite con un contesto precedente
  (default: ultimi 200 caratteri)
- ogni tabella diventa un chunk autonomo con contesto

Vantaggi:
- ottimo per documenti tecnici con molte tabelle
- preserva la struttura logica
- evita che le tabelle perdano significato

Limiti:
- leggermente più complesso
- richiede tuning dei parametri

------------------------------------------------------------
D) FUNZIONI DI SUPPORTO
------------------------------------------------------------

_char_split():
    Divide una stringa in segmenti con overlap.

_load_spacy():
    Carica un modello spaCy italiano o inglese.


5. SALVATAGGIO DEI CHUNK
-------------------------

save_chunks_txt():
    Salva i chunk in formato leggibile, con header e metadati.

save_chunks_jsonl():
    Salva i chunk in formato JSONL, uno per riga.
    Questo è il formato richiesto dai retriever.


6. ENTRY POINT 
---------------------

Il file può essere eseguito da terminale:

    python chunker.py documento.pdf --strategy fixed

Parametri principali:
    --strategy (fixed, sliding, hybrid, all)
    --chunk-size
    --overlap
    --sent-per-chunk
    --sent-overlap
    --output-dir
    --save-blocks

Output:
    chunks_<strategy>.txt
    chunks_<strategy>.jsonl
    blocks.json (opzionale)


7. NOTE IMPORTANTI
-------------------

- Le tabelle vengono sempre trattate come chunk autonomi.
- HYBRID è la strategia consigliata per documenti tecnici.
- FIXED è la più veloce e stabile.
- SLIDING è semantica ma anche la più lenta.
- Il formato JSONL prodotto è compatibile con FAISS, BM25 e RRF.

