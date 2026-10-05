GROUND TRUTH VIEWER — VISUALIZZATORE JSONL PER RAG
==================================================

Questo modulo implementa un visualizzatore HTML offline per file JSONL
contenenti ground truth utilizzati nelle pipeline RAG. Il suo scopo è
permettere l’ispezione rapida e leggibile di domande, risposte e chunk
rilevanti, senza dipendenze esterne e senza server.

Il file HTML contiene:
- parsing JSONL (una riga = un oggetto domanda)
- rendering delle domande con risposta e chunk
- sidebar indice per navigazione rapida
- evidenziazione della domanda selezionata
- header fisso con nome del file aperto
- pulsante per apertura file JSONL
- supporto completo offline


1. STRUTTURE DATI
------------------

Il viewer si basa su una struttura JSONL dove ogni riga rappresenta una domanda.

Oggetto domanda:
    question_id (str)
    question_text (str)
    answer_summary (str)
    relevant_chunks (list[Chunk])

Chunk:
    chunk_id (int)
    text (str)

Ogni riga del file JSONL deve essere un JSON valido e indipendente.


2. PARSING JSONL
------------------

La funzione render():
- legge il contenuto del file JSONL
- divide il file in righe tramite split("\n")
- filtra righe vuote
- esegue JSON.parse() su ogni riga
- costruisce una lista di oggetti domanda

Il formato JSONL deve essere:
{"question_id":"Q1", ...}
{"question_id":"Q2", ...}
...


3. RENDERING HTML
------------------

Il viewer costruisce dinamicamente:

A) Sidebar indice
    - elenco delle domande
    - ogni voce punta a un anchor interno (#q_<id>)
    - cliccando una domanda viene evidenziata

B) Header fisso
    - mostra il nome del file JSONL aperto
    - rimane visibile durante lo scroll
    - aggiornato automaticamente quando si apre un file

C) Contenuto principale
    - box con question_id
    - testo della domanda
    - risposta sintetica
    - sezione "Relevant Chunks" con tutti i chunk associati

D) Evidenziazione
    - la domanda selezionata viene evidenziata con bordo luminoso


4. APERTURA FILE JSONL
------------------------

Il viewer funziona completamente offline.

Pulsante "Apri file JSONL":
    - apre un file picker
    - legge il file tramite FileReader()
    - inserisce il contenuto nel viewer
    - aggiorna l’header fisso con il nome del file

Formati accettati:
    - .jsonl
    - .json (se contiene un array di oggetti)


5. FUNZIONI DI SUPPORTO
------------------------

loadLocalJSON():
    legge il file selezionato e aggiorna l’header fisso

updateFileHeader(name):
    mostra il nome del file nell’header fisso

highlight(id):
    evidenzia la domanda selezionata

render():
    esegue il parsing del JSONL e costruisce la pagina


6. UTILIZZO
---------------------

1. Aprire il file ground_truth_viewer.html con doppio clic
2. Cliccare "Apri file JSONL"
3. Selezionare il file dal computer
4. Premere "Render"
5. Navigare tramite la sidebar


7. NOTE IMPORTANTI
-------------------

- Il viewer non può accedere automaticamente alle cartelle del filesystem:
  il file deve essere selezionato manualmente.
- Ogni riga del JSONL deve essere un JSON valido.
- La sidebar viene generata automaticamente.
- Il viewer è completamente offline e non richiede server.

