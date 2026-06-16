# eval — Metriche IR ed Evaluation

Questo modulo valuta il retriever usando metriche standard.

------------------------------------------------------------
METRICHE IMPLEMENTATE
------------------------------------------------------------

- Precision@k
- Recall@k
- nDCG@k
- MAP (Mean Average Precision)

------------------------------------------------------------
FUNZIONE PRINCIPALE
------------------------------------------------------------

evaluate_retriever(queries, retrieve_fn, k_values, json_path)

Produce:
- riepilogo globale
- risultati per query
- file JSON opzionale

------------------------------------------------------------
NOTE
------------------------------------------------------------

- Il ground truth deve essere in formato JSONL.
- retrieve_fn deve restituire Document con metadata["chunk_id"].

