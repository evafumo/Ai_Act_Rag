# eval — Valutazione IR, baseline, CSV/grafici e report

Questo pacchetto valuta il retriever della pipeline RAG con metriche IR
standard, lo confronta con una baseline casuale, aggrega i risultati di
più esperimenti in un CSV e genera grafici comparativi.

------------------------------------------------------------
CONTENUTO DEL PACCHETTO
------------------------------------------------------------

eval/
  __init__.py          Esporta le funzioni pubbliche del pacchetto.
  eval_IR.py           Loader del ground truth, metriche, funzioni di
                       valutazione (aggregata e granulare).
  baseline_dummy.py    Baseline random e valori attesi teorici.
  eval_IR_plots.py     CSV combinato dei risultati + grafici comparativi.
  report_generator.py  Report JSON di coverage (missing / false positive).
                       NON importato da __init__.py né da main_rag_new.py.
  README.txt           Questo file.

Fuori dal pacchetto:
  main_rag_new.py      Pipeline completa: chiama le funzioni di eval.
  make_report.py       Report HTML dai file eval_*.json (vedi sotto).

------------------------------------------------------------
GROUND TRUTH (formato JSONL)
------------------------------------------------------------

Aggregato (domanda intera) -> load_ground_truth(path)
  Per riga: "question_text" e "relevant_chunks": [{"chunk_id", "source"}]
  Output:   [{"query", "relevant_ids": [(chunk_id, source), ...]}]

Granulare (sotto-domande) -> load_ground_truth_subq(path)
  Per riga: "subquestion_id", "question_id" (domanda madre),
            "subquestion_text", "relevant_chunks", "coverage" (opzionale)
  Output:   [{"question_id", "parent_id", "query", "relevant_ids"}]
  Le sotto-domande senza relevant_chunks vengono escluse (con messaggio).

Selezione automatica (in main_rag_new.py), nella cartella del PDF:
  aggregato:  *Fixed.jsonl   *Sliding.jsonl   *Hybrid.jsonl
  granulare:  *Fixed_mq.jsonl  *Sliding_mq.jsonl  *Hybrid_mq.jsonl
Il GT dipende quindi dal chunking usato.

CHIAVE DI MATCHING: ogni chunk è identificato dalla coppia
(str(chunk_id), str(source)). Retrieval, RRF, deduplica e valutazione
usano la stessa chiave: un chunk è rilevante solo se coincidono entrambi.

------------------------------------------------------------
METRICHE (eval_IR.py)
------------------------------------------------------------

- precision_at_k   chunk rilevanti nei primi k, divisi per k.
- recall_at_k      chunk rilevanti trovati nei primi k, divisi per il
                   numero totale di rilevanti nel GT.
- average_precision (AP)
                   somma delle precisioni a ogni rilevante trovato, divisa
                   per il numero totale di rilevanti (quelli mai trovati
                   contano 0).
- dcg_at_k / ndcg_at_k
                   rilevanza binaria, sconto log2(posizione + 1);
                   IDCG calcolato sul ranking ideale.
- MAP              media dell'AP sulle query.

Tutte le metriche @k (k = 5, 10, 20 di default) sono medie semplici
sulle query (macro-media).

------------------------------------------------------------
FUNZIONI DI VALUTAZIONE
------------------------------------------------------------

evaluate_retriever(queries, retrieve_fn, k_values, json_path)
  GT aggregato. Salva (se json_path) e restituisce:
    {
      "summary":   {"MAP", "macro_metrics": {"@k": {precision, recall, ndcg}}},
      "per_query": [{"query", "relevant_ids", "retrieved_ids", "AP",
                     "metrics": {"@k": {..., "retrieved_ids_at_k"}}}]
    }

evaluate_retriever_subq(queries, retrieve_fn, k_values, json_path)
  GT granulare. Stesso formato, più:
    "by_parent": {"D1": {"MAP", "macro_metrics"}, ...}   (MAP per domanda madre)
  e in per_query anche "question_id" e "parent_id".

retrieve_fn(query) deve restituire Document con metadata["chunk_id"]
e metadata["source"].

------------------------------------------------------------
BASELINE RANDOM (baseline_dummy.py)
------------------------------------------------------------

Serve da pavimento: se il retriever reale non la batte nettamente, il
task o il ground truth sono sospetti.

build_random_baseline(all_docs, seed=42) -> retrieve_fn(query, k=20)
  Campiona k chunk a caso senza reinserimento, ignorando la query.
  Deterministica per (seed, query).

expected_random_metrics(n_corpus, n_relevant, k_values=(5, 10, 20))
  Valori attesi (distribuzione ipergeometrica):
    Precision@k = n_relevant / n_corpus
    Recall@k    = k / n_corpus
  Sanity check: la baseline misurata deve oscillare attorno a questi valori.

Da riga di comando (main_rag_new.py):
  --compare-baseline        valuta la baseline sullo stesso GT e stampa il confronto
  --baseline-seed N         seed di partenza (default 42)
  --baseline-runs N         numero di seed su cui mediare (default 1; consigliato 10+)
Con più seed il file salvato ha forma {"seeds": [...], "summary": {...}} dove
summary contiene MAP, MAP_std, MAP_per_seed, macro_metrics e macro_metrics_std
(deviazione standard di popolazione).

------------------------------------------------------------
CSV COMBINATO E GRAFICI (eval_IR_plots.py)
------------------------------------------------------------

CSV in formato "long", una riga per (k, metrica):
  colonne: model, dataset, chunking, k, metric, value
  il MAP ha k vuoto (metrica scalare); precision/recall/ndcg hanno k.

Funzioni principali:
  load_eval_json(json_path, model, dataset, chunking)   JSON -> DataFrame
  build_combined_csv(runs, output_csv)                  più JSON -> un CSV
  append_to_combined_csv(json_path, model, dataset, chunking, combined_csv)
      aggiunge un run al CSV. Se esiste già la stessa combinazione
      (model, dataset, chunking, k, metric) la sostituisce: vince l'ultimo run.
      Viene chiamata da main_rag_new.py dopo ogni valutazione, su
      output/eval_results_combined.csv (anche per le baseline random).
  generate_all_plots(df, output_dir, k_for_heatmap)

Grafici prodotti:
  precision_recall_curve.png   curva Precision vs Recall, una linea per modello
  precision_vs_k.png           Precision@k per modello
  recall_vs_k.png              Recall@k per modello
  ndcg_vs_k.png                nDCG@k per modello
  scalar_metrics_bar.png       barre delle metriche scalari (MAP)
  heatmap_k<K>.png             heatmap modello x metrica a k fisso

Uso da riga di comando:

  # da un CSV già costruito dalla pipeline
  python eval/eval_IR_plots.py --csv-in output/eval_results_combined.csv \
      --dataset-filter B_Adult --chunking-filter fixed \
      --output-dir plots/ --k-heatmap 10

  # oppure da singoli JSON (formato json_path:model[:dataset[:chunking]])
  python eval/eval_IR_plots.py \
      --run output/B_Adult/eval_B_Adult_single_bge_fixed.json:single_bge:B_Adult:fixed \
      --run output/B_Adult/eval_B_Adult_bm25_e5_fixed.json:bm25_e5:B_Adult:fixed \
      --output-dir plots/ --csv-out eval_results_combined.csv

Filtri disponibili: --dataset-filter, --chunking-filter, --model-filter.
Le baseline random compaiono nei grafici come "modelli" a sé
(es. single_bge_random_baseline_avg10).
Dipendenze: pandas, matplotlib.

------------------------------------------------------------
REPORT JSON DI COVERAGE (report_generator.py)
------------------------------------------------------------

generate_full_report(gt_agg, gt_subq, eval_agg_baseline,
                     eval_agg_decomposed, eval_subq, output_path)

Scrive un JSON con:
  aggregated_comparison    MAP e metriche macro: senza vs con decomposizione
                           (GT aggregato), con delta_MAP
  parent_coverage          per domanda: chunk rilevanti, recuperati, mancanti
                           (relevant - retrieved) e falsi positivi
                           (retrieved - relevant), nei due scenari
  granular_coverage        stesso confronto per ogni sotto-domanda, con AP e metriche
  parent_granular_summary  per domanda madre: sotto-domande, AP media,
                           chunk mancanti e falsi positivi aggregati

STATO: non è richiamato da main_rag_new.py e, leggendo il codice, si
aspetta strutture diverse da quelle prodotte dai loader attuali:
  - gt_agg: usa "question_id" e "relevant_chunks", ma load_ground_truth()
    restituisce solo "query" e "relevant_ids".
  - eval_*["per_query"] è una lista, mentre qui viene indicizzata con
    l'id della domanda come fosse un dizionario.
  - retrieved_ids letti da JSON sono liste (non tuple): vanno convertiti
    in tuple prima di costruire i set.
Va adattato prima dell'uso.

------------------------------------------------------------
REPORT HTML (make_report.py)
------------------------------------------------------------

  python make_report.py <dataset>      # legge output/<dataset>/

Legge direttamente i file eval_*.json (non il CSV) e produce
output/<dataset>/report_<dataset>.html con: spiegazione della pipeline,
ranking per MAP, dettaglio per k, confronto con baseline random (misurata
e attesa), effetto di decomposizione e chunking, MAP per domanda madre e
query più difficili. Poiché usa i JSON, distingue run con GT aggregato,
granulare e decomposizione, cosa che il CSV combinato non fa.

------------------------------------------------------------
CONVENZIONE DEI FILE DI OUTPUT (output/<dataset>/)
------------------------------------------------------------

  eval_<dataset>_<retriever>_<chunking>[_decomposed_<mode>][_subq].json
  ...<nome sopra>_baseline_random.json            baseline a seed singolo
  ...<nome sopra>_baseline_random_avg<N>.json     baseline mediata su N seed
  chunks_<chunking>.jsonl                         chunk generati
  output/eval_results_combined.csv                CSV combinato (tutti i dataset)

------------------------------------------------------------
NOTE E LIMITI
------------------------------------------------------------

- Con --only-parent la valutazione usa solo le sotto-domande di una madre:
  quei risultati non sono confrontabili con quelli sull'intero GT.
- Con poche query (decine) il MAP è instabile e anche la baseline ha una
  deviazione standard alta.
