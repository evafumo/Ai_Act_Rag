"""
Script di valutazione: fa girare process_document (da pipeline.py) su un
sottoinsieme di OmniDocBench e confronta l'output con il ground truth.

FIX rispetto alla versione precedente: il report era inaffidabile per due
motivi.

1. `predicted_text` includeva le tabelle serializzate in Markdown, mentre
   `gt_text` le escludeva (extract_ground_truth_text prende solo
   text_block/title/list). Su ogni pagina con tabelle l'edit distance si
   gonfiava per un motivo che non aveva niente a che fare con la qualità
   dell'OCR: si confrontava testo con testo+tabelle. Ora testo e tabelle
   sono valutati separatamente, ciascuno contro il proprio ground truth.

2. Un singolo numero di edit distance normalizzata sull'intero testo della
   pagina è poco diagnostico: un paragrafo saltato all'inizio disallinea
   tutto il resto e fa sembrare pessima anche una trascrizione quasi
   perfetta. Ora il report include anche il Word Error Rate (WER, più
   leggibile: "quante parole" invece di "quanti caratteri") e, per le
   pagine peggiori, un estratto delle righe più divergenti.

ASSUNZIONE SULLO SCHEMA DI process_document: si assume che ogni entry
restituita abbia un campo "type" con valore "table" per le tabelle e
qualsiasi altro valore (es. "text") per il resto. Se pipeline.py usa un
nome/valore diverso, aggiorna `separate_predicted_content` di conseguenza.

Genera, per ogni pagina:
    - output/eval/<nome>_predicted.txt        → testo estratto (solo testo)
    - output/eval/<nome>_ground_truth.txt     → testo vero, per confronto visivo
    - output/eval/<nome>_tables_predicted.md  → tabelle estratte (se presenti)
    - output/eval/<nome>_tables_ground_truth.md → tabelle vere (se presenti)
    - output/eval/report.json                 → riepilogo con metriche per pagina
"""

from __future__ import annotations  # permette di usare "list[str]", "int|None" ecc. come type hint anche su Python < 3.10

import json               # per leggere il ground truth e scrivere il report.json
import argparse            # per gestire gli argomenti da riga di comando
import difflib             # per generare il diff leggibile tra predetto e ground truth
from pathlib import Path   # per gestire i path in modo cross-platform invece di stringhe grezze

from pipeline import process_document  # la funzione della TUA pipeline OCR, non toccata da questo script


def normalized_edit_distance(a: str, b: str) -> float:
    """
    Levenshtein normalizzato a livello di carattere: 0 = identico, 1 = completamente diverso.

    INPUT:
        a, b (str): testo predetto e testo ground truth

    OUTPUT:
        float
    """
    # Caso banale: se entrambe le stringhe sono vuote, sono identiche per definizione -> distanza 0
    if not a and not b:
        return 0.0

    m, n = len(a), len(b)

    # Matrice di programmazione dinamica (m+1) x (n+1):
    # dp[i][j] = numero minimo di operazioni per trasformare a[:i] in b[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Riga/colonna 0: trasformare una stringa vuota in una di lunghezza i/j
    # richiede i/j inserimenti -> caso base della DP
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Riempimento della matrice: per ogni coppia di caratteri (a[i-1], b[j-1])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # costo 0 se i caratteri coincidono (nessuna sostituzione necessaria), altrimenti 1
            cost = 0 if a[i - 1] == b[j - 1] else 1
            # si prende il minimo tra: cancellazione, inserimento, sostituzione (o match)
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # cancellazione di un carattere da a
                dp[i][j - 1] + 1,       # inserimento di un carattere per arrivare a b
                dp[i - 1][j - 1] + cost  # sostituzione (o match se cost=0)
            )

    # Normalizzazione: divide per la lunghezza massima, così il risultato è comparabile
    # tra testi di lunghezze diverse (0 = identici, 1 = completamente diversi)
    return dp[m][n] / max(m, n, 1)  # max(..., 1) evita la divisione per zero


def word_error_rate(a: str, b: str) -> float:
    """
    Come normalized_edit_distance ma a livello di parola: più leggibile
    per capire "quante parole ha sbagliato/perso" la pipeline, invece di
    "quanti caratteri". Utile in aggiunta al CER, non al suo posto: i due
    indicano problemi diversi (CER alto + WER basso = tipico di errori di
    riconoscimento carattere singolo; WER alto + CER simile = tipico di
    parole/righe intere saltate).

    INPUT:
        a, b (str): testo predetto e testo ground truth

    OUTPUT:
        float
    """
    # Tokenizzazione semplice per spazi: split() converte le stringhe in liste di parole
    wa, wb = a.split(), b.split()

    # Stessa logica di normalized_edit_distance, ma applicata a liste di parole
    # invece che a stringhe di caratteri (l'algoritmo di Levenshtein è identico,
    # cambia solo l'unità che viene confrontata elemento per elemento)
    if not wa and not wb:
        return 0.0

    m, n = len(wa), len(wb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # confronto tra parole intere invece che singoli caratteri
            cost = 0 if wa[i - 1] == wb[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n] / max(m, n, 1)


def worst_diff_snippet(predicted: str, ground_truth: str, max_lines: int = 8) -> str:
    """
    Estrae le righe più divergenti tra predetto e ground truth, in formato
    diff leggibile, per capire A COLPO D'OCCHIO cosa non va senza dover
    aprire i due file affiancati manualmente.

    INPUT:
        predicted, ground_truth (str)
        max_lines (int): numero massimo di righe di diff da includere

    OUTPUT:
        str: blocco diff testuale (formato unified diff semplificato)
    """
    # splitlines() spezza il testo in righe (senza includere i caratteri di a-capo)
    pred_lines = predicted.splitlines()
    gt_lines = ground_truth.splitlines()

    # unified_diff confronta due sequenze di righe e produce un diff in stile "git diff":
    # righe precedute da "-" sono nel ground truth ma non nel predetto (mancanti/diverse),
    # righe precedute da "+" sono nel predetto ma non nel ground truth (aggiunte/errate).
    # n=0 disattiva le righe di contesto: vogliamo solo le righe che cambiano.
    diff = list(difflib.unified_diff(
        gt_lines, pred_lines,
        fromfile="ground_truth", tofile="predicted",
        lineterm="", n=0
    ))

    # Filtra via l'header del diff (righe "+++"/"---") e le righe "@@" di posizione,
    # tenendo solo le righe di modifica effettiva, poi tronca a max_lines
    changed = [l for l in diff if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    return "\n".join(changed[:max_lines])


def extract_ground_truth_text(entry: dict) -> str:
    """
    Ricostruisce il testo di una pagina di ground truth
    OmniDocBench, concatena tutti i blocchi di tipo testuale
    nell'ordine in cui compaiono nel JSON.

    INPUT:
        entry (dict): una entry del OmniDocBench.json

    OUTPUT:
        str
    """
    # Solo questi tre tipi di blocco vengono considerati "testo" (le tabelle
    # hanno un category_type diverso e vengono gestite a parte)
    text_categories = {"text_block", "title", "list"}

    parts = []
    # "layout_dets" è la lista dei blocchi (elementi di layout) della pagina,
    # nell'ordine originale in cui sono stati annotati nel dataset
    for block in entry.get("layout_dets", []):
        if block.get("category_type") in text_categories:
            txt = block.get("text", "")
            if txt.strip():  # scarta blocchi vuoti o fatti solo di spazi
                parts.append(txt.strip())

    # Unisce i blocchi con doppio a-capo, per simulare la separazione tra paragrafi
    return "\n\n".join(parts)


def extract_ground_truth_tables(entry: dict) -> list[str]:
    """
    Estrae le tabelle di ground truth in formato HTML (se presente).

    INPUT:
        entry (dict)

    OUTPUT:
        list[str]: lista di tabelle HTML
    """
    tables = []
    for block in entry.get("layout_dets", []):
        # A differenza del testo, le tabelle hanno category_type == "table"
        # e il loro contenuto è nel campo "html" (non "text")
        if block.get("category_type") == "table":
            html = block.get("html", "")
            if html.strip():
                tables.append(html.strip())
    return tables


def separate_predicted_content(entries: list[dict]) -> tuple[str, list[str]]:
    """
    Separa il testo dalle tabelle nell'output della pipeline, per garantire
    il confronto con il ground truth

    FORMATO REALE (da pipeline.py): ogni entry è una pagina,
    {"page": int, "text": str}, dove "text" contiene il testo e, se ci
    sono tabelle, il marcatore "<!-- TABELLE -->" seguito dal markdown
    delle tabelle, la funzione inverte la concatenazione

    INPUT:
        entries (list[dict]): output di process_document

    OUTPUT:
        tuple[str, list[str]]: (testo concatenato di tutte le pagine,
                                 lista di blocchi tabella markdown, uno
                                 per pagina che ne contiene)
    """
    # Marcatore usato da pipeline.py per separare, dentro allo stesso campo "text",
    # il testo vero e proprio dalle tabelle serializzate in markdown
    MARKER = "<!-- TABELLE -->"

    text_parts = []
    table_parts = []

    for e in entries:
        page_text = e.get("text", "")

        if MARKER in page_text:
            # split(MARKER, 1) spezza in al massimo 2 pezzi: tutto ciò che precede
            # il marcatore è testo, tutto ciò che segue sono le tabelle
            text_section, tables_section = page_text.split(MARKER, 1)
            text_parts.append(text_section.strip())
            if tables_section.strip():
                table_parts.append(tables_section.strip())
        else:
            # nessuna tabella in questa pagina: tutto il campo è testo
            text_parts.append(page_text.strip())

    # Il testo di tutte le pagine viene unito in un unico blocco (per il confronto
    # CER/WER complessivo), mentre le tabelle restano una lista separata per pagina
    return "\n\n".join(text_parts), table_parts


def run_evaluation(images_dir: str, ground_truth_json: str, out_dir: str, limit: int | None = None, verbose: bool = True):
    """
    Esegue la pipeline su ogni immagine del sottoinsieme e confronta
    il risultato con il ground truth corrispondente.

    INPUT:
        images_dir (str): cartella con le immagini selezionate
                           (es. test_data/omnidocbench_subset)
        ground_truth_json (str): path al JSON filtrato di ground truth
                                  (es. ground_truth_subset.json)
        out_dir (str): cartella dove salvare i risultati leggibili
        limit (int|None): elabora solo le prime N immagini (utile per test rapidi)
        verbose (bool)
    """
    images_path = Path(images_dir)
    out_path = Path(out_dir)
    # crea la cartella di output se non esiste già (parents=True crea anche le sottocartelle mancanti,
    # exist_ok=True evita errori se la cartella c'è già)
    out_path.mkdir(parents=True, exist_ok=True)

    # Carica l'intero JSON di ground truth in memoria (lista di entry, una per pagina/immagine)
    with open(ground_truth_json, encoding="utf-8") as f:
        gt_entries = json.load(f)

    # Costruisce un dizionario nome_file_immagine -> entry di ground truth,
    # così per ogni immagine processata si trova subito il ground truth corrispondente
    # senza dover scorrere la lista ogni volta (lookup O(1) invece di O(n))
    gt_by_image = {}
    for entry in gt_entries:
        img_name = Path(entry["page_info"]["image_path"]).name
        gt_by_image[img_name] = entry

    # Elenca tutte le immagini .jpg e .png nella cartella, ordinate alfabeticamente
    # (sorted() garantisce un ordine deterministico tra esecuzioni diverse)
    image_files = sorted(images_path.glob("*.jpg")) + sorted(images_path.glob("*.png"))

    # Se è stato passato un limite, elabora solo le prime N immagini (utile per test rapidi
    # senza dover girare su tutto il dataset)
    if limit:
        image_files = image_files[:limit]

    report = []  # qui si accumula una riga di metriche per ogni pagina elaborata

    for img_file in image_files:
        # Se per questa immagine non esiste un ground truth corrispondente, non ha senso valutarla
        if img_file.name not in gt_by_image:
            if verbose:
                print(f"  [skip] nessun ground truth per {img_file.name}")
            continue

        if verbose:
            print(f"\n[eval] {img_file.name}")

        # --- Esecuzione della TUA pipeline, invariata ---
        # Qui avviene la vera chiamata al sistema da valutare: OCR + parsing del layout
        entries = process_document(str(img_file), verbose=verbose)
        # Separa il testo dalle tabelle nell'output predetto (vedi separate_predicted_content)
        predicted_text, predicted_tables = separate_predicted_content(entries)

        # --- Ground truth corrispondente ---
        gt_entry = gt_by_image[img_file.name]
        gt_text = extract_ground_truth_text(gt_entry)
        gt_tables = extract_ground_truth_tables(gt_entry)

        # --- Metriche testo: CER (char) + WER (word), ora confrontabili 1:1 ---
        # entrambe confrontano SOLO testo con SOLO testo (niente più contaminazione da tabelle)
        cer = normalized_edit_distance(predicted_text, gt_text)
        wer = word_error_rate(predicted_text, gt_text)

        # --- Metriche tabelle: conteggio + edit distance media sulle coppie ---
        # controlla se il numero di tabelle rilevate coincide con quello reale
        table_count_match = len(predicted_tables) == len(gt_tables)

        table_cer = None
        if gt_tables and predicted_tables:
            # confronta le tabelle a coppie, nell'ordine in cui compaiono (assunzione:
            # l'ordine delle tabelle predette rispecchia quello del ground truth).
            # Se il numero non combacia, si limita al minimo tra i due conteggi.
            n_pairs = min(len(gt_tables), len(predicted_tables))
            table_cer = sum(
                normalized_edit_distance(predicted_tables[i], gt_tables[i])
                for i in range(n_pairs)
            ) / n_pairs  # media della edit distance sulle coppie confrontate

        # --- Salva testo affiancato ---
        stem = img_file.stem  # nome del file senza estensione, usato come prefisso per gli output
        (out_path / f"{stem}_predicted.txt").write_text(predicted_text, encoding="utf-8")
        (out_path / f"{stem}_ground_truth.txt").write_text(gt_text, encoding="utf-8")

        # --- Salva tabelle affiancate, se presenti ---
        # i file vengono creati solo se ci sono effettivamente tabelle, per non
        # riempire la cartella di file vuoti
        if predicted_tables:
            (out_path / f"{stem}_tables_predicted.md").write_text(
                "\n\n---\n\n".join(predicted_tables), encoding="utf-8")
        if gt_tables:
            (out_path / f"{stem}_tables_ground_truth.md").write_text(
                "\n\n---\n\n".join(gt_tables), encoding="utf-8")

        if verbose:
            print(f"    CER (char): {cer:.3f}  |  WER (word): {wer:.3f}")
            print(f"    tabelle: {len(predicted_tables)} predette vs {len(gt_tables)} ground truth"
                  + (f"  |  table CER: {table_cer:.3f}" if table_cer is not None else ""))

        # Aggiunge la riga di metriche per questa pagina al report complessivo
        report.append({
            "image": img_file.name,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "predicted_chars": len(predicted_text),
            "ground_truth_chars": len(gt_text),
            "predicted_table_count": len(predicted_tables),
            "ground_truth_table_count": len(gt_tables),
            "table_count_match": table_count_match,
            "table_cer": round(table_cer, 4) if table_cer is not None else None,
            "worst_diff": worst_diff_snippet(predicted_text, gt_text),
        })

    # --- Report riassuntivo, ordinato dal caso peggiore al migliore (per WER: più leggibile del CER) ---
    # ordina in place la lista di dict per WER decrescente, così le pagine più problematiche
    # sono in cima al report.json
    report.sort(key=lambda r: r["wer"], reverse=True)

    report_path = out_path / "report.json"
    # ensure_ascii=False mantiene i caratteri accentati/non-ASCII leggibili nel file invece
    # di convertirli in sequenze di escape unicode; indent=2 rende il JSON leggibile
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print(f"\n[report] salvato in {report_path}")
        if report:  # evita divisioni per zero se non è stata valutata nessuna pagina
            avg_cer = sum(r["cer"] for r in report) / len(report)
            avg_wer = sum(r["wer"] for r in report) / len(report)
            # conta quante pagine hanno un numero di tabelle rilevate sbagliato
            mismatched_tables = sum(1 for r in report if not r["table_count_match"])
            print(f"[report] CER medio: {avg_cer:.3f}  |  WER medio: {avg_wer:.3f}")
            print(f"[report] pagine con numero di tabelle sbagliato: {mismatched_tables}/{len(report)}")
            # report[0] è la pagina peggiore, perché la lista è stata ordinata per WER decrescente
            print(f"[report] pagina peggiore (WER): {report[0]['image']} ({report[0]['wer']})")
            print(f"[report] diff pagina peggiore:\n{report[0]['worst_diff']}")


if __name__ == "__main__":
    # Configurazione della CLI: permette di lanciare lo script da terminale
    # personalizzando cartelle di input/output e numero di immagini da valutare
    parser = argparse.ArgumentParser(description="Valuta la pipeline su un sottoinsieme di OmniDocBench")
    parser.add_argument("--images-dir", default="../test_data/omnidocbench_subset")
    parser.add_argument("--ground-truth", default="../test_data/omnidocbench_subset/ground_truth_subset.json")
    parser.add_argument("--output-dir", default="../output/eval")
    parser.add_argument("--limit", type=int, default=None, help="Elabora solo le prime N immagini")
    parser.add_argument("--quiet", action="store_true")  # se presente, disattiva i print di verbose

    args = parser.parse_args()
    run_evaluation(
        images_dir=args.images_dir,
        ground_truth_json=args.ground_truth,
        out_dir=args.output_dir,
        limit=args.limit,
        verbose=not args.quiet,  # --quiet inverte il flag verbose
    )
