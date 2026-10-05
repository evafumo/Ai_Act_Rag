

OCR PIPELINE — SISTEMA DI ESTRAZIONE TESTO DA PDF SCANNERIZZATI  
==================================================================

Questo modulo implementa l’intera pipeline OCR utilizzata nella fase iniziale della pipeline RAG.  
Il suo scopo è trasformare un PDF scannerizzato (o un’immagine) in un JSON strutturato contenente testo e tabelle, pronto per il chunker.

Il sistema include:

- rasterizzazione PDF → immagini (PyMuPDF)
- preprocessing immagini (deskew, denoise, upscale, binarizzazione adattiva)
- OCR con RapidOCR (testo + confidenza)
- separazione testo/tabelle
- salvataggio JSON compatibile con il chunker
- valutazione OCR (CER, WER, diff, confronto testo/tabelle)

------------------------------------------------------------------

1. PREPROCESSING IMMAGINI  
--------------------------

Il modulo `image_cleaning.py` applica una serie di trasformazioni alle immagini scannerizzate per migliorare la qualità del riconoscimento OCR.

### deskew(img)
Corregge l’inclinazione della pagina:
- converte in grayscale  
- binarizza con Otsu  
- stima l’angolo con `cv2.minAreaRect`  
- ruota l’immagine per raddrizzarla  

### denoise(img)
Riduce il rumore tipico degli scanner:
- usa `cv2.fastNlMeansDenoisingColored`  
- rimuove grana, artefatti e rumore “sale e pepe”

### adaptive_binarize(img)
Binarizzazione adattiva:
- utile solo per OCR classici (Tesseract legacy)  
- **non consigliata** per RapidOCR/PaddleOCR (perdita di dettagli)

### upscale_if_needed(img)
Ingrandisce immagini troppo piccole:
- se altezza < 1500px → upscale bicubico  
- migliora la leggibilità dei caratteri

### clean_pipeline(img, binarize=False)
Pipeline completa:
```
upscale → denoise → deskew → (binarize opzionale)
```

Note:
- il default è `binarize=False`  
- RapidOCR è addestrato su immagini grayscale/colore  
- binarizzare può peggiorare il riconoscimento

------------------------------------------------------------------

2. OCR CON RAPIDOCR  
---------------------

Il modulo `ocr_table_engine.py` gestisce l’interazione con RapidOCR.

### get_ocr_engine()
Crea un’istanza RapidOCR **singleton**:
- evita ricaricamenti del modello  
- riduce l’uso di RAM  
- migliora le prestazioni

### extract_page_content(img)
Esegue OCR su una pagina pulita:
- `result.txts` → testi riconosciuti  
- `result.scores` → confidenze  
- `result.boxes` → bounding box  

Restituisce:
```
{
    "text_blocks": [...],
    "tables_md": [],        # tabelle non ancora implementate
    "low_confidence": bool
}
```

### html_table_to_markdown(html)
Converte una tabella HTML in Markdown.  
Serve per quando verrà integrato `rapid_table`.

### debug_dump_raw_result(img)
Stampa output grezzo RapidOCR (testo, confidenze, box).  
Utile per debug.

------------------------------------------------------------------

3. RASTERIZZAZIONE PDF  
------------------------

Il modulo `pdf_to_images.py` converte ogni pagina del PDF in un’immagine.

### pdf_to_images(pdf_path, dpi=250)
- usa PyMuPDF (`fitz`)  
- rasterizza ogni pagina  
- converte in array NumPy BGR  
- restituisce una lista di immagini

### load_single_image(path)
Carica una singola immagine `.jpg/.png`.

------------------------------------------------------------------

4. PIPELINE COMPLETA  
----------------------------

Il modulo `pipeline.py` unisce tutti gli step in un’unica pipeline.

### process_document(input_path)
Per ogni pagina:
1. rasterizza PDF → immagini  
2. pulisce immagine (`clean_pipeline`)  
3. esegue OCR (`extract_page_content`)  
4. costruisce il testo finale:

```
<paragrafi>

<!-- TABELLE -->

<tabelle md>
```

Restituisce:
```
[
  {"page": 1, "text": "..."},
  {"page": 2, "text": "..."},
  ...
]
```

### save_json(entries, out_path)
Salva il JSON nel formato richiesto dal chunker.

------------------------------------------------------------------

5. VALUTAZIONE OCR  
--------------------

Il modulo di valutazione confronta l’output della pipeline con il ground truth OmniDocBench.

### 5.1 Metriche OCR (con formule)

#### **CER — Character Error Rate**
Misura gli errori a livello di carattere.

Formula:

```
CER(a, b) = Levenshtein(a, b) / max(len(a), len(b), 1)
```

dove:
- `a` = testo predetto  
- `b` = testo ground truth  
- `Levenshtein(a,b)` = numero minimo di inserzioni, cancellazioni, sostituzioni  

---

#### **WER — Word Error Rate**
Misura gli errori a livello di parola.

Formula:

```
WER(a, b) = Levenshtein(words(a), words(b)) / max(len(words(a)), len(words(b)), 1)
```

dove:
- `words(a)` = lista di parole del testo predetto  
- `words(b)` = lista di parole del ground truth  

---

#### **TABLE CER — Character Error Rate sulle tabelle**
Confronta solo le tabelle, una per una.

Formula:

```
TableCER = (1/N) * Σ CER(T_pred[i], T_gt[i])
```

dove:
- `T_pred[i]` = tabella predetta i-esima  
- `T_gt[i]` = tabella ground truth i-esima  
- `N = min(num_predette, num_ground_truth)`  

---

### 5.2 Funzioni di valutazione

#### normalized_edit_distance(a, b)
CER normalizzato.

#### word_error_rate(a, b)
WER normalizzato.

#### worst_diff_snippet(predicted, ground_truth)
Mostra le righe più divergenti in formato diff.

#### extract_ground_truth_text(entry)
Estrae solo i blocchi testuali del ground truth.

#### extract_ground_truth_tables(entry)
Estrae le tabelle HTML del ground truth.

#### separate_predicted_content(entries)
Divide testo e tabelle predette usando il marcatore:
```
<!-- TABELLE -->
```

### run_evaluation(...)
Per ogni immagine:
- esegue la pipeline OCR  
- confronta testo e tabelle separatamente  
- calcola CER, WER, table CER  
- salva file affiancati (predicted vs ground truth)  
- genera `report.json` ordinato per WER decrescente  

Output:
- `<nome>_predicted.txt`  
- `<nome>_ground_truth.txt`  
- `<nome>_tables_predicted.md`  
- `<nome>_tables_ground_truth.md`  
- `report.json`

------------------------------------------------------------------

6. ENTRY POINT  
----------------

Esecuzione da terminale:

```
python pipeline.py documento.pdf --output output/extracted.json
```

Parametri:
- `input` → PDF o immagine  
- `output` → JSON estratto  

------------------------------------------------------------------

7. NOTE IMPORTANTI  
--------------------

- RapidOCR è leggero e non richiede torch/faiss/modelscope.  
- La pipeline è ottimizzata per PDF scannerizzati.  
- Le tabelle verranno integrate con `rapid_layout` + `rapid_table`.  
- Il JSON prodotto è compatibile con il chunker RAG.  
- La valutazione separa testo e tabelle per evitare metriche fuorvianti.
- la pipeline è stat testata su 50 immagini in inglese di OmniDocBench


