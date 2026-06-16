# extractor — Estrazione PDF

Questo modulo estrae testo e tabelle da PDF e produce un JSON compatibile con il chunker.

------------------------------------------------------------
FUNZIONI PRINCIPALI
------------------------------------------------------------

extract_native_text(page)
    Estrae testo digitale tramite PyMuPDF.

extract_tables_pdfplumber(pdf_path, page_index)
    Estrae tabelle e le converte in Markdown.

extract_pdf_for_rag(pdf_path, verbose)
    Combina testo + tabelle in un unico campo "text".
    Restituisce una lista di dizionari, uno per pagina.

save_to_json(pages, output_path)
    Salva l’output in formato JSON.

------------------------------------------------------------
FORMATO OUTPUT
------------------------------------------------------------

{
  "page": 1,
  "text": "contenuto pagina + tabelle",
  "has_tables": true,
  "source": "path.pdf"
}

------------------------------------------------------------
NOTE
------------------------------------------------------------

- Le tabelle vengono marcate con il separatore <!-- TABELLE -->
- Il JSON generato è compatibile con parse_json_to_blocks del chunker

