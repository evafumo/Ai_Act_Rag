# llm — Generazione Risposta (Qwen 0.5B GGUF)

Questo modulo gestisce il caricamento del modello LLM e la generazione delle risposte.

------------------------------------------------------------
FUNZIONI
------------------------------------------------------------

load_qwen()
    Carica Qwen 0.5B Instruct in formato GGUF tramite llama.cpp.
    Usa caching per evitare ricaricamenti.

generate_answer(query, context)
    Genera una risposta usando SOLO il contesto recuperato.

------------------------------------------------------------
NOTE
------------------------------------------------------------

- Il prompt forza il modello a non inventare informazioni.
- stop=["###"] impedisce output indesiderati.

