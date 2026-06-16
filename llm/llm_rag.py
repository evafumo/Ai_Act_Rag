from llama_cpp import Llama

_llm = None

MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
# scaricabile da: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF

def load_qwen():
    """
        Caricare il modello Qwen2.5 0.5B in formato GGUF 
        tramite llama.cpp, usando lazy loading + caching.

    INPUT:
        Nessuno.

    OUTPUT:
        Llama:
            Istanza del modello già caricata.

    NOTE:
        - Usa variabile globale _llm come cache.
        - n_ctx=2048 definisce la finestra di contesto.
        (ho usato Claude per questa funzione, avevo poca RAM spero abbia senso)
    """
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
    return _llm


def generate_answer(query: str, context: str) -> str:
    """
    Genera una risposta usando solo il contesto fornito

    INPUT:
        query (str):
            Domanda dell’utente.
        context (str):
            Testo recuperato dal retriever.

    OUTPUT:
        str:
            Risposta generata dal modello.

    NOTE:
        - Se la risposta non è nel contesto, il prompt chiede di dirlo esplicitamente.
        - stop=["###"] evita che il modello continui oltre la risposta.
    """
    llm = load_qwen()
    prompt = f"""Usa SOLO il seguente contesto per rispondere alla domanda.
          Se il contesto non contiene la risposta, dì che non è presente.

          ### CONTEXT:
            {context}

          ### DOMANDA:
            {query}
    
          ### RISPOSTA:
         """
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.0,
        stop=["###"],
    )
    return output["choices"][0]["text"].strip()

