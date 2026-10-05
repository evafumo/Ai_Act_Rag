# decomposition/query_decomposition.py
# ============================================================
#  DECOMPOSIZIONE DELLA QUERY UTENTE IN SOTTO-DOMANDE
# ============================================================

import json
import re
from llm.llm_rag import load_qwen  


def decompose_question(query: str) -> list[str]:
    """
    Scompone una domanda complessa in sotto-domande atomiche usando l'LLM.

    INPUT:
        query (str):
            Domanda originale dell'utente.

    OUTPUT:
        list[str]:
            Lista di sotto-domande atomiche. Se la domanda è già semplice,
            o se il parsing fallisce, ritorna [query] (fallback sicuro).

    NOTE:
        - Usa lo stesso modello Qwen già caricato (load_qwen()).
    
    """
    llm = load_qwen()

    prompt = f"""Scomponi la seguente domanda in sotto-domande atomiche, una per ogni
concetto distinto richiesto. Se la domanda e' gia' atomica (chiede una sola cosa),
restituiscila cosi' com'e'.

Rispondi SOLO con un JSON valido in questo formato, nessun altro testo:
{{"sub_questions": ["domanda 1", "domanda 2"]}}

Domanda: {query}

### JSON:
"""

    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.0,
        stop=["###"],
    )
    raw_text = output["choices"][0]["text"].strip()

    return _parse_decomposition_output(raw_text, fallback_query=query)


def _parse_decomposition_output(raw_text: str, fallback_query: str) -> list[str]:
    """
    Estrae e valida la lista di sotto-domande dal testo grezzo generato dall'LLM.

    INPUT:
        raw_text (str):
            Testo grezzo restituito dal modello.
        fallback_query (str):
            Domanda originale, usata come fallback in caso di parsing fallito.

    OUTPUT:
        list[str]:
            Lista di sotto-domande valide, oppure [fallback_query] in caso di errore.

    NOTE:
        - Isolata dalla funzione principale per essere testabile separatamente
          (utile per gli unit test sul parsing, senza dover invocare l'LLM).
    """
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return [fallback_query]

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [fallback_query]

    sub_questions = parsed.get("sub_questions", [])

    # Validazione: deve essere una lista non vuota di stringhe non vuote
    if not isinstance(sub_questions, list) or not sub_questions:
        return [fallback_query]

    cleaned = [q.strip() for q in sub_questions if isinstance(q, str) and q.strip()]
    if not cleaned:
        return [fallback_query]

    return cleaned
