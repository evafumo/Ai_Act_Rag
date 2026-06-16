import json
import csv
from typing import List, Dict, Callable
from rouge_score import rouge_scorer


# ---------------------------------------------------------
# METRICHE LESSICALI SEMPLICI (senza LLM judge)
# ---------------------------------------------------------

_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge(generated: str, reference: str) -> Dict[str, float]:
    scores = _scorer.score(reference, generated)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def token_overlap(generated: str, reference: str) -> float:
    """
    Overlap lessicale semplice: % di token della reference
    presenti nella risposta generata. Proxy di 'recall' di contenuto.
    """
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    overlap = gen_tokens & ref_tokens
    return len(overlap) / len(ref_tokens)


def context_overlap(generated: str, context: str) -> float:
    """
    Proxy di 'faithfulness' senza LLM: quanti token della risposta
    sono presenti nel contesto recuperato (alto = risposta ancorata al contesto).
    """
    gen_tokens = set(generated.lower().split())
    ctx_tokens = set(context.lower().split())
    if not gen_tokens:
        return 0.0
    overlap = gen_tokens & ctx_tokens
    return len(overlap) / len(gen_tokens)


# ---------------------------------------------------------
# LOOP DI VALUTAZIONE RAG END-TO-END
# ---------------------------------------------------------

def evaluate_rag(
    ground_truth: List[Dict],
    retrieve_fn: Callable[[str], List],
    generate_fn: Callable[[str, str], str],
    csv_path: str = None,
    json_path: str = None,
    top_k_context: int = 3,
) -> Dict:
    """
    ground_truth: lista di dict con almeno "query" e "answer_summary"
    retrieve_fn: query -> lista di Document (già rerankati)
    generate_fn: (query, context) -> risposta generata (es. generate_answer)
    top_k_context: quanti chunk usare come contesto per il LLM
    """

    results = []

    for q in ground_truth:
        query_text = q["query"]
        reference_answer = q.get("answer_summary", "")

        docs = retrieve_fn(query_text)
        top_docs = docs[:top_k_context]
        context = "\n\n".join(d.page_content for d in top_docs)

        generated_answer = generate_fn(query_text, context)

        rouge = compute_rouge(generated_answer, reference_answer)
        overlap_ref = token_overlap(generated_answer, reference_answer)
        overlap_ctx = context_overlap(generated_answer, context)

        row = {
            "query": query_text,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "context_used": context,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "token_overlap_with_reference": overlap_ref,
            "token_overlap_with_context": overlap_ctx,
        }
        results.append(row)

        print(f"\nQUERY: {query_text}")
        print(f"  ROUGE-1: {rouge['rouge1']:.3f} | ROUGE-2: {rouge['rouge2']:.3f} | ROUGE-L: {rouge['rougeL']:.3f}")
        print(f"  Overlap vs reference: {overlap_ref:.3f} | Overlap vs context (faithfulness proxy): {overlap_ctx:.3f}")

    # Medie aggregate
    n = len(results)
    avg = {
        "avg_rouge1": sum(r["rouge1"] for r in results) / n,
        "avg_rouge2": sum(r["rouge2"] for r in results) / n,
        "avg_rougeL": sum(r["rougeL"] for r in results) / n,
        "avg_token_overlap_with_reference": sum(r["token_overlap_with_reference"] for r in results) / n,
        "avg_token_overlap_with_context": sum(r["token_overlap_with_context"] for r in results) / n,
    }

    print("\n=== MEDIE RAG ===")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

    # Salvataggio CSV
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for r in results:
                # rimuovo newline interni per CSV pulito
                r_clean = {k: (v.replace("\n", " ") if isinstance(v, str) else v) for k, v in r.items()}
                writer.writerow(r_clean)
        print(f"\nCSV salvato in: {csv_path}")

    # Salvataggio JSON (utile per ispezione completa)
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"per_query": results, "averages": avg}, f, ensure_ascii=False, indent=2)
        print(f"JSON salvato in: {json_path}")

    return {"per_query": results, "averages": avg}
