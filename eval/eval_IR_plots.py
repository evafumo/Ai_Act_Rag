# ============================================================
# eval_plots.py
# Aggrega i risultati di evaluate_retriever() (eval_*.json) in un
# unico CSV "long format" e genera grafici comparativi tra retriever
# diversi (precision, recall, F1, NDCG, MAP, MRR).
#
# INSTALLAZIONE (aggiungi a requirements.txt se non presenti):
#   pandas==2.2.2
#   matplotlib==3.8.4
# ============================================================

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1) CARICAMENTO + NORMALIZZAZIONE risultati eval.py -> long format
# ============================================================

def load_eval_json(
	json_path: str,
	model: str, 
	dataset: str = "unknown", 
	chunking: str = "unknown"
	) -> pd.DataFrame:
    """
    Carica un file eval_*.json prodotto da evaluate_retriever() e lo
    converte in un DataFrame long format, una riga per (k, metrica).

    INPUT:
        json_path (str): percorso al file eval_*.json
        model (str): nome/etichetta del retriever (es. "single_bge")
        dataset (str): nome del dataset/PDF valutato (es. "Adult")
        chunking (str): strategia di chunking usata (es. "hybrid")

    OUTPUT:
        pd.DataFrame con colonne:
            model, dataset, chunking, k, metric, value

    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    # --- MAP come metrica scalare ---
    if "summary" in data and "MAP" in data["summary"]:
        rows.append({
            "model": model,
            "dataset": dataset,
            "chunking": chunking,
            "k": None,
            "metric": "map",
            "value": float(data["summary"]["MAP"]),
        })

    # --- macro_metrics: precision/recall/ndcg per-k ---
    macro = data["summary"].get("macro_metrics", {})
    for k_key, metrics in macro.items():
        k = int(k_key.replace("@", ""))  # "@5" → 5

        for metric_name, v in metrics.items():
            rows.append({
                "model": model,
                "dataset": dataset,
                "chunking": chunking,
                "k": k,
                "metric": metric_name.lower(),  # precision, recall, ndcg
                "value": float(v),
            })

    return pd.DataFrame(rows)



def build_combined_csv(
    runs: list[dict],
    output_csv: str = "eval_results_combined.csv",
) -> pd.DataFrame:
    """
    Unisce più eval_*.json (uno per retriever/dataset/chunking) in un
    unico CSV long format, utile per confrontare N modelli insieme.

    INPUT:
        runs (list[dict]): lista di run, ognuno con chiavi
            {"json_path": ..., "model": ..., "dataset": ..., "chunking": ...}
        output_csv (str): percorso del CSV combinato in output

    OUTPUT:
        pd.DataFrame: il dataframe combinato (già salvato su disco)

    NOTE:
        Ogni volta che valuti un nuovo retriever, aggiungi una riga a
        `runs` (o accoda al CSV esistente con pd.concat) invece di
        rigenerare grafici ad-hoc: mantieni un'unica fonte di verità.
    """
    frames = [
        load_eval_json(r["json_path"], r["model"], r.get("dataset", "unknown"), r.get("chunking", "unknown"))
        for r in runs
    ]
    combined = pd.concat(frames, ignore_index=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"[INFO] CSV combinato salvato in: {output_csv} ({len(combined)} righe)")

    return combined


# ============================================================
# 2) GRAFICI
# ============================================================

def plot_precision_recall_curve(df: pd.DataFrame, output_path: str) -> None:
    """
    Curva Precision@k vs Recall@k, una linea per modello.
    Il grafico classico per confrontare retriever IR: a parità di
    recall, chi ha precision più alta è il retriever migliore.

    INPUT:
        df (pd.DataFrame): dataframe long format (colonne: model, k, metric, value)
        output_path (str): percorso file immagine di output (.png)
    OUTPUT:
        None (salva il file)
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for model, group in df[df["metric"].isin(["precision", "recall"])].groupby("model"):
        pivot = group.pivot_table(index="k", columns="metric", values="value")
        pivot = pivot.sort_index()
        ax.plot(pivot["recall"], pivot["precision"], marker="o", label=model)
        for k_val, row in pivot.iterrows():
            ax.annotate(f"k={k_val}", (row["recall"], row["precision"]),
                        fontsize=7, textcoords="offset points", xytext=(4, 4))

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall per retriever")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Salvato: {output_path}")


def plot_metric_vs_k(df: pd.DataFrame, metric: str, output_path: str) -> None:
    """
    Linea del valore di una metrica (precision, recall, ndcg, ...)
    in funzione di k, una linea per modello.

    INPUT:
        df (pd.DataFrame): dataframe long format
        metric (str): nome della metrica da plottare (es. "precision", "ndcg")
        output_path (str): percorso file immagine di output (.png)
    OUTPUT:
        None (salva il file)
    """
    subset = df[df["metric"] == metric].dropna(subset=["k"])
    if subset.empty:
        print(f"[WARN] Nessun dato per la metrica '{metric}', skip grafico.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, group in subset.groupby("model"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["value"], marker="o", linestyle="None", label=model)

    ax.set_xlabel("k")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()}@k per retriever")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Salvato: {output_path}")


def plot_scalar_metrics_bar(df: pd.DataFrame, output_path: str) -> None:
    """
    Bar chart raggruppato per confrontare metriche scalari
    (es. MAP, MRR) tra i vari modelli.

    INPUT:
        df (pd.DataFrame): dataframe long format
        output_path (str): percorso file immagine di output (.png)
    OUTPUT:
        None (salva il file)

    NOTE:
        Usa le righe con k is None (metriche scalari, non per-k).
        Se vuoi confrontare anche precision@10/recall@10 in questo
        bar chart, filtra df su k==10 prima di chiamare la funzione
        e concatenalo alle metriche scalari.
    """
    subset = df[df["k"].isna()]
    if subset.empty:
        print("[WARN] Nessuna metrica scalare (MAP/MRR) trovata, skip grafico.")
        return

    pivot = subset.pivot_table(index="model", columns="metric", values="value")

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Valore")
    ax.set_title("Confronto metriche aggregate per retriever")
    ax.set_ylim(0, 1)
    ax.legend(title="Metrica")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Salvato: {output_path}")


def plot_heatmap_model_metric(df: pd.DataFrame, k: int, output_path: str) -> None:
    """
    Heatmap modello x metrica a un k fisso: colpo d'occhio su quale
    retriever vince su quale metrica, utile con molti modelli insieme.

    INPUT:
        df (pd.DataFrame): dataframe long format
        k (int): valore di k su cui fissare il confronto (es. 10)
        output_path (str): percorso file immagine di output (.png)
    OUTPUT:
        None (salva il file)
    """
    subset = df[df["k"] == k]
    if subset.empty:
        print(f"[WARN] Nessun dato per k={k}, skip heatmap.")
        return

    pivot = subset.pivot_table(index="model", columns="metric", values="value")

    fig, ax = plt.subplots(figsize=(1.5 * len(pivot.columns) + 2, 0.6 * len(pivot.index) + 2))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(f"Metriche a k={k} per retriever")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Valore")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Salvato: {output_path}")


def append_to_combined_csv(
    json_path: str,
    model: str,
    dataset: str,
    chunking: str,
    combined_csv: str = "output/eval_results_combined.csv",
) -> pd.DataFrame:
    """
    Aggiunge i risultati di un singolo run (un eval_*.json) al CSV
    combinato, senza sovrascrivere i run precedenti. Se un run con
    la stessa combinazione (model, dataset, chunking, k, metric)
    esiste già, lo sostituisce (utile se rilanci lo stesso esperimento).

    INPUT:
        json_path (str): percorso al file eval_*.json appena generato
        model (str): nome del retriever usato in questo run (es. args.retriever)
        dataset (str): nome del PDF/dataset (es. pdf_name)
        chunking (str): strategia di chunking usata (es. args.chunking)
        combined_csv (str): percorso del CSV combinato, persistente tra run

    OUTPUT:
        pd.DataFrame: il dataframe combinato aggiornato (già salvato su disco)

    NOTE:
        Chiamala da main.py subito dopo evaluate_retriever(), con i
        parametri che main.py già conosce (args.retriever, pdf_name,
        args.chunking). eval.py resta agnostico rispetto a queste info.
    """
    new_rows = load_eval_json(json_path, model=model, dataset=dataset, chunking=chunking)

    combined_csv = Path(combined_csv)
    key_cols = ["model", "dataset", "chunking", "k", "metric"]

    if combined_csv.exists():
        existing = pd.read_csv(combined_csv)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        # tiene l'ultima occorrenza per ogni combinazione (run più recente vince)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = new_rows

    combined = combined.sort_values(["dataset", "model", "chunking", "metric", "k"])
    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_csv, index=False)
    print(f"[INFO] CSV combinato aggiornato: {combined_csv} ({len(combined)} righe totali)")

    return combined


def generate_all_plots(df: pd.DataFrame, output_dir: str, k_for_heatmap: int = 5) -> None:
    """
    Genera tutti i grafici consigliati per la valutazione IR a partire
    dal dataframe combinato.

    INPUT:
        df (pd.DataFrame): dataframe long format (output di build_combined_csv)
        output_dir (str): cartella dove salvare i grafici
        k_for_heatmap (int): valore di k da usare per la heatmap
    OUTPUT:
        None (salva i file immagine nella cartella indicata)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_precision_recall_curve(df, output_dir / "precision_recall_curve.png")
    plot_metric_vs_k(df, "precision", output_dir / "precision_vs_k.png")
    plot_metric_vs_k(df, "recall", output_dir / "recall_vs_k.png")
    plot_metric_vs_k(df, "ndcg", output_dir / "ndcg_vs_k.png")
    plot_scalar_metrics_bar(df, output_dir / "scalar_metrics_bar.png")
    plot_heatmap_model_metric(df, k_for_heatmap, output_dir / f"heatmap_k{k_for_heatmap}.png")


# ============================================================
# MAIN — uso da linea di comando
# ============================================================

"""
ESEMPIO D'USO:

    python eval_plots.py \
        --run eval_Adult_single_bge.json:single_bge:Adult:hybrid \
        --run eval_Adult_single_e5.json:single_e5:Adult:hybrid \
        --run eval_Adult_bm25_bge.json:bm25_bge:Adult:hybrid \
        --output-dir plots/ \
        --csv-out eval_results_combined.csv

Ogni --run è nel formato: json_path:model:dataset:chunking
(dataset e chunking sono opzionali, default "unknown")
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggrega eval_*.json in un CSV e genera grafici comparativi tra retriever."
    )
    parser.add_argument(
        "--run", action="append", default=[],
        help="Formato json_path:model[:dataset[:chunking]]. Ripeti --run per ogni retriever da confrontare. "
             "Non serve se usi --csv-in su un CSV già costruito (es. da main_rag.py)."
    )
    parser.add_argument(
        "--csv-in", default=None,
        help="Percorso a un CSV combinato GIA' esistente (es. quello aggiornato automaticamente "
             "da main_rag.py via append_to_combined_csv). Se passato, salta la costruzione da --run "
             "e genera i grafici direttamente da questo file."
    )
    parser.add_argument("--csv-out", default="eval_results_combined.csv",
                        help="Percorso del CSV combinato in output (usato solo se costruisci da --run)")
    parser.add_argument("--output-dir", default="plots",
                        help="Cartella dove salvare i grafici")
    parser.add_argument("--k-heatmap", type=int, default=10,
                        help="Valore di k da usare per la heatmap modello x metrica")
    parser.add_argument("--dataset-filter", default=None,
                        help="Filtra i grafici su un singolo dataset (es. B_Adult, C_COMPAS).")
    parser.add_argument("--chunking-filter", default=None,
    		         help="Filtra i grafici su una specifica strategia di chunking (es. fixed, sliding, hybrid)")
    parser.add_argument("--model-filter", default=None,
    			 help="Filtra i grafici su un singolo modello (es. single_bge, dense_bge_e5, bm25_bge)")
		         
    args = parser.parse_args()

    # --- CARICAMENTO DF ---
    if args.csv_in:
        df = pd.read_csv(args.csv_in)
        print(f"[INFO] Caricato CSV combinato esistente: {args.csv_in} ({len(df)} righe)")
    elif args.run:
        runs = []
        for r in args.run:
            parts = r.split(":")
            json_path = parts[0]
            model = parts[1]
            dataset = parts[2] if len(parts) > 2 else "unknown"
            chunking = parts[3] if len(parts) > 3 else "unknown"
            runs.append({"json_path": json_path, "model": model, "dataset": dataset, "chunking": chunking})
        df = build_combined_csv(runs, output_csv=args.csv_out)
    else:
        parser.error("Serve --csv-in oppure almeno un --run")

    # --- FILTRO PER DATASET ---
    if args.dataset_filter:
        df = df[df["dataset"] == args.dataset_filter]
        print(f"[INFO] Filtrato dataset: {args.dataset_filter} ({len(df)} righe)")
        if df.empty:
            print("[WARN] Nessun dato trovato per questo dataset.")
    
    # --- FILTRO PER CHUNKING ---
    if args.chunking_filter:
    	df = df[df["chunking"] == args.chunking_filter]
    	print(f"[INFO] Filtrato chunking: {args.chunking_filter} ({len(df)} righe)")
    	if df.empty:
            print("[WARN] Nessun dato trovato per questo chunking.")
            
    # --- FILTRO PER MODELLO ---
    if args.model_filter:
        df = df[df["model"] == args.model_filter]
        print(f"[INFO] Filtrato modello: {args.model_filter} ({len(df)} righe)")
        if df.empty:
            print("[WARN] Nessun dato trovato per questo modello.")
        


    generate_all_plots(df, output_dir=args.output_dir, k_for_heatmap=args.k_heatmap)

    print("\n[DONE] Grafici generati in:", args.output_dir)
