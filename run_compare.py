"""
This file compares all runs by collecting their test metrics from saved artifacts. It generates:
- A detailed CSV with one row per run, including all metadata and metrics.
- A summary CSV with aggregated mean/std metrics for groups of runs.
- Summary tables in the console for both detailed and summary views.
- A graph comparing performance and stability
"""

# Imported libraries
import os
import json
import argparse
import csv
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd

# Define type order for sorting
TYPE_ORDER = {"full": 0, "teacher": 1, "student": 2, "ensemble": 3, "unknown": 9}

# Utility functions for run comparison
def detect_run_type(run_name: str) -> str:
    if run_name.startswith("full_"):
        return "full"
    if run_name.startswith("teacher_"):
        return "teacher"
    if run_name.startswith("student_"):
        return "student"
    if run_name.startswith("ensemble"):
        return "ensemble"
    return "unknown"

# Processing functions
def safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# Extraction functions
def format_float(x: Any, digits: int = 4) -> str:
    if x is None:
        return "None"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "None"

# Extract accuracy and f1_macro from meta
def extract_metrics(meta: dict) -> Tuple[Optional[float], Optional[float]]:
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    acc = metrics.get("accuracy", None)
    f1 = metrics.get("f1_macro", None)
    return acc, f1

# Infer ensemble models from teacher_dirs
def infer_ensemble_models(runs_root: str, teacher_dirs: List[str]) -> List[str]:
    models = []
    for d in teacher_dirs:
        dpath = d
        if not os.path.isabs(dpath):
            dpath = os.path.join(runs_root, os.path.normpath(d).replace("\\", "/").lstrip("./"))
        meta = safe_read_json(os.path.join(dpath, "meta.json"))
        if meta:
            m = meta.get("model", None)
            if m:
                models.append(str(m))
    seen = set()
    uniq = []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq

# Data Loading and Processing
def load_all_runs(runs_root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(runs_root):
        return rows
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue
        meta_path = os.path.join(run_dir, "meta.json")
        if not os.path.exists(meta_path):
            continue
        meta = safe_read_json(meta_path)
        if not meta:
            continue
        rtype = detect_run_type(name)
        acc, f1 = extract_metrics(meta)
        model = meta.get("model", None)
        if model is None:
            model = meta.get("student_model", None)
        if model is None and rtype == "ensemble":
            model = "ensemble"
        if model is None:
            model = "n/a"
        teacher_dirs = meta.get("teacher_dirs", None)
        ensemble_models = None
        if rtype == "ensemble" and isinstance(teacher_dirs, list):
            ensemble_models_list = infer_ensemble_models(runs_root, teacher_dirs)
            if ensemble_models_list:
                ensemble_models = ",".join(ensemble_models_list)
        row = {
            "run_name": name,
            "type": rtype,
            "model": model,
            "ensemble_models": ensemble_models,
            "accuracy": acc,
            "f1_macro": f1,
            "retain_ratio_vs_full": None,
            "train_seed": meta.get("train_seed"),
            "support_seed": meta.get("support_seed"),
            "n_per_class": meta.get("n_per_class"),
            "tau": meta.get("tau"),
            "alpha": meta.get("alpha"),
            "epochs": meta.get("epochs"),
            "lr": meta.get("lr"),
            "batch_size": meta.get("batch_size"),
            "max_len": meta.get("max_len"),
        }
        rows.append(row)
    return rows

# Retain Ratio Calculation
def compute_retain_ratio(rows: List[Dict[str, Any]]) -> None:
    full_acc_by_model: Dict[str, float] = {}
    for r in rows:
        if r["type"] == "full" and r.get("accuracy") is not None:
            full_acc_by_model[str(r["model"])] = float(r["accuracy"])
    for r in rows:
        if r["type"] in {"teacher", "student"}:
            m = str(r["model"])
            if m in full_acc_by_model and r.get("accuracy") is not None:
                r["retain_ratio_vs_full"] = float(r["accuracy"]) / full_acc_by_model[m]

# Sorting function
def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]):
        t = TYPE_ORDER.get(r.get("type", "unknown"), 9)
        model = str(r.get("model", ""))
        acc = r.get("accuracy", None)
        acc_key = -float(acc) if acc is not None else 1e9
        return (t, model, acc_key, str(r.get("run_name", "")))
    return sorted(rows, key=key)

# Grouping and Aggregation
def group_key(run: Dict[str, Any]) -> Tuple: 
    rtype = run.get("type")
    if rtype == "teacher":
        return (rtype, run.get("model"), run.get("support_seed"), run.get("n_per_class"))
    if rtype == "student":
        return (rtype, run.get("model"), run.get("support_seed"), run.get("n_per_class"), run.get("tau"), run.get("alpha"))
    if rtype == "full":
        return (rtype, run.get("model"), run.get("epochs"), run.get("lr"), run.get("batch_size"))
    if rtype == "ensemble":
        return (rtype, run.get("ensemble_models"), run.get("support_seed"), run.get("n_per_class"))
    return (rtype, run.get("model"))

# Compute mean and std
def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values: return None, None
    if len(values) == 1: return values[0], 0.0
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, var ** 0.5

# Build summary from grouped runs
def build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    summary_rows: List[Dict[str, Any]] = []
    for k, items in groups.items():
        accs = [float(x["accuracy"]) for x in items if x.get("accuracy") is not None]
        f1s = [float(x["f1_macro"]) for x in items if x.get("f1_macro") is not None]
        retains = [float(x["retain_ratio_vs_full"]) for x in items if x.get("retain_ratio_vs_full") is not None]
        acc_m, acc_s = mean_std(accs)
        f1_m, f1_s = mean_std(f1s)
        ret_m, ret_s = mean_std(retains)
        t = k[0]
        row = {
            "group_type": t,
            "group_model": items[0].get("model") if t != "ensemble" else "ensemble",
            "n_runs": len(items),
            "accuracy_mean": acc_m,
            "accuracy_std": acc_s,
            "f1_macro_mean": f1_m,
            "f1_macro_std": f1_s,
            "retain_mean": ret_m,
            "retain_std": ret_s,
        }
        summary_rows.append(row)
    return sorted(summary_rows, key=lambda x: (TYPE_ORDER.get(x["group_type"], 9), str(x["group_model"]), -(x["accuracy_mean"] or 0)))

# Output and Visualization
def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

# Generate Summary Graph
def save_summary_plot(summary_data: List[Dict[str, Any]], out_path: str):
    if not summary_data:
        return
    df = pd.DataFrame(summary_data)
    df['group_model'] = df['group_model'].fillna('n/a')
    plt.figure(figsize=(12, 7))
    labels = df['group_type'] + ": " + df['group_model'].astype(str).str[:15]
    plt.bar(labels, df['accuracy_mean'], yerr=df['accuracy_std'], 
            capsize=5, color='skyblue', edgecolor='navy', alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy (Mean ± SD)')
    plt.title('Few-Shot Learning Performance & Stability Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\n[Visual] Performance plot saved to: {out_path}")

# Main function to execute the comparison
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--out", type=str, default="runs/comparison.csv")
    ap.add_argument("--out_summary", type=str, default="runs/comparison_summary.csv")
    ap.add_argument("--out_plot", type=str, default="runs/results_vis.png")
    args = ap.parse_args()

    # Load and process
    rows = load_all_runs(args.runs_root)
    if not rows:
        print(f"No runs found in {args.runs_root}")
        return

    compute_retain_ratio(rows)
    rows = sort_rows(rows)
    save_csv(rows, args.out)

    # Build summary
    summary_rows = build_summary(rows)
    save_csv(summary_rows, args.out_summary)

    # Output paths
    print(f"Detailed CSV: {args.out}")
    print(f"Summary CSV:  {args.out_summary}")
    
    # Generate Plot
    save_summary_plot(summary_rows, args.out_plot)

if __name__ == "__main__":
    main()