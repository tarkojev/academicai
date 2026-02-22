"""
This file compares all runs by collecting their test metrics from saved artifacts. It generates:
- A detailed CSV with one row per run, including all metadata and metrics.
- A summary CSV with aggregated mean/std metrics for groups of runs (e.g., all students with the same model and support size).
- Summary tables in the console for both detailed and summary views.
"""

# Imported libraries
import os
import json
import argparse
import csv
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Define a consistent order for run types for sorting and grouping
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

# Safe JSON reading with error handling
def safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# Formatting utilities for printing
def format_float(x: Any, digits: int = 4) -> str:
    if x is None:
        return "None"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "None"

# Extracting metrics from meta.json with safe access
def extract_metrics(meta: dict) -> Tuple[Optional[float], Optional[float]]:
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    acc = metrics.get("accuracy", None)
    f1 = metrics.get("f1_macro", None)
    return acc, f1

# Inferring ensemble model names from teacher run metadata
def infer_ensemble_models(runs_root: str, teacher_dirs: List[str]) -> List[str]:
    models = []
    # Read meta for each teacher
    for d in teacher_dirs:
        dpath = d
        if not os.path.isabs(dpath):
            dpath = os.path.join(runs_root, os.path.normpath(d).replace("\\", "/").lstrip("./"))
        meta = safe_read_json(os.path.join(dpath, "meta.json"))
        if meta:
            m = meta.get("model", None)
            if m:
                models.append(str(m))
    # Remove duplicates
    seen = set()
    uniq = []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq

# Loading runs
def load_all_runs(runs_root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

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
        
        # Runs fields
        rtype = detect_run_type(name)
        acc, f1 = extract_metrics(meta)
        model = meta.get("model", None)
        if model is None:
            model = meta.get("student_model", None)
        if model is None and rtype == "ensemble":
            model = "ensemble"
        if model is None:
            model = "n/a"
        train_seed = meta.get("train_seed", None)
        support_seed = meta.get("support_seed", None)
        n_per_class = meta.get("n_per_class", None)
        tau = meta.get("tau", None)
        alpha = meta.get("alpha", None)
        epochs = meta.get("epochs", None)
        lr = meta.get("lr", None)
        batch_size = meta.get("batch_size", None)
        max_len = meta.get("max_len", None)
        teacher_dirs = meta.get("teacher_dirs", None)
        ensemble_models = None
        if rtype == "ensemble" and isinstance(teacher_dirs, list):
            ensemble_models_list = infer_ensemble_models(runs_root, teacher_dirs)
            if ensemble_models_list:
                ensemble_models = ",".join(ensemble_models_list)

        # Rows output
        row = {
            "run_name": name,
            "type": rtype,
            "model": model,
            "ensemble_models": ensemble_models,  # only for ensemble
            "accuracy": acc,
            "f1_macro": f1,
            "retain_ratio_vs_full": None,  # filled later
            "train_seed": train_seed,
            "support_seed": support_seed,
            "n_per_class": n_per_class,
            "tau": tau,
            "alpha": alpha,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "max_len": max_len,
        }
        rows.append(row)
    return rows

# Compute retain_ratio_vs_full for teacher and student runs
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
            else:
                r["retain_ratio_vs_full"] = None
        else:
            r["retain_ratio_vs_full"] = None

# Sorting rows: type -> model -> accuracy desc -> run_name
def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]):
        t = TYPE_ORDER.get(r.get("type", "unknown"), 9)
        model = str(r.get("model", ""))
        # None goes last
        acc = r.get("accuracy", None)
        acc_key = -float(acc) if acc is not None else 1e9
        return (t, model, acc_key, str(r.get("run_name", "")))
    return sorted(rows, key=key)

# CSV saving utility
def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        print("No runs found.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_detailed_table(rows: List[Dict[str, Any]], limit: int = 200) -> None:
    print("\nComparison Table\n")
    header = (
        f"{'Run Name':60} | {'Type':9} | {'Model':28} | "
        f"{'Acc':8} | {'F1':8} | {'Retain':8}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(rows):
        if i >= limit:
            print(f"... ({len(rows) - limit} more rows)")
            break

        model_str = str(r.get("model", ""))
        if r.get("type") == "ensemble" and r.get("ensemble_models"):
            # show ensemble components compactly
            model_str = f"ensemble[{r['ensemble_models']}]"

        print(
            f"{r['run_name'][:60]:60} | "
            f"{str(r.get('type')):9} | "
            f"{model_str[:28]:28} | "
            f"{format_float(r.get('accuracy'), 4):8} | "
            f"{format_float(r.get('f1_macro'), 4):8} | "
            f"{format_float(r.get('retain_ratio_vs_full'), 3):8}"
        )

# Grouping key for summary
def group_key(run: Dict[str, Any]) -> Tuple: 
    type = run.get("type")
    if type == "teacher":
        return (type, run.get("model"), run.get("support_seed"), run.get("n_per_class"))
    if type == "student":
        return (
            type,
            run.get("model"),
            run.get("support_seed"),
            run.get("n_per_class"),
            run.get("tau"),
            run.get("alpha"),
        )
    if type == "full":
        return (
            type,
            run.get("model"),
            run.get("epochs"),
            run.get("lr"),
            run.get("batch_size"),
            run.get("max_len"),
        )
    if type == "ensemble":
        return (
            type,
            run.get("ensemble_models"),
            run.get("support_seed"),
            run.get("n_per_class"),
        )

    return (type, run.get("model"))

# Compute mean and std with handling for empty and single-value lists
def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, var ** 0.5

# Building summary
def build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    summary_rows: List[Dict[str, Any]] = []
    for k, items in groups.items():
        accs = [float(x["accuracy"]) for x in items if x.get("accuracy") is not None]
        f1s = [float(x["f1_macro"]) for x in items if x.get("f1_macro") is not None]
        retains = [float(x["retain_ratio_vs_full"]) for x in items if x.get("retain_ratio_vs_full") is not None]
        acc_mean, acc_std = mean_std(accs)
        f1_mean, f1_std = mean_std(f1s)
        ret_mean, ret_std = mean_std(retains)
        t = k[0]
        row: Dict[str, Any] = {
            "group_type": t,
            "group_model": None,
            "group_details": None,
            "n_runs": len(items),
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "f1_macro_mean": f1_mean,
            "f1_macro_std": f1_std,
            "retain_mean": ret_mean,
            "retain_std": ret_std,
        }

        if t == "teacher":
            _, model, support_seed, n_per_class = k
            row["group_model"] = model
            row["group_details"] = f"supp={support_seed}, fs={n_per_class}"

        elif t == "student":
            _, model, support_seed, n_per_class, tau, alpha = k
            row["group_model"] = model
            row["group_details"] = f"supp={support_seed}, fs={n_per_class}, tau={tau}, alpha={alpha}"

        elif t == "full":
            _, model, epochs, lr, bs, ml = k
            row["group_model"] = model
            row["group_details"] = f"ep={epochs}, lr={lr}, bs={bs}, ml={ml}"

        elif t == "ensemble":
            _, emodels, support_seed, n_per_class = k
            row["group_model"] = "ensemble"
            row["group_details"] = f"models={emodels}, supp={support_seed}, fs={n_per_class}"

        else:
            row["group_model"] = k[1] if len(k) > 1 else None
            row["group_details"] = "n/a"

        summary_rows.append(row)
    # Sort summary rows by type -> model -> accuracy desc -> details
    def sk(r: Dict[str, Any]):
        t = TYPE_ORDER.get(r.get("group_type", "unknown"), 9)
        model = str(r.get("group_model", ""))
        acc = r.get("accuracy_mean", None)
        acc_key = -float(acc) if acc is not None else 1e9
        return (t, model, acc_key, str(r.get("group_details", "")))

    return sorted(summary_rows, key=sk)

# Printing summary table
def print_summary_table(summary_rows: List[Dict[str, Any]]) -> None:
    print("\nCompare mean/std\n")
    header = (
        f"{'Group':10} | {'Model':28} | {'Details':40} | "
        f"{'n':3} | {'Acc(m±s)':14} | {'F1(m±s)':14} | {'Ret(m±s)':14}"
    )
    print(header)
    print("-" * len(header))
    # Format mean+/-std for accuracy, f1, and retain ratio
    for r in summary_rows:
        acc = f"{format_float(r.get('accuracy_mean'), 4)}±{format_float(r.get('accuracy_std'), 4)}"
        f1 = f"{format_float(r.get('f1_macro_mean'), 4)}±{format_float(r.get('f1_macro_std'), 4)}"
        ret = f"{format_float(r.get('retain_mean'), 3)}±{format_float(r.get('retain_std'), 3)}"

        print(
            f"{str(r.get('group_type')):10} | "
            f"{str(r.get('group_model'))[:28]:28} | "
            f"{str(r.get('group_details'))[:40]:40} | "
            f"{int(r.get('n_runs', 0)):3d} | "
            f"{acc:14} | "
            f"{f1:14} | "
            f"{ret:14}"
        )

# Main function to run the comparison and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--out", type=str, default="runs/comparison.csv")
    ap.add_argument("--out_summary", type=str, default="runs/comparison_summary.csv")
    ap.add_argument("--limit", type=int, default=200, help="Max rows to print in detailed table.")
    args = ap.parse_args()

    rows = load_all_runs(args.runs_root)
    compute_retain_ratio(rows)
    rows = sort_rows(rows)

    save_csv(rows, args.out)
    print_detailed_table(rows, limit=args.limit)

    summary_rows = build_summary(rows)
    save_csv(summary_rows, args.out_summary)
    print_summary_table(summary_rows)

    print(f"\nSaved detailed CSV to: {args.out}")
    print(f"Saved summary CSV  to: {args.out_summary}")


if __name__ == "__main__":
    main()