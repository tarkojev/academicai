"""
This file implements the evaluation script for multiple runs (teachers or students).
It collects the test logits and labels from matching run folders, computes aggregated metrics:
- accuracy mean/std
- macro-F1 mean/std
- label flipping rate across runs
- number of runs

Saves summary JSON into runs_root.
"""

# Imported libraries
import os
import argparse
import numpy as np
from utils import label_flipping_rate, metrics_from_logits, save_json

# Collect metrics from multiple student runs and compute aggregated results
def collect(run_dirs):
    accs, f1s = [], []
    preds_all = []

    for d in run_dirs:
        logits = np.load(os.path.join(d, "test_logits.npy"))
        labels = np.load(os.path.join(d, "test_labels.npy"))

        m = metrics_from_logits(logits, labels)
        accs.append(m["accuracy"])
        f1s.append(m["f1_macro"])

        preds_all.append(logits.argmax(axis=1))

    pred_matrix = np.stack(preds_all, axis=0)

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "label_flipping_rate": float(label_flipping_rate(pred_matrix)),
        "n_runs": int(len(run_dirs)),
        "run_dirs": run_dirs,
    }

# Main function to run the evaluation and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument(
        "--run",
        type=str,
        required=True,
        help="Prefix to match folders. Example: teacher_bert-base-uncased_fs10_supp123",
    )
    args = ap.parse_args()
    run_dirs = []
    for name in os.listdir(args.runs_root):
        if name.startswith(args.run):
            run_dirs.append(os.path.join(args.runs_root, name))
    run_dirs = sorted(run_dirs)
    if not run_dirs:
        raise ValueError(f"No runs found under {args.runs_root} with prefix: {args.run}")
    summary = collect(run_dirs)
    out_path = os.path.join(args.runs_root, f"summary_{args.run}.json")
    save_json(out_path, summary)
    print(summary)

if __name__ == "__main__":
    main()