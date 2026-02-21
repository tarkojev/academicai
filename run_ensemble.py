"""
This file implements an ensemble baseline evaluation for the AG News classification task. It includes:
- Loading test_logits.npy from multiple teacher runs
- Building ensemble probs by averaging softmax(logits)
- Computeing accuracy + macro-F1
- Saveing run artifacts in runs/<name>/
"""

# Imported libraries
import os
import argparse
import numpy as np
from utils import metrics_from_logits, save_json

# Calculate temperature-scaled softmax probabilities from logits
def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

# Main function to run the ensemble evaluation and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--teacher_dirs", type=str, nargs="+", required=True)
    ap.add_argument("--name", type=str, default="ensemble_run")
    args = ap.parse_args()
    probs = []
    labels_ref = None
    for d in args.teacher_dirs:
        logits = np.load(os.path.join(d, "test_logits.npy"))
        labels = np.load(os.path.join(d, "test_labels.npy"))

        if labels_ref is None:
            labels_ref = labels
        else:
            if not np.array_equal(labels_ref, labels):
                raise ValueError(
                    f"Label mismatch across teacher runs.\n"
                    f"Likely different dataset ordering or incompatible artifacts.\n"
                    f"Dir: {d}"
                )

        probs.append(softmax_np(logits))
    # Average the probabilities across teachers to get ensemble soft targets
    p_ens = np.mean(np.stack(probs, axis=0), axis=0)  # (N, C)

    # metrics_from_logits expects logits-like array; argmax works on log-probs same as probs
    logits_like = np.log(p_ens + 1e-12)
    m = metrics_from_logits(logits_like, labels_ref)
    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "test_probs.npy"), p_ens)
    np.save(os.path.join(run_dir, "test_labels.npy"), labels_ref)

    # Save summary JSON with metrics and teacher run dirs
    save_json(
        os.path.join(run_dir, "meta.json"),
        {
            "teacher_dirs": args.teacher_dirs,
            "metrics": m,
        },
    )
    print(m)

if __name__ == "__main__":
    main()