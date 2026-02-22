"""
This file implements the ensemble script that combines predictions from multiple teacher runs:
- It loads the test logits/probs and labels from multiple teacher run folders
- It checks that the labels are the same across runs (sanity check for correct artifacts)
- It averages the teacher probabilities to get ensemble soft targets
- It computes ensemble logits (log of averaged probs) and metrics on the test set
- It optionally does the same for the support set (for later KD training)
- It saves the ensemble test logits/probs, labels, and metrics into a new run folder
- It saves a summary JSON with metadata about the teacher runs and ensemble metrics for later analysis.
"""

# Imported libraries
import os
import argparse
import numpy as np
from utils import metrics_from_logits, save_json, logits_to_probs

# Loading logits or probabilities from a run, with support for both formats
def load_probs(run_dir: str, split: str) -> np.ndarray:
    p_path = os.path.join(run_dir, f"{split}_probs.npy")
    if os.path.exists(p_path):
        return np.load(p_path)
    l_path = os.path.join(run_dir, f"{split}_logits.npy")
    if os.path.exists(l_path):
        logits = np.load(l_path)
        return logits_to_probs(logits)

    raise FileNotFoundError(f"Missing both {split}_probs.npy and {split}_logits.npy in {run_dir}")

# Loading labels from a run
def load_labels(run_dir: str, split: str) -> np.ndarray:
    y_path = os.path.join(run_dir, f"{split}_labels.npy")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing {split}_labels.npy in {run_dir}")
    return np.load(y_path)

# Confirm labels are the same
def ensure_same_labels(label_list):
    ref = label_list[0]
    for y in label_list[1:]:
        if not np.array_equal(ref, y):
            raise ValueError("Label mismatch across teacher runs. Check artifacts and ordering.")
    return ref

# Main function to run the ensemble and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--teacher_dirs", type=str, nargs="+", required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--also_support", action="store_true", help="Also store support_probs for KD")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)

    test_probs_list = []
    test_labels_list = []

    for d in args.teacher_dirs:
        test_probs_list.append(load_probs(d, "test"))
        test_labels_list.append(load_labels(d, "test"))

    test_labels = ensure_same_labels(test_labels_list)
    test_probs_ens = np.mean(np.stack(test_probs_list, axis=0), axis=0)

    test_logits_like = np.log(test_probs_ens + 1e-12)
    test_metrics = metrics_from_logits(test_logits_like, test_labels)

    np.save(os.path.join(run_dir, "test_probs.npy"), test_probs_ens)
    np.save(os.path.join(run_dir, "test_logits.npy"), test_logits_like)
    np.save(os.path.join(run_dir, "test_labels.npy"), test_labels)

    support_metrics = None
    if args.also_support:
        support_probs_list = []
        support_labels_list = []

        for d in args.teacher_dirs:
            support_probs_list.append(load_probs(d, "support"))
            support_labels_list.append(load_labels(d, "support"))

        support_labels = ensure_same_labels(support_labels_list)
        support_probs_ens = np.mean(np.stack(support_probs_list, axis=0), axis=0)

        support_logits_like = np.log(support_probs_ens + 1e-12)
        support_metrics = metrics_from_logits(support_logits_like, support_labels)

        np.save(os.path.join(run_dir, "support_probs.npy"), support_probs_ens)
        np.save(os.path.join(run_dir, "support_logits.npy"), support_logits_like)
        np.save(os.path.join(run_dir, "support_labels.npy"), support_labels)

    save_json(
        os.path.join(run_dir, "meta.json"),
        {
            "type": "ensemble",
            "teacher_dirs": args.teacher_dirs,
            "also_support": bool(args.also_support),
            "metrics_test": test_metrics,
            "metrics_support": support_metrics,
        },
    )

    print(test_metrics)


if __name__ == "__main__":
    main()