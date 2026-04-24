import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
runs_root = "runs"

zero_shot_runs = [
    "full_bert-base-uncased_seed0_ep0_lr2e-05_bs16_ml128",
    "full_distilbert-base-uncased_seed0_ep0_lr2e-05_bs16_ml128",
    "full_t5-small_seed0_ep0_lr2e-05_bs16_ml128",
]

label_names = ["World", "Sports", "Business", "Sci/Tech"]

# Save OUTSIDE runs
output_dir = "zero_shot_confusion_analysis"
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# STEP 1: inspect unique labels first
# --------------------------------------------------
run_data = []
all_unique_sets = []

print("=" * 100)
print("STEP 1: CHECK UNIQUE LABELS")
print("=" * 100)

for run_name in zero_shot_runs:
    run_dir = os.path.join(runs_root, run_name)

    labels_path = os.path.join(run_dir, "test_labels.npy")
    logits_path = os.path.join(run_dir, "test_logits.npy")
    probs_path = os.path.join(run_dir, "test_probs.npy")

    if not os.path.exists(labels_path):
        print(f"[SKIP] {run_name}: test_labels.npy not found")
        continue

    y_true = np.load(labels_path)

    if os.path.exists(logits_path):
        y_scores = np.load(logits_path)
        y_pred = np.argmax(y_scores, axis=1)
    elif os.path.exists(probs_path):
        y_scores = np.load(probs_path)
        y_pred = np.argmax(y_scores, axis=1)
    else:
        print(f"[SKIP] {run_name}: no test_logits.npy or test_probs.npy found")
        continue

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    print(f"{run_name}")
    print(f"  unique true labels      : {unique_true}")
    print(f"  unique predicted labels : {unique_pred}")

    all_unique_sets.append(tuple(unique_true.tolist()))
    run_data.append({
        "run_name": run_name,
        "run_dir": run_dir,
        "y_true": y_true,
        "y_pred": y_pred,
        "unique_true": unique_true,
        "unique_pred": unique_pred,
    })

# Check consistency across runs
consistent_labels = len(set(all_unique_sets)) == 1
print("\nLabel consistency across runs:", consistent_labels)

with open(os.path.join(output_dir, "unique_labels_check.txt"), "w", encoding="utf-8") as f:
    for item in run_data:
        f.write(f"{item['run_name']}\n")
        f.write(f"  unique true labels      : {item['unique_true']}\n")
        f.write(f"  unique predicted labels : {item['unique_pred']}\n\n")
    f.write(f"Label consistency across runs: {consistent_labels}\n")

# --------------------------------------------------
# STEP 2: process each run and collect one summary
# --------------------------------------------------
summary_rows = []
combined_cm_true = []

print("\n" + "=" * 100)
print("STEP 2: BUILD PER-RUN REPORTS + COMBINED SUMMARY")
print("=" * 100)

for item in run_data:
    run_name = item["run_name"]
    y_true = item["y_true"]
    y_pred = item["y_pred"]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(label_names)), zero_division=0
    )

    cm_raw = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    cm_true = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)), normalize="true")
    cm_pred = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)), normalize="pred")

    combined_cm_true.append((run_name, cm_true))

    true_counts = np.bincount(y_true, minlength=len(label_names))
    pred_counts = np.bincount(y_pred, minlength=len(label_names))

    # Per-run text report
    report_text = classification_report(
        y_true, y_pred, target_names=label_names, digits=4, zero_division=0
    )

    with open(os.path.join(output_dir, f"{run_name}_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Accuracy : {acc:.6f}\n")
        f.write(f"Macro-F1 : {macro_f1:.6f}\n\n")
        f.write("True class counts:\n")
        for cls, cnt in zip(label_names, true_counts):
            f.write(f"  {cls}: {cnt}\n")
        f.write("\nPredicted class counts:\n")
        for cls, cnt in zip(label_names, pred_counts):
            f.write(f"  {cls}: {cnt}\n")
        f.write("\nRaw confusion matrix:\n")
        f.write(np.array2string(cm_raw))
        f.write("\n\nRow-normalized confusion matrix (recall view):\n")
        f.write(np.array2string(np.round(cm_true, 4)))
        f.write("\n\nColumn-normalized confusion matrix (precision view):\n")
        f.write(np.array2string(np.round(cm_pred, 4)))
        f.write("\n\nClassification report:\n")
        f.write(report_text)

    # Per-run confusion matrix figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ConfusionMatrixDisplay(cm_raw, display_labels=label_names).plot(
        ax=axes[0], cmap="Blues", values_format="d", colorbar=False
    )
    axes[0].set_title("Raw")

    ConfusionMatrixDisplay(cm_true, display_labels=label_names).plot(
        ax=axes[1], cmap="Blues", values_format=".2f", colorbar=False
    )
    axes[1].set_title("Normalized by True\n(Recall View)")

    ConfusionMatrixDisplay(cm_pred, display_labels=label_names).plot(
        ax=axes[2], cmap="Blues", values_format=".2f", colorbar=False
    )
    axes[2].set_title("Normalized by Pred\n(Precision View)")

    fig.suptitle(run_name, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_name}_confusion_matrices.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Per-run class distribution figure
    x = np.arange(len(label_names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, true_counts, width, label="True")
    plt.bar(x + width / 2, pred_counts, width, label="Predicted")
    plt.xticks(x, label_names, rotation=20)
    plt.ylabel("Count")
    plt.title(f"True vs Predicted Class Distribution\n{run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_name}_class_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Summary row
    row = {
        "run_name": run_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "unique_true_labels": json.dumps(item["unique_true"].tolist()),
        "unique_pred_labels": json.dumps(item["unique_pred"].tolist()),
    }

    for i, cls in enumerate(label_names):
        row[f"{cls}_precision"] = prec[i]
        row[f"{cls}_recall"] = rec[i]
        row[f"{cls}_f1"] = f1[i]
        row[f"{cls}_support"] = support[i]
        row[f"{cls}_true_count"] = true_counts[i]
        row[f"{cls}_pred_count"] = pred_counts[i]

    summary_rows.append(row)

    print(f"{run_name}: acc={acc:.6f}, macro_f1={macro_f1:.6f}")

# --------------------------------------------------
# STEP 3: save one combined CSV summary
# --------------------------------------------------
summary_csv_path = os.path.join(output_dir, "zero_shot_combined_summary.csv")

fieldnames = list(summary_rows[0].keys()) if summary_rows else []
with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

# Also save a short human-readable summary
summary_txt_path = os.path.join(output_dir, "zero_shot_combined_summary.txt")
with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("ZERO-SHOT COMBINED SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    for row in summary_rows:
        f.write(f"Run: {row['run_name']}\n")
        f.write(f"  Accuracy : {row['accuracy']:.6f}\n")
        f.write(f"  Macro-F1 : {row['macro_f1']:.6f}\n")
        f.write(f"  Unique true labels      : {row['unique_true_labels']}\n")
        f.write(f"  Unique predicted labels : {row['unique_pred_labels']}\n")
        f.write("  Per-class metrics:\n")
        for cls in label_names:
            f.write(
                f"    {cls:<10} "
                f"P={row[f'{cls}_precision']:.4f} "
                f"R={row[f'{cls}_recall']:.4f} "
                f"F1={row[f'{cls}_f1']:.4f} "
                f"true={row[f'{cls}_true_count']} "
                f"pred={row[f'{cls}_pred_count']}\n"
            )
        f.write("\n")

# --------------------------------------------------
# STEP 4: one combined figure for all models
# --------------------------------------------------
if combined_cm_true:
    fig, axes = plt.subplots(1, len(combined_cm_true), figsize=(6 * len(combined_cm_true), 5))

    if len(combined_cm_true) == 1:
        axes = [axes]

    for ax, (run_name, cm_true) in zip(axes, combined_cm_true):
        disp = ConfusionMatrixDisplay(cm_true, display_labels=label_names)
        disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
        ax.set_title(run_name.replace("full_", "").replace("_seed0_ep0_lr2e-05_bs16_ml128", ""))

    fig.suptitle("Zero-Shot Confusion Matrices", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zero_shot_combined_confusion_matrices.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

print(f"\nDone. All outputs saved to: {output_dir}")
