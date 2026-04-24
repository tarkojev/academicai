"""
This file compares all runs by collecting their test metrics from saved artifacts. It generates:
- A detailed CSV with one row per run, including all metadata and metrics.
- A summary CSV with aggregated mean/std metrics for grouped runs.
- A student-table CSV tailored for KD appendix tables.
- Per-model/per-few-shot CSV files tailored for KD appendix tables.
- A trade-off graph: accuracy vs latency.
- An accuracy ranking graph.
- A latency ranking graph.
"""

import os
import json
import argparse
import csv
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


TYPE_ORDER = {
    "zero-shot": 0,
    "full": 1,
    "teacher": 2,
    "student": 3,
    "ensemble": 4,
    "unknown": 9,
}

# Base marker shape = run type
TYPE_MARKERS = {
    "zero-shot": "D",
    "full": "s",
    "teacher": "X",
    "student": "^",   # student uses special FS-dependent orientation later
    "ensemble": "o",
    "unknown": "P",
}

# Student FS orientation mapping
STUDENT_FS_MARKERS = {
    10: "v",      # down triangle
    100: ">",     # right triangle
    1000: "^",    # up triangle
}

# Ensemble FS shown via circle size
ENSEMBLE_FS_SIZES = {
    10: 110,
    100: 150,
    1000: 190,
}

# Color = model family
MODEL_COLORS = {
    "BERT": "#4C78A8",
    "DistilBERT": "#F58518",
    "T5-small": "#54A24B",
    "Ensemble": "#9E9E9E",
    "Other": "#BDBDBD",
}

# Teacher FS orientation mapping for the teacher-only scarcity plot
TEACHER_FS_MARKERS = {
    10: "v",      # down triangle
    100: ">",     # right triangle
    1000: "^",    # up triangle
}

def safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics(meta: dict) -> Tuple[Optional[float], Optional[float]]:
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    acc = metrics.get("accuracy", None)
    f1 = metrics.get("f1_macro", None)
    return acc, f1


def detect_run_type(run_name: str, meta: Optional[dict] = None) -> str:
    if run_name.startswith("full_"):
        if meta is not None and meta.get("epochs") == 0:
            return "zero-shot"
        return "full"
    if run_name.startswith("teacher_"):
        return "teacher"
    if run_name.startswith("student_"):
        return "student"
    if run_name.startswith("ensemble"):
        return "ensemble"
    return "unknown"


def infer_ensemble_models(runs_root: str, teacher_dirs: List[str]) -> List[str]:
    models = []
    for d in teacher_dirs:
        dpath = d
        if not os.path.isabs(dpath):
            dpath = os.path.join(
                runs_root,
                os.path.normpath(d).replace("\\", "/").lstrip("./")
            )
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

        rtype = detect_run_type(name, meta)
        acc, f1 = extract_metrics(meta)

        eff = meta.get("efficiency", {})
        params = eff.get("params")
        latency = eff.get("latency_ms")
        size_mb = eff.get("size_mb")

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
            "params": params,
            "latency_ms": latency,
            "size_mb": size_mb,
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


def fmt_pm(mean_val: Optional[float], std_val: Optional[float], digits: int = 4, pm: str = "±") -> str:
    if mean_val is None:
        return ""
    if std_val is None:
        return f"{mean_val:.{digits}f}"
    return f"{mean_val:.{digits}f} {pm} {std_val:.{digits}f}"


def compute_retain_ratio(rows: List[Dict[str, Any]]) -> None:
    full_acc_by_model: Dict[str, float] = {}

    for r in rows:
        if (
            r["type"] == "full"
            and r.get("accuracy") is not None
            and r.get("epochs") is not None
            and int(r["epochs"]) > 0
        ):
            model_name = str(r["model"])
            acc = float(r["accuracy"])
            if model_name not in full_acc_by_model or acc > full_acc_by_model[model_name]:
                full_acc_by_model[model_name] = acc

    for r in rows:
        if r["type"] in {"teacher", "student"}:
            model_name = str(r["model"])
            if model_name in full_acc_by_model and r.get("accuracy") is not None:
                r["retain_ratio_vs_full"] = float(r["accuracy"]) / full_acc_by_model[model_name]


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]):
        t = TYPE_ORDER.get(r.get("type", "unknown"), 9)
        model = str(r.get("model", ""))
        acc = r.get("accuracy", None)
        acc_key = -float(acc) if acc is not None else 1e9
        return (t, model, acc_key, str(r.get("run_name", "")))

    return sorted(rows, key=key)


def group_key(run: Dict[str, Any]) -> Tuple:
    rtype = run.get("type")

    if rtype == "teacher":
        return (
            rtype,
            run.get("model"),
            run.get("support_seed"),
            run.get("n_per_class"),
        )

    if rtype == "student":
        return (
            rtype,
            run.get("model"),
            run.get("support_seed"),
            run.get("n_per_class"),
            run.get("tau"),
            run.get("alpha"),
        )

    if rtype in {"full", "zero-shot"}:
        return (
            rtype,
            run.get("model"),
            run.get("epochs"),
            run.get("lr"),
            run.get("batch_size"),
        )

    if rtype == "ensemble":
        return (
            rtype,
            run.get("ensemble_models"),
            run.get("support_seed"),
            run.get("n_per_class"),
        )

    return (rtype, run.get("model"))


def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0

    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, var ** 0.5


def build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    summary_rows: List[Dict[str, Any]] = []

    for _, items in groups.items():
        accs = [float(x["accuracy"]) for x in items if x.get("accuracy") is not None]
        f1s = [float(x["f1_macro"]) for x in items if x.get("f1_macro") is not None]
        retains = [
            float(x["retain_ratio_vs_full"])
            for x in items
            if x.get("retain_ratio_vs_full") is not None
        ]
        latencies = [
            float(x["latency_ms"])
            for x in items
            if x.get("latency_ms") is not None
        ]

        acc_m, acc_s = mean_std(accs)
        f1_m, f1_s = mean_std(f1s)
        ret_m, ret_s = mean_std(retains)
        lat_m, lat_s = mean_std(latencies)

        t = items[0].get("type")
        if t == "ensemble":
            group_model = items[0].get("ensemble_models")
        else:
            group_model = items[0].get("model")

        row = {
            "group_type": t,
            "group_model": group_model,
            "fs": items[0].get("n_per_class"),
            "n_runs": len(items),
            "accuracy_mean": acc_m,
            "accuracy_std": acc_s,
            "f1_macro_mean": f1_m,
            "f1_macro_std": f1_s,
            "latency_mean": lat_m,
            "latency_std": lat_s,
            "retain_mean": ret_m,
            "retain_std": ret_s,
            "accuracy_fmt": fmt_pm(acc_m, acc_s, 4),
            "f1_fmt": fmt_pm(f1_m, f1_s, 4),
            "latency_fmt": fmt_pm(lat_m, lat_s, 2),
            "retain_fmt": fmt_pm(ret_m, ret_s, 4),
        }

        if t == "student":
            row["tau"] = items[0].get("tau")
            row["alpha"] = items[0].get("alpha")
            row["support_seed"] = items[0].get("support_seed")

        if t == "teacher":
            row["support_seed"] = items[0].get("support_seed")

        if t == "ensemble":
            row["support_seed"] = items[0].get("support_seed")

        if t in {"full", "zero-shot"}:
            row["epochs"] = items[0].get("epochs")
            row["lr"] = items[0].get("lr")
            row["batch_size"] = items[0].get("batch_size")

        summary_rows.append(row)

    return sorted(
        summary_rows,
        key=lambda x: (
            TYPE_ORDER.get(x["group_type"], 9),
            str(x["group_model"]),
            x["fs"] if x["fs"] is not None else 10**9,
            -(x["accuracy_mean"] or 0),
        ),
    )


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_short_model_name(model: str) -> str:
    return (
        str(model)
        .replace("bert-base-uncased", "BERT")
        .replace("distilbert-base-uncased", "DistilBERT")
        .replace("t5-small", "T5-small")
    )


def slugify_model_name(model: str) -> str:
    return (
        make_short_model_name(model)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "")
    )


def make_short_label(row: pd.Series) -> str:
    run_type = str(row.get("group_type", "n/a"))
    model = make_short_model_name(str(row.get("group_model", "n/a")))
    fs = row.get("fs", None)
    tau = row.get("tau", None)
    alpha = row.get("alpha", None)

    if run_type == "student":
        parts = [model]
        if pd.notna(fs):
            parts.append(f"FS={int(fs)}")
        if pd.notna(tau):
            parts.append(f"τ={tau}")
        if pd.notna(alpha):
            parts.append(f"α={alpha}")
        return ", ".join(parts)

    if run_type == "ensemble":
        return f"Ensemble, FS={int(fs)}" if pd.notna(fs) else "Ensemble"

    if run_type == "full":
        return f"Full {model}"

    if run_type == "zero-shot":
        return f"Zero-shot {model}"

    if run_type == "teacher":
        return f"Teacher {model}, FS={int(fs)}" if pd.notna(fs) else f"Teacher {model}"

    return f"{run_type} {model}"


def get_model_family(group_type: str, group_model: str) -> str:
    if str(group_type) == "ensemble":
        return "Ensemble"

    model = str(group_model).lower()

    if "distilbert" in model:
        return "DistilBERT"
    if "bert" in model:
        return "BERT"
    if "t5" in model:
        return "T5-small"
    return "Other"


def marker_for_row(row: pd.Series) -> str:
    run_type = str(row.get("group_type", "unknown"))

    if run_type == "student":
        fs = row.get("fs")
        if pd.notna(fs):
            try:
                fs = int(fs)
                return STUDENT_FS_MARKERS.get(fs, "^")
            except Exception:
                return "^"
        return "^"

    return TYPE_MARKERS.get(run_type, "o")


def marker_size_for_row(row: pd.Series) -> int:
    run_type = str(row.get("group_type", "unknown"))

    if run_type == "ensemble":
        fs = row.get("fs")
        if pd.notna(fs):
            try:
                fs = int(fs)
                return ENSEMBLE_FS_SIZES.get(fs, 150)
            except Exception:
                return 150
        return 150

    if run_type in {"full", "zero-shot"}:
        return 140
    if run_type == "student":
        return 130
    if run_type == "teacher":
        return 130
    return 130


def build_student_table_rows(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build rows tailored to the KD appendix tables:
    one row per student (model, fs, tau, alpha) summary entry,
    sorted within each (model, fs) block by accuracy descending and latency ascending.
    Also marks the best accuracy and best latency row in each block.
    """
    student_rows = [dict(r) for r in summary_rows if r.get("group_type") == "student"]
    if not student_rows:
        return []

    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in student_rows:
        model = str(row.get("group_model"))
        fs = row.get("fs")
        if fs is None:
            continue
        grouped[(model, int(fs))].append(row)

    output_rows: List[Dict[str, Any]] = []

    for (model, fs), items in grouped.items():
        items_sorted = sorted(
            items,
            key=lambda r: (
                -(r.get("accuracy_mean") or float("-inf")),
                (r.get("latency_mean") if r.get("latency_mean") is not None else float("inf")),
                (r.get("tau") if r.get("tau") is not None else float("inf")),
                (r.get("alpha") if r.get("alpha") is not None else float("inf")),
            ),
        )

        best_acc_idx = None
        best_lat_idx = None

        acc_candidates = [
            (idx, r["accuracy_mean"]) for idx, r in enumerate(items_sorted)
            if r.get("accuracy_mean") is not None
        ]
        if acc_candidates:
            best_acc_idx = max(acc_candidates, key=lambda x: x[1])[0]

        lat_candidates = [
            (idx, r["latency_mean"]) for idx, r in enumerate(items_sorted)
            if r.get("latency_mean") is not None
        ]
        if lat_candidates:
            best_lat_idx = min(lat_candidates, key=lambda x: x[1])[0]

        for idx, r in enumerate(items_sorted):
            row = {
                "section_model": make_short_model_name(model),
                "section_fs": fs,
                "caption": f"{make_short_model_name(model)} student results for FS={fs}.",
                "label": f"tab:{slugify_model_name(model)}_fs{fs}",
                "tau": r.get("tau"),
                "alpha": r.get("alpha"),
                "accuracy_mean": r.get("accuracy_mean"),
                "accuracy_std": r.get("accuracy_std"),
                "f1_macro_mean": r.get("f1_macro_mean"),
                "f1_macro_std": r.get("f1_macro_std"),
                "latency_mean": r.get("latency_mean"),
                "latency_std": r.get("latency_std"),
                "retain_mean": r.get("retain_mean"),
                "retain_std": r.get("retain_std"),
                "n_runs": r.get("n_runs"),
                "support_seed": r.get("support_seed"),
                "rank_in_section": idx + 1,
                "is_best_accuracy": idx == best_acc_idx,
                "is_best_latency": idx == best_lat_idx,
                "row_color": (
                    "green!15" if idx == best_acc_idx
                    else "blue!12" if idx == best_lat_idx
                    else ""
                ),
                "Accuracy (± std)": fmt_pm(r.get("accuracy_mean"), r.get("accuracy_std"), 4),
                "F1 Macro (± std)": fmt_pm(r.get("f1_macro_mean"), r.get("f1_macro_std"), 4),
                "Latency (± std, ms)": fmt_pm(r.get("latency_mean"), r.get("latency_std"), 2),
                "Retain (± std)": fmt_pm(r.get("retain_mean"), r.get("retain_std"), 4),
                "Accuracy (LaTeX)": fmt_pm(r.get("accuracy_mean"), r.get("accuracy_std"), 4, pm=r"$\pm$"),
                "F1 Macro (LaTeX)": fmt_pm(r.get("f1_macro_mean"), r.get("f1_macro_std"), 4, pm=r"$\pm$"),
                "Latency (LaTeX)": fmt_pm(r.get("latency_mean"), r.get("latency_std"), 2, pm=r"$\pm$"),
                "Retain (LaTeX)": fmt_pm(r.get("retain_mean"), r.get("retain_std"), 4, pm=r"$\pm$"),
            }
            output_rows.append(row)

    output_rows = sorted(
        output_rows,
        key=lambda r: (
            r["section_model"],
            int(r["section_fs"]),
            int(r["rank_in_section"]),
        ),
    )
    return output_rows


def save_student_table_csvs(summary_rows: List[Dict[str, Any]], combined_path: str, tables_dir: str) -> None:
    rows = build_student_table_rows(summary_rows)
    if not rows:
        print("No student summary rows found for appendix table CSV export.")
        return

    save_csv(rows, combined_path)
    print(f"Student table CSV: {combined_path}")

    os.makedirs(tables_dir, exist_ok=True)
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["section_model"], int(row["section_fs"]))].append(row)

    for (model, fs), items in grouped.items():
        fname = f"{model.lower().replace('/', '_').replace(' ', '_')}_fs{fs}.csv"
        out_path = os.path.join(tables_dir, fname)
        save_csv(items, out_path)

    print(f"Student table section CSVs saved in: {tables_dir}")


def save_efficiency_tradeoff_plot(summary_rows: List[Dict[str, Any]], out_path: str) -> None:
    if not summary_rows:
        return

    df = pd.DataFrame(summary_rows)
    x_col = "latency_mean"
    y_col = "accuracy_mean"
    type_col = "group_type"

    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        print("No valid data for efficiency trade-off plot.")
        return

    plot_df["model_family"] = plot_df.apply(
        lambda r: get_model_family(r["group_type"], r["group_model"]),
        axis=1
    )
    plot_df["plot_marker"] = plot_df.apply(marker_for_row, axis=1)
    plot_df["plot_size"] = plot_df.apply(marker_size_for_row, axis=1)

    fig, ax = plt.subplots(figsize=(18, 9))

    # Plot points
    for (run_type, model_family, plot_marker), sub in plot_df.groupby([type_col, "model_family", "plot_marker"]):
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=sub["plot_size"],
            alpha=0.75,
            marker=plot_marker,
            c=MODEL_COLORS.get(str(model_family), MODEL_COLORS["Other"]),
            edgecolors="black",
            linewidths=0.6,
        )

    ax.set_title("Efficiency Trade-off: Accuracy vs Latency")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.margins(x=0.05, y=0.06)

    # Leave room on the right for legends
    fig.subplots_adjust(right=0.78)

    # ---------------------------
    # Legend 1: Type (shape)
    # ---------------------------
    present_types = plot_df[type_col].dropna().unique()
    ordered_types = [t for t in TYPE_MARKERS.keys() if t in present_types]

    type_handles = []
    for t in ordered_types:
        marker = "^" if t == "student" else TYPE_MARKERS[t]
        type_handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="",
                color="w",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=10,
                label=t
            )
        )

    # ---------------------------
    # Legend 2: Model (colour)
    # ---------------------------
    present_models = plot_df["model_family"].dropna().unique()
    ordered_models = [m for m in MODEL_COLORS.keys() if m in present_models]

    model_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            color="w",
            markerfacecolor=MODEL_COLORS[m],
            markeredgecolor="black",
            markersize=10,
            label=m
        )
        for m in ordered_models
    ]

    # ---------------------------
    # Legend 3: Ensemble FS (circle size)
    # ---------------------------
    present_ensemble_fs = []
    if "fs" in plot_df.columns:
        ensemble_df = plot_df[plot_df["group_type"] == "ensemble"].copy()
        if not ensemble_df.empty:
            present_ensemble_fs = sorted(set(int(v) for v in ensemble_df["fs"].dropna().tolist()))

    ensemble_fs_handles = []
    for fs in [10, 100, 1000]:
        if fs in present_ensemble_fs:
            ensemble_fs_handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=(ENSEMBLE_FS_SIZES[fs] ** 0.5) / 1.6,
                    label=f"FS={fs}"
                )
            )

    # ---------------------------
    # Legend 4: Student FS (triangle orientation)
    # ---------------------------
    present_student_fs = []
    if "fs" in plot_df.columns:
        student_df = plot_df[plot_df["group_type"] == "student"].copy()
        if not student_df.empty:
            present_student_fs = sorted(set(int(v) for v in student_df["fs"].dropna().tolist()))

    student_fs_handles = []
    for fs in [10, 100, 1000]:
        if fs in present_student_fs:
            student_fs_handles.append(
                Line2D(
                    [0], [0],
                    marker=STUDENT_FS_MARKERS[fs],
                    linestyle="",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=10,
                    label=f"FS={fs}"
                )
            )

    legend_style = {
        "frameon": True,
        "facecolor": "white",
        "framealpha": 0.95,
        "edgecolor": "black",
        "borderpad": 0.6,
        "labelspacing": 0.4,
    }

    if type_handles:
        fig.legend(
            handles=type_handles,
            title="Type (Shape)",
            loc="upper left",
            bbox_to_anchor=(0.80, 0.88),
            **legend_style
        )

    if model_handles:
        fig.legend(
            handles=model_handles,
            title="Model (Colour)",
            loc="upper left",
            bbox_to_anchor=(0.80, 0.60),
            **legend_style
        )

    if ensemble_fs_handles:
        fig.legend(
            handles=ensemble_fs_handles,
            title="Ensemble FS (Circle size)",
            loc="upper left",
            bbox_to_anchor=(0.80, 0.40),
            **legend_style
        )

    if student_fs_handles:
        fig.legend(
            handles=student_fs_handles,
            title="Student FS (Triangle)",
            loc="upper left",
            bbox_to_anchor=(0.80, 0.18),
            **legend_style
        )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visual] Saved: {out_path}")

def save_teacher_baseline_tradeoff_plot(summary_rows: List[Dict[str, Any]], out_path: str) -> None:
    """
    Plot only zero-shot baselines, full-data baselines, and few-shot teacher runs.

    Visual encoding:
    - zero-shot = diamond
    - full = square
    - teacher FS=10 = down triangle
    - teacher FS=100 = right triangle
    - teacher FS=1000 = up triangle
    - colour = model family
    """
    if not summary_rows:
        return

    df = pd.DataFrame(summary_rows)
    x_col = "latency_mean"
    y_col = "accuracy_mean"

    plot_df = df[
        df["group_type"].isin(["zero-shot", "full", "teacher"])
    ].dropna(subset=[x_col, y_col]).copy()

    if plot_df.empty:
        print("No valid data for teacher/baseline trade-off plot.")
        return

    plot_df["model_family"] = plot_df.apply(
        lambda r: get_model_family(r["group_type"], r["group_model"]),
        axis=1
    )

    def teacher_baseline_marker_for_row(row: pd.Series) -> str:
        run_type = str(row.get("group_type", "unknown"))

        if run_type == "teacher":
            fs = row.get("fs")
            if pd.notna(fs):
                try:
                    return TEACHER_FS_MARKERS.get(int(fs), "X")
                except Exception:
                    return "X"
            return "X"

        return TYPE_MARKERS.get(run_type, "o")

    def teacher_baseline_size_for_row(row: pd.Series) -> int:
        run_type = str(row.get("group_type", "unknown"))
        if run_type == "zero-shot":
            return 150
        if run_type == "full":
            return 150
        if run_type == "teacher":
            return 135
        return 130

    plot_df["plot_marker"] = plot_df.apply(teacher_baseline_marker_for_row, axis=1)
    plot_df["plot_size"] = plot_df.apply(teacher_baseline_size_for_row, axis=1)

    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot grouped points
    for (run_type, model_family, plot_marker), sub in plot_df.groupby(
        ["group_type", "model_family", "plot_marker"]
    ):
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=sub["plot_size"],
            alpha=0.8,
            marker=plot_marker,
            c=MODEL_COLORS.get(str(model_family), MODEL_COLORS["Other"]),
            edgecolors="black",
            linewidths=0.7,
        )

    ax.set_title("Teacher Scarcity Trade-off: Zero-shot, Full, and Few-shot Teachers")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.margins(x=0.06, y=0.06)

    # Leave space on the right for legends
    fig.subplots_adjust(right=0.79)

    present_types = set(plot_df["group_type"].dropna().tolist())
    present_models = list(plot_df["model_family"].dropna().unique())

    # Legend 1: baseline type markers
    baseline_handles = []
    if "zero-shot" in present_types:
        baseline_handles.append(
            Line2D(
                [0], [0],
                marker=TYPE_MARKERS["zero-shot"],
                linestyle="",
                color="w",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=10,
                label="zero-shot"
            )
        )
    if "full" in present_types:
        baseline_handles.append(
            Line2D(
                [0], [0],
                marker=TYPE_MARKERS["full"],
                linestyle="",
                color="w",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=10,
                label="full"
            )
        )

    # Legend 2: model family colours
    ordered_models = [m for m in MODEL_COLORS.keys() if m in present_models]
    model_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            color="w",
            markerfacecolor=MODEL_COLORS[m],
            markeredgecolor="black",
            markersize=10,
            label=m
        )
        for m in ordered_models
    ]

    # Legend 3: teacher FS marker orientation
    teacher_fs_handles = []
    present_teacher_fs = []
    teacher_df = plot_df[plot_df["group_type"] == "teacher"].copy()
    if not teacher_df.empty and "fs" in teacher_df.columns:
        present_teacher_fs = sorted(set(int(v) for v in teacher_df["fs"].dropna().tolist()))

    for fs in [10, 100, 1000]:
        if fs in present_teacher_fs:
            teacher_fs_handles.append(
                Line2D(
                    [0], [0],
                    marker=TEACHER_FS_MARKERS[fs],
                    linestyle="",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=10,
                    label=f"FS={fs}"
                )
            )

    legend_style = {
        "frameon": True,
        "facecolor": "white",
        "framealpha": 0.95,
        "edgecolor": "black",
        "borderpad": 0.6,
        "labelspacing": 0.4,
    }

    if baseline_handles:
        fig.legend(
            handles=baseline_handles,
            title="Baseline Type",
            loc="upper left",
            bbox_to_anchor=(0.80, 0.86),
            **legend_style
        )

    if model_handles:
        fig.legend(
            handles=model_handles,
            title="Model (Colour)",
            loc="upper left",
            bbox_to_anchor=(0.80, 0.58),
            **legend_style
        )

    if teacher_fs_handles:
        fig.legend(
            handles=teacher_fs_handles,
            title="Teacher FS (Triangle)",
            loc="upper left",
            bbox_to_anchor=(0.80, 0.30),
            **legend_style
        )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visual] Saved: {out_path}")

def save_accuracy_ranking_plot(summary_rows: List[Dict[str, Any]], out_path: str, top_n: int = 25) -> None:
    if not summary_rows:
        return

    df = pd.DataFrame(summary_rows)
    df = df.dropna(subset=["accuracy_mean"]).copy()
    if df.empty:
        return

    df = df.sort_values("accuracy_mean", ascending=False).head(top_n).copy()
    df["label"] = df.apply(make_short_label, axis=1)

    plt.figure(figsize=(12, 8))
    plt.barh(df["label"], df["accuracy_mean"], xerr=df["accuracy_std"], alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Accuracy (Mean ± SD)")
    plt.ylabel("Configuration")
    plt.title(f"Top {top_n} Accuracy Ranking")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Visual] Saved: {out_path}")


def save_latency_ranking_plot(summary_rows: List[Dict[str, Any]], out_path: str, top_n: int = 25) -> None:
    if not summary_rows:
        return

    df = pd.DataFrame(summary_rows)
    df = df.dropna(subset=["latency_mean"]).copy()
    if df.empty:
        return

    df = df.sort_values("latency_mean", ascending=True).head(top_n).copy()
    df["label"] = df.apply(make_short_label, axis=1)

    plt.figure(figsize=(12, 8))
    plt.barh(df["label"], df["latency_mean"], xerr=df["latency_std"], alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Latency (ms, Mean ± SD)")
    plt.ylabel("Configuration")
    plt.title(f"Top {top_n} Lowest Latency Ranking")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Visual] Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--out", type=str, default="runs/comparison.csv")
    ap.add_argument("--out_summary", type=str, default="runs/comparison_summary.csv")
    ap.add_argument("--out_student_tables", type=str, default="runs/student_table_summary.csv")
    ap.add_argument("--student_tables_dir", type=str, default="runs/student_table_sections")
    ap.add_argument("--plots_dir", type=str, default="plots")
    args = ap.parse_args()

    rows = load_all_runs(args.runs_root)
    if not rows:
        print(f"No runs found in {args.runs_root}")
        return

    compute_retain_ratio(rows)
    rows = sort_rows(rows)
    save_csv(rows, args.out)

    summary_rows = build_summary(rows)
    save_csv(summary_rows, args.out_summary)
    save_student_table_csvs(summary_rows, args.out_student_tables, args.student_tables_dir)

    print(f"Detailed CSV:       {args.out}")
    print(f"Summary CSV:        {args.out_summary}")
    print(f"Student table CSV:  {args.out_student_tables}")

    os.makedirs(args.plots_dir, exist_ok=True)

    save_efficiency_tradeoff_plot(
        summary_rows,
        os.path.join(args.plots_dir, "efficiency_tradeoff.png"),
    )

    save_teacher_baseline_tradeoff_plot(
        summary_rows,
        os.path.join(args.plots_dir, "teacher_baseline_tradeoff.png"),
    )

    save_accuracy_ranking_plot(
        summary_rows,
        os.path.join(args.plots_dir, "accuracy_ranking.png"),
        top_n=25,
    )

    save_latency_ranking_plot(
        summary_rows,
        os.path.join(args.plots_dir, "latency_ranking.png"),
        top_n=25,
    )


if __name__ == "__main__":
    main()
