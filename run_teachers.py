"""
This file implements the teacher training script:
- It trains multiple teacher models (e.g., BERT, DistilBERT, T5) on a few-shot support set from AG News.
- It saves the trained models' predictions (logits and probabilities) on both the support set and the test set for later use in student distillation.
- It benchmarks the efficiency of each teacher model in terms of parameter count, model size, and inference latency (with a special T5 label scoring benchmark).
- All results and artifacts are saved into structured JSON files and NumPy arrays for later analysis and reporting.
"""

# Imported libraries
import os
import argparse
import time
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
)
from utils import (
    set_seed,
    load_ag_news,
    sample_few_shot_support_set,
    make_loader,
    metrics_from_logits,
    logits_to_probs,
    save_json,
    AG_LABELS,
    get_device,
    count_params,
    estimate_model_size_mb,
    latency_ms,
)

# Label texts for T5 scoring
T5_LABEL_TEXTS = ["world", "sports", "business", "sci tech"]

# T5 naming helper
def is_t5(name: str) -> bool:
    return name.startswith("t5-")

# Build teacher model and tokenizer
def build_teacher(model_name: str, num_labels: int = 4):
    tok = AutoTokenizer.from_pretrained(model_name)
    if is_t5(model_name):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return model, tok
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tok

def prompt_texts(texts: List[str]) -> List[str]:
    return [f"classify: {t}" for t in texts]

def labels_to_text(labels: List[int]) -> List[str]:
    return [T5_LABEL_TEXTS[int(y)] for y in labels]

# Sync device for timing
def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

# Training loop for standard classifier teachers
def train_one_classifier(
    model,
    tokenizer,
    support_ds,
    train_seed: int,
    device,
    lr: float,
    epochs: int,
    batch_size: int,
    max_len: int,
):
    set_seed(train_seed)
    model.to(device)
    model.train()
    loader = make_loader(support_ds, tokenizer, batch_size=batch_size, max_len=max_len, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
            labels = labels.to(device)
            opt.zero_grad()
            out = model(**enc)
            logits = out.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()

# T5 seq2seq training loop
def train_one_t5(
    model: T5ForConditionalGeneration,
    tokenizer,
    support_ds,
    train_seed: int,
    device,
    lr: float,
    epochs: int,
    batch_size: int,
    max_len: int,
    max_tgt_len: int = 8,
):
    set_seed(train_seed)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = make_loader(support_ds, tokenizer, batch_size=batch_size, max_len=max_len, shuffle=True)
    for _ in range(epochs):
        for enc_in, labels in loader:
            input_ids = enc_in["input_ids"].to(device)
            attention_mask = enc_in["attention_mask"].to(device)

            y = labels.tolist()
            tgt_text = labels_to_text(y)
            tgt = tokenizer(
                tgt_text,
                padding=True,
                truncation=True,
                max_length=max_tgt_len,
                return_tensors="pt",
            )
            labels_ids = tgt["input_ids"].to(device)
            # Mask padding tokens in labels for T5 loss
            labels_ids_masked = labels_ids.clone()
            labels_ids_masked[labels_ids_masked == tokenizer.pad_token_id] = -100
            opt.zero_grad()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_ids_masked,
            )
            loss = out.loss
            loss.backward()
            opt.step()

# Ensemble inference latency benchmark
@torch.no_grad()
def predict_logits_classifier(model, tokenizer, hf_ds, device, batch_size: int, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    loader = make_loader(hf_ds, tokenizer, batch_size=batch_size, max_len=max_len, shuffle=False)
    all_logits = []
    all_labels = []
    for enc, labels in loader:
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        out = model(**enc)
        logits = out.logits
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

# T5 label scoring latency benchmark (ms/sample)
@torch.no_grad()
def t5_label_probs(
    model: T5ForConditionalGeneration,
    tokenizer,
    hf_ds,
    device,
    batch_size: int,
    max_len: int,
    label_texts: List[str],
    max_tgt_len: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    loader = make_loader(hf_ds, tokenizer, batch_size=batch_size, max_len=max_len, shuffle=False)
    all_probs = []
    all_labels = []
    for enc_in, labels in loader:
        input_ids = enc_in["input_ids"].to(device)
        attention_mask = enc_in["attention_mask"].to(device)
        scores = []
        for lab in label_texts:
            tgt = tokenizer(
                [lab] * labels.size(0),
                padding=True,
                truncation=True,
                max_length=max_tgt_len,
                return_tensors="pt",
            )
            tgt_ids = tgt["input_ids"].to(device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=tgt_ids,
            )
            logits = out.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tgt_ids[:, 1:].contiguous()
            # Compute average log-probability of the target label text under the model's output distribution
            logp = F.log_softmax(shift_logits, dim=-1)
            tgt_logp = logp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            pad_mask = (shift_labels != tokenizer.pad_token_id).float()
            sum_logp = (tgt_logp * pad_mask).sum(dim=1)
            len_tokens = pad_mask.sum(dim=1).clamp_min(1.0)
            avg_logp = sum_logp / len_tokens
            scores.append(avg_logp.detach().cpu().numpy())
        # scores shape: (batch_size, num_labels)
        scores = np.stack(scores, axis=1)
        z = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(axis=1, keepdims=True)
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

# Latecy benchmark for T5 label scoring (ms/sample)
@torch.no_grad()
def benchmark_t5_label_scoring_latency_ms(model, tok, texts, device, max_len=128, n_warmup=5, n_iters=15):
    model.eval()
    model.to(device)
    def run_one(text: str):
        enc = tok(
            f"classify: {text}",
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        for lab in T5_LABEL_TEXTS:
            tgt = tok(lab, return_tensors="pt").input_ids.to(device)
            _ = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=tgt)
    warm_n = min(n_warmup, len(texts))
    for i in range(warm_n):
        run_one(texts[i])
        _sync_device(device)
    times = []
    for i in range(n_iters):
        t = texts[i % len(texts)]
        start = time.perf_counter()
        run_one(t)
        _sync_device(device)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return float(np.mean(times))

# Main function to run teacher training and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--support_seed", type=int, default=123)
    ap.add_argument("--n_per_class", type=int, default=10)
    ap.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", type=str, nargs="+",
                    default=["bert-base-uncased", "distilbert-base-uncased", "t5-small"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--bench_texts", type=int, default=50)
    ap.add_argument("--bench_warmup", type=int, default=10)
    ap.add_argument("--bench_iters", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    train_ds, test_ds = load_ag_news()
    support_ds, support_indices = sample_few_shot_support_set(train_ds, args.n_per_class, args.support_seed)
    results = []
    for model_name in args.models:
        for seed in args.train_seeds:
            run_name = f"teacher_{model_name}_fs{args.n_per_class}_supp{args.support_seed}_seed{seed}".replace("/", "_")
            run_path = os.path.join(args.out_dir, run_name)
            os.makedirs(run_path, exist_ok=True)
            model, tok = build_teacher(model_name, num_labels=4)
            t0 = time.perf_counter()
            if is_t5(model_name):
                train_one_t5(
                    model=model,
                    tokenizer=tok,
                    support_ds=support_ds,
                    train_seed=seed,
                    device=device,
                    lr=args.lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                )
            else:
                train_one_classifier(
                    model=model,
                    tokenizer=tok,
                    support_ds=support_ds,
                    train_seed=seed,
                    device=device,
                    lr=args.lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                )
            train_time_sec = float(time.perf_counter() - t0)
            t1 = time.perf_counter()
            # If T5, get label scoring probs; if classifier, get logits and convert to probs
            if is_t5(model_name):
                support_probs, support_labels = t5_label_probs(
                    model=model,
                    tokenizer=tok,
                    hf_ds=support_ds,
                    device=device,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    label_texts=T5_LABEL_TEXTS,
                )
                test_probs, test_labels = t5_label_probs(
                    model=model,
                    tokenizer=tok,
                    hf_ds=test_ds,
                    device=device,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    label_texts=T5_LABEL_TEXTS,
                )
                test_logits = np.log(test_probs + 1e-12)
            else:
                support_logits, support_labels = predict_logits_classifier(
                    model=model,
                    tokenizer=tok,
                    hf_ds=support_ds,
                    device=device,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                )
                test_logits, test_labels = predict_logits_classifier(
                    model=model,
                    tokenizer=tok,
                    hf_ds=test_ds,
                    device=device,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                )
                support_probs = logits_to_probs(support_logits)
                test_probs = logits_to_probs(test_logits)
                np.save(os.path.join(run_path, "support_logits.npy"), support_logits)
            test_inference_sec = float(time.perf_counter() - t1)
            np.save(os.path.join(run_path, "support_probs.npy"), support_probs)
            np.save(os.path.join(run_path, "support_labels.npy"), support_labels)
            save_json(os.path.join(run_path, "support_indices.json"), {"support_indices": support_indices})

            np.save(os.path.join(run_path, "test_probs.npy"), test_probs)
            np.save(os.path.join(run_path, "test_labels.npy"), test_labels)
            np.save(os.path.join(run_path, "test_logits.npy"), test_logits)

            m = metrics_from_logits(test_logits, test_labels)

            # Count parameters and estimate model size
            params = int(count_params(model))
            size_mb = float(estimate_model_size_mb(model))
            n_bench = min(args.bench_texts, len(test_ds))
            bench_texts = [test_ds[i]["text"] for i in range(n_bench)]

            # If T5, run the T5 label scoring latency benchmark; otherwise, run the standard classifier forward pass latency benchmark
            if is_t5(model_name):
                lat_ms = float(
                    benchmark_t5_label_scoring_latency_ms(
                        model=model,
                        tok=tok,
                        texts=bench_texts,
                        device=device,
                        max_len=args.max_len,
                        n_warmup=min(args.bench_warmup, 10),  # T5 is slower; don't over-warmup
                        n_iters=min(args.bench_iters, 20),
                    )
                )
            else:
                lat_ms = float(
                    latency_ms(
                        model=model,
                        tokenizer=tok,
                        texts=bench_texts,
                        device=device,
                        max_len=args.max_len,
                        n_warmup=args.bench_warmup,
                        n_iters=args.bench_iters,
                    )
                )
            # Total inference time where latency is multiplied by the number of samples in the test set and converted from ms to seconds
            total_test_inference_est_sec = float(lat_ms * len(test_ds) / 1000.0)

            # Current run summary
            save_json(
                os.path.join(run_path, "meta.json"),
                {
                    "model": model_name,
                    "train_seed": seed,
                    "support_seed": args.support_seed,
                    "n_per_class": args.n_per_class,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len,
                    "prob_interface": "label_likelihood" if is_t5(model_name) else "classifier_head",
                    "label_space": AG_LABELS,
                    "t5_label_texts": T5_LABEL_TEXTS if is_t5(model_name) else None,
                    "metrics": m,
                    "efficiency": {
                        "params": params,
                        "size_mb": size_mb,
                        "latency_ms": lat_ms,
                        "latency_bench_n": n_bench,
                        "bench_warmup": args.bench_warmup,
                        "bench_iters": args.bench_iters,
                        "train_time_sec": train_time_sec,
                        "test_inference_sec": test_inference_sec,
                        "total_test_inference_est_sec": total_test_inference_est_sec,
                        "device": str(device),
                        "latency_mode": "t5_label_scoring" if is_t5(model_name) else "classifier_forward",
                    },
                },
            )
            results.append({"run": run_name, "metrics": m})

    # All runs summary
    save_json(
        os.path.join(args.out_dir, "teachers_index.json"),
        {
            "support_seed": args.support_seed,
            "n_per_class": args.n_per_class,
            "train_seeds": args.train_seeds,
            "models": args.models,
            "results": results,
        },
    )

if __name__ == "__main__":
    main()