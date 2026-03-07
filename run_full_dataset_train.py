"""
This file trains a full supervised baseline on AG News. It contains:
- Standard classifier-head training for encoder classifiers such as BERT and DistilBERT.
- T5 full-supervision training using generative label-text targets and label-likelihood scoring.
- Prediction and evaluation on the test set.
- Efficiency benchmarking for parameter count, size, and inference latency.
- Results saved into JSON and NumPy artifacts for later analysis.
"""

# Imported libraries
import os
import argparse
import time
from typing import Tuple
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
import torch.nn.functional as F
from utils import (
    set_seed,
    load_ag_news,
    make_loader,
    metrics_from_logits,
    save_json,
    get_device,
    count_params,
    estimate_model_size_mb,
    latency_ms,
)

# T5 label likelihood
T5_LABEL_TEXTS = ["world", "sports", "business", "sci tech"]
def is_t5(name: str) -> bool:
    return name.startswith("t5-")

def labels_to_text(labels):
    return [T5_LABEL_TEXTS[int(y)] for y in labels]

def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

# Building model and tokenizer based on the specified name
def build_model(name: str, num_labels: int = 4):
    tok = AutoTokenizer.from_pretrained(name)
    if is_t5(name):
        model = T5ForConditionalGeneration.from_pretrained(name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return model, tok

# Training loop for classifier
def train_full(model, loader, device, epochs, lr, grad_accum_steps=1, max_steps=-1, log_every=50):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(epochs):
        model.train()
        for enc, labels in loader:
            global_step += 1
            # Move inputs and labels to device
            enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
            labels = labels.to(device)
            logits = model(**enc)
            if hasattr(logits, "logits"):
                logits = logits.logits
            # Compute loss, perform backpropagation, and update parameters with gradient accumulation
            loss = criterion(logits, labels) / grad_accum_steps
            loss.backward()
            if global_step % grad_accum_steps == 0:
                opt.step()
                opt.zero_grad()
            if global_step % log_every == 0:
                print(f"Epoch {epoch} Step {global_step} Loss: {loss.item() * grad_accum_steps:.4f}")
            if 0 < max_steps <= global_step:
                break
        # If max_steps is reached in the middle of an accumulation cycle then AdamW remaining gradients
        if global_step % grad_accum_steps != 0:
            opt.step()
            opt.zero_grad()
        if 0 < max_steps <= global_step:
            break

# Training loop for T5
def train_full_t5(model, tokenizer, loader, device, epochs, lr, grad_accum_steps=1, max_steps=-1, log_every=50, max_tgt_len=8):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    global_step = 0

    for epoch in range(epochs):
        model.train()
        for enc, labels in loader:
            global_step += 1
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            tgt_text = labels_to_text(labels.tolist())
            tgt = tokenizer(
                tgt_text,
                padding=True,
                truncation=True,
                max_length=max_tgt_len,
                return_tensors="pt",
            )
            labels_ids = tgt["input_ids"].to(device)
            labels_ids_masked = labels_ids.clone()
            labels_ids_masked[labels_ids_masked == tokenizer.pad_token_id] = -100

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_ids_masked,
            )
            loss = out.loss / grad_accum_steps
            loss.backward()

            if global_step % grad_accum_steps == 0:
                opt.step()
                opt.zero_grad()

            if global_step % log_every == 0:
                print(f"Epoch {epoch} Step {global_step} Loss: {loss.item() * grad_accum_steps:.4f}")

            if 0 < max_steps <= global_step:
                break

        if global_step % grad_accum_steps != 0:
            opt.step()
            opt.zero_grad()

        if 0 < max_steps <= global_step:
            break

# Predict logits for classifier
@torch.no_grad()
def predict_logits(model, tokenizer, ds, device, batch_size=16, max_len=128):
    model.eval()
    loader = make_loader(ds, tokenizer, batch_size, max_len, shuffle=False)
    all_logits = []
    all_labels = []
    for enc, labels in loader:
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        logits = model(**enc)
        if hasattr(logits, "logits"):
            logits = logits.logits
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

# Predict label probabilities for T5 via label-likelihood scoring
@torch.no_grad()
def predict_t5_label_probs(model, tokenizer, ds, device, batch_size=16, max_len=128, max_tgt_len=8):
    model.eval()
    loader = make_loader(ds, tokenizer, batch_size, max_len, shuffle=False)
    all_probs = []
    all_labels = []

    for enc, labels in loader:
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        scores = []
        for lab in T5_LABEL_TEXTS:
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

            logp = F.log_softmax(shift_logits, dim=-1)
            tgt_logp = logp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            pad_mask = (shift_labels != tokenizer.pad_token_id).float()
            sum_logp = (tgt_logp * pad_mask).sum(dim=1)
            len_tokens = pad_mask.sum(dim=1).clamp_min(1.0)
            avg_logp = sum_logp / len_tokens
            scores.append(avg_logp.detach().cpu().numpy())

        scores = np.stack(scores, axis=1)
        z = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(axis=1, keepdims=True)

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

# T5 latency benchmark
@torch.no_grad()
def benchmark_t5_label_scoring_latency_ms(model, tok, texts, device, max_len=128, n_warmup=5, n_iters=15):
    model.eval()
    model.to(device)

    def run_one(text: str):
        enc = tok(
            text,
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

# # Main function to run the full dataset training and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--bench_texts", type=int, default=50)
    ap.add_argument("--bench_warmup", type=int, default=10)
    ap.add_argument("--bench_iters", type=int, default=30)

    args = ap.parse_args()

    set_seed(args.train_seed)
    device = get_device()
    train_ds, test_ds = load_ag_news()
    model, tok = build_model(args.model)
    loader = make_loader(train_ds, tok, args.batch_size, args.max_len, shuffle=True)

    t0 = time.perf_counter()
    if is_t5(args.model):
        train_full_t5(
            model=model,
            tokenizer=tok,
            loader=loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            grad_accum_steps=args.grad_accum_steps,
            max_steps=args.max_steps,
            log_every=args.log_every,
        )
    else:
        train_full(
            model=model,
            loader=loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            grad_accum_steps=args.grad_accum_steps,
            max_steps=args.max_steps,
            log_every=args.log_every,
        )
    train_time_sec = float(time.perf_counter() - t0)
    t1 = time.perf_counter()
    if is_t5(args.model):
        test_probs, test_labels = predict_t5_label_probs(
            model, tok, test_ds, device, args.batch_size, args.max_len
        )
        test_logits = np.log(test_probs + 1e-12)
    else:
        test_logits, test_labels = predict_logits(
            model, tok, test_ds, device, args.batch_size, args.max_len
        )
    test_inference_sec = float(time.perf_counter() - t1)
    m = metrics_from_logits(test_logits, test_labels)
    print("Full baseline metrics:", m)
    params = int(count_params(model))
    size_mb = float(estimate_model_size_mb(model))
    n_bench = min(args.bench_texts, len(test_ds))
    bench_texts = [test_ds[i]["text"] for i in range(n_bench)]
    benchwarmup = min(args.bench_warmup, 10) if is_t5(args.model) else args.bench_warmup
    benchiter = min(args.bench_iters, 20) if is_t5(args.model) else args.bench_iters
    if is_t5(args.model):
        lat_ms = float(
            benchmark_t5_label_scoring_latency_ms(
                model=model,
                tok=tok,
                texts=bench_texts,
                device=device,
                max_len=args.max_len,
                n_warmup=benchwarmup,
                n_iters=benchiter,
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
                n_warmup=benchwarmup,
                n_iters=benchiter,
            )
        )
    # A held-out test set which model haven't seen and measures how long it takes for a single piece of news text to travel through the model to produce a prediction
    total_test_inference_est_sec = float(lat_ms * len(test_ds) / 1000.0)
    run_name = f"full_{args.model}_seed{args.train_seed}_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}_ml{args.max_len}".replace("/", "_")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "test_logits.npy"), test_logits)
    np.save(os.path.join(run_dir, "test_labels.npy"), test_labels)
    save_json(
        os.path.join(run_dir, "meta.json"),
        {
            "type": "full_supervision_baseline",
            "model": args.model,
            "train_seed": args.train_seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "metrics": m,
            "prob_interface": "label_likelihood" if is_t5(args.model) else "classifier_head",
            "t5_label_texts": T5_LABEL_TEXTS if is_t5(args.model) else None,
            "data": {
                "train_set_size": len(train_ds),
                "test_set_size": len(test_ds),
            },
            "efficiency": {
                "params": params,
                "size_mb": size_mb,
                "latency_ms": lat_ms,
                "latency_bench_n": n_bench,
                "latency_mode": "t5_label_scoring" if is_t5(args.model) else "classifier_forward",
                "bench_warmup": benchwarmup,
                "bench_iters": benchiter,
                "train_time_sec": train_time_sec,
                "test_inference_sec": test_inference_sec,
                "total_test_inference_est_sec": total_test_inference_est_sec,
                "device": str(device),
            },
        },
    )

if __name__ == "__main__":
    main()