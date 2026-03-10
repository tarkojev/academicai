"""
This file implements the teacher training script:
- It trains multiple teacher models (e.g., BERT, DistilBERT, T5) on a few-shot support set from AG News.
- It saves the trained models' predictions (logits and probabilities) on both the support set and the test set for later use in student distillation.
- It benchmarks the efficiency of each teacher model in terms of parameter count, model size, and inference latency.
- All results and artifacts are saved into structured JSON files and NumPy arrays for later analysis and reporting.
"""

# Imported libraries
import os
import argparse
import time
import json
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5EncoderModel,
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

def is_t5(name: str) -> bool:
    return name.startswith("t5-")

class T5EncoderForClassification(nn.Module):
    def __init__(self, name: str, num_labels: int):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(name)
        hidden = self.encoder.config.d_model
        self.classifier = nn.Linear(hidden, num_labels)
    def forward(self, input_ids=None, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(x)
        return logits

# Build teacher model and tokenizer
def build_teacher(model_name: str, num_labels: int = 4):
    tok = AutoTokenizer.from_pretrained(model_name)
    if is_t5(model_name):
        model = T5EncoderForClassification(model_name, num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tok

# Training loop for classifier teachers
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
    loader = make_loader(
        support_ds,
        tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
            labels = labels.to(device)
            opt.zero_grad()
            out = model(**enc)
            logits = out.logits if hasattr(out, "logits") else out
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()

# Predict logits for classifier teachers
@torch.no_grad()
def predict_logits_classifier(model, tokenizer, hf_ds, device, batch_size: int, max_len: int):
    model.eval()
    model.to(device)
    loader = make_loader(
        hf_ds,
        tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=False,
    )
    all_logits = []
    all_labels = []
    for enc, labels in loader:
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

# Main function to run teacher training and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--support_seed", type=int, default=123)
    ap.add_argument("--n_per_class", type=int, default=10)
    ap.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["bert-base-uncased", "distilbert-base-uncased", "t5-small"],
    )
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
    support_ds, support_indices = sample_few_shot_support_set(
        train_ds,
        args.n_per_class,
        args.support_seed,
    )
    results = []
    for model_name in args.models:
        for seed in args.train_seeds:
            run_name = (
                f"teacher_{model_name}_fs{args.n_per_class}_supp{args.support_seed}_seed{seed}"
            ).replace("/", "_")
            run_path = os.path.join(args.out_dir, run_name)
            os.makedirs(run_path, exist_ok=True)
            model, tok = build_teacher(model_name, num_labels=4)

            # Train
            t0 = time.perf_counter()
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

            # Predict on support and test
            t1 = time.perf_counter()
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
            test_inference_sec = float(time.perf_counter() - t1)
            support_probs = logits_to_probs(support_logits)
            test_probs = logits_to_probs(test_logits)

            # Save artifacts
            np.save(os.path.join(run_path, "support_logits.npy"), support_logits)
            np.save(os.path.join(run_path, "support_probs.npy"), support_probs)
            np.save(os.path.join(run_path, "support_labels.npy"), support_labels)
            save_json(
                os.path.join(run_path, "support_indices.json"),
                {"support_indices": support_indices},
            )
            np.save(os.path.join(run_path, "test_logits.npy"), test_logits)
            np.save(os.path.join(run_path, "test_probs.npy"), test_probs)
            np.save(os.path.join(run_path, "test_labels.npy"), test_labels)

            # Metrics
            m = metrics_from_logits(test_logits, test_labels)

            # Efficiency
            params = int(count_params(model))
            size_mb = float(estimate_model_size_mb(model))
            n_bench = min(args.bench_texts, len(test_ds))
            bench_texts = [test_ds[i]["text"] for i in range(n_bench)]
            benchwarmup = args.bench_warmup
            benchiter = args.bench_iters
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
            total_test_inference_est_sec = float(lat_ms * len(test_ds) / 1000.0)

            # Save metadata
            save_json(
                os.path.join(run_path, "meta.json"),
                {
                    "type": "few_shot_teacher",
                    "model": model_name,
                    "train_seed": seed,
                    "support_seed": args.support_seed,
                    "n_per_class": args.n_per_class,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len,
                    "prob_interface": "classifier_head",
                    "label_space": AG_LABELS,
                    "t5_label_texts": None,
                    "metrics": m,
                    "data": {
                        "train_set_size": len(train_ds),
                        "test_set_size": len(test_ds),
                        "support_set_size": len(support_ds),
                    },
                    "efficiency": {
                        "params": params,
                        "size_mb": size_mb,
                        "latency_ms": lat_ms,
                        "latency_bench_n": n_bench,
                        "latency_mode": "classifier_forward",
                        "bench_warmup": benchwarmup,
                        "bench_iters": benchiter,
                        "train_time_sec": train_time_sec,
                        "test_inference_sec": test_inference_sec,
                        "total_test_inference_est_sec": total_test_inference_est_sec,
                        "device": str(device),
                    },
                },
            )
            results.append(
                {
                    "run": run_name,
                    "metrics": m,
                }
            )

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
