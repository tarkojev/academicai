"""
This file trains a full supervised baseline on AG News. It contains:
- Standard classifier-head training for encoder classifiers such as BERT and DistilBERT.
- A custom T5 encoder-based classifier for direct multi-class classification.
- Prediction and evaluation on the test set.
- Efficiency benchmarking for parameter count, size, and inference latency.
- Results saved into JSON and NumPy artifacts for later analysis.
"""

# Imported libraries
import os
import argparse
import time
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5EncoderModel
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
    logits_to_probs,
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

# Building model and tokenizer based on the specified name
def build_model(name: str, num_labels: int = 4):
    tok = AutoTokenizer.from_pretrained(name)
    if is_t5(name):
        model = T5EncoderForClassification(name, num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return model, tok

# Training loop for classifier models
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
            out = model(**enc)
            logits = out.logits if hasattr(out, "logits") else out
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
        # If max_steps is reached in the middle of an accumulation cycle then apply remaining gradients
        if global_step % grad_accum_steps != 0:
            opt.step()
            opt.zero_grad()
        if 0 < max_steps <= global_step:
            break

# Predict logits for classifier models
@torch.no_grad()
def predict_logits(model, tokenizer, ds, device, batch_size=16, max_len=128):
    model.eval()
    model.to(device)
    loader = make_loader(ds, tokenizer, batch_size, max_len, shuffle=False)
    all_logits = []
    all_labels = []
    for enc, labels in loader:
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

# Main function to run the full dataset training and save results
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
    # Train
    t0 = time.perf_counter()
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

    # Predict on test set
    t1 = time.perf_counter()
    test_logits, test_labels = predict_logits(
        model, tok, test_ds, device, args.batch_size, args.max_len
    )
    test_inference_sec = float(time.perf_counter() - t1)

    # Metrics
    m = metrics_from_logits(test_logits, test_labels)
    print("Full baseline metrics:", m)

    # Efficiency metrics
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

    # Estimated total time to run inference across the full test set
    total_test_inference_est_sec = float(lat_ms * len(test_ds) / 1000.0)

    # Save run
    run_name = (
        f"full_{args.model}_seed{args.train_seed}_ep{args.epochs}_"
        f"lr{args.lr}_bs{args.batch_size}_ml{args.max_len}"
    ).replace("/", "_")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    test_probs = logits_to_probs(test_logits)
    np.save(os.path.join(run_dir, "test_logits.npy"), test_logits)
    np.save(os.path.join(run_dir, "test_probs.npy"), test_probs)
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
            "prob_interface": "classifier_head",
            "t5_label_texts": None,
            "data": {
                "train_set_size": len(train_ds),
                "test_set_size": len(test_ds),
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

if __name__ == "__main__":
    main()
