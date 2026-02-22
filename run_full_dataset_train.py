"""
This file trains a full supervised baseline on AG News. It contains:
- A training loop with cross-entropy loss and AdamW optimizer.
- Optional gradient accumulation and max_steps for quick debugging.
- A prediction function to get logits and labels on the test set.
- Saving of logits, labels, and metadata for later analysis.
"""

# Imported libraries
import os
import argparse
from typing import Tuple
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
)


# Custom T5 encoder-based classifier for student models
class T5EncoderForClassification(nn.Module):
    # Pooling and linear layer for classification
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

# Building model and tokenizer with T5 support.
def build_model_and_tokenizer(model_name: str, num_labels: int = 4):
    if model_name.startswith("t5-"):
        model = T5EncoderForClassification(model_name, num_labels=num_labels)
        tok = AutoTokenizer.from_pretrained(model_name)
        return model, tok
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tok = AutoTokenizer.from_pretrained(model_name)
    return model, tok

# Training loop
def train_full(
    model,
    tokenizer,
    train_ds,
    device: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
    max_len: int,
    grad_accum_steps: int = 1,
    max_steps: int = -1,
    log_every: int = 200,
):
    # Setting model to training mode
    model.to(device)
    model.train()

    loader = make_loader(tokenizer, train_ds, batch_size=batch_size, max_len=max_len, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0

        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            out = model(**enc)
            logits = out.logits if hasattr(out, "logits") else out

            loss = loss_fn(logits, labels) / grad_accum_steps
            loss.backward()

            if (global_step + 1) % grad_accum_steps == 0:
                opt.step()
                opt.zero_grad()

            running_loss += float(loss.item()) * grad_accum_steps
            n_batches += 1
            global_step += 1

            if log_every > 0 and (global_step % log_every == 0):
                avg = running_loss / max(1, n_batches)
                print(f"[epoch {epoch+1}/{epochs}] step={global_step} avg_loss={avg:.4f}")

            if max_steps > 0 and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}. Stopping training early.")
                return

# Prediction function to get logits and labels on the test set
@torch.no_grad()
def predict_logits(
    model,
    tokenizer,
    ds,
    device: torch.device,
    batch_size: int,
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Setting model to evaluation mode
    model.eval()
    model.to(device)
    loader = make_loader(tokenizer, ds, batch_size=batch_size, max_len=max_len, shuffle=False)
    all_logits = []
    all_labels = []

    # Looping through the test set and collecting logits and labels
    for enc, labels in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

# Main function to run the training and evaluation, and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs")

    # Model hyperparameter
    ap.add_argument("--model", type=str, default="distilbert-base-uncased")

    # Sample Seeding
    ap.add_argument("--train_seed", type=int, default=0)

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=2)          # full data often needs fewer epochs
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1, help="For debugging. -1 means no limit.")
    ap.add_argument("--log_every", type=int, default=200)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.train_seed)

    train_ds, test_ds = load_ag_news()

    model, tok = build_model_and_tokenizer(args.model, num_labels=4)

    print(f"Training FULL baseline: model={args.model} device={device}")
    print(f"Train size={len(train_ds)} Test size={len(test_ds)}")

    train_full(
        model=model,
        tokenizer=tok,
        train_ds=train_ds,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        log_every=args.log_every,
    )

    test_logits, test_labels = predict_logits(
        model=model,
        tokenizer=tok,
        ds=test_ds,
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )

    m = metrics_from_logits(test_logits, test_labels)
    print("Full baseline metrics:", m)

    run_name = f"full_{args.model}_seed{args.train_seed}_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}_ml{args.max_len}"
    run_name = run_name.replace("/", "_")
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
            "train_size": int(len(train_ds)),
            "test_size": int(len(test_ds)),
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "grad_accum_steps": args.grad_accum_steps,
            "max_steps": args.max_steps,
            "metrics": m,
            "device": str(device),
        },
    )


if __name__ == "__main__":
    main()