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
    get_device
)

# Custom T5 encoder-based classifier
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
    if name.startswith("t5-"):
        model = T5EncoderForClassification(name, num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return model, tok

# Training loop for full supervised training
def train_full(model, loader, device, epochs, lr, grad_accum_steps=1, max_steps=-1, log_every=50):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            global_step += 1
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)
            
            logits = model(**inputs)
            if hasattr(logits, "logits"):
                logits = logits.logits
                
            loss = criterion(logits, labels) / grad_accum_steps
            loss.backward()

            if global_step % grad_accum_steps == 0:
                opt.step()
                opt.zero_grad()

            if global_step % log_every == 0:
                print(f"Epoch {epoch} Step {global_step} Loss: {loss.item()*grad_accum_steps:.4f}")
            
            if 0 < max_steps <= global_step:
                break
        
        if global_step % grad_accum_steps != 0:
            opt.step()
            opt.zero_grad()
            
        if 0 < max_steps <= global_step:
            break

# Prediction function to get logits and labels on the test set
@torch.no_grad()
def predict_logits(model, tokenizer, ds, device, batch_size=16, max_len=128):
    model.eval()
    loader = make_loader(ds, tokenizer, batch_size, max_len, shuffle=False)
    all_logits = []
    all_labels = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"]
        logits = model(**inputs)
        if hasattr(logits, "logits"):
            logits = logits.logits
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


# Main function to run the full supervised training and save results
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
    args = ap.parse_args()

    set_seed(args.train_seed)
    device = get_device()
    train_ds, test_ds = load_ag_news()
    model, tok = build_model(args.model)
    loader = make_loader(train_ds, tok, args.batch_size, args.max_len, shuffle=True)

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

    test_logits, test_labels = predict_logits(model, tok, test_ds, device, args.batch_size, args.max_len)
    m = metrics_from_logits(test_logits, test_labels)
    print("Full baseline metrics:", m)

    run_name = f"full_{args.model}_seed{args.train_seed}_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}_ml{args.max_len}".replace("/", "_")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    np.save(os.path.join(run_dir, "test_logits.npy"), test_logits)
    np.save(os.path.join(run_dir, "test_labels.npy"), test_labels)
    save_json(os.path.join(run_dir, "meta.json"), {
        "type": "full_supervision_baseline",
        "model": args.model,
        "train_seed": args.train_seed,
        "metrics": m
    })

if __name__ == "__main__":
    main()