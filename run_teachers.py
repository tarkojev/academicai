"""
This file implements the training of teacher models for the AG News classification task. It includes:
- Building and training standard HuggingFace classification models
- Training T5 encoder-decoder models in seq2seq mode
- Saving trained teachers to disk
"""

# Imported libraries
import os
import argparse
from typing import List, Tuple, Dict, Any
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
)

# Use cleaner label texts for T5 decoding / scoring (avoid "Sci/Tech" slash token issues)
T5_LABEL_TEXTS = ["world", "sports", "business", "sci tech"]

# Check if model name corresponds to a T5 variant
def is_t5(name: str) -> bool:
    return name.startswith("t5-")

# Building teacher models
def build_teacher(model_name: str, num_labels: int = 4):
    tok = AutoTokenizer.from_pretrained(model_name)
    if is_t5(model_name):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return model, tok
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tok

# Converting labels to text
def prompt_texts(texts: List[str]) -> List[str]:
    return [f"classify: {t}" for t in texts]
def labels_to_text(labels: List[int]) -> List[str]:
    return [T5_LABEL_TEXTS[int(y)] for y in labels]

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
    loader = make_loader(tokenizer, support_ds, batch_size=batch_size, max_len=max_len, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for _ in range(epochs):
        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
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
    loader = make_loader(tokenizer, support_ds, batch_size=batch_size, max_len=max_len, shuffle=True)

    # Training loop
    for _ in range(epochs):
        for enc_in, labels in loader:
            input_ids = enc_in["input_ids"].to(device)
            attention_mask = enc_in["attention_mask"].to(device)

            # Convert numeric labels to target strings and tokenize
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

            # Avoid penalizing shorter sequences
            labels_ids_masked = labels_ids.clone()
            labels_ids_masked[labels_ids_masked == tokenizer.pad_token_id] = -100

            # Forward pass with teacher forcing and compute loss
            opt.zero_grad()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_ids_masked,
            )
            loss = out.loss
            loss.backward()
            opt.step()

# Predicting logits for standard classifiers
@torch.no_grad()
def predict_logits_classifier(model, tokenizer, hf_ds, device, batch_size: int, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    loader = make_loader(tokenizer, hf_ds, batch_size=batch_size, max_len=max_len, shuffle=False)
    all_logits = []
    all_labels = []
    for enc, labels in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = out.logits
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

# Predicting label probabilities for T5
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
    loader = make_loader(tokenizer, hf_ds, batch_size=batch_size, max_len=max_len, shuffle=False)
    all_probs = []
    all_labels = []

    # Loop through the dataset and compute label likelihoods for each example
    for enc_in, labels in loader:
        input_ids = enc_in["input_ids"].to(device)
        attention_mask = enc_in["attention_mask"].to(device)

        # Score each label
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

            # Teacher forcing forward
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=tgt_ids,
            )
            logits = out.logits 

            # Shift for token-level likelihood
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tgt_ids[:, 1:].contiguous()

            # Compute log probabilities of the target sequence and average over tokens
            logp = F.log_softmax(shift_logits, dim=-1)
            tgt_logp = logp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) 
            pad_mask = (shift_labels != tokenizer.pad_token_id).float()
            sum_logp = (tgt_logp * pad_mask).sum(dim=1)
            len_tokens = pad_mask.sum(dim=1).clamp_min(1.0)
            avg_logp = sum_logp / len_tokens
            scores.append(avg_logp.detach().cpu().numpy())

        # Convert scores to probabilities
        scores = np.stack(scores, axis=1)
        z = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(axis=1, keepdims=True)
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

# Main function to run the teacher training and save results
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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = load_ag_news()
    support_ds, support_indices = sample_few_shot_support_set(train_ds, args.n_per_class, args.support_seed)

    results = []

    # Loop through model architectures and training seeds to train multiple teacher runs
    for model_name in args.models:
        for seed in args.train_seeds:
            run_name = f"teacher_{model_name}_fs{args.n_per_class}_supp{args.support_seed}_seed{seed}".replace("/", "_")
            run_path = os.path.join(args.out_dir, run_name)
            os.makedirs(run_path, exist_ok=True)

            model, tok = build_teacher(model_name, num_labels=4)

            # Train models
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

            # Get predictions on support and test sets
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
            np.save(os.path.join(run_path, "support_probs.npy"), support_probs)
            np.save(os.path.join(run_path, "support_labels.npy"), support_labels)
            save_json(os.path.join(run_path, "support_indices.json"), {"support_indices": support_indices})
            np.save(os.path.join(run_path, "test_probs.npy"), test_probs)
            np.save(os.path.join(run_path, "test_labels.npy"), test_labels)
            np.save(os.path.join(run_path, "test_logits.npy"), test_logits)
            m = metrics_from_logits(test_logits, test_labels)

            # This teacher runs
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
                },
            )

            results.append({"run": run_name, "metrics": m})

    # All teacher runs
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