"""
This file implements student training with Knowledge Distillation (KD) for the AG News classification task. It includes:
- Temperature-scaled softmax computation for teacher logits
- Construction of ensemble soft targets from multiple teacher runs
- Student training with combined loss: Cross-Entropy + KL-divergence (KD)
- Alignment between support samples and teacher soft targets
- Evaluation of the trained student on the held-out test set
- Saving run artifacts (test_logits.npy, test_labels.npy, meta.json) into per-run folders
- Writing a single index JSON with summaries of all teacher runs for later analysis
"""

# Imported libraries
import os
import argparse
from typing import List
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import (
    set_seed,
    load_ag_news,
    sample_few_shot_support_set,
    make_loader,
    metrics_from_logits,
    save_json,
)

# Function to compute temperature-scaled softmax probabilities from logits
def softmax_np(x: np.ndarray, tau: float) -> np.ndarray:
    # Formula: p_i = exp(logit_i / tau) / sum_j exp(logit_j / tau)
    z = x / tau
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# Loading teacher logits on the support set and constructing ensemble soft targets
@torch.no_grad()
def teacher_probs_on_support(teacher_run_dirs: List[str], tau: float) -> np.ndarray:
    # For each teacher run, load the support logits and compute softmax probabilities with temperature tau
    probs = []
    for d in teacher_run_dirs:
        p = os.path.join(d, "support_logits.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing support_logits.npy in teacher dir: {d}\n"
                f"Fix: run run_teachers.py after updating it to save support logits."
            )
        logits = np.load(p)
        probs.append(softmax_np(logits, tau))

    # Average the probabilities across teachers to get ensemble soft targets
    return np.mean(np.stack(probs, axis=0), axis=0)  # Support set size multiple by number of classes (N, C)

# Running inference with the student model to get logits and labels for the test set
@torch.no_grad()
def predict_logits(model, tokenizer, hf_ds, device, batch_size: int, max_len: int):
    model.eval()

    # shuffle=False ensures that the order of examples matches the order of soft targets during training
    loader = make_loader(tokenizer, hf_ds, batch_size=batch_size, max_len=max_len, shuffle=False)
    all_logits = []
    all_labels = []
    for enc, labels in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        all_logits.append(out.logits.detach().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits, 0), np.concatenate(all_labels, 0)

# Main function to run the student training with KD and save results
def main():
    ap = argparse.ArgumentParser()
    # Output directory for student runs
    ap.add_argument("--out_dir", type=str, default="runs")

    # Student model architecture
    ap.add_argument("--student_model", type=str, default="distilbert-base-uncased")

    # Few-shot support seed for sampling the same support set across all teacher runs
    ap.add_argument("--support_seed", type=int, default=123)
    
    # Number of examples per class in the few-shot support set
    ap.add_argument("--n_per_class", type=int, default=10)

    # Student training seed
    ap.add_argument("--train_seed", type=int, default=0)

    # List of teacher run directories to load soft targets from
    ap.add_argument("--teacher_dirs", type=str, nargs="+", required=True)

    # KD
    ap.add_argument("--tau", type=float, default=2.0)     # temperature
    ap.add_argument("--alpha", type=float, default=0.1)   # CE vs KD weighting

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    # Creating output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed
    set_seed(args.train_seed)

    # Loading AG News dataset and sampling few-shot support set
    train_ds, test_ds = load_ag_news()
    support_ds, support_indices = sample_few_shot_support_set(train_ds, args.n_per_class, args.support_seed)

    # Build student model and tokenizer
    tok = AutoTokenizer.from_pretrained(args.student_model)
    student = AutoModelForSequenceClassification.from_pretrained(
        args.student_model,
        num_labels=4,
    ).to(device)
    student.train()

    # Load ensemble soft targets on support set
    ptau = teacher_probs_on_support(args.teacher_dirs, tau=args.tau)

    # Ensure that the number of support examples matches the number of soft target rows
    expected_support = 4 * args.n_per_class
    if ptau.shape[0] != expected_support:
        raise ValueError(
            f"Teacher support probs size mismatch: got {ptau.shape[0]}, expected {expected_support}.\n"
            f"Check that teachers were trained with the same --support_seed and --n_per_class."
        )
    
    # Convert teacher probabilities to a PyTorch tensor for KD loss computation
    ptau_t = torch.tensor(ptau, dtype=torch.float32, device=device)

    # Set up DataLoader for the support set
    loader = make_loader(
        tok,
        support_ds,
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=False, # is False in order to align with ptau ordering
    )

    #  AdamW optimizer for student training and loss functions for CE and KD
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    # Training loop
    for _ in range(args.epochs):
        idx0 = 0 
        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            bsz = labels.size(0)
            logits = student(**enc).logits  # (B, C)

            # Cross-Entropy loss with hard labels
            l_ce = ce(logits, labels)

            # KD loss: KL( p_teacher || p_student )
            log_p_s = torch.log_softmax(logits / args.tau, dim=1)
            p_t = ptau_t[idx0:idx0 + bsz]
            l_kd = kl(log_p_s, p_t)

            # Combined loss
            loss = (
                args.alpha * l_ce
                + (1.0 - args.alpha) * (args.tau ** 2) * l_kd
            )

            # Backpropagation and optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()
            idx0 += bsz

    # Evaluate student on test set
    test_logits, test_labels = predict_logits(
        student,
        tok,
        test_ds,
        device,
        args.batch_size,
        args.max_len,
    )
    
    eval_metrics = metrics_from_logits(test_logits, test_labels)

    # Create run directory
    run_name = (
        f"student_{args.student_model}_fs{args.n_per_class}"
        f"_supp{args.support_seed}_seed{args.train_seed}"
        f"_tau{args.tau}_a{args.alpha}"
    ).replace("/", "_")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save test logits and labels as numpy arrays
    np.save(os.path.join(run_dir, "test_logits.npy"), test_logits)
    np.save(os.path.join(run_dir, "test_labels.npy"), test_labels)

    # Save Summary JSON 
    save_json(
        os.path.join(run_dir, "meta.json"),
        {
            "student_model": args.student_model,
            "teacher_dirs": args.teacher_dirs,
            "tau": args.tau,
            "alpha": args.alpha,
            "train_seed": args.train_seed,
            "support_seed": args.support_seed,
            "n_per_class": args.n_per_class,
            "support_indices": support_indices,
            "metrics": eval_metrics,
        },
    )
    print(eval_metrics)

if __name__ == "__main__":
    main()