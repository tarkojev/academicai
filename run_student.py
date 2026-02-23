"""
This file implements the student training script that distills knowledge from teacher models:
- It supports two modes of operation: distillation from multiple teacher runs or from an already-stored ensemble.
- It samples a few-shot support set from the training data (same as teachers) and gets soft targets.
- It trains a student model on the support set using CE and KL Divergence loss.
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
    apply_temperature_to_probs,
    logits_to_probs,
    get_device,
    count_params,
    estimate_model_size_mb,
    latency_ms
)

# Averaging teacher probabilities with temperature scaling to get ensemble soft targets
@torch.no_grad()
def ensemble_probs_on_support_from_teacher_dirs(teacher_run_dirs: List[str], tau: float) -> np.ndarray:
    probs = []
    for d in teacher_run_dirs:
        p_path = os.path.join(d, "support_probs.npy")
        if not os.path.exists(p_path):
            raise FileNotFoundError(f"Missing support_probs.npy in teacher dir: {d}")
        p = np.load(p_path)
        probs.append(apply_temperature_to_probs(p, tau))
    return np.mean(np.stack(probs, axis=0), axis=0)

# Prediction function to get logits and labels on the test set
@torch.no_grad()
def predict_logits(model, tokenizer, ds, device, batch_size=16, max_len=128):
    model.eval()
    loader = make_loader(ds, tokenizer, batch_size, max_len, shuffle=False)
    all_logits = []
    all_labels = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "text"]}
        labels = batch["label"]
        logits = model(**inputs).logits
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

# Main function to run the student training and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_model", type=str, required=True)
    ap.add_argument("--teacher_dirs", type=str, nargs="+", default=[])
    ap.add_argument("--ensemble_dir", type=str, default=None)
    ap.add_argument("--bench_warmup", type=int, default=5)
    ap.add_argument("--bench_iters", type=int, default=15)
    ap.add_argument("--n_per_class", type=int, default=10)
    ap.add_argument("--support_seed", type=int, default=123)
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    set_seed(args.train_seed)
    device = get_device()
    train_ds, test_ds = load_ag_news()
    support_ds, _ = sample_few_shot_support_set(train_ds, args.n_per_class, args.support_seed)

    if args.ensemble_dir:
        ensemble_probs = np.load(os.path.join(args.ensemble_dir, "support_probs.npy"))
        ensemble_labels = np.load(os.path.join(args.ensemble_dir, "support_labels.npy"))
        ptau_t = apply_temperature_to_probs(ensemble_probs, args.tau)
    else:
        ptau_t = ensemble_probs_on_support_from_teacher_dirs(args.teacher_dirs, args.tau)
        ensemble_labels = np.load(os.path.join(args.teacher_dirs[0], "support_labels.npy"))

    support_labels_from_ds = np.array([sample['label'] for sample in support_ds])
    if not np.array_equal(support_labels_from_ds, ensemble_labels):
        raise ValueError("Error: Support dataset order does not match saved ensemble labels.")

    tok = AutoTokenizer.from_pretrained(args.student_model)
    student = AutoModelForSequenceClassification.from_pretrained(args.student_model, num_labels=4).to(device)
    loader = make_loader(support_ds, tok, args.batch_size, args.max_len, shuffle=False)
    
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    # Configure training loop with CE + KD loss
    for epoch in range(args.epochs):
        student.train()
        idx0 = 0
        for batch in loader:
            bsz = batch["label"].size(0)
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "text"]}
            labels = batch["label"].to(device)
            p_t = torch.from_numpy(ptau_t[idx0:idx0 + bsz]).to(device)

            logits_s = student(**inputs).logits
            l_ce = ce_loss(logits_s, labels)
            
            logp_s = torch.log_softmax(logits_s / args.tau, dim=1)
            l_kd = kl_loss(logp_s, p_t)
            
            loss = args.alpha * l_ce + (1.0 - args.alpha) * (args.tau ** 2) * l_kd

            opt.zero_grad()
            loss.backward()
            opt.step()
            idx0 += bsz

    test_logits, test_labels = predict_logits(student, tok, test_ds, device, args.batch_size, args.max_len)
    eval_metrics = metrics_from_logits(test_logits, test_labels)

    # Efficiency benchmarking
    params = count_params(student)
    size_mb = estimate_model_size_mb(student)
    bench_texts = [test_ds[i]["text"] for i in range(min(50, len(test_ds)))]
    lat_ms = latency_ms(
        model=student,
        tokenizer=tok,
        texts=bench_texts,
        device=device,
        max_len=args.max_len,
        n_warmup=args.bench_warmup,
        n_iters=args.bench_iters,
    )
    total_test_inference_est_sec = float(lat_ms * len(test_ds) / 1000.0)
    run_name = f"student_{args.student_model}_fs{args.n_per_class}_supp{args.support_seed}_seed{args.train_seed}_tau{args.tau}_a{args.alpha}".replace("/", "_")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "test_logits.npy"), test_logits)
    np.save(os.path.join(run_dir, "test_labels.npy"), test_labels)
    save_json(os.path.join(run_dir, "meta.json"), {
        "student_model": args.student_model,
        "train_seed": args.train_seed,
        "support_seed": args.support_seed,
        "n_per_class": args.n_per_class,
        "tau": args.tau,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "metrics": eval_metrics,
        "efficiency": {
            "params": params,
            "size_mb": size_mb,
            "latency_ms": lat_ms,
            "total_test_inference_est_sec": total_test_inference_est_sec,
            "device": str(device)
        }
    })

if __name__ == "__main__":
    main()
