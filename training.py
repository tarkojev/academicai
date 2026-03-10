import os
import argparse
import time
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


# ---------------------------------------------------------
# Model helpers
# ---------------------------------------------------------

def is_t5(name: str) -> bool:
    return name.startswith("t5-")


class T5EncoderForClassification(nn.Module):
    """
    Simple encoder-only T5 classifier:
    - run T5 encoder
    - mean-pool token embeddings
    - pass pooled vector into linear classifier
    """
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


def build_model(name: str, num_labels: int = 4):
    """
    Build tokenizer + classification model.
    Uses a custom encoder-classifier for T5 and standard HF classifier heads
    for BERT / DistilBERT / similar encoder models.
    """
    tok = AutoTokenizer.from_pretrained(name)
    if is_t5(name):
        model = T5EncoderForClassification(name, num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=num_labels
        )
    return model, tok


# ---------------------------------------------------------
# Core training / inference
# ---------------------------------------------------------

def train_classifier(
    model,
    loader,
    device,
    epochs,
    lr,
    grad_accum_steps=1,
    max_steps=-1,
    log_every=50,
    train_seed=None,
):
    """
    Generic classifier training loop with CrossEntropy loss.
    Works for both:
    - single-model baseline training
    - few-shot teacher training
    """
    if train_seed is not None:
        set_seed(train_seed)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()

        for enc, labels in loader:
            global_step += 1

            # Move tensors to GPU / CPU device
            enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
            labels = labels.to(device)

            # Forward pass
            out = model(**enc)
            logits = out.logits if hasattr(out, "logits") else out

            # Standard CE loss
            loss = criterion(logits, labels) / grad_accum_steps
            loss.backward()

            # Optimizer step after gradient accumulation
            if global_step % grad_accum_steps == 0:
                opt.step()
                opt.zero_grad()

            # Periodic training log
            if global_step % log_every == 0:
                print(
                    f"Epoch {epoch} Step {global_step} "
                    f"Loss: {loss.item() * grad_accum_steps:.4f}"
                )

            # Optional hard limit on number of training steps
            if 0 < max_steps <= global_step:
                break

        # Flush remaining gradients if epoch ended mid accumulation cycle
        if global_step % grad_accum_steps != 0:
            opt.step()
            opt.zero_grad()

        if 0 < max_steps <= global_step:
            break


@torch.no_grad()
def predict_logits(model, tokenizer, ds, device, batch_size=16, max_len=128):
    """
    Run batched inference and return:
    - logits
    - ground truth labels
    """
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


def benchmark_model(model, tokenizer, test_ds, device, max_len, bench_texts, bench_warmup, bench_iters):
    """
    Collect simple efficiency statistics:
    - parameter count
    - estimated model size
    - average single-text latency
    """
    params = int(count_params(model))
    size_mb = float(estimate_model_size_mb(model))

    n_bench = min(bench_texts, len(test_ds))
    texts = [test_ds[i]["text"] for i in range(n_bench)]

    lat_ms = float(
        latency_ms(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_len=max_len,
            n_warmup=bench_warmup,
            n_iters=bench_iters,
        )
    )

    total_test_inference_est_sec = float(lat_ms * len(test_ds) / 1000.0)

    return {
        "params": params,
        "size_mb": size_mb,
        "latency_ms": lat_ms,
        "latency_bench_n": n_bench,
        "latency_mode": "classifier_forward",
        "bench_warmup": bench_warmup,
        "bench_iters": bench_iters,
        "total_test_inference_est_sec": total_test_inference_est_sec,
        "device": str(device),
    }


# ---------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------

def prepare_train_subset(train_ds, n_per_class=None, support_seed=123):
    """
    Select either:
    - full training set
    - few-shot support subset
    """
    if n_per_class is None:
        return train_ds, None, "full_supervision_baseline"

    support_ds, support_indices = sample_few_shot_support_set(
        train_ds,
        n_per_class,
        support_seed,
    )
    return support_ds, support_indices, "single_model_few_shot_baseline"


# ---------------------------------------------------------
# Baseline mode
# ---------------------------------------------------------

def run_baseline(args, device, train_ds, test_ds):
    """
    Baseline mode:
    - full-data single-model training
    - or few-shot single-model training if n_per_class is provided
    """
    train_subset, support_indices, run_type = prepare_train_subset(
        train_ds=train_ds,
        n_per_class=args.n_per_class,
        support_seed=args.support_seed,
    )

    model, tok = build_model(args.model, num_labels=4)

    loader = make_loader(
        train_subset,
        tok,
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=True,
    )

    # Train model
    t0 = time.perf_counter()
    train_classifier(
        model=model,
        loader=loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        log_every=args.log_every,
        train_seed=args.train_seed,
    )
    train_time_sec = float(time.perf_counter() - t0)

    # Evaluate on test set
    t1 = time.perf_counter()
    test_logits, test_labels = predict_logits(
        model=model,
        tokenizer=tok,
        ds=test_ds,
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    test_inference_sec = float(time.perf_counter() - t1)

    m = metrics_from_logits(test_logits, test_labels)
    print("Baseline metrics:", m)

    efficiency = benchmark_model(
        model=model,
        tokenizer=tok,
        test_ds=test_ds,
        device=device,
        max_len=args.max_len,
        bench_texts=args.bench_texts,
        bench_warmup=args.bench_warmup,
        bench_iters=args.bench_iters,
    )
    efficiency["train_time_sec"] = train_time_sec
    efficiency["test_inference_sec"] = test_inference_sec

    # Build run name
    if args.n_per_class is None:
        run_name = (
            f"full_{args.model}_seed{args.train_seed}_ep{args.epochs}_"
            f"lr{args.lr}_bs{args.batch_size}_ml{args.max_len}"
        )
    else:
        run_name = (
            f"single_{args.model}_fs{args.n_per_class}_supp{args.support_seed}_"
            f"seed{args.train_seed}_ep{args.epochs}_lr{args.lr}_"
            f"bs{args.batch_size}_ml{args.max_len}"
        )

    run_name = run_name.replace("/", "_")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save test predictions
    test_probs = logits_to_probs(test_logits)
    np.save(os.path.join(run_dir, "test_logits.npy"), test_logits)
    np.save(os.path.join(run_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(run_dir, "test_labels.npy"), test_labels)

    # Save metadata
    save_json(
        os.path.join(run_dir, "meta.json"),
        {
            "type": run_type,
            "model": args.model,
            "train_seed": args.train_seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "metrics": m,
            "data": {
                "train_set_size": int(len(train_ds)),
                "train_subset_size": int(len(train_subset)),
                "test_set_size": int(len(test_ds)),
                "support_set_size": int(len(train_subset)) if args.n_per_class is not None else None,
            },
            "few_shot": {
                "enabled": args.n_per_class is not None,
                "n_per_class": args.n_per_class,
                "support_seed": args.support_seed if args.n_per_class is not None else None,
                "support_indices": support_indices,
            },
            "efficiency": efficiency,
        },
    )


# ---------------------------------------------------------
# Teachers mode
# ---------------------------------------------------------

def run_teachers(args, device, train_ds, test_ds):
    """
    Teachers mode:
    - sample one support set
    - train multiple teacher models on it
    - save support and test predictions for ensemble / KD
    """
    if args.n_per_class is None:
        raise ValueError("Teachers mode requires --n_per_class.")

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

            run_dir = os.path.join(args.out_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            model, tok = build_model(model_name, num_labels=4)

            loader = make_loader(
                support_ds,
                tok,
                batch_size=args.batch_size,
                max_len=args.max_len,
                shuffle=True,
            )

            # Train teacher on support set
            t0 = time.perf_counter()
            train_classifier(
                model=model,
                loader=loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                grad_accum_steps=1,
                max_steps=-1,
                log_every=max(999999, args.log_every),
                train_seed=seed,
            )
            train_time_sec = float(time.perf_counter() - t0)

            # Predict on support and test for downstream ensemble / KD
            t1 = time.perf_counter()
            support_logits, support_labels = predict_logits(
                model=model,
                tokenizer=tok,
                ds=support_ds,
                device=device,
                batch_size=args.batch_size,
                max_len=args.max_len,
            )
            test_logits, test_labels = predict_logits(
                model=model,
                tokenizer=tok,
                ds=test_ds,
                device=device,
                batch_size=args.batch_size,
                max_len=args.max_len,
            )
            test_inference_sec = float(time.perf_counter() - t1)

            support_probs = logits_to_probs(support_logits)
            test_probs = logits_to_probs(test_logits)

            # Save arrays needed later by ensemble.py / student.py
            np.save(os.path.join(run_dir, "support_logits.npy"), support_logits)
            np.save(os.path.join(run_dir, "support_probs.npy"), support_probs)
            np.save(os.path.join(run_dir, "support_labels.npy"), support_labels)
            save_json(
                os.path.join(run_dir, "support_indices.json"),
                {"support_indices": support_indices},
            )

            np.save(os.path.join(run_dir, "test_logits.npy"), test_logits)
            np.save(os.path.join(run_dir, "test_probs.npy"), test_probs)
            np.save(os.path.join(run_dir, "test_labels.npy"), test_labels)

            m = metrics_from_logits(test_logits, test_labels)

            efficiency = benchmark_model(
                model=model,
                tokenizer=tok,
                test_ds=test_ds,
                device=device,
                max_len=args.max_len,
                bench_texts=args.bench_texts,
                bench_warmup=args.bench_warmup,
                bench_iters=args.bench_iters,
            )
            efficiency["train_time_sec"] = train_time_sec
            efficiency["test_inference_sec"] = test_inference_sec

            save_json(
                os.path.join(run_dir, "meta.json"),
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
                    "efficiency": efficiency,
                },
            )

            results.append(
                {
                    "run": run_name,
                    "model": model_name,
                    "train_seed": seed,
                    "support_seed": args.support_seed,
                    "n_per_class": args.n_per_class,
                    "metrics": m,
                    "accuracy": m["accuracy"],
                    "f1_macro": m["f1_macro"],
                    "latency_ms": efficiency["latency_ms"],
                    "params": efficiency["params"],
                    "size_mb": efficiency["size_mb"],
                    "train_time_sec": efficiency["train_time_sec"],
                    "test_inference_sec": efficiency["test_inference_sec"],
                }
            )

    # Save one summary index for all teacher runs
    index_name = f"teachers_index_fs{args.n_per_class}.json"

    save_json(
        os.path.join(args.out_dir, index_name),
        {
            "support_seed": args.support_seed,
            "n_per_class": args.n_per_class,
            "train_seeds": args.train_seeds,
            "models": args.models,
            "results": results,
        },
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    # Main mode switch:
    # baseline -> single model
    # teachers -> multiple teacher models + saved support predictions
    ap.add_argument("--mode", type=str, choices=["baseline", "teachers"], required=True)

    # Common args
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--support_seed", type=int, default=123)
    ap.add_argument("--n_per_class", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--bench_texts", type=int, default=50)
    ap.add_argument("--bench_warmup", type=int, default=10)
    ap.add_argument("--bench_iters", type=int, default=30)
    ap.add_argument("--log_every", type=int, default=100)

    # Baseline mode args
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--grad_accum_steps", type=int, default=1)

    # Teachers mode args
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["bert-base-uncased", "distilbert-base-uncased", "t5-small"],
    )
    ap.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2])

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    train_ds, test_ds = load_ag_news()

    if args.mode == "baseline":
        if args.model is None:
            raise ValueError("Baseline mode requires --model.")
        set_seed(args.train_seed)
        run_baseline(args, device, train_ds, test_ds)

    elif args.mode == "teachers":
        run_teachers(args, device, train_ds, test_ds)


if __name__ == "__main__":
    main()