"""
This file contains utility functions and classes for the AG News classification task, including:
- Data loading and few-shot sampling
- Dataset and DataLoader construction
- Metrics computation
- Model size and latency estimation
- JSON saving
"""

# Imported libraries
import time
import json
import random
from typing import Dict, List
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

#  Detect device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Set random seeds for reproducibility
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Load AG News dataset
def load_ag_news():
    ds = load_dataset("ag_news")
    return ds["train"], ds["test"]

# Sample a few-shot support set from the training data
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# Sample a few-shot support set from the training data
def estimate_model_size_mb(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

# Dataset class for AG News
def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1_macro": float(f1)}

# Compute label flipping rate across multiple runs
def label_flipping_rate(pred_matrix: np.ndarray) -> float:
    if pred_matrix.shape[0] < 2:
        return 0.0
    n_runs, n_samples = pred_matrix.shape
    flips = 0
    for i in range(n_samples):
        preds = pred_matrix[:, i]
        if not np.all(preds == preds[0]):
            flips += 1
    return flips / n_samples

# Latency measurement function for a model on a set of texts
def latency_ms(model, tokenizer, texts, device, max_len=128, n_warmup=10, n_iters=30):
    model.eval()
    def inference_run(t):
        inputs = tokenizer(t, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len).to(device)
        with torch.no_grad():
            model(**inputs)
    for i in range(min(n_warmup, len(texts))):
        inference_run(texts[i])
    times = []
    for i in range(n_iters):
        t = texts[i % len(texts)]
        start = time.perf_counter()
        inference_run(t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return float(np.mean(times))

# JSON saving utility
def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# Convert logits to probabilities
def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
