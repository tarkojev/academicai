"""
This file contains utility functions and classes for the AG News classification task, including:
- Dataset loading and processing for AG News- Hardware detection and random seed setting for reproducibility
- Metrics computation (accuracy, F1-macro) and stability analysis (label flipping rate)
- Model efficiency calculations (parameter count, estimated size in MB, latency benchmarking)
- Probability conversions and temperature scaling for knowledge distillation
- JSON saving utility for results and metadata
"""

# Imported libraries
import time
import json
import random
from typing import Dict, List, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Labels for AG News classes
AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Dataset class for AG News, with tokenization
class AGNewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }, torch.tensor(label, dtype=torch.long)

# Loading AG News dataset from Hugging Face Datasets library
def load_ag_news():
    ds = load_dataset("ag_news")
    return ds["train"], ds["test"]

# Convert to data loader
def make_loader(dataset, tokenizer, batch_size: int, max_len: int, shuffle: bool = False):
    if not isinstance(dataset, AGNewsDataset):
        texts = dataset["text"]
        labels = dataset["label"]
        ds = AGNewsDataset(texts, labels, tokenizer, max_len)
    else:
        ds = dataset
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# Few-Shot Support Set Sampling
def sample_few_shot_support_set(dataset, n_per_class: int, seed: int):
    """Samples N examples per class and returns BOTH the dataset and the indices."""
    random.seed(seed)
    np.random.seed(seed)
    
    all_indices = np.arange(len(dataset))
    labels = np.array(dataset["label"])
    
    support_indices = []
    for label_idx in range(4): # AG News has 4 classes
        label_indices = all_indices[labels == label_idx]
        selected = np.random.choice(label_indices, n_per_class, replace=False)
        support_indices.extend(selected.tolist()) # Convert to list for JSON compatibility
    
    random.shuffle(support_indices)
    return dataset.select(support_indices), support_indices

# Hardware handler
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Set random seeds
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

# Accuracy and F1-Macro 
def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1_macro": float(f1)}

# Stability analysis 
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

# Counting trainable parameters
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# Estimating model memory footprint in MB
def estimate_model_size_mb(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

# Inference latency benchmarking
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

# Softmaxing logits to get probabilities
def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# Applying temperature scaling for knowledge distillation
def apply_temperature_to_probs(probs: np.ndarray, tau: float) -> np.ndarray:
    logits = np.log(probs + 1e-12) / tau
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Saving results to JSON
def save_json(path: str, obj: dict) -> None:
    """Saves dictionary results to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
