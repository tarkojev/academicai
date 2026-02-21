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

# 4 Classes in AG News, 4 labels (0..3)
AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Setting random seeds and deterministic behavior
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

# AG News dataset is from Hugging Face Datasets library
def load_ag_news():
    ds = load_dataset("ag_news")
    return ds["train"], ds["test"]

# Sampling a few-shot support set from the training dataset for few-shot supervised adaptation
def sample_few_shot_support_set(train_ds, n_per_class: int, support_seed: int):
    """
    This function samples a few-shot support set from the training dataset:
    - Select n_per_class examples for each class
    - Total support size = 4 * n_per_class
    - support_seed controls which examples are chosen
    """
    # Random number generator is used for support selection
    rng = np.random.default_rng(support_seed)

    # Group indices of training examples by class label
    by_class = {i: [] for i in range(4)}
    for idx, row in enumerate(train_ds):
        by_class[int(row["label"])].append(idx)

    # For each class, randomly choose n_per_class indices without replacement and sort them by index
    support_indices = []
    for c in range(4):
        idxs = by_class[c]
        chosen = rng.choice(idxs, size=n_per_class, replace=False)
        support_indices.extend(chosen.tolist())
    support_indices = sorted(support_indices)
    support = train_ds.select(support_indices)
    # Return HF dataset subset and list of indices used from the original train set
    return support, support_indices

# Dataset wrapper
class TextClsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        # List of raw text samples
        self.texts = texts

        # List of integer labels (0..3)
        self.labels = labels

    # Number of samples in the dataset
    def __len__(self):
        return len(self.texts)

    # Return sample
    def __getitem__(self, i):
        return {"text": self.texts[i], "label": int(self.labels[i])}

# Creating a DataLoader for the AG News dataset and a tokenizer
def make_loader(tokenizer, hf_ds, batch_size: int, max_len: int, shuffle: bool):
    # Extract raw texts and labels
    texts = [x["text"] for x in hf_ds]
    labels = [int(x["label"]) for x in hf_ds]
    ds = TextClsDataset(texts, labels)

    # Collate function is used to convert a batch of raw text samples into tokenized tensors and labels
    def collate(batch):
        texts_b = [x["text"] for x in batch]
        labels_b = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tokenizer(
            texts_b,                # batch of raw text strings
            padding=True,           # pad to the longest sequence in the batch
            truncation=True,        # truncate sequences longer than max_len
            max_length=max_len,     # maximum sequence length
            return_tensors="pt",    # return PyTorch tensors
        )
        return enc, labels_b
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

# Computing accuracy and macro F1 score from model logits and true labels
def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    # Highest logit score corresponds to predicted class label
    preds = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }

# Computing label flipping rate across multiple runs of the same samples
def label_flipping_rate(predicted_matrix: np.ndarray) -> float:
    # 2d array that contains predicted labels for each sample across multiple runs
    n_runs, n_samples = predicted_matrix.shape
    # If there are less than 2 runs or no samples, flipping rate is 0 by definition
    if n_runs < 2 or n_samples == 0:
        return 0.0
    flips = 0

    # For each sample, check how many unique predicted labels exist across runs
    # If more than 1 unique label -> it flips at least once
    for j in range(n_samples):
        if len(np.unique(predicted_matrix[:, j])) > 1:
            flips += 1
    
    # Returning fraction of samples that flip at least once
    return float(flips / n_samples)

# Counting total number of parameters in a model
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# Estimating model size in MB based on parameter storage
def estimate_model_size_mb(model: nn.Module) -> float:
    # Total bytes = number of parameters * bytes per parameter
    bytes_total = 0
    for p in model.parameters():
        bytes_total += p.numel() * p.element_size()

    # Converting bytes to MB by dividing by 1024^2
    return float(bytes_total / (1024 ** 2))

# no_grad measures latency of a model on a list of texts by running inference multiple times and averaging the time taken
@torch.no_grad()
def latency_ms(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_len: int,
    n_warmup: int = 10,
    n_iters: int = 30
) -> float:
    # Setting model to evaluation mode
    model.eval()
    model.to(device)

    # Run inference on a single text sample
    def inference_run(t: str):
        enc = tokenizer(t, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        _ = model(**enc)

    # Warmup ensures setup time is not included in latency measurement
    for i in range(min(n_warmup, len(texts))):
        inference_run(texts[i])

    # Running inference on multiple samples and averaging the time taken
    times = []
    for i in range(n_iters):
        t = texts[i % len(texts)]
        start = time.perf_counter()
        inference_run(t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)  # seconds -> ms

    # Return average latency
    return float(np.mean(times))

# Save output dictionary to a JSON file
def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
