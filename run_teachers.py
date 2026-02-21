"""
This file implements the training of teacher models for the AG News classification task. It includes:
- Definition of a T5 encoder-based classifier (encoder + pooling + linear head)
- Building either a standard HuggingFace classification model or the custom T5 classifier
- Training loop for fine-tuning teacher models on a few-shot support set (Cross-Entropy loss)
- Evaluation of trained teachers on the held-out test set (logits + labels)
- Saving run artifacts into per-run folders
- Writing a single index JSON with summaries of all teacher runs for later analysis
"""

# Imported libraries
import os
import argparse
from typing import Dict, Any
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5EncoderModel
from utils import (
    set_seed,
    load_ag_news,
    sample_few_shot_support_set,
    make_loader,
    metrics_from_logits,
    save_json,
)

# Custom T5 encoder-based classifier for AG News classification
class T5EncoderForClassification(nn.Module):
    """
    Minimal classifier built on top of a T5 encoder:
    - Uses the encoder output hidden states (B, T, H: where B=batch size, T=sequence length, H=hidden size)
    - Applies mean pooling over the token dimension using the attention mask
    - Feeds pooled vector into a linear classifier head for 4-way classification
    """

    # Initialization loads the T5 encoder and sets up the classification head
    def __init__(self, name: str, num_labels: int):
        super().__init__()
        # Loading the T5 encoder model without the decoder
        self.encoder = T5EncoderModel.from_pretrained(name)
        hidden = self.encoder.config.d_model
        self.classifier = nn.Linear(hidden, num_labels)

    # Forward pass takes input_ids and attention_mask, returns logits for classification
    def forward(self, input_ids=None, attention_mask=None):
        # Pass inputs through the T5 encoder
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        # Calculate mean pooling over the token dimension, accounting for padding with the mask
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(x)
        return logits

# Build a teacher model and tokenizer based on the specified model name
def build_model(model_name: str, num_labels: int):
    # If the model name starts with "t5-", use custom T5 encoder-based classifier
    if model_name.startswith("t5-"):
        model = T5EncoderForClassification(model_name, num_labels)
        tok = AutoTokenizer.from_pretrained(model_name)
        return model, tok
    # Otherwise, use HuggingFace sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tok = AutoTokenizer.from_pretrained(model_name)
    return model, tok

# Train one teacher model on the few-shot support set with the specified training seed and hyperparameters
def train_one(
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
    # Set random seed and enable training mode
    set_seed(train_seed)
    model.to(device)
    model.train()

    # Load support set into a DataLoader for training
    loader = make_loader(tokenizer, support_ds, batch_size=batch_size, max_len=max_len, shuffle=True)

    # Use AdamW optimizer and Cross-Entropy loss for classification
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop over epochs and batches
    for _ in range(epochs):
        # Batch loop
        for enc, labels in loader:
            # Move batch to device
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            # Reset gradients
            opt.zero_grad()
            # Output logits from the model forward pass
            out = model(**enc)
            # Extract logits from the model output
            logits = out.logits if hasattr(out, "logits") else out

            # Compute loss between predicted logits and true labels
            loss = loss_fn(logits, labels)

            # Enable backpropagation and update model parameters
            loss.backward()
            opt.step()

# Run inference on the test set and return logits and labels as numpy arrays
@torch.no_grad()
def predict_logits(model, tokenizer, hf_ds, device, batch_size: int, max_len: int):
    # Enable evaluation mode
    model.eval()
    model.to(device)

    # Load test set into a DataLoader for inference testing
    loader = make_loader(tokenizer, hf_ds, batch_size=batch_size, max_len=max_len, shuffle=False)
    all_logits = []
    all_labels = []
    # Inference loop over batches
    for enc, labels in loader:
        enc = {k: v.to(device) for k, v in enc.items()}

        # Forward pass
        out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out

        # Store logits and labels on CPU as numpy arrays
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.numpy())

    # Concatenate all batch logits and labels into single numpy arrays for the entire test set
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

# Main function to run the teacher training and evaluation across multiple models and seeds, and save results
def main():
    ap = argparse.ArgumentParser()
    # Output location for all run folders
    ap.add_argument("--out_dir", type=str, default="runs")

    # Few-shot support seed for sampling the same support set across all teacher runs
    ap.add_argument("--support_seed", type=int, default=123)
    
    # Number of examples per class in the few-shot support set
    ap.add_argument("--n_per_class", type=int, default=10)

    # Training seeds
    ap.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2])

    # Supported models
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["bert-base-uncased", "distilbert-base-uncased", "t5-small"],
    )

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

    # Load dataset and construct few-shot support set
    train_ds, test_ds = load_ag_news()
    support_ds, support_indices = sample_few_shot_support_set(train_ds, args.n_per_class, args.support_seed)

    # Results list
    results = []

    # Train/evaluate each teacher model across seeds
    for model_name in args.models:
        for seed in args.train_seeds:
            # Building a unique run name and path for this teacher model and seed
            run_name = f"teacher_{model_name}_fs{args.n_per_class}_supp{args.support_seed}_seed{seed}".replace("/", "_")
            run_path = os.path.join(args.out_dir, run_name)
            os.makedirs(run_path, exist_ok=True)

            # Build model + tokenizer for current teacher run
            model, tok = build_model(model_name, num_labels=4)

            # Train teacher on support set
            train_one(
                model,
                tok,
                support_ds,
                train_seed=seed,
                device=device,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_len=args.max_len,
            )

            # Save support set logits and labels for the current teacher run
            support_logits, support_labels = predict_logits(model, tok, support_ds, device, args.batch_size, args.max_len)
            np.save(os.path.join(run_path, "support_logits.npy"), support_logits)
            np.save(os.path.join(run_path, "support_labels.npy"), support_labels)
            save_json(os.path.join(run_path, "support_indices.json"), {"support_indices": support_indices})

            # Evaluate teacher on the test set
            test_logits, test_labels = predict_logits(model, tok, test_ds, device, args.batch_size, args.max_len)

            # Compute evaluation metrics
            m = metrics_from_logits(test_logits, test_labels)

            # Save test predictions per run
            np.save(os.path.join(run_path, "test_logits.npy"), test_logits)
            np.save(os.path.join(run_path, "test_labels.npy"), test_labels)

            # Save run metadata
            save_json(
                os.path.join(run_path, "meta.json"),
                {
                    "model": model_name,
                    "train_seed": seed,
                    "support_seed": args.support_seed,
                    "n_per_class": args.n_per_class,
                    "metrics": m,
                },
            )

            # Add this run to the global index list
            results.append({"run": run_name, "metrics": m})

    # Save Summary JSON
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