"""
This file implements efficiency measurement for a given Hugging Face model architecture. It computes:
- param count
- approximate size (MB)
- simple single-sample latency benchmark
It runs inference on a small number of test samples and averages the latency over multiple runs. Results are saved into a JSON file for later analysis.
"""

# Imported libraries
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_ag_news, count_params, estimate_model_size_mb, latency_ms, save_json

# Function to compute costs for a given model and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/efficiency.json")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--n_texts", type=int, default=50)
    ap.add_argument("--n_warmup", type=int, default=10)
    ap.add_argument("--n_iters", type=int, default=30)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=4)
    _, test_ds = load_ag_news()
    texts = [test_ds[i]["text"] for i in range(min(args.n_texts, len(test_ds)))]

    res = {
        "model_name": args.model_name,
        "device": str(device),
        "params": int(count_params(model)),
        "size_mb": float(estimate_model_size_mb(model)),
        "latency_ms": float(
            latency_ms(
                model,
                tok,
                texts,
                device,
                max_len=args.max_len,
                n_warmup=args.n_warmup,
                n_iters=args.n_iters,
            )
        ),
        "n_texts": int(len(texts)),
        "max_len": int(args.max_len),
        "n_warmup": int(args.n_warmup),
        "n_iters": int(args.n_iters),
    }
    # Output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_json(args.out, res)
    print(res)

if __name__ == "__main__":
    main()