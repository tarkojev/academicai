"""
This file implements the ensemble cost benchmarking script:
- It loads multiple teacher-model architectures into memory and runs inference on a small set of test samples.
- It measures the total parameter count, estimated memory size, and average latency per sample for the ensemble.
- It saves the results into a JSON file for later analysis and reporting.
"""

# Imported libraries
import os
import argparse
import time
import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5EncoderModel
from utils import load_ag_news, count_params, estimate_model_size_mb, save_json, get_device

def is_t5(name: str) -> bool:
    return name.startswith("t5-")

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

# Benchmark an ensemble of models on a set of texts, returning total params, size, and latency
def benchmark_ensemble(model_names, texts, device, max_len=128, n_warmup=5, n_iters=15):
    models_toks = []
    total_params = 0
    total_size_mb = 0.0
    print(f"Loading {len(model_names)} models into memory")
    for name in model_names:
        tok = AutoTokenizer.from_pretrained(name)
        if is_t5(name):
            model = T5EncoderForClassification(name, num_labels=4)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=4)
        model.to(device)
        model.eval()
        total_params += int(count_params(model))
        total_size_mb += float(estimate_model_size_mb(model))
        models_toks.append((model, tok))

    # Run one inference pass through the ensemble for a given text
    def run_one_ensemble_inference(text: str):
        with torch.no_grad():
            for model, tok in models_toks:
                inputs = tok(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = model(**inputs)
                logits = out.logits if hasattr(out, "logits") else out
                _ = torch.softmax(logits, dim=-1)
    print("Running ensemble inference to measure latency.")
    for i in range(min(n_warmup, len(texts))):
        run_one_ensemble_inference(texts[i % len(texts)])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
    print("Measuring latency")
    times = []
    for i in range(n_iters):
        t = texts[i % len(texts)]
        start = time.perf_counter()
        run_one_ensemble_inference(t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return total_params, total_size_mb, float(np.mean(times))

# Main function to run the ensemble cost benchmarking and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=["bert-base-uncased", "distilbert-base-uncased", "t5-small"],
    )
    ap.add_argument("--out", type=str, default="runs/eff_ensemble.json")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--bench_texts", type=int, default=50)
    ap.add_argument("--bench_warmup", type=int, default=10)
    ap.add_argument("--bench_iters", type=int, default=30)
    args = ap.parse_args()
    device = get_device()
    _, test_ds = load_ag_news()
    test_set_size = len(test_ds)
    n_bench = min(args.bench_texts, test_set_size)
    bench_texts = [test_ds[i]["text"] for i in range(n_bench)]
    params, size, latency = benchmark_ensemble(
        model_names=args.models,
        texts=bench_texts,
        device=device,
        max_len=args.max_len,
        n_warmup=args.bench_warmup,
        n_iters=args.bench_iters,
    )
    total_test_time_ms = latency * test_set_size
    total_test_time_sec = total_test_time_ms / 1000.0
    total_test_time_min = total_test_time_sec / 60.0
    res = {
        "type": "ensemble_cost",
        "models": args.models,
        "prob_interface": "classifier_head",
        "data": {
            "test_set_size": test_set_size,
        },
        "efficiency": {
            "params": int(params),
            "size_mb": float(size),
            "latency_ms": float(latency),
            "latency_bench_n": int(n_bench),
            "latency_mode": "classifier_forward_sum",
            "bench_warmup": int(args.bench_warmup),
            "bench_iters": int(args.bench_iters),
            "total_test_inference_sec": float(total_test_time_sec),
            "total_test_inference_min": float(total_test_time_min),
            "device": str(device),
        },
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_json(args.out, res)
    print("Ensemble cost benchmarking results:")
    print(f"Models:               {', '.join(args.models)}")
    print(f"Total Params:         {params:,}")
    print(f"Memory Footprint:     {size:.2f} MB")
    print(f"Inference Latency:    {latency:.2f} ms/sample")
    print(f"Benchmark Samples:    {n_bench}")
    print(f"Test Set Size:        {test_set_size} samples")
    print(f"Total Run Time:       {total_test_time_sec:.2f} seconds (~{total_test_time_min:.2f} minutes)")

if __name__ == "__main__":
    main()
