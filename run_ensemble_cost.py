"""
This file implements the ensemble cost benchmarking script:
- It loads multiple teacher models into memory and runs inference on a small set of test samples.
- It measures the total parameter count, estimated memory size, and average latency per sample for the ensemble.
- It saves the results into a JSON file for later analysis and reporting.
"""

# Imported libraries
import os
import argparse
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from utils import load_ag_news, count_params, estimate_model_size_mb, save_json, get_device

# Label texts used for T5 scoring in the ensemble benchmarking
T5_LABEL_TEXTS = ["world", "sports", "business", "sci tech"]

# Benchmark an ensemble of models on a set of texts, returning total params, size, and latency
def benchmark_ensemble(model_names, texts, device, max_len=128, n_warmup=5, n_iters=15):
    models_toks = []
    total_params = 0
    total_size_mb = 0
    
    print(f"Loading {len(model_names)} models into memory")
    for name in model_names:
        tok = AutoTokenizer.from_pretrained(name)
        if "t5-" in name.lower():
            model = T5ForConditionalGeneration.from_pretrained(name)
            is_t5 = True
        else:
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=4)
            is_t5 = False
        
        model.to(device)
        model.eval()
        total_params += count_params(model)
        total_size_mb += estimate_model_size_mb(model)
        models_toks.append((model, tok, is_t5))

    # Run one inference pass through the ensemble for a given text
    def run_one_ensemble_inference(text):
        with torch.no_grad():
            for model, tok, is_t5 in models_toks:
                inputs = tok(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len).to(device)
                if is_t5:
                    for label_text in T5_LABEL_TEXTS:
                        target = tok(label_text, return_tensors="pt").input_ids.to(device)
                        model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=target)
                else:
                    model(**inputs)
            _ = np.zeros((1, 4)) 

    print("Running ensemble inference to measure latency.")
    for i in range(n_warmup):
        run_one_ensemble_inference(texts[i % len(texts)])
    
    print("Measuring latency")
    times = []
    for i in range(n_iters):
        t = texts[i % len(texts)]
        start = time.perf_counter()
        run_one_ensemble_inference(t)
        if device.type == "cuda": torch.cuda.synchronize()
        elif device.type == "mps": torch.mps.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return total_params, total_size_mb, np.mean(times)

# Main function to run the ensemble cost benchmarking and save results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["bert-base-uncased", "distilbert-base-uncased", "t5-small"])
    ap.add_argument("--out", type=str, default="runs/eff_ensemble.json")
    args = ap.parse_args()
    
    device = get_device()
    _, test_ds = load_ag_news()
    test_set_size = len(test_ds)
    
    bench_texts = [test_ds[i]["text"] for i in range(min(50, test_set_size))]
    
    params, size, latency = benchmark_ensemble(args.models, bench_texts, device)
    
    total_test_time_ms = latency * test_set_size
    total_test_time_sec = total_test_time_ms / 1000.0
    total_test_time_min = total_test_time_sec / 60.0

    res = {
        "models": args.models,
        "params": int(params),
        "size_mb": float(size),
        "latency_ms": float(latency),
        "test_set_size": test_set_size,
        "total_test_inference_sec": float(total_test_time_sec),
        "total_test_inference_min": float(total_test_time_min),
        "device": str(device)
    }
    
    save_json(args.out, res)
    print("Ensemble cost benchmarking results:")
    print(f"Models:          {', '.join(args.models)}")
    print(f"Total Params:    {params:,}")
    print(f"Memory Footprint: {size:.2f} MB")
    print(f"Inference Latency: {latency:.2f} ms/sample")
    print(f"Test Set Size:   {test_set_size} samples")
    print(f"Total Run Time:  {total_test_time_sec:.2f} seconds (~{total_test_time_min:.2f} minutes)")

if __name__ == "__main__":
    main()
