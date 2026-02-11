#!/usr/bin/env python3
"""
BF16 vs FP8 vs INT4(W4A16) 3-way Comparison Benchmark

Qwen3-VL-Embedding-2B 모델을 BF16, FP8, INT4(W4A16)로 서빙하고
Batch=1,16 / Tokens=1024에 대해 latency를 비교합니다.

Usage:
    python run_3way_benchmark.py
"""

import subprocess
import time
import json
import random
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, asdict

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SERVER_PORT = 8100
MAX_MODEL_LEN = 4096
OVERHEAD = 2
RUNS = 5
BATCH_SIZES = [1, 16]
TOKEN_LENGTH = 1024

WORDS = [
    "the", "cat", "dog", "house", "car", "book", "computer", "phone",
    "water", "food", "happy", "sad", "big", "small", "red", "blue",
    "run", "walk", "eat", "sleep", "work", "play", "think", "feel",
    "good", "bad", "new", "old", "first", "last", "long", "short",
    "man", "woman", "child", "world", "life", "day", "night", "time",
    "mountain", "river", "ocean", "forest", "desert", "island", "valley", "bridge",
    "guitar", "piano", "violin", "trumpet", "flute", "cello", "drum", "harp",
    "python", "rust", "java", "swift", "ruby", "perl", "scala", "kotlin",
    "matrix", "vector", "tensor", "graph", "queue", "stack", "tree", "node",
    "quantum", "photon", "neutron", "proton", "electron", "plasma", "gravity", "orbit",
    "dolphin", "eagle", "tiger", "panda", "falcon", "whale", "shark", "cobra",
    "crystal", "diamond", "emerald", "topaz", "sapphire", "opal", "jade", "amber",
    "volcano", "glacier", "canyon", "plateau", "mesa", "fjord", "delta", "lagoon",
    "algebra", "calculus", "geometry", "topology", "entropy", "inertia", "momentum", "friction",
    "crimson", "indigo", "violet", "scarlet", "ivory", "bronze", "silver", "copper",
    "abstract", "concrete", "dynamic", "static", "volatile", "mutable", "frozen", "elastic",
    "whisper", "thunder", "silence", "rhythm", "harmony", "melody", "chorus", "echo",
]

# (label, model_id, extra_vllm_args)
CONFIGS = [
    ("BF16", "Qwen/Qwen3-VL-Embedding-2B", []),
    ("FP8", "Forturne/Qwen3-VL-Embedding-2B-FP8", []),
    ("INT4-W4A16", "Forturne/Qwen3-VL-Embedding-2B-W4A16", []),
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_random_text(num_words: int) -> str:
    return " ".join(random.choices(WORDS, k=num_words))


def get_gpu_memory_mib() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def start_server(model: str, extra_args: List[str]) -> subprocess.Popen:
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--runner", "pooling",
        "--dtype", "auto",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.85",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--port", str(SERVER_PORT),
    ] + extra_args
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_for_server(timeout: int = 600) -> bool:
    url = f"http://localhost:{SERVER_PORT}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def benchmark(model_name: str, batch_size: int) -> dict:
    actual_tokens = min(TOKEN_LENGTH, MAX_MODEL_LEN - OVERHEAD)
    url = f"http://localhost:{SERVER_PORT}/v1/embeddings"

    # Warmup
    texts = [make_random_text(actual_tokens) for _ in range(batch_size)]
    requests.post(url, json={"model": model_name, "input": texts}, timeout=300)

    # Measure
    latencies = []
    for _ in range(RUNS):
        texts = [make_random_text(actual_tokens) for _ in range(batch_size)]
        t0 = time.time()
        resp = requests.post(url, json={"model": model_name, "input": texts}, timeout=300)
        latencies.append((time.time() - t0) * 1000)
        if resp.status_code != 200:
            return {"batch_size": batch_size, "error": resp.text[:200]}

    avg = float(np.mean(latencies))
    p50 = float(np.percentile(latencies, 50))
    p99 = float(np.percentile(latencies, 99))
    throughput = (batch_size * actual_tokens) / (avg / 1000)

    return {
        "batch_size": batch_size,
        "tokens": TOKEN_LENGTH,
        "avg_ms": round(avg, 2),
        "p50_ms": round(p50, 2),
        "p99_ms": round(p99, 2),
        "throughput_tok_s": round(throughput, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    for label, model_id, extra_args in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {label}: {model_id}")
        print(f"{'='*60}")

        proc = start_server(model_id, extra_args)
        try:
            print("  Starting server...")
            if not wait_for_server():
                print("  FAILED: server timeout")
                all_results[label] = {"error": "server timeout"}
                continue

            time.sleep(3)
            vram = get_gpu_memory_mib()
            print(f"  Server ready! VRAM: {vram} MiB")

            results = []
            for batch in BATCH_SIZES:
                print(f"  Benchmarking batch={batch}, tokens={TOKEN_LENGTH}...", end=" ", flush=True)
                r = benchmark(model_id, batch)
                results.append(r)
                if "error" in r:
                    print(f"ERROR: {r['error']}")
                else:
                    print(f"avg={r['avg_ms']:.1f}ms  throughput={r['throughput_tok_s']:.0f} tok/s")

            all_results[label] = {"model": model_id, "vram_mib": vram, "results": results}

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[label] = {"error": str(e)}
        finally:
            proc.terminate()
            proc.wait()
            time.sleep(5)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Qwen3-VL-Embedding-2B  |  Batch={{1,16}}, Tokens={TOKEN_LENGTH}")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'VRAM':>8} {'B=1 Avg':>10} {'B=1 Thr':>12} {'B=16 Avg':>10} {'B=16 Thr':>12}")
    print("-" * 70)

    for label in ["BF16", "FP8", "INT4-W4A16"]:
        if label not in all_results or "error" in all_results[label]:
            print(f"{label:<15} {'ERROR':>8}")
            continue
        d = all_results[label]
        vram = d.get("vram_mib", "?")
        r = {r["batch_size"]: r for r in d["results"]}
        b1 = r.get(1, {})
        b16 = r.get(16, {})
        print(f"{label:<15} {vram:>7} {'MiB':1}"
              f" {b1.get('avg_ms', 0):>8.1f}ms"
              f" {b1.get('throughput_tok_s', 0):>10.0f}t/s"
              f" {b16.get('avg_ms', 0):>8.1f}ms"
              f" {b16.get('throughput_tok_s', 0):>10.0f}t/s")

    # Save
    out = "/home/ubuntu/fp8_inference_toolkit/benchmark/3way_benchmark_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
