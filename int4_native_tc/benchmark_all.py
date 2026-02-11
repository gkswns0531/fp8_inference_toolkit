#!/usr/bin/env python3
"""
Comprehensive benchmark: Latency + Cosine Similarity across all INT4 quantization configs.

Configs tested:
  1. BF16 (baseline)
  2. W4A16 Marlin (g128) - highest quality INT4
  3. INT4-TC (naive INT4x INT4, per-channel)
  4. W4A4A8-Mixed (per-channel, Marlin post-act)
  5. Enhanced-PG+Asym+Clip(0.95) (our best)
"""

import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

SERVER_PORT = 8100
MODEL = "Qwen/Qwen3-Embedding-4B"
MAX_MODEL_LEN = 512
NUM_TEST_SENTENCES = 10
LATENCY_WARMUP = 3
LATENCY_ITERS = 10

WORDS = [
    "mountain", "river", "ocean", "forest", "desert", "island", "valley", "bridge",
    "guitar", "piano", "violin", "trumpet", "flute", "cello", "drum", "harp",
    "python", "rust", "java", "swift", "ruby", "perl", "scala", "kotlin",
    "matrix", "vector", "tensor", "graph", "queue", "stack", "tree", "node",
    "quantum", "photon", "neutron", "proton", "electron", "plasma", "gravity", "orbit",
    "dolphin", "eagle", "tiger", "panda", "falcon", "whale", "shark", "cobra",
    "crystal", "diamond", "emerald", "topaz", "sapphire", "opal", "jade", "amber",
    "volcano", "glacier", "canyon", "plateau", "mesa", "fjord", "delta", "lagoon",
]

# (label, extra_args, env_vars)
CONFIGS = [
    # enforce-eager (baseline — no torch.compile, no CUDA graph)
    ("BF16(eager)", ["--enforce-eager"], {}),
    ("PerCh(eager)", ["--quantization", "w4a8-enhanced-int4tc", "--enforce-eager"],
     {"INT4TC_PER_CHANNEL": "1"}),
    ("PG-sym(eager)", ["--quantization", "w4a8-enhanced-int4tc", "--enforce-eager"],
     {}),
    ("PG+AC(eager)", ["--quantization", "w4a8-enhanced-int4tc", "--enforce-eager"],
     {"INT4TC_ASYMMETRIC": "1", "INT4TC_CLIP_RATIO": "0.95"}),
    # torch.compile + PIECEWISE CUDA graph (default O2)
    ("BF16(compile)", [], {}),
    ("PerCh(compile)", ["--quantization", "w4a8-enhanced-int4tc"],
     {"INT4TC_PER_CHANNEL": "1"}),
    ("PG-sym(compile)", ["--quantization", "w4a8-enhanced-int4tc"],
     {}),
    ("PG+AC(compile)", ["--quantization", "w4a8-enhanced-int4tc"],
     {"INT4TC_ASYMMETRIC": "1", "INT4TC_CLIP_RATIO": "0.95"}),
]


def clear_compile_cache():
    """Clear torch.compile cache to avoid shape mismatches between configs."""
    cache_dir = Path.home() / ".cache" / "vllm" / "torch_compile_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("  Cleared torch.compile cache", flush=True)


def start_server(extra_args, env_vars):
    clear_compile_cache()
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--runner", "pooling",
        "--dtype", "auto", "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90", "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--port", str(SERVER_PORT),
    ] + extra_args
    env = os.environ.copy()
    env.update(env_vars)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)


def wait_for_server(timeout=300):
    url = f"http://localhost:{SERVER_PORT}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def get_embeddings(texts):
    resp = requests.post(
        f"http://localhost:{SERVER_PORT}/v1/embeddings",
        json={"model": MODEL, "input": texts}, timeout=300)
    data = resp.json()["data"]
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


def measure_latency(texts, warmup=LATENCY_WARMUP, iters=LATENCY_ITERS):
    """Measure average latency over multiple iterations after warmup."""
    # Warmup
    for _ in range(warmup):
        get_embeddings(texts)

    # Timed runs
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        get_embeddings(texts)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    return latencies


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def kill_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    time.sleep(3)


def main():
    random.seed(42)
    sentences = [" ".join(random.choices(WORDS, k=50)) for _ in range(NUM_TEST_SENTENCES)]

    results = {}

    for label, extra_args, env_vars in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {label}")
        if env_vars:
            print(f"  Env: {env_vars}")
        print(f"{'='*60}")
        print("  Starting server...", flush=True)

        proc = start_server(extra_args, env_vars)
        try:
            if not wait_for_server():
                print("  FAILED: timeout")
                try:
                    out, _ = proc.communicate(timeout=5)
                    for line in out.strip().split('\n')[-5:]:
                        print(f"    {line}")
                except Exception:
                    pass
                continue

            time.sleep(2)
            print("  Server ready!", flush=True)

            # 1) Get embeddings for cosine comparison
            embs = get_embeddings(sentences)
            embs_np = np.array(embs)
            print(f"  Embeddings: {embs_np.shape}")

            # 2) Measure latency
            print(f"  Latency: warmup={LATENCY_WARMUP}, iters={LATENCY_ITERS}...", flush=True)
            latencies = measure_latency(sentences)
            lat_mean = np.mean(latencies) * 1000  # ms
            lat_std = np.std(latencies) * 1000
            lat_min = np.min(latencies) * 1000
            lat_max = np.max(latencies) * 1000
            lat_p50 = np.percentile(latencies, 50) * 1000
            lat_p99 = np.percentile(latencies, 99) * 1000
            print(f"  Latency: mean={lat_mean:.1f}ms, std={lat_std:.1f}ms, "
                  f"p50={lat_p50:.1f}ms, p99={lat_p99:.1f}ms")

            results[label] = {
                "embeddings": embs_np,
                "latency_ms": latencies,
                "lat_mean": lat_mean,
                "lat_std": lat_std,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lat_p50": lat_p50,
                "lat_p99": lat_p99,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            kill_server(proc)

    # --- Results ---
    baseline_label = "BF16(eager)" if "BF16(eager)" in results else "BF16(compile)"
    baseline = results.get(baseline_label)
    if baseline is None:
        print("No BF16 baseline!")
        return

    bf16_embs = baseline["embeddings"]
    bf16_lat = baseline["lat_mean"]

    # Compute cosine for each config
    for label, data in results.items():
        if label == baseline_label:
            data["cos_mean"] = 1.0
            data["cos_min"] = 1.0
            data["cos_max"] = 1.0
            continue
        embs = data["embeddings"]
        cos_sims = [cosine_sim(bf16_embs[i], embs[i]) for i in range(NUM_TEST_SENTENCES)]
        data["cos_mean"] = float(np.mean(cos_sims))
        data["cos_min"] = float(np.min(cos_sims))
        data["cos_max"] = float(np.max(cos_sims))

    # Print summary
    print(f"\n\n{'='*90}")
    print(f"  COMPREHENSIVE BENCHMARK RESULTS")
    print(f"{'='*90}")
    print(f"  Model: {MODEL}")
    print(f"  Batch: {NUM_TEST_SENTENCES} sentences x {MAX_MODEL_LEN} max tokens")
    print(f"  Latency: {LATENCY_WARMUP} warmup + {LATENCY_ITERS} timed iterations")
    print(f"{'='*90}\n")

    # Header
    hdr = (f"{'Config':<25} {'Cosine':>8} {'(min)':>8} {'(max)':>8} "
           f"{'Lat(ms)':>9} {'std':>7} {'p50':>9} {'p99':>9} {'Speedup':>8}")
    print(hdr)
    print("-" * len(hdr))

    for label in [c[0] for c in CONFIGS]:
        if label not in results:
            continue
        d = results[label]
        speedup = bf16_lat / d["lat_mean"] if d["lat_mean"] > 0 else 0
        print(f"{label:<25} {d['cos_mean']:>8.4f} {d['cos_min']:>8.4f} {d['cos_max']:>8.4f} "
              f"{d['lat_mean']:>9.1f} {d['lat_std']:>7.1f} {d['lat_p50']:>9.1f} {d['lat_p99']:>9.1f} "
              f"{speedup:>7.2f}x")

    print()


if __name__ == "__main__":
    main()
