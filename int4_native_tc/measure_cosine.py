#!/usr/bin/env python3
"""
INT4 Quantization Quality Measurement: BF16 vs quantized configs.

Launches vLLM server with each config, collects embeddings, computes cosine similarity.

Configs tested:
  1. BF16 (baseline)
  2. W4A16 (per-group g128)
  3. W4A4A8-Mixed (per-channel)
  4. Enhanced-PG (per-group g128, no calibration)
  5. Enhanced-PG+SQ (per-group g128 + SmoothQuant calibration)
"""

import json
import os
import random
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

# Ensure our custom quant modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    ("BF16", [], {}),
    ("W4A16 (g128)", ["--quantization", "w4a16-int4tc"], {}),
    ("W4A4A8-Mixed", ["--quantization", "w4a4a8-mixed-int4tc"], {}),
    ("Enhanced-PG", ["--quantization", "w4a8-enhanced-int4tc"], {}),
    ("Enhanced-PG+SQ", ["--quantization", "w4a8-enhanced-int4tc"],
     {"INT4TC_CALIBRATION_STATS": "/tmp/calib_stats.pt"}),
]


def start_server(extra_args: list, env_vars: dict) -> subprocess.Popen:
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--runner", "pooling",
        "--dtype", "auto",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
        "--trust-remote-code",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--port", str(SERVER_PORT),
    ] + extra_args

    env = os.environ.copy()
    env.update(env_vars)

    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env)


def wait_for_server(timeout: int = 300) -> bool:
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


def get_embeddings(texts: list[str]) -> list[list[float]]:
    url = f"http://localhost:{SERVER_PORT}/v1/embeddings"
    resp = requests.post(
        url, json={"model": MODEL, "input": texts}, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()["data"]
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    return float(dot / (np.linalg.norm(a) * np.linalg.norm(b)))


def kill_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    time.sleep(3)


def dump_server_log(proc: subprocess.Popen, label: str) -> None:
    """Read remaining stdout and print last few lines on failure."""
    try:
        out, _ = proc.communicate(timeout=5)
        lines = out.strip().split('\n')
        print(f"  [Last 10 log lines for {label}]:")
        for line in lines[-10:]:
            print(f"    {line}")
    except Exception:
        pass


def main():
    random.seed(42)

    # Generate fixed test sentences
    sentences = [
        " ".join(random.choices(WORDS, k=50))
        for _ in range(NUM_TEST_SENTENCES)
    ]

    print(f"Model: {MODEL}")
    print(f"Test sentences: {NUM_TEST_SENTENCES}")
    print(f"Max model len: {MAX_MODEL_LEN}")
    print()

    all_embeddings = {}

    for label, extra_args, env_vars in CONFIGS:
        print(f"{'='*60}")
        print(f"  Config: {label}")
        if extra_args:
            print(f"  Args: {' '.join(extra_args)}")
        if env_vars:
            print(f"  Env: {env_vars}")
        print(f"{'='*60}")

        proc = start_server(extra_args, env_vars)
        try:
            print("  Starting server...", flush=True)
            if not wait_for_server():
                print("  FAILED: Server did not start within timeout!")
                dump_server_log(proc, label)
                continue

            time.sleep(2)
            print("  Server ready! Getting embeddings...", flush=True)

            embs = get_embeddings(sentences)
            all_embeddings[label] = np.array(embs)
            print(f"  Got embeddings: shape={all_embeddings[label].shape}")

        except Exception as e:
            print(f"  ERROR: {e}")
            dump_server_log(proc, label)
        finally:
            kill_server(proc)
            print(f"  Server stopped.\n")

    # --- Comparison ---
    print(f"\n{'='*70}")
    print(f"  COSINE SIMILARITY RESULTS (vs BF16 baseline)")
    print(f"{'='*70}\n")

    baseline_label = "BF16"
    if baseline_label not in all_embeddings:
        print("ERROR: BF16 baseline not available. Cannot compare.")
        return

    baseline = all_embeddings[baseline_label]
    results = []

    for label, embs in all_embeddings.items():
        if label == baseline_label:
            continue

        # Per-sentence cosine similarity
        cos_sims = [cosine_sim(baseline[i], embs[i])
                    for i in range(len(sentences))]

        mean_cos = np.mean(cos_sims)
        min_cos = np.min(cos_sims)
        max_cos = np.max(cos_sims)

        results.append({
            "config": label,
            "cosine_mean": float(mean_cos),
            "cosine_min": float(min_cos),
            "cosine_max": float(max_cos),
            "cosine_per_sentence": [float(c) for c in cos_sims],
        })

        print(f"  {label}:")
        print(f"    Mean cosine:  {mean_cos:.6f}")
        print(f"    Min cosine:   {min_cos:.6f}")
        print(f"    Max cosine:   {max_cos:.6f}")
        print()

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'Mean Cosine':>12} {'Min Cosine':>12} {'Max Cosine':>12}")
    print("-" * 70)
    print(f"{'BF16 (baseline)':<35} {'1.000000':>12} {'1.000000':>12} {'1.000000':>12}")
    for r in results:
        print(f"{r['config']:<35} {r['cosine_mean']:>12.6f} "
              f"{r['cosine_min']:>12.6f} {r['cosine_max']:>12.6f}")

    # Save results
    out_path = str(Path(__file__).parent / "cosine_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
