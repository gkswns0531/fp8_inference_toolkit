#!/usr/bin/env python3
"""
Test outlier clipping and asymmetric INT4 features.
Compares against BF16 baseline and Enhanced-PG (no clipping, symmetric).
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
    ("Enhanced-PG (baseline)", ["--quantization", "w4a8-enhanced-int4tc"], {}),
    ("PG+Clip(0.99)", ["--quantization", "w4a8-enhanced-int4tc"],
     {"INT4TC_CLIP_RATIO": "0.99"}),
    ("PG+Asymmetric", ["--quantization", "w4a8-enhanced-int4tc"],
     {"INT4TC_ASYMMETRIC": "1"}),
    ("PG+Asym+Clip(0.99)", ["--quantization", "w4a8-enhanced-int4tc"],
     {"INT4TC_ASYMMETRIC": "1", "INT4TC_CLIP_RATIO": "0.99"}),
    ("PG+Asym+Clip(0.95)", ["--quantization", "w4a8-enhanced-int4tc"],
     {"INT4TC_ASYMMETRIC": "1", "INT4TC_CLIP_RATIO": "0.95"}),
]


def start_server(extra_args, env_vars):
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--runner", "pooling",
        "--dtype", "auto", "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90", "--trust-remote-code",
        "--enforce-eager", "--no-enable-prefix-caching",
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

    all_embeddings = {}

    for label, extra_args, env_vars in CONFIGS:
        print(f"=== {label} ===")
        if env_vars:
            print(f"  Env: {env_vars}")
        print("  Starting...", flush=True)

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
            embs = get_embeddings(sentences)
            all_embeddings[label] = np.array(embs)
            print(f"  Got ({all_embeddings[label].shape[0]}, {all_embeddings[label].shape[1]})")
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            kill_server(proc)

    baseline = all_embeddings.get("BF16")
    if baseline is None:
        print("No BF16 baseline!")
        return

    print(f"\n{'='*60}")
    print(f"  FEATURE TEST RESULTS (vs BF16)")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    print(f"{'BF16':<30} {'1.000000':>10} {'1.000000':>10} {'1.000000':>10}")

    for label, embs in all_embeddings.items():
        if label == "BF16":
            continue
        cos_sims = [cosine_sim(baseline[i], embs[i]) for i in range(len(sentences))]
        mean_c = np.mean(cos_sims)
        min_c = np.min(cos_sims)
        max_c = np.max(cos_sims)
        print(f"{label:<30} {mean_c:>10.6f} {min_c:>10.6f} {max_c:>10.6f}")


if __name__ == "__main__":
    main()
