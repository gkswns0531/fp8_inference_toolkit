#!/usr/bin/env python3
"""
Additional accuracy configs: Asymmetric + Clip variants
"""

import subprocess
import time
import json
import random
import numpy as np
import os
from typing import List, Optional
import requests

SERVER_PORT = 8100
MAX_MODEL_LEN = 2048
TOKEN_LENGTH = 512
MODEL_ID = "Qwen/Qwen3-Embedding-4B"

REAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Machine learning models require large amounts of training data to achieve good performance.",
    "Quantum computing leverages superposition and entanglement to solve complex problems.",
    "The global economy is influenced by factors such as trade policies and currency exchange rates.",
    "Neural networks with attention mechanisms have revolutionized natural language processing.",
    "Climate change is causing rising sea levels and more frequent extreme weather events.",
    "The theory of relativity describes the relationship between space, time, and gravity.",
    "Renewable energy sources like solar and wind power are becoming increasingly cost effective.",
]

# Additional configs to test
CONFIGS = [
    ("PG+Asym", {"INT4TC_ASYMMETRIC": "1"}),
    ("PG+Clip0.95", {"INT4TC_CLIP_RATIO": "0.95"}),
    ("PG+Asym+Clip0.95", {"INT4TC_ASYMMETRIC": "1", "INT4TC_CLIP_RATIO": "0.95"}),
    ("PG+Clip0.90", {"INT4TC_CLIP_RATIO": "0.90"}),
]


def kill_server_on_port(port):
    try:
        out = subprocess.check_output(["fuser", f"{port}/tcp"], stderr=subprocess.STDOUT, text=True)
        for pid in out.strip().split():
            subprocess.run(["kill", "-9", pid.strip()], capture_output=True)
        time.sleep(2)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        for pid in out.strip().split("\n"):
            if pid.strip():
                subprocess.run(["kill", "-9", pid.strip()], capture_output=True)
        time.sleep(3)
    except Exception:
        pass


def start_server(env_vars, label):
    log_path = f"/tmp/vllm_server_{label}.log"
    log_file = open(log_path, "w")
    env = os.environ.copy()
    env.update(env_vars)

    pre_import = "vllm.model_executor.layers.quantization.w4a8_enhanced_int4tc"
    base_args = [
        "--model", MODEL_ID, "--runner", "pooling",
        "--dtype", "auto", "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90", "--trust-remote-code",
        "--no-enable-prefix-caching", "--port", str(SERVER_PORT),
        "--quantization", "w4a8-enhanced-int4tc", "--enforce-eager",
    ]
    arg_str = ", ".join(f"'{a}'" for a in base_args)
    cmd = [
        "python3", "-c",
        f"import {pre_import}; "
        f"from vllm.utils.argparse_utils import FlexibleArgumentParser; "
        f"from vllm.entrypoints.openai.api_server import make_arg_parser, run_server; "
        f"parser = make_arg_parser(FlexibleArgumentParser()); "
        f"args = parser.parse_args([{arg_str}]); "
        f"import asyncio; asyncio.run(run_server(args))"
    ]
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            text=True, cwd="/tmp", env=env)


def wait_for_server(proc, timeout=600):
    url = f"http://localhost:{SERVER_PORT}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def get_embeddings(texts):
    url = f"http://localhost:{SERVER_PORT}/v1/embeddings"
    resp = requests.post(url, json={"model": MODEL_ID, "input": texts}, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()["data"]
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    # Load BF16 baseline from previous run
    prev_path = "/home/ubuntu/fp8_inference_toolkit/benchmark/vllm_inference_comparison.json"
    bf16_embs = None

    # First get BF16 baseline
    print("Getting BF16 baseline...")
    kill_server_on_port(SERVER_PORT)
    env = os.environ.copy()
    log_file = open("/tmp/vllm_server_bf16_extra.log", "w")
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID, "--runner", "pooling",
        "--dtype", "auto", "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90", "--trust-remote-code",
        "--no-enable-prefix-caching", "--port", str(SERVER_PORT),
    ]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            text=True, cwd="/tmp", env=env)
    try:
        if wait_for_server(proc):
            time.sleep(3)
            bf16_embs = np.array(get_embeddings(REAL_TEXTS))
            print(f"  BF16 baseline shape: {bf16_embs.shape}")
        else:
            print("  BF16 server failed!")
            return
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        kill_server_on_port(SERVER_PORT)
        time.sleep(10)

    # Test each config
    results = []
    for label, env_vars in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {label}: {env_vars}")
        print(f"{'='*60}")

        kill_server_on_port(SERVER_PORT)
        proc = start_server(env_vars, label.replace("+", "_"))
        try:
            print("  Starting server...")
            if not wait_for_server(proc):
                log_path = f"/tmp/vllm_server_{label.replace('+', '_')}.log"
                try:
                    with open(log_path) as f:
                        lines = f.read().strip().split("\n")[-10:]
                    print(f"  FAILED:\n" + "\n".join(lines))
                except Exception:
                    print("  FAILED")
                continue

            time.sleep(3)
            print("  Server ready!")
            embs = np.array(get_embeddings(REAL_TEXTS))

            cos_sims = [cosine_sim(bf16_embs[i], embs[i]) for i in range(len(REAL_TEXTS))]
            diffs = np.abs(embs - bf16_embs).flatten()

            r = {
                "label": label,
                "cosine_sim_mean": round(float(np.mean(cos_sims)), 6),
                "cosine_sim_min": round(float(np.min(cos_sims)), 6),
                "mae": round(float(np.mean(diffs)), 6),
                "max_diff": round(float(np.max(diffs)), 6),
            }
            results.append(r)
            print(f"  CosSim: mean={r['cosine_sim_mean']:.6f}  min={r['cosine_sim_min']:.6f}")
            print(f"  MAE={r['mae']:.6f}  MaxDiff={r['max_diff']:.6f}")

            # Measure latency (single batch)
            url = f"http://localhost:{SERVER_PORT}/v1/embeddings"
            requests.post(url, json={"model": MODEL_ID, "input": REAL_TEXTS[:1]}, timeout=300)
            lats = []
            for _ in range(5):
                t0 = time.time()
                requests.post(url, json={"model": MODEL_ID, "input": REAL_TEXTS[:1]}, timeout=300)
                lats.append((time.time() - t0) * 1000)
            r["latency_b1_ms"] = round(float(np.mean(lats)), 1)
            print(f"  Latency B=1: {r['latency_b1_ms']:.1f}ms")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            kill_server_on_port(SERVER_PORT)
            time.sleep(10)

    # Summary
    print(f"\n{'='*70}")
    print(f"  ADDITIONAL CONFIGS vs BF16 (Real Text, 8 sentences)")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'CosSim':>10} {'CosSim(min)':>12} {'MAE':>10} {'Lat B=1':>10}")
    print("-" * 70)
    # Include previous results for comparison
    print(f"{'FP8 (prev)':>25} {'0.995979':>10} {'0.994823':>12} {'0.001397':>10} {'72.5ms':>10}")
    print(f"{'PG-sym (prev)':>25} {'0.866598':>10} {'0.809623':>12} {'0.008056':>10} {'87.2ms':>10}")
    print(f"{'PerCh (prev)':>25} {'0.340851':>10} {'0.184241':>12} {'0.017897':>10} {'59.7ms':>10}")
    for r in results:
        lat = r.get("latency_b1_ms", "?")
        lat_str = f"{lat}ms" if isinstance(lat, float) else lat
        print(f"{r['label']:<25} {r['cosine_sim_mean']:>10.6f} {r['cosine_sim_min']:>12.6f} "
              f"{r['mae']:>10.6f} {lat_str:>10}")

    out_path = "/home/ubuntu/fp8_inference_toolkit/benchmark/vllm_accuracy_extra.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
