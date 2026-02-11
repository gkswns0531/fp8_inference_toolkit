#!/usr/bin/env python3
"""
vLLM Inference Accuracy & Latency Comparison

Qwen3-Embedding-4B 모델에 대해 다양한 quantization 방식의
실제 임베딩 출력 정확도와 레이턴시를 비교합니다.

Configs:
  1. BF16:              No quantization (baseline)
  2. FP8:               FP8 dynamic quantization
  3. Enhanced-PG(sym):   W4A8-Enhanced per-group INT4 (symmetric)
  4. Enhanced-PerCh:     W4A8-Enhanced per-channel INT4

Usage:
    python compare_vllm_inference.py
"""

import subprocess
import time
import json
import random
import numpy as np
import os
from typing import List, Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
SERVER_PORT = 8100
MAX_MODEL_LEN = 2048
OVERHEAD = 2
BATCH_SIZES = [1, 8]
TOKEN_LENGTH = 512
LATENCY_RUNS = 5

WORDS = [
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

# Real text sentences for semantic comparison
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

MODEL_ID = "Qwen/Qwen3-Embedding-4B"

# (label, extra_vllm_args, env_vars, pre_import_module)
CONFIGS = [
    ("BF16", [], {}, None),
    ("FP8", ["--quantization", "fp8"], {}, None),
    ("W4A8-FP8-Marlin", ["--quantization", "w4a8fp8-int4tc"], {},
     "vllm.model_executor.layers.quantization.w4a8fp8_int4tc"),
    ("W4A8-INT8-Marlin", ["--quantization", "w4a8-fused-int4tc"], {},
     "vllm.model_executor.layers.quantization.w4a8_fused_int4tc"),
    ("Enhanced-PG(sym)", ["--quantization", "w4a8-enhanced-int4tc"], {},
     "vllm.model_executor.layers.quantization.w4a8_enhanced_int4tc"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def kill_server_on_port(port: int) -> None:
    try:
        out = subprocess.check_output(
            ["fuser", f"{port}/tcp"], stderr=subprocess.STDOUT, text=True)
        pids = out.strip().split()
        for pid in pids:
            subprocess.run(["kill", "-9", pid.strip()], capture_output=True)
        time.sleep(2)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True)
        for pid in out.strip().split("\n"):
            pid = pid.strip()
            if pid:
                subprocess.run(["kill", "-9", pid], capture_output=True)
        time.sleep(3)
    except Exception:
        pass


def start_server(extra_args: List[str], env_vars: dict,
                 pre_import: Optional[str] = None,
                 label: str = "") -> subprocess.Popen:
    log_path = f"/tmp/vllm_server_{label}.log"
    log_file = open(log_path, "w")

    env = os.environ.copy()
    env.update(env_vars)

    base_args = [
        "--model", MODEL_ID,
        "--runner", "pooling",
        "--dtype", "auto",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--port", str(SERVER_PORT),
    ] + extra_args

    if pre_import:
        if "--enforce-eager" not in base_args:
            base_args.append("--enforce-eager")
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
    else:
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
        ] + base_args

    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            text=True, cwd="/tmp", env=env)


def wait_for_server(proc: subprocess.Popen, timeout: int = 600) -> bool:
    url = f"http://localhost:{SERVER_PORT}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            print(f"  Server process exited with code {proc.returncode}")
            return False
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def get_embeddings(texts: List[str]) -> List[List[float]]:
    url = f"http://localhost:{SERVER_PORT}/v1/embeddings"
    resp = requests.post(url, json={"model": MODEL_ID, "input": texts}, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()["data"]
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


def measure_latency(texts: List[str]) -> dict:
    url = f"http://localhost:{SERVER_PORT}/v1/embeddings"
    # Warmup
    requests.post(url, json={"model": MODEL_ID, "input": texts}, timeout=300)

    latencies = []
    for _ in range(LATENCY_RUNS):
        t0 = time.time()
        resp = requests.post(url, json={"model": MODEL_ID, "input": texts}, timeout=300)
        latencies.append((time.time() - t0) * 1000)
        if resp.status_code != 200:
            return {"error": resp.text[:200]}

    return {
        "avg_ms": round(float(np.mean(latencies)), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
    }


def get_gpu_memory_mib() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def analyze_embeddings(base: np.ndarray, target: np.ndarray, label: str) -> dict:
    """두 임베딩 세트의 차이를 분석."""
    n = base.shape[0]
    cos_sims = [cosine_sim(base[i], target[i]) for i in range(n)]
    diffs = np.abs(target - base).flatten()

    return {
        "label": label,
        "n_samples": n,
        "embed_dim": int(base.shape[1]),
        "cosine_sim_mean": round(float(np.mean(cos_sims)), 8),
        "cosine_sim_min": round(float(np.min(cos_sims)), 8),
        "cosine_sim_max": round(float(np.max(cos_sims)), 8),
        "mae": round(float(np.mean(diffs)), 8),
        "max_diff": round(float(np.max(diffs)), 8),
        "p99_diff": round(float(np.percentile(diffs, 99)), 8),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    # Fixed test inputs
    actual_tokens = min(TOKEN_LENGTH, MAX_MODEL_LEN - OVERHEAD)
    random_texts = {
        bs: [" ".join(random.choices(WORDS, k=actual_tokens)) for _ in range(bs)]
        for bs in BATCH_SIZES
    }

    all_embeddings = {}  # label -> {"random": {bs: np.array}, "real": np.array}
    all_latency = {}     # label -> {bs: latency_dict}
    all_vram = {}        # label -> int

    for label, extra_args, env_vars, pre_import in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {label}")
        if extra_args:
            print(f"  Args: {' '.join(extra_args)}")
        if env_vars:
            print(f"  Env:  {env_vars}")
        print(f"{'='*60}")

        kill_server_on_port(SERVER_PORT)
        proc = start_server(extra_args, env_vars, pre_import, label=label)

        try:
            print("  Starting server...")
            if not wait_for_server(proc):
                log_path = f"/tmp/vllm_server_{label}.log"
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                try:
                    with open(log_path) as f:
                        lines = f.read().strip().split("\n")[-15:]
                    print(f"  FAILED:\n" + "\n".join(lines))
                except Exception:
                    print("  FAILED: server timeout")
                continue

            time.sleep(3)
            vram = get_gpu_memory_mib()
            all_vram[label] = vram
            print(f"  Server ready! VRAM: {vram} MiB")

            # Get embeddings for real texts
            print(f"  Getting real text embeddings ({len(REAL_TEXTS)} sentences)...", flush=True)
            real_embs = get_embeddings(REAL_TEXTS)
            all_embeddings.setdefault(label, {})["real"] = np.array(real_embs)
            print(f"    Shape: {all_embeddings[label]['real'].shape}")

            # Get embeddings for random texts + latency
            all_embeddings[label]["random"] = {}
            all_latency[label] = {}
            for bs in BATCH_SIZES:
                texts = random_texts[bs]
                print(f"  Getting random text embeddings batch={bs}...", flush=True)
                embs = get_embeddings(texts)
                all_embeddings[label]["random"][bs] = np.array(embs)
                print(f"    Shape: {all_embeddings[label]['random'][bs].shape}")

                print(f"  Measuring latency batch={bs}...", end=" ", flush=True)
                lat = measure_latency(texts)
                all_latency[label][bs] = lat
                if "error" in lat:
                    print(f"ERROR: {lat['error']}")
                else:
                    print(f"avg={lat['avg_ms']:.1f}ms")

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

    # ─── Analysis ───
    base_label = "BF16"
    if base_label not in all_embeddings:
        print("ERROR: BF16 baseline not available")
        return

    print(f"\n{'='*80}")
    print(f"  ACCURACY COMPARISON vs BF16 (Qwen3-Embedding-4B)")
    print(f"{'='*80}")

    results = []

    # 1. Real text embeddings comparison
    print(f"\n── Real Text Embeddings ({len(REAL_TEXTS)} sentences) ──")
    base_real = all_embeddings[base_label]["real"]

    for label, _, _, _ in CONFIGS:
        if label == base_label or label not in all_embeddings:
            continue
        target_real = all_embeddings[label]["real"]
        r = analyze_embeddings(base_real, target_real, f"BF16 vs {label} (real)")
        results.append(r)
        print(f"\n  {label}:")
        print(f"    Cosine Sim:  mean={r['cosine_sim_mean']:.6f}  "
              f"min={r['cosine_sim_min']:.6f}  max={r['cosine_sim_max']:.6f}")
        print(f"    MAE={r['mae']:.6f}  MaxDiff={r['max_diff']:.6f}  P99={r['p99_diff']:.6f}")

    # 2. Random text embeddings comparison
    for bs in BATCH_SIZES:
        print(f"\n── Random Text Embeddings (batch={bs}, tokens={TOKEN_LENGTH}) ──")
        base_rand = all_embeddings[base_label]["random"][bs]

        for label, _, _, _ in CONFIGS:
            if label == base_label or label not in all_embeddings:
                continue
            target_rand = all_embeddings[label]["random"][bs]
            r = analyze_embeddings(base_rand, target_rand, f"BF16 vs {label} (random B={bs})")
            results.append(r)
            print(f"\n  {label}:")
            print(f"    Cosine Sim:  mean={r['cosine_sim_mean']:.6f}  "
                  f"min={r['cosine_sim_min']:.6f}")
            print(f"    MAE={r['mae']:.6f}  MaxDiff={r['max_diff']:.6f}")

    # ─── Summary Table ───
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Config':<22} {'VRAM':>6} ", end="")
    for bs in BATCH_SIZES:
        print(f"  {'B='+str(bs)+' Lat':>8}", end="")
    print(f"  {'CosSim(real)':>12}  {'MAE(real)':>10}")
    print("-" * 80)

    for label, _, _, _ in CONFIGS:
        if label not in all_embeddings:
            print(f"{label:<22} {'ERR':>6}")
            continue

        vram = all_vram.get(label, "?")
        line = f"{label:<22} {str(vram)+'M':>6} "

        for bs in BATCH_SIZES:
            lat = all_latency.get(label, {}).get(bs, {})
            avg = lat.get("avg_ms", 0)
            line += f"  {avg:>7.1f}ms"

        # Find real-text accuracy result
        real_result = None
        for r in results:
            if r["label"] == f"BF16 vs {label} (real)":
                real_result = r
                break

        if label == base_label:
            line += f"  {'1.000000':>12}  {'0.000000':>10}"
        elif real_result:
            line += f"  {real_result['cosine_sim_mean']:>12.6f}"
            line += f"  {real_result['mae']:>10.6f}"
        else:
            line += f"  {'N/A':>12}  {'N/A':>10}"

        print(line)

    # Save
    out_path = "/home/ubuntu/fp8_inference_toolkit/benchmark/vllm_inference_comparison.json"
    save_data = {
        "model": MODEL_ID,
        "token_length": TOKEN_LENGTH,
        "batch_sizes": BATCH_SIZES,
        "latency_runs": LATENCY_RUNS,
        "vram": all_vram,
        "latency": {k: {str(bs): v for bs, v in d.items()} for k, d in all_latency.items()},
        "accuracy": results,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
