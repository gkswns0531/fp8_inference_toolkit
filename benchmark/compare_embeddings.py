#!/usr/bin/env python3
"""
BF16 vs FP8(online) vs FP8(offline) 임베딩 출력 텐서 비교

동일한 입력에 대해 3가지 설정의 임베딩 출력 차이를 측정합니다.
"""

import subprocess
import time
import json
import random
import numpy as np
from typing import List, Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
SERVER_PORT = 8100
MAX_MODEL_LEN = 4096
OVERHEAD = 2
BATCH_SIZES = [1, 16]
TOKEN_LENGTH = 1024

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

# (label, model_id, extra_args)
CONFIGS = [
    ("BF16",          "Qwen/Qwen3-VL-Embedding-2B", []),
    ("FP8-online",    "Qwen/Qwen3-VL-Embedding-2B", ["--quantization", "fp8"]),
    ("FP8-offline",   "Forturne/Qwen3-VL-Embedding-2B-FP8", []),
]


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


def get_embeddings(model_name: str, texts: List[str]) -> List[List[float]]:
    url = f"http://localhost:{SERVER_PORT}/v1/embeddings"
    resp = requests.post(url, json={"model": model_name, "input": texts}, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()["data"]
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


def analyze_diff(base: np.ndarray, target: np.ndarray, label: str, batch_size: int):
    """두 임베딩 세트의 차이를 분석."""
    # base, target: (batch_size, embed_dim)
    diffs = target - base  # (batch_size, embed_dim)
    abs_diffs = np.abs(diffs)

    # 전체 배치에 대한 통계
    all_abs = abs_diffs.flatten()

    print(f"\n  [{label}] Batch={batch_size}, Tokens={TOKEN_LENGTH}")
    print(f"  Embedding dim: {base.shape[1]}")
    print(f"  Total diff values: {len(all_abs)} ({batch_size} x {base.shape[1]})")
    print(f"  ── Absolute Difference ──")
    print(f"    Mean:   {np.mean(all_abs):.8f}")
    print(f"    Median: {np.median(all_abs):.8f}")
    print(f"    Std:    {np.std(all_abs):.8f}")
    print(f"    Max:    {np.max(all_abs):.8f}")
    print(f"    Min:    {np.min(all_abs):.8f}")

    # Percentile 분포
    percentiles = [50, 90, 95, 99, 99.9]
    pvals = np.percentile(all_abs, percentiles)
    print(f"  ── Percentile Distribution ──")
    for p, v in zip(percentiles, pvals):
        print(f"    P{p:<5}: {v:.8f}")

    # Cosine similarity (per sample)
    cos_sims = []
    for i in range(batch_size):
        cos = np.dot(base[i], target[i]) / (np.linalg.norm(base[i]) * np.linalg.norm(target[i]))
        cos_sims.append(cos)
    print(f"  ── Cosine Similarity ──")
    print(f"    Mean:   {np.mean(cos_sims):.10f}")
    print(f"    Min:    {np.min(cos_sims):.10f}")
    print(f"    Max:    {np.max(cos_sims):.10f}")

    # 히스토그램 (텍스트)
    bins = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    hist, _ = np.histogram(all_abs, bins=bins)
    print(f"  ── Distribution Histogram ──")
    for i in range(len(bins) - 1):
        bar = "█" * int(hist[i] / len(all_abs) * 50)
        pct = hist[i] / len(all_abs) * 100
        print(f"    [{bins[i]:.0e}, {bins[i+1]:.0e}): {hist[i]:>6} ({pct:5.1f}%) {bar}")

    return {
        "label": label,
        "batch_size": batch_size,
        "embed_dim": int(base.shape[1]),
        "mean_abs_diff": float(np.mean(all_abs)),
        "median_abs_diff": float(np.median(all_abs)),
        "max_abs_diff": float(np.max(all_abs)),
        "std_abs_diff": float(np.std(all_abs)),
        "cosine_sim_mean": float(np.mean(cos_sims)),
        "cosine_sim_min": float(np.min(cos_sims)),
        "percentiles": {f"p{p}": float(v) for p, v in zip(percentiles, pvals)},
    }


def main():
    random.seed(42)
    np.random.seed(42)

    # 고정 입력 텍스트 생성
    actual_tokens = min(TOKEN_LENGTH, MAX_MODEL_LEN - OVERHEAD)
    test_inputs = {}
    for batch_size in BATCH_SIZES:
        test_inputs[batch_size] = [
            " ".join(random.choices(WORDS, k=actual_tokens))
            for _ in range(batch_size)
        ]

    # 각 설정별 임베딩 수집
    all_embeddings = {}

    for label, model_id, extra_args in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {label}: {model_id}")
        if extra_args:
            print(f"  Extra args: {' '.join(extra_args)}")
        print(f"{'='*60}")

        proc = start_server(model_id, extra_args)
        try:
            print("  Starting server...")
            if not wait_for_server():
                print("  FAILED: server timeout")
                continue

            time.sleep(3)
            print("  Server ready!")

            all_embeddings[label] = {}
            for batch_size in BATCH_SIZES:
                texts = test_inputs[batch_size]
                print(f"  Getting embeddings batch={batch_size}...")
                embs = get_embeddings(model_id, texts)
                all_embeddings[label][batch_size] = np.array(embs)
                print(f"    Shape: {all_embeddings[label][batch_size].shape}")

        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            proc.terminate()
            proc.wait()
            time.sleep(5)

    # 비교 분석
    print(f"\n{'='*70}")
    print(f"  COMPARISON: BF16 vs FP8(online) vs FP8(offline)")
    print(f"{'='*70}")

    results = []
    base_label = "BF16"
    if base_label not in all_embeddings:
        print("ERROR: BF16 baseline not available")
        return

    for cmp_label in ["FP8-online", "FP8-offline"]:
        if cmp_label not in all_embeddings:
            print(f"  SKIP: {cmp_label} not available")
            continue

        for batch_size in BATCH_SIZES:
            base = all_embeddings[base_label][batch_size]
            target = all_embeddings[cmp_label][batch_size]
            r = analyze_diff(base, target, f"BF16 vs {cmp_label}", batch_size)
            results.append(r)

    # FP8-online vs FP8-offline 직접 비교
    if "FP8-online" in all_embeddings and "FP8-offline" in all_embeddings:
        for batch_size in BATCH_SIZES:
            base = all_embeddings["FP8-online"][batch_size]
            target = all_embeddings["FP8-offline"][batch_size]
            r = analyze_diff(base, target, "FP8-online vs FP8-offline", batch_size)
            results.append(r)

    # 요약 테이블
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Comparison':<30} {'Batch':>5} {'Mean Diff':>12} {'Max Diff':>12} {'Cosine Sim':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<30} {r['batch_size']:>5}"
              f" {r['mean_abs_diff']:>12.8f}"
              f" {r['max_abs_diff']:>12.8f}"
              f" {r['cosine_sim_mean']:>12.10f}")

    # Save
    out = "/home/ubuntu/fp8_inference_toolkit/benchmark/embedding_diff_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
