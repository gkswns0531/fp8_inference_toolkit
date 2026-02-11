#!/usr/bin/env python3
"""
BF16 vs FP8 vs INT4-TC embedding output comparison.

BF16을 ground truth로 두고, FP8/INT4-TC의 출력 오차를 측정합니다.

Metrics:
  - Cosine similarity (1.0 = identical direction)
  - L2 distance (normalized)
  - Max absolute error
  - Mean absolute error
  - Relative L2 error (||delta|| / ||ref||)
"""

import subprocess
import time
import json
import sys
import numpy as np
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SERVER_PORT = 8100
MODEL_ID = "Qwen/Qwen3-Embedding-4B"

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we interact with technology.",
    "CUDA tensor cores enable efficient matrix multiplication for deep learning workloads.",
    "Climate change poses significant challenges to global food security and biodiversity.",
    "The Pythagorean theorem states that a squared plus b squared equals c squared.",
    "Quantum computing leverages superposition and entanglement to solve complex problems.",
    "Natural language processing enables machines to understand human language.",
    "The mitochondria is the powerhouse of the cell, producing ATP through oxidative phosphorylation.",
]

CONFIGS = [
    ("BF16", [], None),
    ("FP8", ["--quantization", "fp8"], None),
    ("INT4-TC", ["--quantization", "int4-tc"],
     "vllm.model_executor.layers.quantization.w4a4_int4tc"),
    ("W4A16-Marlin", ["--quantization", "w4a16-int4tc"],
     "vllm.model_executor.layers.quantization.w4a16_int4tc"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Server management (reused from benchmark)
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


def start_server(extra_args, pre_import=None, label=""):
    log_path = f"/tmp/vllm_compare_{label}.log"
    log_file = open(log_path, "w")
    base_args = [
        "--model", MODEL_ID,
        "--convert", "embed",
        "--dtype", "auto",
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.85",
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
        cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server"] + base_args

    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            text=True, cwd="/tmp")


def wait_for_server(proc, timeout=300):
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


def get_embeddings(sentences):
    """Get embeddings for a list of sentences. Returns np array [N, dim]."""
    url = f"http://localhost:{SERVER_PORT}/v1/embeddings"
    # Send one by one to avoid output size issues
    all_embs = []
    for sent in sentences:
        resp = requests.post(url, json={"model": MODEL_ID, "input": sent}, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"API error: {resp.text[:200]}")
        data = resp.json()
        all_embs.append(data["data"][0]["embedding"])
    return np.array(all_embs, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison metrics
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    """Per-row cosine similarity."""
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-12)


def compute_metrics(ref, test):
    """Compute comparison metrics. ref, test: [N, dim]."""
    cos_sim = cosine_similarity(ref, test)
    abs_err = np.abs(ref - test)
    diff_norms = np.linalg.norm(ref - test, axis=1)
    ref_norms = np.linalg.norm(ref, axis=1)
    rel_l2 = diff_norms / (ref_norms + 1e-12)

    return {
        "cosine_sim_mean": float(np.mean(cos_sim)),
        "cosine_sim_min": float(np.min(cos_sim)),
        "cosine_sim_std": float(np.std(cos_sim)),
        "max_abs_err": float(np.max(abs_err)),
        "mean_abs_err": float(np.mean(abs_err)),
        "rel_l2_mean": float(np.mean(rel_l2)),
        "rel_l2_max": float(np.max(rel_l2)),
        "per_sentence_cosine": cos_sim.tolist(),
        "per_sentence_rel_l2": rel_l2.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_embeddings = {}

    for label, extra_args, pre_import in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Collecting embeddings: {label}")
        print(f"{'='*60}")

        kill_server_on_port(SERVER_PORT)
        proc = start_server(extra_args, pre_import, label=label)

        try:
            print("  Starting server...", flush=True)
            if not wait_for_server(proc):
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                log_path = f"/tmp/vllm_compare_{label}.log"
                try:
                    with open(log_path) as f:
                        lines = f.read().strip().split("\n")[-15:]
                    print(f"  FAILED:\n" + "\n".join(lines))
                except Exception:
                    print("  FAILED: no log")
                continue

            time.sleep(3)
            print("  Server ready. Collecting embeddings...", flush=True)

            # Warmup
            requests.post(
                f"http://localhost:{SERVER_PORT}/v1/embeddings",
                json={"model": MODEL_ID, "input": "warmup"}, timeout=60)

            embs = get_embeddings(TEST_SENTENCES)
            all_embeddings[label] = embs
            print(f"  Got {embs.shape[0]} embeddings, dim={embs.shape[1]}")

        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            kill_server_on_port(SERVER_PORT)
            time.sleep(10)

    # ─── Compare ───
    if "BF16" not in all_embeddings:
        print("\nERROR: BF16 baseline missing, cannot compare.")
        return

    ref = all_embeddings["BF16"]

    print(f"\n{'='*70}")
    print(f"  OUTPUT COMPARISON (BF16 = ground truth)")
    print(f"{'='*70}")

    results = {}
    for label in ["FP8", "INT4-TC", "W4A16-Marlin"]:
        if label not in all_embeddings:
            print(f"\n  {label}: SKIPPED (no data)")
            continue

        test = all_embeddings[label]
        m = compute_metrics(ref, test)
        results[label] = m

        print(f"\n  {label} vs BF16:")
        print(f"  {'─'*50}")
        print(f"  Cosine similarity:  mean={m['cosine_sim_mean']:.6f}  "
              f"min={m['cosine_sim_min']:.6f}  std={m['cosine_sim_std']:.2e}")
        print(f"  Absolute error:     max={m['max_abs_err']:.6f}  "
              f"mean={m['mean_abs_err']:.6f}")
        print(f"  Relative L2 error:  mean={m['rel_l2_mean']:.6f}  "
              f"max={m['rel_l2_max']:.6f}")
        print()
        print(f"  Per-sentence cosine similarity:")
        for i, (sent, cos) in enumerate(zip(TEST_SENTENCES, m["per_sentence_cosine"])):
            short = sent[:60] + "..." if len(sent) > 60 else sent
            print(f"    [{i}] {cos:.6f}  \"{short}\"")

    # Save
    out_path = "/home/ubuntu/fp8_inference_toolkit/benchmark/output_comparison.json"
    save_data = {
        "sentences": TEST_SENTENCES,
        "metrics": results,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
