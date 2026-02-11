#!/usr/bin/env python3
"""
BF16 vs FP8 vs INT4-TC 3-way Benchmark

Qwen3-Embedding-4B 모델에 대해 3가지 quantization 방식의 embedding latency를 비교합니다.

Configs:
  1. BF16:     No quantization (baseline)
  2. FP8:      FP8 dynamic quantization
  3. INT4-TC:  INT4x INT4 native TC GEMM with fused EVT epilogue

Usage:
    python run_5way_benchmark.py
"""

import subprocess
import time
import json
import random
import numpy as np
from typing import List, Optional

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SERVER_PORT = 8100
MAX_MODEL_LEN = 2048
OVERHEAD = 2
RUNS = 5
BATCH_SIZES = [1, 8]
TOKEN_LENGTH = 512

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

MODEL_ID = "Qwen/Qwen3-Embedding-4B"

# (label, model_id, extra_vllm_args, pre_import_module)
# pre_import_module: module to import before starting vLLM (for registration)
CONFIGS = [
    ("BF16", MODEL_ID, [], None),
    ("FP8", MODEL_ID, ["--quantization", "fp8"], None),
    ("INT4-TC", MODEL_ID, ["--quantization", "int4-tc"],
     "vllm.model_executor.layers.quantization.w4a4_int4tc"),
    ("W4A16-Marlin", MODEL_ID, ["--quantization", "w4a16-int4tc"],
     "vllm.model_executor.layers.quantization.w4a16_int4tc"),
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


SERVER_LOG = "/tmp/vllm_server.log"


def start_server(model: str, extra_args: List[str],
                 pre_import: Optional[str] = None,
                 label: str = "") -> subprocess.Popen:
    log_path = f"/tmp/vllm_server_{label}.log" if label else SERVER_LOG
    log_file = open(log_path, "w")
    base_args = [
        "--model", model,
        "--convert", "embed",
        "--dtype", "auto",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.85",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--port", str(SERVER_PORT),
    ] + extra_args

    if pre_import:
        # Custom ops need enforce-eager (not compatible with torch.compile/dynamo)
        if "--enforce-eager" not in base_args:
            base_args.append("--enforce-eager")
        # Import registration module first, then use make_arg_parser + run_server
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
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, cwd="/tmp")


def wait_for_server(proc: subprocess.Popen, timeout: int = 300) -> bool:
    url = f"http://localhost:{SERVER_PORT}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        # Check if server process has died
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


def kill_server_on_port(port: int) -> None:
    """Kill any process using the given port and related GPU processes."""
    try:
        out = subprocess.check_output(
            ["fuser", f"{port}/tcp"], stderr=subprocess.STDOUT, text=True)
        pids = out.strip().split()
        for pid in pids:
            subprocess.run(["kill", "-9", pid.strip()], capture_output=True)
        time.sleep(2)
    except Exception:
        pass
    # Also kill any VLLM EngineCore processes that might be holding GPU memory
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    all_results = {}

    for label, model_id, extra_args, pre_import in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {label}: {model_id}")
        if extra_args:
            print(f"  Args: {' '.join(extra_args)}")
        print(f"{'='*60}")

        # Ensure port is free
        kill_server_on_port(SERVER_PORT)

        proc = start_server(model_id, extra_args, pre_import, label=label)
        try:
            print("  Starting server...")
            if not wait_for_server(proc):
                # Read server log for error details
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                try:
                    log_path = f"/tmp/vllm_server_{label}.log"
                    with open(log_path, "r") as f:
                        log_content = f.read()
                    last_lines = log_content.strip().split("\n")[-20:]
                    err_msg = "\n".join(last_lines)
                except Exception:
                    err_msg = "server timeout, no log"
                print(f"  FAILED:\n{err_msg}")
                all_results[label] = {"error": err_msg}
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
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            kill_server_on_port(SERVER_PORT)
            time.sleep(10)

    # Print summary
    labels = [label for label, _, _, _ in CONFIGS]
    print(f"\n{'='*90}")
    print(f"  SUMMARY: Qwen3-Embedding-4B  |  Batch={{1,8}}, Tokens={TOKEN_LENGTH}")
    print(f"{'='*90}")

    header = f"{'Config':<18} {'VRAM':>8}"
    for b in BATCH_SIZES:
        header += f" {'B='+str(b)+' Avg':>10} {'B='+str(b)+' Thr':>12}"
    print(header)
    print("-" * 90)

    for label in labels:
        if label not in all_results or "error" in all_results[label]:
            err = all_results.get(label, {}).get("error", "unknown")
            print(f"{label:<18} {'ERROR':>8}  {str(err)[:60]}")
            continue
        d = all_results[label]
        vram = d.get("vram_mib", "?")
        r = {r["batch_size"]: r for r in d["results"]}

        line = f"{label:<18} {vram:>7} {'M':1}"
        for b in BATCH_SIZES:
            br = r.get(b, {})
            avg = br.get("avg_ms", 0)
            thr = br.get("throughput_tok_s", 0)
            line += f" {avg:>8.1f}ms {thr:>10.0f}t/s"
        print(line)

    # Save
    out_path = "/home/ubuntu/fp8_inference_toolkit/benchmark/3way_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
