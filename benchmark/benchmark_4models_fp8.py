#!/usr/bin/env python3
"""
4-Model FP8 Benchmark via vLLM Direct

4개 FP8 모델을 순차적으로 vLLM 서버에 올려 추론 레이턴시를 비교합니다.
각 모델마다: 서버 시작 → 벤치마크 (batch=1,16 / tokens=1024) → 서버 종료

Usage:
    python benchmark_4models_fp8.py
"""

import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "Qwen3-Embedding-0.6B-FP8",
        "model_path": "Forturne/Qwen3-Embedding-0.6B-FP8",
        "dtype": "auto",
        "max_model_len": 4096,
        "gpu_mem": 0.80,
    },
    {
        "name": "bge-m3-FP8",
        "model_path": "Forturne/bge-m3-FP8",
        "dtype": "auto",
        "max_model_len": 4096,
        "gpu_mem": 0.80,
    },
    {
        "name": "Qwen3-VL-Embedding-2B-FP8",
        "model_path": "Forturne/Qwen3-VL-Embedding-2B-FP8",
        "dtype": "auto",
        "max_model_len": 4096,
        "gpu_mem": 0.80,
    },
    {
        "name": "Qwen3-VL-Embedding-8B-FP8",
        "model_path": "Forturne/Qwen3-VL-Embedding-8B-FP8",
        "dtype": "auto",
        "max_model_len": 4096,
        "gpu_mem": 0.85,
    },
]

PORT = 8000
BATCH_SIZES = [1, 16]
TOKEN_LENGTH = 1024
RUNS = 5
OVERHEAD = 2
WARMUP_RUNS = 2
SERVER_STARTUP_TIMEOUT = 300  # seconds

WORDS = [
    "the", "cat", "dog", "house", "car", "book", "computer", "phone",
    "water", "food", "happy", "sad", "big", "small", "red", "blue",
    "run", "walk", "eat", "sleep", "work", "play", "think", "feel",
    "mountain", "river", "ocean", "forest", "desert", "island", "valley", "bridge",
    "python", "rust", "java", "swift", "ruby", "perl", "scala", "kotlin",
    "matrix", "vector", "tensor", "graph", "queue", "stack", "tree", "node",
    "quantum", "photon", "neutron", "proton", "electron", "plasma", "gravity", "orbit",
    "dolphin", "eagle", "tiger", "panda", "falcon", "whale", "shark", "cobra",
    "crystal", "diamond", "emerald", "sapphire", "topaz", "opal", "jade", "amber",
    "volcano", "glacier", "canyon", "plateau", "mesa", "fjord", "delta", "lagoon",
    "algebra", "calculus", "geometry", "topology", "entropy", "inertia", "momentum", "friction",
    "whisper", "thunder", "silence", "rhythm", "harmony", "melody", "chorus", "echo",
]

RESULT_FILE = "/home/ubuntu/fp8_inference_toolkit/benchmark/4model_fp8_benchmark_results.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_random_text(num_words: int) -> str:
    import random
    return " ".join(random.choices(WORDS, k=num_words))


def get_gpu_memory_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def kill_port(port: int):
    """포트를 점유 중인 프로세스 강제 종료."""
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        for pid in out.split("\n"):
            if pid:
                os.kill(int(pid), signal.SIGKILL)
        time.sleep(3)
    except Exception:
        pass


def wait_for_server(port: int, proc: subprocess.Popen, timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        # 프로세스가 죽었으면 즉시 실패
        if proc.poll() is not None:
            return False
        try:
            resp = requests.get(f"http://localhost:{port}/v1/models", timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_vllm_server(model_cfg: dict) -> subprocess.Popen:
    # 포트 정리
    kill_port(PORT)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_cfg["model_path"],
        "--dtype", model_cfg["dtype"],
        "--max-model-len", str(model_cfg["max_model_len"]),
        "--gpu-memory-utilization", str(model_cfg["gpu_mem"]),
        "--trust-remote-code",
        "--runner", "pooling",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--no-enable-prefix-caching",
    ]
    log_path = f"/tmp/vllm_{model_cfg['name']}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    return proc


def stop_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass
    # 포트와 GPU 메모리 해제 대기
    kill_port(PORT)
    time.sleep(5)


def run_benchmark(model_name: str, model_path: str) -> list[dict]:
    actual_tokens = min(TOKEN_LENGTH, 4096 - OVERHEAD)
    results = []

    for batch_size in BATCH_SIZES:
        label = f"  batch={batch_size}, tokens={TOKEN_LENGTH}"
        print(label, end="", flush=True)

        # Warmup
        for _ in range(WARMUP_RUNS):
            texts = [make_random_text(actual_tokens) for _ in range(batch_size)]
            try:
                requests.post(
                    f"http://localhost:{PORT}/v1/embeddings",
                    json={"model": model_path, "input": texts},
                    timeout=300,
                )
            except Exception as e:
                print(f" WARMUP FAILED: {e}")
                results.append({
                    "model": model_name, "batch_size": batch_size,
                    "token_length": TOKEN_LENGTH, "success": False,
                })
                continue

        # Measure
        latencies = []
        for _ in range(RUNS):
            texts = [make_random_text(actual_tokens) for _ in range(batch_size)]
            t0 = time.time()
            resp = requests.post(
                f"http://localhost:{PORT}/v1/embeddings",
                json={"model": model_path, "input": texts},
                timeout=300,
            )
            elapsed_ms = (time.time() - t0) * 1000
            if resp.status_code != 200:
                detail = resp.text[:200] if resp.text else ""
                print(f" FAILED (HTTP {resp.status_code}: {detail})")
                results.append({
                    "model": model_name, "batch_size": batch_size,
                    "token_length": TOKEN_LENGTH, "success": False,
                })
                break
            latencies.append(elapsed_ms)

        if latencies and len(latencies) == RUNS:
            avg = float(np.mean(latencies))
            p50 = float(np.percentile(latencies, 50))
            p99 = float(np.percentile(latencies, 99))
            throughput = (batch_size * TOKEN_LENGTH) / (avg / 1000)
            gpu_mem = get_gpu_memory_mib()

            result = {
                "model": model_name,
                "batch_size": batch_size,
                "token_length": TOKEN_LENGTH,
                "avg_latency_ms": round(avg, 2),
                "p50_latency_ms": round(p50, 2),
                "p99_latency_ms": round(p99, 2),
                "throughput_tok_s": round(throughput, 0),
                "gpu_memory_mib": gpu_mem,
                "success": True,
            }
            results.append(result)
            print(f"  -> avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms  "
                  f"throughput={throughput:.0f} tok/s  GPU={gpu_mem}MiB")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("4-Model FP8 Benchmark (vLLM Direct)")
    print(f"Configs: batch_sizes={BATCH_SIZES}, token_length={TOKEN_LENGTH}, runs={RUNS}")
    print("=" * 70)

    all_results = []

    for i, model_cfg in enumerate(MODELS):
        name = model_cfg["name"]
        print(f"\n{'─' * 70}")
        print(f"[{i+1}/{len(MODELS)}] {name}")
        print(f"  Model: {model_cfg['model_path']}")
        print(f"{'─' * 70}")

        # Start server
        print("  Starting vLLM server...", end=" ", flush=True)
        proc = start_vllm_server(model_cfg)

        if not wait_for_server(PORT, proc):
            print("TIMEOUT - server failed to start")
            print(f"  Check log: /tmp/vllm_{name}.log")
            # 로그 마지막 10줄 출력
            try:
                with open(f"/tmp/vllm_{name}.log") as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(f"    {line.rstrip()}")
            except Exception:
                pass
            stop_server(proc)
            for bs in BATCH_SIZES:
                all_results.append({
                    "model": name, "batch_size": bs,
                    "token_length": TOKEN_LENGTH, "success": False,
                })
            continue
        print("READY")

        # Benchmark
        results = run_benchmark(name, model_cfg["model_path"])
        all_results.extend(results)

        # Stop server
        print(f"  Stopping server...", end=" ", flush=True)
        stop_server(proc)
        print("OK")

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<32} {'Batch':>5} {'Avg(ms)':>9} {'P50(ms)':>9} {'P99(ms)':>9} {'Tok/s':>10} {'GPU(MiB)':>9}")
    print("─" * 90)

    for r in all_results:
        if r["success"]:
            print(f"{r['model']:<32} {r['batch_size']:>5} "
                  f"{r['avg_latency_ms']:>9.1f} {r['p50_latency_ms']:>9.1f} "
                  f"{r['p99_latency_ms']:>9.1f} {r['throughput_tok_s']:>10.0f} "
                  f"{r.get('gpu_memory_mib', 'N/A'):>9}")
        else:
            print(f"{r['model']:<32} {r['batch_size']:>5}      FAILED")

    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "batch_sizes": BATCH_SIZES,
            "token_length": TOKEN_LENGTH,
            "runs": RUNS,
            "warmup_runs": WARMUP_RUNS,
        },
        "results": all_results,
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
