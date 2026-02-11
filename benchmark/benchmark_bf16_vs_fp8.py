#!/usr/bin/env python3
"""
BF16 vs FP8 Comparison Benchmark

BF16과 FP8 모델의 성능을 비교합니다.
각 설정에 대해 vLLM 서버를 자동으로 시작하고 벤치마크를 실행합니다.

Requirements:
    pip install vllm requests numpy

Usage:
    python benchmark_bf16_vs_fp8.py

주의사항:
    1. 충분한 GPU 메모리 필요 (8B 모델: ~24GB for BF16, ~16GB for FP8)
    2. llmcompressor로 생성된 FP8 모델은 quantization 파라미터 없이 자동 감지됨
    3. 서버 시작에 시간이 걸릴 수 있음 (VL 8B: ~5-10분)
"""

import subprocess
import time
import json
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests library required. Install with: pip install requests")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_MODEL_LEN = 4096
OVERHEAD = 2
RUNS = 3

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
    "crystal", "diamond", "emerald", "ruby", "sapphire", "topaz", "opal", "jade",
    "volcano", "glacier", "canyon", "plateau", "mesa", "fjord", "delta", "lagoon",
    "algebra", "calculus", "geometry", "topology", "entropy", "inertia", "momentum", "friction",
    "crimson", "indigo", "violet", "amber", "scarlet", "ivory", "bronze", "silver",
    "abstract", "concrete", "dynamic", "static", "volatile", "mutable", "frozen", "elastic",
    "whisper", "thunder", "silence", "rhythm", "harmony", "melody", "chorus", "echo",
]

BATCH_SIZES = [1, 2, 4, 8, 16]
TOKEN_LENGTHS = [128, 256, 512, 1024, 2048, 4096]

# 비교할 모델 목록
# (Label, BF16 model ID, FP8 model ID, FP8 quantization method)
# FP8 quantization이 None이면 compressed-tensors 포맷으로 자동 감지
MODELS = [
    ("VL-2B", "Qwen/Qwen3-VL-Embedding-2B", "Forturne/Qwen3-VL-Embedding-2B-FP8", None),
    ("VL-8B", "Qwen/Qwen3-VL-Embedding-8B", "Forturne/Qwen3-VL-Embedding-8B-FP8", None),
]

SERVER_PORT = 8100

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_random_text(num_words: int) -> str:
    return ' '.join(random.choices(WORDS, k=num_words))


def get_gpu_memory_mib() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


@dataclass
class BenchmarkResult:
    batch_size: int
    target_tokens: int
    avg_latency_ms: float
    throughput_tokens_per_sec: float
    success: bool


class VLLMClient:
    def __init__(self, base_url: str, model_name: str, timeout: float = 300.0):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout

    def is_ready(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def embed(self, texts: List[str]) -> bool:
        payload = {"model": self.model_name, "input": texts}
        resp = requests.post(f"{self.base_url}/v1/embeddings", json=payload, timeout=self.timeout)
        return resp.status_code == 200


def start_vllm_server(model: str, quantization: Optional[str] = None, port: int = SERVER_PORT):
    """vLLM 서버 시작."""
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--runner", "pooling",
        "--convert", "embed",
        "--dtype", "auto",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.9",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--port", str(port),
    ]

    if quantization:
        cmd.extend(["--quantization", quantization])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc


def wait_for_server(url: str, timeout: int = 900) -> bool:
    """서버가 준비될 때까지 대기."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"{url}/v1/models", timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_single(client: VLLMClient, batch_size: int, target_tokens: int) -> BenchmarkResult:
    actual_tokens = min(target_tokens, MAX_MODEL_LEN - OVERHEAD)

    # Warmup
    try:
        client.embed([make_random_text(actual_tokens) for _ in range(batch_size)])
    except Exception:
        return BenchmarkResult(batch_size, target_tokens, 0, 0, False)

    latencies = []
    for _ in range(RUNS):
        texts = [make_random_text(actual_tokens) for _ in range(batch_size)]
        start = time.time()
        try:
            client.embed(texts)
            latencies.append((time.time() - start) * 1000)
        except Exception:
            return BenchmarkResult(batch_size, target_tokens, 0, 0, False)

    avg = np.mean(latencies)
    throughput = (batch_size * actual_tokens) / (avg / 1000)
    return BenchmarkResult(batch_size, target_tokens, float(avg), float(throughput), True)


def run_benchmark(client: VLLMClient, label: str) -> Dict:
    vram_idle = get_gpu_memory_mib()
    print(f"  VRAM (idle): {vram_idle} MiB")

    results = []
    for batch in BATCH_SIZES:
        for tokens in TOKEN_LENGTHS:
            result = benchmark_single(client, batch, tokens)
            results.append(result)
            status = f"{result.avg_latency_ms:.1f}ms" if result.success else "FAIL"
            print(f"    Batch={batch}, Tokens={tokens} -> {status}")

    vram_after = get_gpu_memory_mib()

    return {
        "label": label,
        "vram_idle_mib": vram_idle,
        "vram_after_mib": vram_after,
        "results": [asdict(r) for r in results],
    }


def main():
    base_url = f"http://localhost:{SERVER_PORT}"
    all_results = {}

    for model_label, bf16_model, fp8_model, fp8_quant in MODELS:
        print(f"\n{'#'*70}")
        print(f"# Model: {model_label}")
        print(f"{'#'*70}")

        model_results = {}

        # BF16과 FP8 설정
        configs = [
            ("BF16", bf16_model, None),
            ("FP8", fp8_model, fp8_quant),
        ]

        for dtype_label, model_id, quant in configs:
            print(f"\n## {dtype_label}: {model_id}")
            if quant:
                print(f"   Quantization: {quant}")
            else:
                print(f"   Quantization: auto-detect")

            proc = start_vllm_server(model_id, quant, SERVER_PORT)
            try:
                print("  Waiting for server...")
                if not wait_for_server(base_url, timeout=900):
                    print("  FAILED to start server!")
                    model_results[dtype_label] = {"error": "Server failed to start"}
                    proc.terminate()
                    continue

                print("  Server ready!")
                time.sleep(5)

                client = VLLMClient(base_url, model_id)
                model_results[dtype_label] = run_benchmark(client, f"{model_label} {dtype_label}")

            except Exception as e:
                print(f"  ERROR: {e}")
                model_results[dtype_label] = {"error": str(e)}
            finally:
                print("  Stopping server...")
                proc.terminate()
                proc.wait()
                time.sleep(5)

        all_results[model_label] = model_results

        # 중간 결과 저장
        with open("/home/ubuntu/benchmark_bf16_vs_fp8_progress.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # 최종 결과 저장
    output_file = f"/home/ubuntu/benchmark_bf16_vs_fp8_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # 요약 출력
    print("\n" + "="*80)
    print("SUMMARY: BF16 vs FP8")
    print("="*80)

    for model_label in all_results:
        mr = all_results[model_label]
        if "BF16" in mr and "FP8" in mr and "error" not in mr["BF16"] and "error" not in mr["FP8"]:
            bf16_vram = mr["BF16"]["vram_idle_mib"]
            fp8_vram = mr["FP8"]["vram_idle_mib"]
            vram_saving = bf16_vram - fp8_vram

            # Batch=8, Tokens=1024에서 speedup 계산
            bf16_res = {(r['batch_size'], r['target_tokens']): r for r in mr["BF16"]["results"]}
            fp8_res = {(r['batch_size'], r['target_tokens']): r for r in mr["FP8"]["results"]}

            key = (8, 1024)
            if key in bf16_res and key in fp8_res:
                bf16_ms = bf16_res[key]['avg_latency_ms']
                fp8_ms = fp8_res[key]['avg_latency_ms']
                speedup = bf16_ms / fp8_ms if fp8_ms > 0 else 0
            else:
                speedup = 0

            print(f"\n{model_label}:")
            print(f"  VRAM: BF16={bf16_vram} MiB, FP8={fp8_vram} MiB, Saving={vram_saving} MiB")
            print(f"  Speedup (B=8, T=1024): {speedup:.2f}x")


if __name__ == "__main__":
    main()
