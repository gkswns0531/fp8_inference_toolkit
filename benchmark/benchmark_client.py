#!/usr/bin/env python3
"""
Benchmark Client for Triton + vLLM Embedding Server

Triton gRPC streaming 또는 vLLM Direct API를 통한 벤치마크를 수행합니다.
다양한 batch size와 token length에 대한 latency/throughput을 측정합니다.

Requirements:
    pip install tritonclient[all] requests numpy

Usage:
    # Triton benchmark (gRPC)
    python benchmark_client.py --grpc-port 8001 --model-name qwen3_vl_embedding

    # vLLM Direct benchmark (HTTP)
    python benchmark_client.py --use-vllm-direct --vllm-port 8000 --vllm-model Qwen/Qwen3-VL-Embedding-8B

    # Save results to JSON
    python benchmark_client.py --output results.json

주의사항:
    1. Triton vLLM backend은 gRPC streaming만 지원 (HTTP 501 에러)
    2. OVERHEAD 2 tokens (EOS + safety margin) 적용
    3. 랜덤 텍스트로 캐시 영향 최소화
"""

import argparse
import json
import time
import random
import subprocess
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, asdict

try:
    import tritonclient.grpc as grpcclient
    TRITON_CLIENT_AVAILABLE = True
except ImportError:
    TRITON_CLIENT_AVAILABLE = False
    print("Warning: tritonclient not installed. Install with: pip install tritonclient[all]")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_MODEL_LEN = 4096
OVERHEAD = 2  # EOS token + safety margin

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

BATCH_SIZES = [1, 16]
TOKEN_LENGTHS = [1024]
RUNS = 3


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    batch_size: int
    target_tokens: int
    actual_tokens: int
    truncated: bool
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    throughput_tokens_per_sec: float
    success: bool


TOKENIZER = None


def init_tokenizer(model_name: str):
    """Tokenizer 로드 (정확한 토큰 수 truncation용)."""
    global TOKENIZER
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"Tokenizer loaded: {model_name}")


def make_random_text(num_tokens: int) -> str:
    """Generate random text truncated to exactly num_tokens tokens."""
    # 여유 있게 단어를 생성 (토큰:단어 비율 ~1.1x)
    raw = ' '.join(random.choices(WORDS, k=int(num_tokens * 1.5)))
    if TOKENIZER is None:
        return raw
    ids = TOKENIZER.encode(raw, add_special_tokens=False)[:num_tokens - OVERHEAD]
    return TOKENIZER.decode(ids, skip_special_tokens=True)


def get_gpu_memory_mib() -> Optional[int]:
    """Get current GPU memory usage in MiB."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Triton Client (gRPC Streaming)
# ─────────────────────────────────────────────────────────────────────────────

class TritonEmbeddingClient:
    """Client for Triton vLLM Backend embedding via gRPC streaming."""

    def __init__(
        self,
        host: str = "localhost",
        grpc_port: int = 8001,
        model_name: str = "qwen3_vl_embedding",
        timeout: float = 300.0,
    ):
        self.host = host
        self.grpc_port = grpc_port
        self.model_name = model_name
        self.timeout = timeout
        self._init_client()

    def _init_client(self):
        if not TRITON_CLIENT_AVAILABLE:
            raise RuntimeError("tritonclient is not available")
        self.client = grpcclient.InferenceServerClient(
            url=f"{self.host}:{self.grpc_port}")

    def is_server_ready(self) -> bool:
        try:
            return self.client.is_server_ready()
        except Exception:
            return False

    def is_model_ready(self) -> bool:
        try:
            return self.client.is_model_ready(self.model_name)
        except Exception:
            return False

    def embed(self, texts: List[str]) -> bool:
        """Send embedding request via gRPC streaming (decoupled model)."""
        import queue

        result_queue: queue.Queue = queue.Queue()
        expected = len(texts)

        def callback(result, error):
            if error:
                result_queue.put(error)
            else:
                result_queue.put(result)

        self.client.start_stream(callback=callback)
        try:
            for text in texts:
                embed_req = json.dumps({"input": text})
                inputs = [
                    grpcclient.InferInput("text_input", [1], "BYTES"),
                    grpcclient.InferInput("embedding_request", [1], "BYTES"),
                ]
                inputs[0].set_data_from_numpy(np.array([""], dtype=object))
                inputs[1].set_data_from_numpy(np.array([embed_req], dtype=object))
                self.client.async_stream_infer(
                    model_name=self.model_name, inputs=inputs)

            # Collect all responses
            for _ in range(expected):
                result = result_queue.get(timeout=self.timeout)
                if isinstance(result, Exception):
                    raise result
        finally:
            self.client.stop_stream()

        return True


# ─────────────────────────────────────────────────────────────────────────────
# vLLM Direct Client (HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class VLLMDirectClient:
    """Direct client for vLLM OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        model_name: str = "Qwen/Qwen3-VL-Embedding-8B",
        timeout: float = 300.0,
    ):
        self.base_url = f"http://{host}:{port}"
        self.model_name = model_name
        self.timeout = timeout

    def is_server_ready(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def is_model_ready(self) -> bool:
        return self.is_server_ready()

    def embed(self, texts: List[str]) -> bool:
        """Send embedding request. Returns True on success."""
        payload = {"model": self.model_name, "input": texts}
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_single(
    client,
    batch_size: int,
    target_tokens: int,
    runs: int = RUNS,
) -> BenchmarkResult:
    """Run benchmark for one (batch_size, token_length) pair."""
    if TOKENIZER is not None:
        actual_tokens = target_tokens  # tokenizer가 정확히 truncate
    else:
        actual_tokens = min(target_tokens, MAX_MODEL_LEN - OVERHEAD)
    truncated = actual_tokens < target_tokens

    # Warmup
    warmup_texts = [make_random_text(actual_tokens) for _ in range(batch_size)]
    try:
        client.embed(warmup_texts)
    except Exception as e:
        print(f" [Warmup failed: {e}]", end="")
        return BenchmarkResult(
            batch_size=batch_size, target_tokens=target_tokens,
            actual_tokens=actual_tokens, truncated=truncated,
            avg_latency_ms=0, p50_latency_ms=0, p99_latency_ms=0,
            throughput_tokens_per_sec=0, success=False,
        )

    # Measure
    latencies = []
    for _ in range(runs):
        texts = [make_random_text(actual_tokens) for _ in range(batch_size)]
        start = time.time()
        try:
            client.embed(texts)
        except Exception as e:
            print(f" [Run failed: {e}]", end="")
            return BenchmarkResult(
                batch_size=batch_size, target_tokens=target_tokens,
                actual_tokens=actual_tokens, truncated=truncated,
                avg_latency_ms=0, p50_latency_ms=0, p99_latency_ms=0,
                throughput_tokens_per_sec=0, success=False,
            )
        latencies.append((time.time() - start) * 1000)

    avg = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    total_tokens = batch_size * target_tokens
    throughput = total_tokens / (avg / 1000)

    return BenchmarkResult(
        batch_size=batch_size, target_tokens=target_tokens,
        actual_tokens=actual_tokens, truncated=truncated,
        avg_latency_ms=float(avg), p50_latency_ms=float(p50),
        p99_latency_ms=float(p99),
        throughput_tokens_per_sec=float(throughput), success=True,
    )


def run_full_benchmark(client, label: str) -> List[BenchmarkResult]:
    """Run the full benchmark suite."""
    mem = get_gpu_memory_mib()

    print("=" * 70)
    print(f"Benchmark: {label}")
    print(f"Overhead: {OVERHEAD} token(s)")
    if mem is not None:
        print(f"GPU Memory: {mem} MiB")
    print("=" * 70)

    total = len(BATCH_SIZES) * len(TOKEN_LENGTHS)
    idx = 0
    results = []

    for batch in BATCH_SIZES:
        for tokens in TOKEN_LENGTHS:
            idx += 1
            print(f"\n[{idx}/{total}] Batch={batch}, Tokens={tokens}", end="", flush=True)

            result = benchmark_single(client, batch, tokens)
            results.append(result)

            if result.success:
                print()
                print(f"  Avg: {result.avg_latency_ms:.2f}ms | "
                      f"P50: {result.p50_latency_ms:.2f}ms | "
                      f"P99: {result.p99_latency_ms:.2f}ms | "
                      f"Throughput: {result.throughput_tokens_per_sec:.0f} tok/s")
            else:
                print(" -> FAILED")

    print("\n" + "=" * 70)
    print("Benchmark Complete!")

    mem_after = get_gpu_memory_mib()
    if mem_after is not None:
        print(f"GPU Memory (after): {mem_after} MiB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton/vLLM Embedding Server")

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--grpc-port", type=int, default=8001,
                        help="Triton gRPC port")
    parser.add_argument("--model-name", type=str, default="qwen3_vl_embedding",
                        help="Triton model name")
    parser.add_argument("--use-vllm-direct", action="store_true",
                        help="Use vLLM direct API instead of Triton")
    parser.add_argument("--vllm-port", type=int, default=8000,
                        help="vLLM direct server port")
    parser.add_argument("--vllm-model", type=str,
                        default="Qwen/Qwen3-VL-Embedding-8B",
                        help="vLLM model name for API calls")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated batch sizes (e.g. 1,16)")
    parser.add_argument("--token-lengths", type=str, default=None,
                        help="Comma-separated token lengths (e.g. 64,128,256)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer model name for exact token truncation")

    args = parser.parse_args()

    # Override globals if CLI args provided
    global BATCH_SIZES, TOKEN_LENGTHS
    if args.batch_sizes:
        BATCH_SIZES = [int(x) for x in args.batch_sizes.split(",")]
    if args.token_lengths:
        TOKEN_LENGTHS = [int(x) for x in args.token_lengths.split(",")]
    if args.tokenizer:
        init_tokenizer(args.tokenizer)

    # Initialize client
    if args.use_vllm_direct:
        if not REQUESTS_AVAILABLE:
            print("ERROR: requests library required for vLLM direct mode")
            return 1
        label = f"vLLM Direct - {args.vllm_model}"
        print(f"Mode: vLLM Direct HTTP (port {args.vllm_port})")
        client = VLLMDirectClient(
            host=args.host, port=args.vllm_port,
            model_name=args.vllm_model,
        )
    else:
        if not TRITON_CLIENT_AVAILABLE:
            print("ERROR: tritonclient required for Triton mode")
            return 1
        label = f"Triton vLLM Backend - {args.model_name}"
        print(f"Mode: Triton gRPC streaming (port {args.grpc_port})")
        client = TritonEmbeddingClient(
            host=args.host, grpc_port=args.grpc_port,
            model_name=args.model_name,
        )

    # Check server
    print("Checking server...", end=" ", flush=True)
    if not client.is_server_ready():
        print("FAIL - server not ready")
        return 1
    if not client.is_model_ready():
        print("FAIL - model not loaded")
        return 1
    print("OK")

    # Run benchmark
    results = run_full_benchmark(client, label)

    # Save results
    if args.output:
        data = {
            "label": label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_memory_mib": get_gpu_memory_mib(),
            "config": {
                "batch_sizes": BATCH_SIZES,
                "token_lengths": TOKEN_LENGTHS,
                "runs": RUNS,
                "overhead": OVERHEAD,
            },
            "results": [asdict(r) for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
