#!/usr/bin/env python3
"""
Embedding Model Latency Benchmark

다양한 임베딩 모델의 레이턴시를 측정합니다.
vLLM의 LLM(runner="pooling")을 사용하여 직접 로드하고 embed()로 측정합니다.
test_data/에서 미리 준비된 텍스트를 로드합니다.

Requirements:
    pip install vllm numpy

Usage:
    # 모든 모델 벤치마크
    python benchmark/benchmark_embedding_latency.py

    # 특정 모델만
    python benchmark/benchmark_embedding_latency.py --model BAAI/bge-m3

    # 커스텀 설정
    python benchmark/benchmark_embedding_latency.py \
        --batch-sizes 1,8,16 \
        --input-lengths 128,512,2048 \
        --num-runs 20
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODELS = [
    {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "max_context": 32768,
        "tokenizer_dir": "qwen3-embedding",
    },
    {
        "model_id": "Qwen/Qwen3-Embedding-4B",
        "max_context": 32768,
        "tokenizer_dir": "qwen3-embedding",
    },
    {
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "max_context": 32768,
        "tokenizer_dir": "qwen3-embedding",
    },
    {
        "model_id": "Qwen/Qwen3-VL-Embedding-2B",
        "max_context": 32768,
        "tokenizer_dir": "qwen3-vl-embedding",
    },
    {
        "model_id": "Qwen/Qwen3-VL-Embedding-8B",
        "max_context": 32768,
        "tokenizer_dir": "qwen3-vl-embedding",
    },
    {
        "model_id": "BAAI/bge-m3",
        "max_context": 8192,
        "tokenizer_dir": "bge-m3",
    },
]

DEFAULT_BATCH_SIZES = [1, 4, 8, 16]
DEFAULT_INPUT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_NUM_WARMUP = 3
DEFAULT_NUM_RUNS = 10
DEFAULT_GPU_MEM = 0.90
DEFAULT_MAX_MODEL_LEN = 8192

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

WORDS = [
    "the", "cat", "dog", "house", "car", "book", "computer", "phone",
    "water", "food", "happy", "sad", "big", "small", "red", "blue",
    "mountain", "river", "ocean", "forest", "quantum", "photon", "neutron",
    "algebra", "calculus", "geometry", "topology", "entropy", "inertia",
    "whisper", "thunder", "silence", "rhythm", "harmony", "melody",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_random_text(num_words: int) -> str:
    return " ".join(random.choices(WORDS, k=num_words))


def load_test_text(tokenizer_dir: str, length: int) -> str | None:
    """Load pre-generated test text from test_data directory."""
    filepath = os.path.join(TEST_DATA_DIR, tokenizer_dir, f"{length}_tokens.txt")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def get_test_text(tokenizer_dir: str, length: int) -> str:
    """Load test text or fall back to random text."""
    text = load_test_text(tokenizer_dir, length)
    if text is not None:
        return text
    # Fallback: approximate tokens as 0.75 * num_words
    return make_random_text(int(length * 1.5))


def get_gpu_memory_mib() -> int | None:
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(
    model_id: str,
    tokenizer_dir: str,
    max_context: int,
    input_lengths: list[int],
    batch_sizes: list[int],
    num_warmup: int,
    num_runs: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> list[dict]:
    """Benchmark a single embedding model."""
    from vllm import LLM

    # Clamp input lengths to model max context
    effective_lengths = [l for l in input_lengths if l <= max_context]
    if len(effective_lengths) < len(input_lengths):
        skipped = [l for l in input_lengths if l > max_context]
        print(f"  Skipping lengths > {max_context}: {skipped}")

    model_max_len = min(max_model_len, max_context)

    print(f"  Loading model with vLLM (runner=pooling, max_model_len={model_max_len}) ...")
    llm = LLM(
        model=model_id,
        task="embed",
        max_model_len=model_max_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )

    gpu_mem_loaded = get_gpu_memory_mib()
    print(f"  GPU memory after load: {gpu_mem_loaded} MiB")

    results = []

    for length in effective_lengths:
        for batch_size in batch_sizes:
            print(f"    tokens={length:>6}, batch={batch_size:>2}", end="", flush=True)

            text = get_test_text(tokenizer_dir, length)

            # Warmup with random text to avoid cache effects
            for _ in range(num_warmup):
                warmup_texts = [make_random_text(int(length * 1.5)) for _ in range(batch_size)]
                llm.embed(warmup_texts)

            # Timed runs
            latencies = []
            for _ in range(num_runs):
                batch_texts = [text] * batch_size
                t0 = time.perf_counter()
                llm.embed(batch_texts)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)

            avg = float(np.mean(latencies))
            std = float(np.std(latencies))
            p50 = float(np.percentile(latencies, 50))
            p99 = float(np.percentile(latencies, 99))
            throughput = (batch_size * length) / (avg / 1000)

            result = {
                "model": model_id,
                "input_tokens": length,
                "batch_size": batch_size,
                "num_runs": num_runs,
                "avg_latency_ms": round(avg, 2),
                "std_latency_ms": round(std, 2),
                "p50_latency_ms": round(p50, 2),
                "p99_latency_ms": round(p99, 2),
                "throughput_tok_s": round(throughput, 0),
                "gpu_memory_mib": gpu_mem_loaded,
            }
            results.append(result)

            print(f"  -> avg={avg:.1f}ms  std={std:.1f}  p50={p50:.1f}ms  p99={p99:.1f}ms  {throughput:.0f} tok/s")

    # Cleanup GPU memory
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(3)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(all_results: list[dict]):
    """Print a comparison table of all results."""
    print("\n" + "=" * 110)
    print("SUMMARY: Embedding Model Latency Comparison")
    print("=" * 110)
    print(f"{'Model':<35} {'Tokens':>6} {'Batch':>5} {'Avg(ms)':>9} {'Std':>7} "
          f"{'P50(ms)':>9} {'P99(ms)':>9} {'Tok/s':>10} {'GPU(MiB)':>9}")
    print("─" * 110)

    for r in all_results:
        model_short = r["model"].split("/")[-1]
        print(f"{model_short:<35} {r['input_tokens']:>6} {r['batch_size']:>5} "
              f"{r['avg_latency_ms']:>9.1f} {r['std_latency_ms']:>7.1f} "
              f"{r['p50_latency_ms']:>9.1f} {r['p99_latency_ms']:>9.1f} "
              f"{r['throughput_tok_s']:>10.0f} {r.get('gpu_memory_mib', 'N/A'):>9}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding model latency with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model ID to benchmark (default: all models)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
        help=f"Comma-separated batch sizes (default: {DEFAULT_BATCH_SIZES})",
    )
    parser.add_argument(
        "--input-lengths",
        type=str,
        default=",".join(str(l) for l in DEFAULT_INPUT_LENGTHS),
        help=f"Comma-separated input token lengths (default: {DEFAULT_INPUT_LENGTHS})",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=DEFAULT_NUM_WARMUP,
        help=f"Number of warmup runs (default: {DEFAULT_NUM_WARMUP})",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"Number of timed runs (default: {DEFAULT_NUM_RUNS})",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"vLLM max_model_len override (default: {DEFAULT_MAX_MODEL_LEN})",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEM,
        help=f"GPU memory utilization (default: {DEFAULT_GPU_MEM})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: benchmark/embedding_latency_results.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    input_lengths = sorted(int(x) for x in args.input_lengths.split(","))

    # Filter models
    if args.model:
        models = [m for m in EMBEDDING_MODELS if m["model_id"] == args.model]
        if not models:
            print(f"ERROR: Model '{args.model}' not found in EMBEDDING_MODELS.")
            print(f"Available: {[m['model_id'] for m in EMBEDDING_MODELS]}")
            sys.exit(1)
    else:
        models = EMBEDDING_MODELS

    print("=" * 70)
    print("Embedding Model Latency Benchmark")
    print("=" * 70)
    print(f"  Models:       {[m['model_id'] for m in models]}")
    print(f"  Batch sizes:  {batch_sizes}")
    print(f"  Input lengths:{input_lengths}")
    print(f"  Warmup:       {args.num_warmup}")
    print(f"  Runs:         {args.num_runs}")
    print(f"  Max model len:{args.max_model_len}")
    print(f"  GPU mem util: {args.gpu_memory_utilization}")
    print()

    all_results = []

    for i, model_cfg in enumerate(models):
        model_id = model_cfg["model_id"]
        print(f"\n{'─' * 70}")
        print(f"[{i+1}/{len(models)}] {model_id}")
        print(f"  max_context={model_cfg['max_context']}, tokenizer_dir={model_cfg['tokenizer_dir']}")
        print(f"{'─' * 70}")

        try:
            results = benchmark_model(
                model_id=model_id,
                tokenizer_dir=model_cfg["tokenizer_dir"],
                max_context=model_cfg["max_context"],
                input_lengths=input_lengths,
                batch_sizes=batch_sizes,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print_summary_table(all_results)

    # Save results
    output_file = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "embedding_latency_results.json",
    )
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "batch_sizes": batch_sizes,
            "input_lengths": input_lengths,
            "num_warmup": args.num_warmup,
            "num_runs": args.num_runs,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "results": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
