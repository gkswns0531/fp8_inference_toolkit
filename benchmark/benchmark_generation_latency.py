#!/usr/bin/env python3
"""
Generation Model Latency Benchmark

생성형 모델의 TTFT(Time To First Token), 전체 생성 시간, 토큰 처리 속도를 측정합니다.
vLLM의 LLM()으로 직접 모델을 로드하고, generate()로 벤치마크합니다.
test_data/에서 미리 준비된 텍스트를 로드합니다.

Requirements:
    pip install vllm numpy

Usage:
    # 단일 모델 벤치마크
    python benchmark/benchmark_generation_latency.py \
        --model Qwen/Qwen3-Next-80B-A3B-Instruct \
        --tensor-parallel-size 2

    # 여러 출력 길이 테스트
    python benchmark/benchmark_generation_latency.py \
        --model openai/gpt-oss-120b \
        --tensor-parallel-size 4 \
        --input-lengths 1024,4096,8192 \
        --output-lengths 256,1024

    # 모든 모델 순차 실행
    python benchmark/benchmark_generation_latency.py
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

GENERATION_MODELS = [
    {
        "model_id": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "max_context": 262144,
        "tokenizer_dir": "qwen3-next",
    },
    {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "max_context": 262144,
        "tokenizer_dir": "qwen3-next",
    },
    {
        "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "max_context": 262144,
        "tokenizer_dir": "qwen3-next",
    },
    {
        "model_id": "openai/gpt-oss-120b",
        "max_context": 131072,
        "tokenizer_dir": "qwen3-next",
    },
]

DEFAULT_INPUT_LENGTHS = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_OUTPUT_LENGTHS = [256, 1024, 4096]
DEFAULT_NUM_WARMUP = 2
DEFAULT_NUM_RUNS = 5
DEFAULT_TP_SIZE = 1
DEFAULT_GPU_MEM = 0.90

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
    output_lengths: list[int],
    num_warmup: int,
    num_runs: int,
    tensor_parallel_size: int,
    max_model_len: int | None,
    gpu_memory_utilization: float,
) -> list[dict]:
    """Benchmark a single generation model."""
    from vllm import LLM, SamplingParams

    effective_max = max_model_len if max_model_len else max_context
    effective_lengths = [l for l in input_lengths if l < effective_max]
    if len(effective_lengths) < len(input_lengths):
        skipped = [l for l in input_lengths if l >= effective_max]
        print(f"  Skipping input lengths >= {effective_max}: {skipped}")

    print(f"  Loading model with vLLM (tp={tensor_parallel_size}, max_model_len={effective_max}) ...")
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=effective_max,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )

    gpu_mem_loaded = get_gpu_memory_mib()
    print(f"  GPU memory after load: {gpu_mem_loaded} MiB")

    results = []

    for input_len in effective_lengths:
        for output_len in output_lengths:
            # Ensure input + output fits within max context
            if input_len + output_len > effective_max:
                print(f"    SKIP input={input_len}, output={output_len} "
                      f"(total {input_len + output_len} > max {effective_max})")
                continue

            print(f"    input={input_len:>6}, output={output_len:>4}", end="", flush=True)

            text = get_test_text(tokenizer_dir, input_len)
            sampling_params = SamplingParams(
                max_tokens=output_len,
                temperature=0,
            )

            # Warmup with random text
            for _ in range(num_warmup):
                warmup_text = make_random_text(int(input_len * 1.5))
                llm.generate([warmup_text], sampling_params)

            # Timed runs
            total_latencies = []
            output_token_counts = []

            for _ in range(num_runs):
                t0 = time.perf_counter()
                outputs = llm.generate([text], sampling_params)
                total_elapsed_ms = (time.perf_counter() - t0) * 1000

                # Extract output info
                output_obj = outputs[0]
                generated_tokens = len(output_obj.outputs[0].token_ids)
                output_token_counts.append(generated_tokens)

                # TTFT: For offline/batch mode, we approximate TTFT as
                # total_time / generated_tokens (time per token) for the first token.
                # In online mode, TTFT would be measured via streaming.
                # Here we report total latency and per-token throughput.
                total_latencies.append(total_elapsed_ms)

            avg_total = float(np.mean(total_latencies))
            std_total = float(np.std(total_latencies))
            p50_total = float(np.percentile(total_latencies, 50))
            p99_total = float(np.percentile(total_latencies, 99))
            avg_output_tokens = float(np.mean(output_token_counts))

            # Throughput: output tokens per second
            output_tok_per_sec = avg_output_tokens / (avg_total / 1000) if avg_total > 0 else 0
            # Total throughput: (input + output) tokens per second
            total_tok_per_sec = (input_len + avg_output_tokens) / (avg_total / 1000) if avg_total > 0 else 0

            result = {
                "model": model_id,
                "input_tokens": input_len,
                "output_tokens_requested": output_len,
                "output_tokens_actual": round(avg_output_tokens, 1),
                "num_runs": num_runs,
                "avg_total_latency_ms": round(avg_total, 2),
                "std_total_latency_ms": round(std_total, 2),
                "p50_total_latency_ms": round(p50_total, 2),
                "p99_total_latency_ms": round(p99_total, 2),
                "output_tok_per_sec": round(output_tok_per_sec, 1),
                "total_tok_per_sec": round(total_tok_per_sec, 1),
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_mib": gpu_mem_loaded,
            }
            results.append(result)

            print(f"  -> avg={avg_total:.0f}ms  p50={p50_total:.0f}ms  p99={p99_total:.0f}ms  "
                  f"out_tok/s={output_tok_per_sec:.1f}  gen_tokens={avg_output_tokens:.0f}")

    # Cleanup
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
    """Print a comparison table organized by input length × output length."""
    print("\n" + "=" * 130)
    print("SUMMARY: Generation Model Latency Comparison")
    print("=" * 130)
    print(f"{'Model':<40} {'In':>7} {'Out':>5} {'Actual':>6} {'Avg(ms)':>10} {'Std':>8} "
          f"{'P50(ms)':>10} {'P99(ms)':>10} {'Out tok/s':>10} {'TP':>3} {'GPU(MiB)':>9}")
    print("─" * 130)

    for r in all_results:
        model_short = r["model"].split("/")[-1]
        if len(model_short) > 38:
            model_short = model_short[:38]
        print(f"{model_short:<40} {r['input_tokens']:>7} {r['output_tokens_requested']:>5} "
              f"{r['output_tokens_actual']:>6.0f} "
              f"{r['avg_total_latency_ms']:>10.0f} {r['std_total_latency_ms']:>8.0f} "
              f"{r['p50_total_latency_ms']:>10.0f} {r['p99_total_latency_ms']:>10.0f} "
              f"{r['output_tok_per_sec']:>10.1f} {r['tensor_parallel_size']:>3} "
              f"{r.get('gpu_memory_mib', 'N/A'):>9}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark generation model latency with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model ID to benchmark (default: all models sequentially)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TP_SIZE,
        help=f"Tensor parallel size (default: {DEFAULT_TP_SIZE})",
    )
    parser.add_argument(
        "--input-lengths",
        type=str,
        default=",".join(str(l) for l in DEFAULT_INPUT_LENGTHS),
        help=f"Comma-separated input token lengths (default: {DEFAULT_INPUT_LENGTHS})",
    )
    parser.add_argument(
        "--output-lengths",
        type=str,
        default=",".join(str(l) for l in DEFAULT_OUTPUT_LENGTHS),
        help=f"Comma-separated output token lengths (default: {DEFAULT_OUTPUT_LENGTHS})",
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
        default=None,
        help="vLLM max_model_len override (default: model's max_context)",
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
        help="Output JSON file path (default: benchmark/generation_latency_results.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_lengths = sorted(int(x) for x in args.input_lengths.split(","))
    output_lengths = sorted(int(x) for x in args.output_lengths.split(","))

    # Filter models
    if args.model:
        models = [m for m in GENERATION_MODELS if m["model_id"] == args.model]
        if not models:
            print(f"ERROR: Model '{args.model}' not found in GENERATION_MODELS.")
            print(f"Available: {[m['model_id'] for m in GENERATION_MODELS]}")
            sys.exit(1)
    else:
        models = GENERATION_MODELS

    print("=" * 70)
    print("Generation Model Latency Benchmark")
    print("=" * 70)
    print(f"  Models:        {[m['model_id'] for m in models]}")
    print(f"  Input lengths: {input_lengths}")
    print(f"  Output lengths:{output_lengths}")
    print(f"  TP size:       {args.tensor_parallel_size}")
    print(f"  Warmup:        {args.num_warmup}")
    print(f"  Runs:          {args.num_runs}")
    print(f"  Max model len: {args.max_model_len or 'auto'}")
    print(f"  GPU mem util:  {args.gpu_memory_utilization}")
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
                output_lengths=output_lengths,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print_summary_table(all_results)

    # Save results
    output_file = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "generation_latency_results.json",
    )
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "input_lengths": input_lengths,
            "output_lengths": output_lengths,
            "tensor_parallel_size": args.tensor_parallel_size,
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
