#!/usr/bin/env python3
"""
GEMM Micro-Benchmark: INT4 TC vs FP8 vs BF16

Qwen3-Embedding-4B GEMM dimensions에 대해 GEMM-only / quant / total latency 비교.

Layers:
  qkv_proj:  K=2560, N=3840
  o_proj:    K=2560, N=2560
  gate_proj: K=2560, N=9728
  up_proj:   K=2560, N=9728
  down_proj: K=9728, N=2560

Batch sizes (M): 1, 4, 16, 64, 128, 256, 512, 1024

Usage:
    python benchmark_int4_native_gemm.py
"""

import torch
import time
import json
import sys
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

LAYERS = [
    ("qkv_proj",  2560, 3840),
    ("o_proj",    2560, 2560),
    ("gate_proj", 2560, 9728),
    ("up_proj",   2560, 9728),
    ("down_proj", 9728, 2560),
]

# Ensure K is multiple of 64 for INT4 alignment (K=9728 → 9728/64=152, OK)
BATCH_SIZES = [1, 4, 16, 64, 128, 256, 512, 1024]
WARMUP = 20
REPEAT = 100
NUM_LAYERS = 36  # Qwen3-Embedding-4B has 36 transformer layers
GEMMS_PER_LAYER = 5  # qkv, o, gate, up, down

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cuda_timer(fn, warmup: int = WARMUP, repeat: int = REPEAT) -> float:
    """Returns median latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]  # us
    times.sort()
    return times[repeat // 2]  # median


def compute_tflops(M: int, N: int, K: int, latency_us: float) -> float:
    """Compute TFLOPS from GEMM dimensions and latency."""
    flops = 2.0 * M * N * K  # multiply-add = 2 ops
    return flops / (latency_us * 1e-6) / 1e12


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark functions
# ─────────────────────────────────────────────────────────────────────────────

def bench_bf16(M: int, K: int, N: int) -> dict:
    """BF16 GEMM via torch.matmul (cuBLAS)."""
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    gemm_us = cuda_timer(lambda: torch.matmul(a, w.T))
    tflops = compute_tflops(M, N, K, gemm_us)
    return {"gemm_us": gemm_us, "quant_us": 0.0, "total_us": gemm_us, "tflops": tflops}


def bench_fp8(M: int, K: int, N: int) -> dict:
    """FP8 GEMM via torch._scaled_mm (cuBLAS FP8 TC)."""
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.ones(1, dtype=torch.float32, device="cuda")
    scale_b = torch.ones(1, dtype=torch.float32, device="cuda")

    # Quant: BF16 → FP8
    def quant_fn():
        return a_bf16.to(torch.float8_e4m3fn)

    quant_us = cuda_timer(quant_fn)
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)

    # GEMM: FP8 × FP8
    def gemm_fn():
        torch._scaled_mm(a_fp8, w_fp8.T, scale_a=scale_a, scale_b=scale_b,
                         out_dtype=torch.bfloat16)

    gemm_us = cuda_timer(gemm_fn)
    total_us = quant_us + gemm_us
    tflops = compute_tflops(M, N, K, gemm_us)
    return {"gemm_us": gemm_us, "quant_us": quant_us, "total_us": total_us, "tflops": tflops}


def bench_int4_w4a4(M: int, K: int, N: int) -> dict:
    """INT4×INT4 GEMM via CUTLASS native TC (W4A4)."""
    import int4_native_tc

    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Pre-quantize weight
    w_packed, w_scale = int4_native_tc.static_int4_weight_quant(w_bf16)

    # Quant: BF16 → INT4
    def quant_fn():
        return int4_native_tc.dynamic_int4_quant(a_bf16)

    quant_us = cuda_timer(quant_fn)
    a_packed, a_scale = int4_native_tc.dynamic_int4_quant(a_bf16)

    # GEMM: INT4 × INT4 with fused EVT epilogue
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def gemm_fn():
        int4_native_tc.cutlass_int4_scaled_mm(
            a_packed, w_packed, a_scale, w_scale, out, M, N, K)

    gemm_us = cuda_timer(gemm_fn)
    total_us = quant_us + gemm_us
    tflops = compute_tflops(M, N, K, gemm_us)
    return {"gemm_us": gemm_us, "quant_us": quant_us, "total_us": total_us, "tflops": tflops}


def bench_int4_w4a8(M: int, K: int, N: int) -> dict:
    """W4A8: INT8 activation × INT4 weight (pre-unpacked to INT8) GEMM via CUTLASS TC."""
    import int4_native_tc

    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Pre-quantize weight (INT4) and unpack to INT8 (done at model load time)
    w_packed, w_scale = int4_native_tc.static_int4_weight_quant(w_bf16)
    w_int8 = int4_native_tc.unpack_int4_to_int8(w_packed, K)

    # Quant: BF16 → INT8
    def quant_fn():
        return int4_native_tc.dynamic_int8_quant(a_bf16)

    quant_us = cuda_timer(quant_fn)
    a_int8, a_scale = int4_native_tc.dynamic_int8_quant(a_bf16)

    # GEMM: INT8 × INT8 with fused EVT epilogue
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def gemm_fn():
        int4_native_tc.cutlass_w4a8_scaled_mm(
            a_int8, w_int8, a_scale, w_scale, out, M, N, K)

    gemm_us = cuda_timer(gemm_fn)
    total_us = quant_us + gemm_us
    tflops = compute_tflops(M, N, K, gemm_us)
    return {"gemm_us": gemm_us, "quant_us": quant_us, "total_us": total_us, "tflops": tflops}


def bench_int4_w4a16(M: int, K: int, N: int) -> dict:
    """W4A16: INT4 weight dequant → BF16 GEMM."""
    import int4_native_tc

    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Pre-quantize weight
    w_packed, w_scale = int4_native_tc.static_int4_weight_quant(w_bf16)

    # Dequant: INT4 → BF16
    def dequant_fn():
        return int4_native_tc.dequant_int4_to_bf16(w_packed, w_scale, K)

    dequant_us = cuda_timer(dequant_fn)
    w_dequant = int4_native_tc.dequant_int4_to_bf16(w_packed, w_scale, K)

    # GEMM: BF16 × BF16
    def gemm_fn():
        torch.matmul(a_bf16, w_dequant.T)

    gemm_us = cuda_timer(gemm_fn)
    total_us = dequant_us + gemm_us
    tflops = compute_tflops(M, N, K, gemm_us)
    return {"gemm_us": gemm_us, "quant_us": dequant_us, "total_us": total_us, "tflops": tflops}


def _prepare_marlin_weight(w_bf16: torch.Tensor, K: int, N: int,
                           is_a_8bit: bool = False) -> tuple:
    """Shared Marlin weight preparation for W4A16-Marlin and W4A8-Fused benchmarks."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.w4a16_int4tc import (
        _quantize_w4_symmetric,
        _pack_to_gptq_format,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        GPTQ_MARLIN_MIN_THREAD_N,
        GPTQ_MARLIN_MAX_PARALLEL,
        marlin_permute_scales,
    )

    q_unsigned, w_scale = _quantize_w4_symmetric(w_bf16)
    q_gptq = _pack_to_gptq_format(q_unsigned)
    perm = torch.empty(0, dtype=torch.int, device="cuda")
    w_marlin = ops.gptq_marlin_repack(q_gptq, perm, K, N, 4, is_a_8bit=is_a_8bit)
    s_for_marlin = w_scale.to(torch.bfloat16).reshape(1, N)
    s_marlin = marlin_permute_scales(
        s_for_marlin, K, N, group_size=-1, is_a_8bit=is_a_8bit)

    old_ws = (N // GPTQ_MARLIN_MIN_THREAD_N) * GPTQ_MARLIN_MAX_PARALLEL
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    workspace = torch.zeros(max(old_ws, sm_count), dtype=torch.int, device="cuda")
    empty = torch.empty(0, dtype=torch.int, device="cuda")

    return w_marlin, s_marlin, workspace, empty


def bench_int4_w4a16_marlin(M: int, K: int, N: int) -> dict:
    """W4A16-Marlin: INT4 weight × BF16 activation via Marlin fused dequant+BF16 TC."""
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    w_marlin, s_marlin, workspace, empty = _prepare_marlin_weight(
        w_bf16, K, N, is_a_8bit=False)

    # GEMM: BF16 activation × INT4 weight (Marlin fused dequant -> BF16 TC)
    def gemm_fn():
        ops.marlin_gemm(
            a_bf16, None, w_marlin, None, s_marlin, None, None,
            empty, empty, empty, workspace,
            scalar_types.uint4b8,
            size_m=M, size_n=N, size_k=K,
            is_k_full=True, use_atomic_add=False,
            use_fp32_reduce=True, is_zp_float=False,
        )

    gemm_us = cuda_timer(gemm_fn)
    tflops = compute_tflops(M, N, K, gemm_us)
    return {"gemm_us": gemm_us, "quant_us": 0.0, "total_us": gemm_us, "tflops": tflops}


def bench_int4_w4a8_fused(M: int, K: int, N: int) -> dict:
    """W4A8-Fused: INT4 weight × INT8 activation via Marlin fused dequant+INT8 TC."""
    import int4_native_tc
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    w_marlin, s_marlin, workspace, empty = _prepare_marlin_weight(
        w_bf16, K, N, is_a_8bit=True)

    # Quant: BF16 → INT8 (fused CUDA kernel)
    def quant_fn():
        return int4_native_tc.dynamic_int8_quant(a_bf16)

    quant_us = cuda_timer(quant_fn)
    x_int8, a_scales = quant_fn()

    # GEMM: INT8 activation × INT4 weight (Marlin fused dequant -> INT8 TC)
    def gemm_fn():
        ops.marlin_gemm(
            x_int8, None, w_marlin, None, s_marlin, a_scales, None,
            empty, empty, empty, workspace,
            scalar_types.uint4b8,
            size_m=M, size_n=N, size_k=K,
            is_k_full=True, use_atomic_add=False,
            use_fp32_reduce=False, is_zp_float=False,
        )

    gemm_us = cuda_timer(gemm_fn)
    total_us = quant_us + gemm_us
    tflops = compute_tflops(M, N, K, gemm_us)
    return {"gemm_us": gemm_us, "quant_us": quant_us, "total_us": total_us, "tflops": tflops}


# Layer classification for mixed W4A4A8
# INT4×INT8 for post-activation layers, INT4×INT4 for others
INT8_ACT_LAYERS = {"o_proj", "down_proj"}


def bench_int4_w4a4a8_mixed(M: int, K: int, N: int, layer_name: str) -> dict:
    """W4A4A8-Mixed: INT4×INT4 for most layers, INT4×INT8 for post-activation layers."""
    if layer_name in INT8_ACT_LAYERS:
        # INT4×INT8 path via Marlin (same as W4A8-Fused)
        return bench_int4_w4a8_fused(M, K, N)
    else:
        # INT4×INT4 path via CUTLASS (same as W4A4)
        return bench_int4_w4a4(M, K, N)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("GEMM Micro-Benchmark: INT4 TC vs FP8 vs BF16")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP}, Repeat: {REPEAT}")
    print("=" * 80)

    all_results = {}

    for layer_name, K, N in LAYERS:
        layer_results = {}
        for M in BATCH_SIZES:
            print(f"\nLayer: {layer_name} (K={K}, N={N}), M={M}")
            print(f"{'Dtype':<16} {'GEMM(us)':>10} {'Quant(us)':>10} {'Total(us)':>10} {'TFLOPS':>8}")
            print("-" * 60)

            results = {}

            # BF16
            r = bench_bf16(M, K, N)
            results["BF16"] = r
            print(f"{'BF16':<16} {r['gemm_us']:>10.1f} {'-':>10} {r['total_us']:>10.1f} {r['tflops']:>8.2f}")

            # FP8
            try:
                r = bench_fp8(M, K, N)
                results["FP8"] = r
                print(f"{'FP8':<16} {r['gemm_us']:>10.1f} {r['quant_us']:>10.1f} {r['total_us']:>10.1f} {r['tflops']:>8.2f}")
            except Exception as e:
                print(f"{'FP8':<16} SKIP ({e})")

            # W4A16 (dequant → BF16 GEMM)
            try:
                r = bench_int4_w4a16(M, K, N)
                results["W4A16"] = r
                print(f"{'W4A16':<16} {r['gemm_us']:>10.1f} {r['quant_us']:>10.1f} {r['total_us']:>10.1f} {r['tflops']:>8.2f}")
            except Exception as e:
                print(f"{'W4A16':<16} SKIP ({e})")

            # W4A8 (INT8 × INT8 CUTLASS TC, weight pre-unpacked)
            try:
                r = bench_int4_w4a8(M, K, N)
                results["W4A8"] = r
                print(f"{'W4A8':<16} {r['gemm_us']:>10.1f} {r['quant_us']:>10.1f} {r['total_us']:>10.1f} {r['tflops']:>8.2f}")
            except Exception as e:
                print(f"{'W4A8':<16} SKIP ({e})")

            # W4A8-Fused (INT4 weight × INT8 activation via Marlin)
            try:
                r = bench_int4_w4a8_fused(M, K, N)
                results["W4A8-Fused"] = r
                print(f"{'W4A8-Fused':<16} {r['gemm_us']:>10.1f} {r['quant_us']:>10.1f} {r['total_us']:>10.1f} {r['tflops']:>8.2f}")
            except Exception as e:
                print(f"{'W4A8-Fused':<16} SKIP ({e})")

            # W4A4 (INT4 × INT4 CUTLASS TC)
            try:
                r = bench_int4_w4a4(M, K, N)
                results["W4A4"] = r
                print(f"{'W4A4-INT4TC':<16} {r['gemm_us']:>10.1f} {r['quant_us']:>10.1f} {r['total_us']:>10.1f} {r['tflops']:>8.2f}")
            except Exception as e:
                print(f"{'W4A4-INT4TC':<16} SKIP ({e})")

            # W4A16-Marlin (INT4 weight × BF16 activation via Marlin)
            try:
                r = bench_int4_w4a16_marlin(M, K, N)
                results["W4A16-Marlin"] = r
                print(f"{'W4A16-Marlin':<16} {r['gemm_us']:>10.1f} {r['quant_us']:>10.1f} {r['total_us']:>10.1f} {r['tflops']:>8.2f}")
            except Exception as e:
                print(f"{'W4A16-Marlin':<16} SKIP ({e})")

            layer_results[M] = results

        all_results[layer_name] = layer_results

    # ─────────────────────────────────────────────────────────────────────
    # Summary: estimated full forward pass (36 layers × 5 GEMMs)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"Full Forward Pass Estimate ({NUM_LAYERS} layers × {GEMMS_PER_LAYER} GEMMs)")
    print(f"{'=' * 120}")
    dtype_cols = ["BF16", "FP8", "W4A16", "W4A8", "W4A8-Fused", "W4A4", "W4A16-Marlin", "W4A4A8-Mixed"]
    header = f"{'M':<8}" + "".join(f" {d+'(ms)':>14}" for d in dtype_cols)
    print(header)
    print("-" * 120)

    summary = {}
    for M in BATCH_SIZES:
        totals = {d: 0.0 for d in dtype_cols}
        for layer_name, _, _ in LAYERS:
            layer_data = all_results.get(layer_name, {}).get(M, {})
            for dtype_name in totals:
                if dtype_name == "W4A4A8-Mixed":
                    # Mixed: o_proj/down_proj use W4A8-Fused, others use W4A4
                    if layer_name in INT8_ACT_LAYERS:
                        if "W4A8-Fused" in layer_data:
                            totals[dtype_name] += layer_data["W4A8-Fused"]["total_us"]
                    else:
                        if "W4A4" in layer_data:
                            totals[dtype_name] += layer_data["W4A4"]["total_us"]
                elif dtype_name in layer_data:
                    totals[dtype_name] += layer_data[dtype_name]["total_us"]

        # Scale by num_layers
        for k in totals:
            totals[k] = totals[k] * NUM_LAYERS / 1000.0  # us → ms

        summary[M] = totals

        parts = []
        for dtype_name in dtype_cols:
            v = totals[dtype_name]
            parts.append(f"{v:>14.2f}" if v > 0 else f"{'N/A':>14}")
        print(f"{M:<8} {''.join(parts)}")

    # Save results
    out_path = "/home/ubuntu/fp8_inference_toolkit/benchmark/int4_gemm_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({"per_layer": all_results, "forward_pass_ms": summary}, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
