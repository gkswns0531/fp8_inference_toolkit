#!/usr/bin/env python3
"""
Profile all INT4 per-group GEMM implementations:
  1. PerCh  — CUTLASS per-channel INT4 GEMM (baseline)
  2. V4     — Fused grouped GEMM V4 (current production)
  3. V5     — Multi-group K-loop V5 (sync reduction)
  4. Dequant+BF16 — INT4→BF16 dequant + cuBLAS BF16 GEMM
  5. Progressive INT8 — INT4→INT8 unpack + INT8 MMA

Measures CUDA kernel time via torch.profiler across all 36 layers of Qwen3-4B.
"""

import gc
import time
import torch
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = False

HIDDEN_SIZE = 2560
NUM_LAYERS = 36
GROUP_SIZE = 128
BATCH_TOKENS = 500

# W4A4 projections per layer
W4A4_LAYERS = {
    "qkv_proj":     (2560, 6144),    # (K, N)
    "gate_up_proj": (2560, 19456),   # (K, N)
}


def make_int4_packed_perchannel(N, K):
    """Create random INT4 packed weights + per-channel scales."""
    w = torch.randn(N, K, dtype=torch.bfloat16)
    absmax = w.abs().amax(dim=1)
    scale = (absmax / 7.0).clamp(min=1e-10)
    w_int4 = torch.clamp(torch.round(w / scale.unsqueeze(1)), -8, 7).to(torch.int8)
    w_even = w_int4[:, 0::2] & 0x0F
    w_odd = (w_int4[:, 1::2] & 0x0F) << 4
    packed = (w_even | w_odd).to(torch.uint8)
    return packed.contiguous().cuda(), scale.to(torch.float32).contiguous().cuda()


def make_int4_packed_pergroup(N, K, gs=128):
    """Create random INT4 packed weights + per-group scales."""
    ng = K // gs
    w = torch.randn(N, K, dtype=torch.bfloat16)
    w_grouped = w.reshape(N, ng, gs).permute(1, 0, 2)  # (ng, N, gs)
    absmax = w_grouped.abs().amax(dim=2)  # (ng, N)
    scale = (absmax / 7.0).clamp(min=1e-10)
    w_int4 = torch.clamp(torch.round(w_grouped / scale.unsqueeze(2)), -8, 7).to(torch.int8)
    w_even = w_int4[:, :, 0::2] & 0x0F
    w_odd = (w_int4[:, :, 1::2] & 0x0F) << 4
    packed = (w_even | w_odd).to(torch.uint8)
    return packed.contiguous().cuda(), scale.to(torch.float32).contiguous().cuda()


def benchmark_config(name, run_fn, warmup=10, iters=20):
    """Benchmark a function with CUDA events."""
    # Warmup
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()

    # Time with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        run_fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    min_ms = np.min(times)
    return mean_ms, std_ms, min_ms


def run_all_profiles(M):
    import int4_native_tc as ops

    print(f"\n{'='*80}")
    print(f"  Profiling all kernels: M={M}, {NUM_LAYERS} layers × {len(W4A4_LAYERS)} projections")
    print(f"{'='*80}")

    # ── Prepare weights for all layers ──
    layers = []
    for layer_idx in range(NUM_LAYERS):
        for proj_name, (K, N) in W4A4_LAYERS.items():
            layers.append((f"L{layer_idx}.{proj_name}", K, N))

    # Per-channel weights
    perch_weights = []
    for name, K, N in layers:
        wp, ws = make_int4_packed_perchannel(N, K)
        perch_weights.append((wp, ws, K, N))

    # Per-group weights
    gs = GROUP_SIZE
    pg_weights = []
    for name, K, N in layers:
        wp, ws = make_int4_packed_pergroup(N, K, gs)
        ng = K // gs
        pg_weights.append((wp, ws, K, N, ng))

    # Pre-dequanted BF16 weights (for dequant+GEMM approach)
    bf16_weights = []
    for wp, ws, K, N, ng in pg_weights:
        w_bf16 = ops.dequant_int4_grouped_to_bf16(wp, ws, gs)  # [N, K]
        w_bf16_t = w_bf16.t().contiguous()  # [K, N]
        bf16_weights.append(w_bf16_t)

    # Pre-unpacked INT8 weights (for progressive approach)
    int8_weights = []
    for wp, ws, K, N, ng in pg_weights:
        # unpack grouped INT4 weights [ng, N, gs/2] -> [N, K] int8
        try:
            w_int8 = ops.unpack_int4_grouped_to_int8_weight(wp, gs)
            int8_weights.append(w_int8)
        except Exception as e:
            print(f"  Warning: unpack_int4_grouped_to_int8_weight failed: {e}")
            # Fallback: manual unpack
            w_int8 = torch.zeros(N, K, dtype=torch.int8, device='cuda')
            for g in range(ng):
                packed_g = wp[g]  # [N, gs/2]
                for k_pair in range(gs // 2):
                    p = packed_g[:, k_pair]
                    v0 = ((p.to(torch.int16) << 4).to(torch.int8) >> 4).to(torch.int8)
                    v1 = (p.to(torch.int8) >> 4).to(torch.int8)
                    w_int8[:, g * gs + k_pair * 2] = v0
                    w_int8[:, g * gs + k_pair * 2 + 1] = v1
            int8_weights.append(w_int8)

    print(f"  Weights prepared. Starting benchmarks...\n")

    results = {}

    # ═══════════════════════════════════════════════════
    # 1. PerCh — CUTLASS per-channel baseline
    # ═══════════════════════════════════════════════════
    def run_perch():
        for wp, ws, K, N in perch_weights:
            x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            x_packed, x_scale = ops.dynamic_int4_quant(x_input)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            ops.cutlass_int4_scaled_mm(x_packed, wp, x_scale, ws, out, M, N, K)

    mean, std, mn = benchmark_config("PerCh", run_perch)
    results["PerCh"] = (mean, std, mn)
    print(f"  PerCh (CUTLASS per-channel):  {mean:8.2f}ms ± {std:.2f}ms  (min={mn:.2f}ms)")

    gc.collect(); torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 2. V4 — Fused grouped GEMM V4 (production)
    # ═══════════════════════════════════════════════════
    def run_v4():
        for wp, ws, K, N, ng in pg_weights:
            x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            x_packed, x_scale = ops.dynamic_int4_quant_grouped(x_input, gs)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            ops.cutlass_int4_fused_grouped_gemm(x_packed, wp, x_scale, ws, out, M, N, gs, ng)

    mean, std, mn = benchmark_config("V4", run_v4)
    results["V4"] = (mean, std, mn)
    print(f"  V4 (fused grouped, current):  {mean:8.2f}ms ± {std:.2f}ms  (min={mn:.2f}ms)")

    gc.collect(); torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 2a. V8 — V4 tile with 16-byte vectorized cp.async
    # ═══════════════════════════════════════════════════
    def run_v8():
        for wp, ws, K, N, ng in pg_weights:
            x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            x_packed, x_scale = ops.dynamic_int4_quant_grouped(x_input, gs)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            ops.cutlass_int4_fused_grouped_gemm_v8(x_packed, wp, x_scale, ws, out, M, N, gs, ng)

    mean, std, mn = benchmark_config("V8", run_v8)
    results["V8"] = (mean, std, mn)
    print(f"  V8 (16B vec cp.async):        {mean:8.2f}ms ± {std:.2f}ms  (min={mn:.2f}ms)")

    gc.collect(); torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 2b. V7 — 128×128 tile, 3-stage pipeline, 256 threads
    # ═══════════════════════════════════════════════════
    def run_v7():
        for wp, ws, K, N, ng in pg_weights:
            x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            x_packed, x_scale = ops.dynamic_int4_quant_grouped(x_input, gs)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            ops.cutlass_int4_fused_grouped_gemm_v7(x_packed, wp, x_scale, ws, out, M, N, gs, ng)

    mean, std, mn = benchmark_config("V7", run_v7)
    results["V7"] = (mean, std, mn)
    print(f"  V7 (128x128, 3-stage):        {mean:8.2f}ms ± {std:.2f}ms  (min={mn:.2f}ms)")

    gc.collect(); torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 3. V5 — Multi-group K-loop (sync reduction)
    # ═══════════════════════════════════════════════════
    def run_v5():
        for wp, ws, K, N, ng in pg_weights:
            x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            x_packed, x_scale = ops.dynamic_int4_quant_grouped(x_input, gs)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            ops.cutlass_int4_fused_grouped_gemm_v5(x_packed, wp, x_scale, ws, out, M, N, gs, ng)

    mean, std, mn = benchmark_config("V5", run_v5)
    results["V5"] = (mean, std, mn)
    print(f"  V5 (multi-group K-loop):      {mean:8.2f}ms ± {std:.2f}ms  (min={mn:.2f}ms)")

    gc.collect(); torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 4. Dequant+BF16 — INT4→BF16 dequant + cuBLAS GEMM
    # ═══════════════════════════════════════════════════
    def run_dequant_bf16():
        for i, (wp, ws, K, N, ng) in enumerate(pg_weights):
            x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            x_packed, x_scale = ops.dynamic_int4_quant_grouped(x_input, gs)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            ops.cutlass_int4_dequant_gemm_grouped(
                x_packed, bf16_weights[i], x_scale, out, M, N, K, gs, ng)

    mean, std, mn = benchmark_config("Dequant+BF16", run_dequant_bf16)
    results["Dequant+BF16"] = (mean, std, mn)
    print(f"  Dequant+BF16 (cuBLAS):        {mean:8.2f}ms ± {std:.2f}ms  (min={mn:.2f}ms)")

    gc.collect(); torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 5. Progressive INT8 — INT4→INT8 + INT8 MMA
    # ═══════════════════════════════════════════════════
    def run_progressive():
        for i, (wp, ws, K, N, ng) in enumerate(pg_weights):
            x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            x_packed, x_scale = ops.dynamic_int4_quant_grouped(x_input, gs)
            # Unpack INT4 activations to contiguous INT8 [M, K]
            x_int8 = ops.unpack_int4_grouped_to_int8_contiguous(x_packed, gs)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            ops.progressive_int4_gemm_grouped(
                x_int8, int8_weights[i], x_scale, ws, out, M, N, K, gs, ng)

    mean, std, mn = benchmark_config("Progressive", run_progressive)
    results["Progressive"] = (mean, std, mn)
    print(f"  Progressive INT8:             {mean:8.2f}ms ± {std:.2f}ms  (min={mn:.2f}ms)")

    gc.collect(); torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 6. BF16 baseline — pure torch.mm
    # ═══════════════════════════════════════════════════
    bf16_w_full = []
    for _, _, K, N in perch_weights:
        w = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
        bf16_w_full.append(w)

    def run_bf16():
        for i, (_, _, K, N) in enumerate(perch_weights):
            x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            out = torch.mm(x_input, bf16_w_full[i])

    mean, std, mn = benchmark_config("BF16", run_bf16)
    results["BF16"] = (mean, std, mn)
    print(f"  BF16 (torch.mm baseline):     {mean:8.2f}ms ± {std:.2f}ms  (min={mn:.2f}ms)")

    del bf16_w_full
    gc.collect(); torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  SUMMARY (M={M}, {NUM_LAYERS} layers × {len(W4A4_LAYERS)} W4A4 projections)")
    print(f"  Total ops per run: {len(layers)} quant+GEMM calls")
    print(f"{'='*80}")
    print(f"  {'Config':<22} {'Mean(ms)':>10} {'Std(ms)':>10} {'Min(ms)':>10} {'vs PerCh':>10} {'vs BF16':>10}")
    print(f"  {'─'*72}")

    perch_mean = results["PerCh"][0]
    bf16_mean = results["BF16"][0]

    for name in ["BF16", "PerCh", "V4", "V8", "V7", "V5", "Dequant+BF16", "Progressive"]:
        mean, std, mn = results[name]
        vs_perch = f"{mean/perch_mean:.2f}x"
        vs_bf16 = f"{mean/bf16_mean:.2f}x"
        print(f"  {name:<22} {mean:>10.2f} {std:>10.2f} {mn:>10.2f} {vs_perch:>10} {vs_bf16:>10}")

    return results


if __name__ == "__main__":
    # GPU info
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, SM{props.major}{props.minor}, "
          f"SMEM/block={props.shared_memory_per_block_optin//1024}KB")

    for M in [500]:
        run_all_profiles(M)
