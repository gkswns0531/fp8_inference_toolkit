#!/usr/bin/env python3
"""
Layer-level profiling: measure exact CUDA time for each operation
in Enhanced-PG W4A4 and W4A16 paths, and compare with BF16 baseline.

Uses CUDA events for accurate GPU timing.
"""
import torch
import sys
sys.path.insert(0, '.')
import int4_native_tc_ops as ops


def cuda_time(fn, warmup=10, iters=100):
    """Measure GPU time in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16

    # Qwen3-Embedding-4B dimensions
    M = 10  # batch size (typical for embedding)
    gs = 128
    layers = 36

    # Layer dimensions in Qwen3-Embedding-4B
    # hidden_size=2560, intermediate_size=9728, num_heads=20, head_dim=128
    # num_kv_heads=4
    layer_specs = {
        # name: (K_in, N_out, path, count_per_layer)
        "qkv_proj":   (2560, 2560 + 512 + 512, "W4A4", 1),  # Q+K+V fused
        "o_proj":     (2560, 2560, "W4A16", 1),
        "gate_proj":  (2560, 9728, "W4A4", 1),
        "up_proj":    (2560, 9728, "W4A4", 1),
        "down_proj":  (9728, 2560, "W4A16", 1),
    }

    print(f"{'='*80}")
    print(f"  Layer-level profiling: M={M}, gs={gs}, dtype={dtype}")
    print(f"{'='*80}\n")

    total_bf16 = 0.0
    total_enhanced = 0.0

    header = f"{'Layer':<14} {'K':>5} {'N':>5} {'Path':<6} " \
             f"{'BF16(ms)':>9} {'Enh(ms)':>9} {'Quant':>7} {'GEMM':>7} {'Other':>7} {'Ratio':>6}"
    print(header)
    print("-" * len(header))

    for name, (K, N, path, count) in layer_specs.items():
        num_groups = K // gs

        # === BF16 baseline ===
        x = torch.randn(M, K, dtype=dtype, device=device)
        w_bf16 = torch.randn(N, K, dtype=dtype, device=device)

        def bf16_mm():
            torch.mm(x, w_bf16.t())

        t_bf16 = cuda_time(bf16_mm)

        # === Enhanced-PG ===
        if path == "W4A4":
            # Weight quantization (done once at load)
            w_packed, w_scale = ops.static_int4_weight_quant_grouped(w_bf16, gs)

            # Pre-compute w_col_sum for AZP
            w_col_sum_groups = []
            for g in range(num_groups):
                unpacked = ops.unpack_int4_to_int8(w_packed[g], gs)
                w_col_sum_groups.append(unpacked.float().sum(dim=1))
            w_col_sum = torch.stack(w_col_sum_groups)

            # Smooth scale (simulate)
            smooth_scale = torch.rand(K, dtype=dtype, device=device) * 2 + 0.5

            out = torch.empty(M, N, dtype=dtype, device=device)
            clip_ratio = 0.95

            # Measure individual steps
            def step_quant():
                return ops.dynamic_int4_quant_asymmetric_clipped_grouped(
                    x, gs, clip_ratio, smooth_scale)

            t_quant = cuda_time(step_quant)

            x_packed, x_scale, azp_adj = step_quant()

            def step_gemm():
                ops.cutlass_int4_fused_grouped_gemm_azp(
                    x_packed, w_packed, x_scale, w_scale,
                    out, azp_adj, w_col_sum,
                    M, N, gs, num_groups)

            t_gemm = cuda_time(step_gemm)

            # Full path
            def full_enhanced():
                xp, xs, azp = ops.dynamic_int4_quant_asymmetric_clipped_grouped(
                    x, gs, clip_ratio, smooth_scale)
                ops.cutlass_int4_fused_grouped_gemm_azp(
                    xp, w_packed, xs, w_scale,
                    out, azp, w_col_sum,
                    M, N, gs, num_groups)

            t_total = cuda_time(full_enhanced)
            t_other = t_total - t_quant - t_gemm

        else:  # W4A16
            # Pre-dequantized weight (same as BF16)
            w_predequant = torch.randn(N, K, dtype=dtype, device=device)
            smooth_scale = torch.rand(K, dtype=dtype, device=device) * 2 + 0.5

            # Measure smooth + mm
            def step_smooth_mm():
                x_s = x / smooth_scale.unsqueeze(0)
                torch.mm(x_s, w_predequant.t())

            def step_mm_only():
                torch.mm(x, w_predequant.t())

            t_total = cuda_time(step_smooth_mm)
            t_mm = cuda_time(step_mm_only)
            t_quant = 0.0
            t_gemm = t_mm
            t_other = t_total - t_mm

        ratio = t_total / t_bf16 if t_bf16 > 0 else 0

        total_bf16 += t_bf16 * count
        total_enhanced += t_total * count

        print(f"{name:<14} {K:>5} {N:>5} {path:<6} "
              f"{t_bf16:>9.4f} {t_total:>9.4f} {t_quant:>7.4f} {t_gemm:>7.4f} {t_other:>7.4f} {ratio:>5.2f}x")

    print()
    print(f"{'Per-layer total (ms)':<40} {total_bf16:>9.4f} {total_enhanced:>9.4f}")
    print(f"{'Full model (×36 layers, ms)':<40} {total_bf16*layers:>9.2f} {total_enhanced*layers:>9.2f}")
    print(f"{'Overhead (ms)':<40} {'':>9} {(total_enhanced-total_bf16)*layers:>9.2f}")
    print(f"{'Ratio':<40} {'':>9} {total_enhanced/total_bf16:>8.2f}x")

    # === Also measure Python dispatch overhead ===
    print(f"\n{'='*80}")
    print(f"  Python dispatch / kernel launch overhead analysis")
    print(f"{'='*80}\n")

    # Empty kernel launch overhead
    dummy = torch.empty(1, device=device)
    def noop():
        torch.add(dummy, dummy)

    t_noop = cuda_time(noop, warmup=100, iters=1000)
    print(f"  Minimal kernel launch: {t_noop:.4f} ms")

    # Count kernels per layer
    print(f"\n  Enhanced-PG kernels per layer:")
    print(f"    W4A4: 1 (fused quant) + 1 (fused GEMM) = 2 launches")
    print(f"    W4A16: 1 (smooth div) + 1 (torch.mm) = 2 launches")
    print(f"    Total per transformer layer: 3×2 + 2×2 = 10 launches")
    print(f"    Full model: 10 × 36 = 360 launches")
    print(f"    Estimated launch overhead: {360 * t_noop:.2f} ms")

    print(f"\n  BF16 kernels per layer:")
    print(f"    5 torch.mm = 5 launches")
    print(f"    Full model: 5 × 36 = 180 launches")
    print(f"    Estimated launch overhead: {180 * t_noop:.2f} ms")

    print(f"\n  Extra launch overhead: {(360-180) * t_noop:.2f} ms")

    # === Non-linear ops (LayerNorm, RMSNorm, attention, etc.) ===
    print(f"\n{'='*80}")
    print(f"  Estimated non-GEMM overhead")
    print(f"{'='*80}")
    bf16_bench = 137.3  # from benchmark
    enh_bench = 165.0   # from benchmark
    print(f"  End-to-end BF16:      {bf16_bench:.1f} ms")
    print(f"  End-to-end Enhanced:  {enh_bench:.1f} ms")
    print(f"  GEMM-only BF16:      {total_bf16*layers:.2f} ms")
    print(f"  GEMM-only Enhanced:  {total_enhanced*layers:.2f} ms")
    non_gemm = bf16_bench - total_bf16*layers
    print(f"  Non-GEMM overhead:   ~{non_gemm:.1f} ms (norm, attention, softmax, etc.)")
    print(f"  GEMM overhead:       {(total_enhanced-total_bf16)*layers:.2f} ms")
    print(f"  Gap explained:       {(total_enhanced-total_bf16)*layers:.2f} / {enh_bench-bf16_bench:.1f} ms")


if __name__ == "__main__":
    main()
