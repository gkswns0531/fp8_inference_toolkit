#!/usr/bin/env python3
"""
Profile CPU-side overhead: Python dispatch, tensor allocation, etc.
This is the dominant bottleneck for small M (batch size).

With --enforce-eager, every layer goes through Python.
GPU finishes kernels in <0.1ms but CPU takes ~0.7ms to dispatch → CPU-bound.
"""
import torch
import time
import sys
sys.path.insert(0, '.')
import int4_native_tc_ops as ops


def wall_time(fn, warmup=10, iters=200):
    """Measure wall-clock time (includes CPU overhead)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000  # ms


def cuda_time(fn, warmup=10, iters=200):
    """Measure GPU-only time."""
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
    M = 10
    gs = 128

    print(f"{'='*80}")
    print(f"  CPU vs GPU overhead comparison (M={M})")
    print(f"{'='*80}\n")

    header = f"{'Operation':<55} {'Wall(ms)':>9} {'CUDA(ms)':>9} {'CPU(ms)':>9}"
    print(header)
    print("-" * len(header))

    # 1. torch.mm (BF16 baseline per layer)
    K, N = 2560, 9728
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    def bf16_mm():
        torch.mm(x, w.t())

    tw = wall_time(bf16_mm)
    tc = cuda_time(bf16_mm)
    print(f"{'torch.mm(x[10,2560], w[9728,2560].t())':<55} {tw:>9.4f} {tc:>9.4f} {tw-tc:>9.4f}")

    # 2. reshape + mm (common in quantized paths)
    x3d = torch.randn(1, M, K, dtype=dtype, device=device)
    def reshape_mm():
        x_2d = x3d.reshape(-1, K)
        torch.mm(x_2d, w.t())

    tw = wall_time(reshape_mm)
    tc = cuda_time(reshape_mm)
    print(f"{'reshape + torch.mm':<55} {tw:>9.4f} {tc:>9.4f} {tw-tc:>9.4f}")

    # 3. Enhanced W4A4 full path (quant + GEMM)
    num_groups = K // gs
    w_packed, w_scale = ops.static_int4_weight_quant_grouped(w, gs)
    w_col_sum_groups = []
    for g in range(num_groups):
        unpacked = ops.unpack_int4_to_int8(w_packed[g], gs)
        w_col_sum_groups.append(unpacked.float().sum(dim=1))
    w_col_sum = torch.stack(w_col_sum_groups)
    smooth = torch.rand(K, dtype=dtype, device=device) * 2 + 0.5
    clip_ratio = 0.95

    out = torch.empty(M, N, dtype=dtype, device=device)

    def enhanced_w4a4():
        xp, xs, azp = ops.dynamic_int4_quant_asymmetric_clipped_grouped(
            x, gs, clip_ratio, smooth)
        ops.cutlass_int4_fused_grouped_gemm_azp(
            xp, w_packed, xs, w_scale,
            out, azp, w_col_sum,
            M, N, gs, num_groups)

    tw = wall_time(enhanced_w4a4)
    tc = cuda_time(enhanced_w4a4)
    print(f"{'Enhanced W4A4 (fused quant + fused GEMM)':<55} {tw:>9.4f} {tc:>9.4f} {tw-tc:>9.4f}")

    # 4. Enhanced W4A4 full path WITH Python apply() overhead
    class FakeLayer:
        pass
    layer = FakeLayer()
    layer.input_size_per_partition = K
    layer.output_size_per_partition = N
    layer.group_size = gs
    layer.num_groups = num_groups
    layer.weight_packed = w_packed
    layer.weight_scale = w_scale
    layer.w_col_sum = w_col_sum
    layer.smooth_scale = smooth

    def enhanced_w4a4_with_apply():
        x_2d = x3d.reshape(-1, x3d.shape[-1])
        M_ = x_2d.shape[0]
        out_ = torch.empty(M_, N, dtype=dtype, device=device)
        xp, xs, azp = ops.dynamic_int4_quant_asymmetric_clipped_grouped(
            x_2d, gs, clip_ratio, layer.smooth_scale)
        ops.cutlass_int4_fused_grouped_gemm_azp(
            xp, layer.weight_packed, xs, layer.weight_scale,
            out_, azp, layer.w_col_sum,
            M_, N, gs, layer.num_groups)
        return out_.reshape(*x3d.shape[:-1], N)

    tw = wall_time(enhanced_w4a4_with_apply)
    tc = cuda_time(enhanced_w4a4_with_apply)
    print(f"{'Enhanced W4A4 with reshape/alloc/reshape':<55} {tw:>9.4f} {tc:>9.4f} {tw-tc:>9.4f}")

    # 5. Enhanced W4A16 (smooth div + mm)
    w_bf16 = torch.randn(N, K, dtype=dtype, device=device)
    def enhanced_w4a16():
        x_s = x / smooth.unsqueeze(0)
        torch.mm(x_s, w_bf16.t())

    tw = wall_time(enhanced_w4a16)
    tc = cuda_time(enhanced_w4a16)
    print(f"{'Enhanced W4A16 (smooth div + torch.mm)':<55} {tw:>9.4f} {tc:>9.4f} {tw-tc:>9.4f}")

    # 6. Just torch.empty allocation
    def alloc_only():
        torch.empty(M, N, dtype=dtype, device=device)
    tw = wall_time(alloc_only)
    print(f"{'torch.empty(10, 9728) allocation only':<55} {tw:>9.4f} {'N/A':>9} {'':>9}")

    # 7. Just attribute access (simulated)
    def attr_access():
        _ = layer.weight_packed
        _ = layer.weight_scale
        _ = layer.w_col_sum
        _ = layer.smooth_scale
        _ = layer.num_groups
        _ = layer.group_size
        _ = layer.input_size_per_partition
        _ = layer.output_size_per_partition
    tw = wall_time(attr_access, iters=10000)
    print(f"{'8× attribute access':<55} {tw:>9.4f} {'N/A':>9} {'':>9}")

    # --- Full model simulation ---
    print(f"\n{'='*80}")
    print(f"  Full model simulation (36 layers × 5 linear ops)")
    print(f"{'='*80}\n")

    # Dimensions for each projection
    projs = [
        ("qkv_proj", 2560, 3584, "W4A4"),
        ("o_proj",   2560, 2560, "W4A16"),
        ("gate",     2560, 9728, "W4A4"),
        ("up",       2560, 9728, "W4A4"),
        ("down",     9728, 2560, "W4A16"),
    ]

    # Pre-create all weights
    weights_bf16 = {}
    weights_enhanced = {}
    for name, K_, N_, path in projs:
        x_in = torch.randn(M, K_, dtype=dtype, device=device)
        w_in = torch.randn(N_, K_, dtype=dtype, device=device)
        sm = torch.rand(K_, dtype=dtype, device=device) * 2 + 0.5
        weights_bf16[(name,)] = (x_in, w_in)

        if path == "W4A4":
            ng = K_ // gs
            wp, ws = ops.static_int4_weight_quant_grouped(w_in, gs)
            wcs = []
            for g in range(ng):
                wcs.append(ops.unpack_int4_to_int8(wp[g], gs).float().sum(dim=1))
            wcs = torch.stack(wcs)
            weights_enhanced[(name,)] = (x_in, wp, ws, wcs, sm, ng, K_, N_)
        else:
            weights_enhanced[(name,)] = (x_in, w_in, sm, K_, N_)

    def sim_bf16_layer():
        for name, K_, N_, path in projs:
            x_in, w_in = weights_bf16[(name,)]
            torch.mm(x_in, w_in.t())

    def sim_enhanced_layer():
        for name, K_, N_, path in projs:
            if path == "W4A4":
                x_in, wp, ws, wcs, sm, ng, K__, N__ = weights_enhanced[(name,)]
                out_ = torch.empty(M, N__, dtype=dtype, device=device)
                xp, xs, azp = ops.dynamic_int4_quant_asymmetric_clipped_grouped(
                    x_in, gs, 0.95, sm)
                ops.cutlass_int4_fused_grouped_gemm_azp(
                    xp, wp, xs, ws, out_, azp, wcs, M, N__, gs, ng)
            else:
                x_in, w_in, sm, K__, N__ = weights_enhanced[(name,)]
                x_s = x_in / sm.unsqueeze(0)
                torch.mm(x_s, w_in.t())

    tw_bf16 = wall_time(sim_bf16_layer, warmup=20, iters=100)
    tc_bf16 = cuda_time(sim_bf16_layer, warmup=20, iters=100)

    tw_enh = wall_time(sim_enhanced_layer, warmup=20, iters=100)
    tc_enh = cuda_time(sim_enhanced_layer, warmup=20, iters=100)

    print(f"{'Metric':<35} {'BF16':>10} {'Enhanced':>10} {'Diff':>10}")
    print("-" * 65)
    print(f"{'Per-layer wall time (ms)':<35} {tw_bf16:>10.4f} {tw_enh:>10.4f} {tw_enh-tw_bf16:>10.4f}")
    print(f"{'Per-layer CUDA time (ms)':<35} {tc_bf16:>10.4f} {tc_enh:>10.4f} {tc_enh-tc_bf16:>10.4f}")
    print(f"{'Per-layer CPU overhead (ms)':<35} {tw_bf16-tc_bf16:>10.4f} {tw_enh-tc_enh:>10.4f} {(tw_enh-tc_enh)-(tw_bf16-tc_bf16):>10.4f}")
    print()
    print(f"{'36 layers wall time (ms)':<35} {tw_bf16*36:>10.2f} {tw_enh*36:>10.2f} {(tw_enh-tw_bf16)*36:>10.2f}")
    print(f"{'36 layers CUDA time (ms)':<35} {tc_bf16*36:>10.2f} {tc_enh*36:>10.2f} {(tc_enh-tc_bf16)*36:>10.2f}")
    print(f"{'36 layers CPU overhead (ms)':<35} {(tw_bf16-tc_bf16)*36:>10.2f} {(tw_enh-tc_enh)*36:>10.2f} {((tw_enh-tc_enh)-(tw_bf16-tc_bf16))*36:>10.2f}")

    print(f"\n  Benchmark BF16:     137.3 ms")
    print(f"  Benchmark Enhanced: 165.0 ms")
    print(f"  Benchmark gap:       27.7 ms")
    print(f"  Wall-time gap (GEMM only, ×36): {(tw_enh-tw_bf16)*36:.1f} ms")
    print(f"  CUDA-time gap (GEMM only, ×36): {(tc_enh-tc_bf16)*36:.1f} ms")
    print(f"  CPU overhead gap (×36):          {((tw_enh-tc_enh)-(tw_bf16-tc_bf16))*36:.1f} ms")


if __name__ == "__main__":
    main()
