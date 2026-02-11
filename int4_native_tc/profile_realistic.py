#!/usr/bin/env python3
"""
Realistic profiling with actual token count (M=700, not M=10).
10 sentences × ~70 tokens = M≈700 tokens.
This measures what actually happens during vLLM forward pass.
"""
import torch
import time
import sys
sys.path.insert(0, '.')
import int4_native_tc_ops as ops


def cuda_time(fn, warmup=10, iters=50):
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


def wall_time(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (t1 := time.perf_counter(), (t1 - t0) / iters * 1000)[1]


def main():
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16
    gs = 128
    layers = 36

    print(f"{'='*90}")
    print(f"  Realistic profiling: varying M (token count)")
    print(f"{'='*90}\n")

    # Qwen3 layer specs
    projs = [
        ("qkv_proj",  2560, 3584,  "W4A4"),
        ("o_proj",    2560, 2560,  "W4A16"),
        ("gate_proj", 2560, 9728,  "W4A4"),
        ("up_proj",   2560, 9728,  "W4A4"),
        ("down_proj", 9728, 2560,  "W4A16"),
    ]

    for M in [1, 10, 100, 700]:
        print(f"\n{'─'*90}")
        print(f"  M = {M} tokens")
        print(f"{'─'*90}")
        header = f"  {'Layer':<14} {'K':>5} {'N':>5} {'Path':<6} " \
                 f"{'BF16(ms)':>9} {'Enh(ms)':>9} {'Quant':>7} {'GEMM':>7} {'Ratio':>6}"
        print(header)

        total_bf16 = 0.0
        total_enh = 0.0
        total_quant = 0.0
        total_gemm = 0.0

        for name, K, N, path in projs:
            x = torch.randn(M, K, dtype=dtype, device=device)
            w = torch.randn(N, K, dtype=dtype, device=device)

            # BF16
            def bf16_fn():
                torch.mm(x, w.t())
            t_bf16 = cuda_time(bf16_fn)

            if path == "W4A4":
                num_groups = K // gs
                w_packed, w_scale = ops.static_int4_weight_quant_grouped(w, gs)
                wcs = []
                for g in range(num_groups):
                    wcs.append(ops.unpack_int4_to_int8(w_packed[g], gs).float().sum(dim=1))
                w_col_sum = torch.stack(wcs)
                out = torch.empty(M, N, dtype=dtype, device=device)

                def quant_fn():
                    return ops.dynamic_int4_quant_asymmetric_clipped_grouped(x, gs, 0.95)
                t_quant = cuda_time(quant_fn)

                xp, xs, azp = quant_fn()
                def gemm_fn():
                    ops.cutlass_int4_fused_grouped_gemm_azp(
                        xp, w_packed, xs, w_scale, out, azp, w_col_sum,
                        M, N, gs, num_groups)
                t_gemm = cuda_time(gemm_fn)

                t_enh = t_quant + t_gemm

            else:  # W4A16 - just torch.mm (no smooth_scale in benchmark)
                t_quant = 0
                t_gemm = t_bf16
                t_enh = t_bf16

            total_bf16 += t_bf16
            total_enh += t_enh
            total_quant += t_quant
            total_gemm += t_gemm if path == "W4A4" else 0

            ratio = t_enh / t_bf16 if t_bf16 > 0 else 0
            print(f"  {name:<14} {K:>5} {N:>5} {path:<6} "
                  f"{t_bf16:>9.4f} {t_enh:>9.4f} {t_quant:>7.4f} {t_gemm:>7.4f} {ratio:>5.2f}x")

        print(f"\n  {'Per-layer total':<30} {total_bf16:>9.4f} {total_enh:>9.4f}")
        print(f"  {'×36 layers':<30} {total_bf16*36:>9.2f} {total_enh*36:>9.2f}")
        print(f"  {'W4A4 quant overhead (×36)':<30} {'':>9} {total_quant*36:>9.2f}")

    # === Detailed comparison for M=700 ===
    M = 700
    K, N = 2560, 9728  # gate_proj (largest GEMM)
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    print(f"\n\n{'='*90}")
    print(f"  Detailed M=700 analysis: gate_proj (K={K}, N={N})")
    print(f"{'='*90}\n")

    # BF16 cuBLAS
    def bf16_fn():
        torch.mm(x, w.t())
    t_bf16 = cuda_time(bf16_fn)
    print(f"  cuBLAS BF16 torch.mm:      {t_bf16:.4f} ms")

    # INT4 fused GEMM
    num_groups = K // gs
    w_packed, w_scale = ops.static_int4_weight_quant_grouped(w, gs)
    wcs = []
    for g in range(num_groups):
        wcs.append(ops.unpack_int4_to_int8(w_packed[g], gs).float().sum(dim=1))
    w_col_sum = torch.stack(wcs)
    out = torch.empty(M, N, dtype=dtype, device=device)

    # Quant
    def quant_fn():
        return ops.dynamic_int4_quant_asymmetric_clipped_grouped(x, gs, 0.95)
    t_quant = cuda_time(quant_fn)
    print(f"  Fused quant kernel:        {t_quant:.4f} ms")

    xp, xs, azp = quant_fn()

    # GEMM only
    def gemm_fn():
        ops.cutlass_int4_fused_grouped_gemm_azp(
            xp, w_packed, xs, w_scale, out, azp, w_col_sum,
            M, N, gs, num_groups)
    t_gemm = cuda_time(gemm_fn)
    print(f"  Fused INT4 GEMM:           {t_gemm:.4f} ms")
    print(f"  Enhanced total:            {t_quant + t_gemm:.4f} ms")
    print(f"  Ratio (Enhanced/BF16):     {(t_quant + t_gemm) / t_bf16:.2f}x")

    # Grid size analysis
    tile_m, tile_n = 32, 32
    grid_m = (M + tile_m - 1) // tile_m
    grid_n = (N + tile_n - 1) // tile_n
    print(f"\n  Fused GEMM grid: {grid_m} × {grid_n} = {grid_m * grid_n} blocks")
    print(f"  Threads/block: 128 (4 warps)")
    print(f"  Total warps: {grid_m * grid_n * 4}")
    print(f"  Theoretical occupancy limited by shared memory and tiles")

    # Compare with symmetric (non-AZP) GEMM
    xp2, xs2 = ops.dynamic_int4_quant_clipped_grouped(x, gs, 0.95)
    def gemm_sym():
        ops.cutlass_int4_fused_grouped_gemm(
            xp2, w_packed, xs2, w_scale, out, M, N, gs, num_groups)
    t_sym = cuda_time(gemm_sym)
    print(f"\n  Symmetric fused GEMM:      {t_sym:.4f} ms (vs AZP: {t_gemm:.4f})")

    # Summary
    print(f"\n\n{'='*90}")
    print(f"  ROOT CAUSE ANALYSIS")
    print(f"{'='*90}")
    print(f"\n  For M=700 (actual benchmark scenario):")

    # Recalculate totals at M=700
    total_bf16_700 = 0
    total_enh_700 = 0
    for name, K_, N_, path in projs:
        x_ = torch.randn(M, K_, dtype=dtype, device=device)
        w_ = torch.randn(N_, K_, dtype=dtype, device=device)
        t_b = cuda_time(lambda: torch.mm(x_, w_.t()))
        total_bf16_700 += t_b
        if path == "W4A4":
            ng = K_ // gs
            wp, ws = ops.static_int4_weight_quant_grouped(w_, gs)
            wcs_ = []
            for g in range(ng):
                wcs_.append(ops.unpack_int4_to_int8(wp[g], gs).float().sum(dim=1))
            wcs_ = torch.stack(wcs_)
            out_ = torch.empty(M, N_, dtype=dtype, device=device)
            xp_, xs_, azp_ = ops.dynamic_int4_quant_asymmetric_clipped_grouped(x_, gs, 0.95)
            t_q = cuda_time(lambda: ops.dynamic_int4_quant_asymmetric_clipped_grouped(x_, gs, 0.95))
            t_g = cuda_time(lambda: ops.cutlass_int4_fused_grouped_gemm_azp(
                xp_, wp, xs_, ws, out_, azp_, wcs_, M, N_, gs, ng))
            total_enh_700 += t_q + t_g
        else:
            total_enh_700 += t_b

    print(f"  Per-layer BF16 GEMM total:       {total_bf16_700:.4f} ms")
    print(f"  Per-layer Enhanced GEMM total:    {total_enh_700:.4f} ms")
    print(f"  Per-layer overhead:               {total_enh_700 - total_bf16_700:.4f} ms")
    print(f"  ×36 layers BF16:                  {total_bf16_700*36:.2f} ms")
    print(f"  ×36 layers Enhanced:              {total_enh_700*36:.2f} ms")
    print(f"  ×36 layers GEMM overhead:         {(total_enh_700 - total_bf16_700)*36:.2f} ms")
    print(f"\n  Benchmark gap:                    27.7 ms")
    print(f"  GEMM-explained gap:               {(total_enh_700 - total_bf16_700)*36:.2f} ms")
    non_gemm = 27.7 - (total_enh_700 - total_bf16_700)*36
    print(f"  Unexplained gap:                  {non_gemm:.2f} ms")
    print(f"  (likely: Python dispatch, vLLM overhead, memory alloc)")


if __name__ == "__main__":
    main()
