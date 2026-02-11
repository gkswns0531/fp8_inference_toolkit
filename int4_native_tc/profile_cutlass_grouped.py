#!/usr/bin/env python3
"""
Profiling: Fused V2 kernel vs old CUTLASS per-group loop vs manual per-group loop vs cuBLAS BF16.

Compares four approaches for gate_proj (M=700, K=2560, N=9728, gs=128, num_groups=20):

1. Fused V2 kernel:  cutlass_int4_fused_grouped_gemm_azp  (single kernel launch)
2. Old CUTLASS loop:  cutlass_int4_scaled_mm_azp_grouped  (20 CUTLASS + 20 add_ in C++)
3. Manual per-group:  cutlass_int4_scaled_mm per group    (Python loop, FP32 accum)
4. cuBLAS BF16:       torch.mm baseline
"""

import torch
import sys

sys.path.insert(0, '.')
import int4_native_tc_ops as ops


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def cuda_time(fn, warmup=20, iters=100):
    """Measure GPU-only time using CUDA events. Returns ms per call."""
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


def cuda_time_single(fn, warmup=10, iters=50):
    """Per-call timing: record events around each individual call and report median."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return mean, median, min(times), max(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16

    M = 700
    K = 2560
    N = 9728
    gs = 128
    num_groups = K // gs  # 20

    print(f"{'=' * 95}")
    print(f"  CUTLASS Grouped GEMM Profiling")
    print(f"  gate_proj: M={M}, K={K}, N={N}, gs={gs}, num_groups={num_groups}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 95}\n")

    # -------------------------------------------------------------------
    # Prepare inputs
    # -------------------------------------------------------------------
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    # Per-group weight quantization: w_packed [G, N, gs/2], w_scale [G, N]
    w_packed, w_scale = ops.static_int4_weight_quant_grouped(w, gs)

    # w_col_sum [G, N] for AZP correction
    wcs_list = []
    for g in range(num_groups):
        wcs_list.append(ops.unpack_int4_to_int8(w_packed[g], gs).float().sum(dim=1))
    w_col_sum = torch.stack(wcs_list)

    # Per-group activation quantization (asymmetric clipped)
    xp, xs, azp = ops.dynamic_int4_quant_asymmetric_clipped_grouped(x, gs, 0.95)
    # xp: [G, M, gs/2],  xs: [G, M],  azp: [G, M]

    out_fused = torch.empty(M, N, dtype=dtype, device=device)
    out_loop = torch.empty(M, N, dtype=dtype, device=device)
    out_manual = torch.empty(M, N, dtype=dtype, device=device)

    # Per-channel weight quantization (for manual loop approach)
    # We'll quantize each group independently and use cutlass_int4_scaled_mm per group
    # cutlass_int4_scaled_mm(a_packed, b_packed, scale_a, scale_b, out, M, N, K_group)

    # -------------------------------------------------------------------
    # 1. Fused V2 kernel
    # -------------------------------------------------------------------
    def fused_v2():
        ops.cutlass_int4_fused_grouped_gemm_azp(
            xp, w_packed, xs, w_scale, out_fused, azp, w_col_sum,
            M, N, gs, num_groups)

    t_fused = cuda_time(fused_v2)

    # -------------------------------------------------------------------
    # 2. Old CUTLASS per-group loop (C++ loop: 20 CUTLASS + 20 add_)
    # -------------------------------------------------------------------
    def old_loop():
        ops.cutlass_int4_scaled_mm_azp_grouped(
            xp, w_packed, xs, w_scale, out_loop, azp, w_col_sum,
            M, N, gs, num_groups)

    t_old_loop = cuda_time(old_loop)

    # -------------------------------------------------------------------
    # 3. Manual Python per-group loop using cutlass_int4_scaled_mm
    # -------------------------------------------------------------------
    def manual_loop():
        acc = torch.zeros(M, N, dtype=torch.float32, device=device)
        temp = torch.empty(M, N, dtype=dtype, device=device)
        for g in range(num_groups):
            a_g = xp[g]           # [M, gs/2]
            b_g = w_packed[g]     # [N, gs/2]
            sa_g = xs[g]          # [M]
            sb_g = w_scale[g]     # [N]
            ops.cutlass_int4_scaled_mm(a_g, b_g, sa_g, sb_g, temp, M, N, gs)
            acc.add_(temp)
            # AZP correction
            azp_g = azp[g]
            wcs_g = w_col_sum[g]
            u = (azp_g * sa_g).unsqueeze(1)
            v = (wcs_g * sb_g).unsqueeze(0)
            acc.add_(u * v)
        out_manual.copy_(acc.to(dtype))

    t_manual = cuda_time(manual_loop)

    # -------------------------------------------------------------------
    # 4. cuBLAS BF16 baseline
    # -------------------------------------------------------------------
    def cublas_bf16():
        torch.mm(x, w.t())

    t_cublas = cuda_time(cublas_bf16)

    # -------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------
    print(f"  {'Approach':<50} {'CUDA time (ms)':>14} {'vs cuBLAS':>10}")
    print(f"  {'-' * 74}")
    print(f"  {'1. Fused V2 (cutlass_int4_fused_grouped_gemm_azp)':<50} {t_fused:>14.4f} {t_fused/t_cublas:>9.2f}x")
    print(f"  {'2. Old loop (cutlass_int4_scaled_mm_azp_grouped)':<50} {t_old_loop:>14.4f} {t_old_loop/t_cublas:>9.2f}x")
    print(f"  {'3. Manual Python loop (cutlass_int4_scaled_mm)':<50} {t_manual:>14.4f} {t_manual/t_cublas:>9.2f}x")
    print(f"  {'4. cuBLAS BF16 (torch.mm)':<50} {t_cublas:>14.4f} {1.0:>9.2f}x")

    # Speedup summary
    print(f"\n  Speedups:")
    print(f"    Fused V2 vs Old loop:    {t_old_loop/t_fused:.2f}x faster")
    print(f"    Fused V2 vs Manual loop: {t_manual/t_fused:.2f}x faster")
    print(f"    Fused V2 vs cuBLAS BF16: {t_cublas/t_fused:.2f}x {'faster' if t_fused < t_cublas else 'slower'}")

    # -------------------------------------------------------------------
    # Breakdown: individual CUTLASS launch at K=128
    # -------------------------------------------------------------------
    print(f"\n\n{'=' * 95}")
    print(f"  Breakdown: Individual CUTLASS launch timing (K={gs})")
    print(f"  Single group GEMM: M={M}, N={N}, K={gs}")
    print(f"{'=' * 95}\n")

    # Time a single cutlass_int4_scaled_mm call (one group)
    a_g0 = xp[0].contiguous()
    b_g0 = w_packed[0].contiguous()
    sa_g0 = xs[0].contiguous()
    sb_g0 = w_scale[0].contiguous()
    temp_single = torch.empty(M, N, dtype=dtype, device=device)

    def single_cutlass():
        ops.cutlass_int4_scaled_mm(a_g0, b_g0, sa_g0, sb_g0, temp_single, M, N, gs)

    mean_s, median_s, min_s, max_s = cuda_time_single(single_cutlass)
    t_single_agg = cuda_time(single_cutlass)

    print(f"  Single CUTLASS launch (cutlass_int4_scaled_mm):")
    print(f"    Aggregate timing:  {t_single_agg:.4f} ms")
    print(f"    Per-call mean:     {mean_s:.4f} ms")
    print(f"    Per-call median:   {median_s:.4f} ms")
    print(f"    Per-call min:      {min_s:.4f} ms")
    print(f"    Per-call max:      {max_s:.4f} ms")

    # Time a single add_ (FP32 MxN matrix)
    acc_dummy = torch.zeros(M, N, dtype=torch.float32, device=device)
    temp_dummy = torch.randn(M, N, dtype=dtype, device=device)

    def single_add():
        acc_dummy.add_(temp_dummy)

    t_add = cuda_time(single_add)
    print(f"\n  Single acc.add_(temp) [FP32 {M}x{N}]:")
    print(f"    CUDA time:         {t_add:.4f} ms")

    # AZP correction for one group
    azp_g0 = azp[0]
    wcs_g0 = w_col_sum[0]

    def single_azp_correction():
        u = (azp_g0 * sa_g0).unsqueeze(1)
        v = (wcs_g0 * sb_g0).unsqueeze(0)
        acc_dummy.add_(u * v)

    t_azp = cuda_time(single_azp_correction)
    print(f"\n  Single AZP correction (outer product + add_):")
    print(f"    CUDA time:         {t_azp:.4f} ms")

    # Final copy
    out_dummy = torch.empty(M, N, dtype=dtype, device=device)
    def final_copy():
        out_dummy.copy_(acc_dummy.to(dtype))

    t_copy = cuda_time(final_copy)
    print(f"\n  Final acc.to(bf16) + copy:")
    print(f"    CUDA time:         {t_copy:.4f} ms")

    # Theoretical breakdown of old loop
    print(f"\n\n{'=' * 95}")
    print(f"  Theoretical vs Measured: Old Loop ({num_groups} groups)")
    print(f"{'=' * 95}\n")

    est_cutlass_total = t_single_agg * num_groups
    est_add_total = t_add * num_groups
    est_azp_total = t_azp * num_groups
    est_total = est_cutlass_total + est_add_total + est_azp_total + t_copy

    print(f"  Component breakdown (per group x {num_groups}):")
    print(f"    {num_groups}x CUTLASS INT4 GEMM:    {t_single_agg:.4f} x {num_groups} = {est_cutlass_total:.4f} ms")
    print(f"    {num_groups}x FP32 add_:            {t_add:.4f} x {num_groups} = {est_add_total:.4f} ms")
    print(f"    {num_groups}x AZP correction:       {t_azp:.4f} x {num_groups} = {est_azp_total:.4f} ms")
    print(f"    1x  final copy to BF16:    {t_copy:.4f} ms")
    print(f"    ---")
    print(f"    Estimated total:           {est_total:.4f} ms")
    print(f"    Measured old loop:          {t_old_loop:.4f} ms")
    print(f"    Measured manual loop:       {t_manual:.4f} ms")
    print(f"    Measured fused V2:          {t_fused:.4f} ms")

    # Time breakdown as percentages
    print(f"\n  Time breakdown of old loop (estimated):")
    if est_total > 0:
        print(f"    CUTLASS GEMM launches:     {est_cutlass_total/est_total*100:>5.1f}%  ({est_cutlass_total:.4f} ms)")
        print(f"    FP32 accumulation:         {est_add_total/est_total*100:>5.1f}%  ({est_add_total:.4f} ms)")
        print(f"    AZP correction:            {est_azp_total/est_total*100:>5.1f}%  ({est_azp_total:.4f} ms)")
        print(f"    Final BF16 copy:           {t_copy/est_total*100:>5.1f}%  ({t_copy:.4f} ms)")

    # -------------------------------------------------------------------
    # Fused V2 kernel breakdown
    # -------------------------------------------------------------------
    print(f"\n\n{'=' * 95}")
    print(f"  Fused V2 Kernel Analysis")
    print(f"{'=' * 95}\n")

    mean_f, median_f, min_f, max_f = cuda_time_single(fused_v2)
    print(f"  Per-call stats (fused V2):")
    print(f"    Mean:     {mean_f:.4f} ms")
    print(f"    Median:   {median_f:.4f} ms")
    print(f"    Min:      {min_f:.4f} ms")
    print(f"    Max:      {max_f:.4f} ms")
    print(f"    Aggregate:{t_fused:.4f} ms")

    print(f"\n  Efficiency:")
    print(f"    Fused V2 does in 1 launch what old loop does in {num_groups} CUTLASS + {num_groups} add_ + {num_groups} AZP")
    print(f"    Launch overhead eliminated: ~{num_groups*2 + num_groups - 1} kernel launches saved")
    print(f"    Speedup: {t_old_loop/t_fused:.2f}x over old C++ loop")
    print(f"    Speedup: {t_manual/t_fused:.2f}x over manual Python loop")

    # -------------------------------------------------------------------
    # Varying M to show scaling
    # -------------------------------------------------------------------
    print(f"\n\n{'=' * 95}")
    print(f"  Scaling across M values")
    print(f"{'=' * 95}\n")

    header = f"  {'M':>5} {'Fused V2':>10} {'Old Loop':>10} {'Manual':>10} {'cuBLAS':>10} {'V2/cuBLAS':>10} {'Old/V2':>8}"
    print(header)
    print(f"  {'-' * 63}")

    for M_test in [1, 10, 100, 300, 700, 1024]:
        x_t = torch.randn(M_test, K, dtype=dtype, device=device)
        w_t = w  # same weights
        xp_t, xs_t, azp_t = ops.dynamic_int4_quant_asymmetric_clipped_grouped(x_t, gs, 0.95)
        out_t = torch.empty(M_test, N, dtype=dtype, device=device)

        tf = cuda_time(lambda: ops.cutlass_int4_fused_grouped_gemm_azp(
            xp_t, w_packed, xs_t, w_scale, out_t, azp_t, w_col_sum,
            M_test, N, gs, num_groups))
        tl = cuda_time(lambda: ops.cutlass_int4_scaled_mm_azp_grouped(
            xp_t, w_packed, xs_t, w_scale, out_t, azp_t, w_col_sum,
            M_test, N, gs, num_groups))

        # Manual loop
        def manual_t():
            acc_t = torch.zeros(M_test, N, dtype=torch.float32, device=device)
            tmp_t = torch.empty(M_test, N, dtype=dtype, device=device)
            for g in range(num_groups):
                ops.cutlass_int4_scaled_mm(
                    xp_t[g], w_packed[g], xs_t[g], w_scale[g], tmp_t,
                    M_test, N, gs)
                acc_t.add_(tmp_t)
                u = (azp_t[g] * xs_t[g]).unsqueeze(1)
                v = (w_col_sum[g] * w_scale[g]).unsqueeze(0)
                acc_t.add_(u * v)
            out_t.copy_(acc_t.to(dtype))

        tm = cuda_time(manual_t)
        tb = cuda_time(lambda: torch.mm(x_t, w_t.t()))

        print(f"  {M_test:>5} {tf:>10.4f} {tl:>10.4f} {tm:>10.4f} {tb:>10.4f} "
              f"{tf/tb:>9.2f}x {tl/tf:>7.2f}x")

    print(f"\n  (All times in ms)")
    print(f"\n{'=' * 95}")
    print(f"  Done.")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
