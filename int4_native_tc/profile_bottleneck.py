#!/usr/bin/env python3
"""
PerCh vs V4 bottleneck analysis: single-layer profiling with nsight-level detail.
"""

import torch
import int4_native_tc as ops

torch.backends.cuda.matmul.allow_tf32 = False

M = 500
GS = 128

# Test both projection sizes
PROJECTIONS = {
    "qkv_proj":     (2560, 6144),
    "gate_up_proj": (2560, 19456),
}


def make_perch_data(N, K):
    w = torch.randn(N, K, dtype=torch.bfloat16)
    absmax = w.abs().amax(dim=1)
    scale = (absmax / 7.0).clamp(min=1e-10)
    w_int4 = torch.clamp(torch.round(w / scale.unsqueeze(1)), -8, 7).to(torch.int8)
    w_even = w_int4[:, 0::2] & 0x0F
    w_odd = (w_int4[:, 1::2] & 0x0F) << 4
    packed = (w_even | w_odd).to(torch.uint8)
    return packed.contiguous().cuda(), scale.to(torch.float32).contiguous().cuda()


def make_pergroup_data(N, K, gs):
    ng = K // gs
    w = torch.randn(N, K, dtype=torch.bfloat16)
    w_grouped = w.reshape(N, ng, gs).permute(1, 0, 2)
    absmax = w_grouped.abs().amax(dim=2)
    scale = (absmax / 7.0).clamp(min=1e-10)
    w_int4 = torch.clamp(torch.round(w_grouped / scale.unsqueeze(2)), -8, 7).to(torch.int8)
    w_even = w_int4[:, :, 0::2] & 0x0F
    w_odd = (w_int4[:, :, 1::2] & 0x0F) << 4
    packed = (w_even | w_odd).to(torch.uint8)
    return packed.contiguous().cuda(), scale.to(torch.float32).contiguous().cuda(), ng


def bench(fn, warmup=50, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    import numpy as np
    return np.median(times), np.mean(times), np.std(times)


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"M={M}, group_size={GS}\n")

for proj_name, (K, N) in PROJECTIONS.items():
    ng = K // GS
    print(f"{'='*70}")
    print(f"  {proj_name}: M={M}, K={K}, N={N}, num_groups={ng}")
    print(f"{'='*70}")

    # ── Data preparation ──
    wp_perch, ws_perch = make_perch_data(N, K)
    wp_pg, ws_pg, ng = make_pergroup_data(N, K, GS)

    # ── 1. PerCh: quant only ──
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    med, avg, std = bench(lambda: ops.dynamic_int4_quant(x))
    print(f"\n  [PerCh] Quant (per-channel):  {avg*1000:.1f}μs")
    perch_quant_us = avg * 1000

    # ── 2. PerCh: GEMM only ──
    x_packed, x_scale = ops.dynamic_int4_quant(x)
    out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    med, avg, std = bench(lambda: ops.cutlass_int4_scaled_mm(
        x_packed, wp_perch, x_scale, ws_perch, out, M, N, K))
    print(f"  [PerCh] GEMM (CUTLASS):       {avg*1000:.1f}μs")
    perch_gemm_us = avg * 1000

    # ── 3. PG: quant only ──
    med, avg, std = bench(lambda: ops.dynamic_int4_quant_grouped(x, GS))
    print(f"\n  [PG]    Quant (per-group):     {avg*1000:.1f}μs")
    pg_quant_us = avg * 1000

    # ── 4. PG V4: GEMM only ──
    xp, xs = ops.dynamic_int4_quant_grouped(x, GS)
    out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    med, avg, std = bench(lambda: ops.cutlass_int4_fused_grouped_gemm(
        xp, wp_pg, xs, ws_pg, out, M, N, GS, ng))
    print(f"  [PG]    GEMM V4 (fused):       {avg*1000:.1f}μs")
    pg_gemm_us = avg * 1000

    # ── Analysis ──
    print(f"\n  ── Breakdown ──")
    print(f"  Quant gap:  {pg_quant_us - perch_quant_us:+.1f}μs  ({pg_quant_us/perch_quant_us:.1f}x)")
    print(f"  GEMM gap:   {pg_gemm_us - perch_gemm_us:+.1f}μs  ({pg_gemm_us/perch_gemm_us:.1f}x)")

    total_perch = perch_quant_us + perch_gemm_us
    total_pg = pg_quant_us + pg_gemm_us
    print(f"  Total gap:  {total_pg - total_perch:+.1f}μs  ({total_pg/total_perch:.1f}x)")

    # ── Theoretical analysis ──
    print(f"\n  ── Why is PG GEMM slower? ──")
    print(f"  PerCh CUTLASS: 128×128 tile, 3-stage pipeline, K-loop={K//64} MMA iters")
    print(f"  PG V4:         64×128 tile, 2-stage pipeline, K-loop={GS//64} MMA iters × {ng} groups")
    print(f"  ")
    print(f"  PerCh MMA iterations:      {K//64} (continuous)")
    print(f"  PG MMA iterations/group:   {GS//64} (then sync + scale + sync)")
    print(f"  PG total MMA iterations:   {ng * (GS//64)} (= same compute)")
    print(f"  PG __syncthreads count:    {ng * 2} barriers")
    print(f"  ")

    # PerCh memory traffic
    perch_a_bytes = M * K // 2  # INT4 packed
    perch_b_bytes = N * K // 2
    perch_scale_bytes = (M + N) * 4  # FP32 scales
    perch_out_bytes = M * N * 2  # BF16 output
    perch_total_bytes = perch_a_bytes + perch_b_bytes + perch_scale_bytes + perch_out_bytes

    # PG memory traffic (V4 loads each group's data from global)
    pg_a_bytes = ng * M * (GS // 2)  # = M * K / 2 (same total)
    pg_b_bytes = ng * N * (GS // 2)  # = N * K / 2 (same total)
    pg_scale_bytes = ng * (M + N) * 4  # More scales! ng sets
    pg_out_bytes = M * N * 2
    pg_total_bytes = pg_a_bytes + pg_b_bytes + pg_scale_bytes + pg_out_bytes

    print(f"  PerCh global memory: A={perch_a_bytes/1e6:.1f}MB + B={perch_b_bytes/1e6:.1f}MB "
          f"+ scales={perch_scale_bytes/1e3:.0f}KB + out={perch_out_bytes/1e6:.1f}MB "
          f"= {perch_total_bytes/1e6:.1f}MB")
    print(f"  PG global memory:    A={pg_a_bytes/1e6:.1f}MB + B={pg_b_bytes/1e6:.1f}MB "
          f"+ scales={pg_scale_bytes/1e3:.0f}KB + out={pg_out_bytes/1e6:.1f}MB "
          f"= {pg_total_bytes/1e6:.1f}MB")
    print(f"  Scale overhead:      {pg_scale_bytes/1e3:.0f}KB vs {perch_scale_bytes/1e3:.0f}KB "
          f"({pg_scale_bytes/perch_scale_bytes:.0f}x)")

    # Compute intensity
    flops = 2 * M * K * N  # MACs counted as 2 FLOPs
    perch_ai = flops / perch_total_bytes
    pg_ai = flops / pg_total_bytes
    print(f"  Arithmetic intensity: PerCh={perch_ai:.1f} FLOP/B, PG={pg_ai:.1f} FLOP/B")

    # L4 specs
    l4_bw = 300e9  # ~300 GB/s
    l4_tflops = 120e12 * 2  # INT4 TC, ~240 TOPS
    perch_bw_time = perch_total_bytes / l4_bw * 1e6
    pg_bw_time = pg_total_bytes / l4_bw * 1e6
    compute_time = flops / l4_tflops * 1e6
    print(f"\n  ── Roofline (L4: 300 GB/s, ~240 INT4 TOPS) ──")
    print(f"  Memory-bound time: PerCh={perch_bw_time:.1f}μs, PG={pg_bw_time:.1f}μs")
    print(f"  Compute-bound time: {compute_time:.1f}μs")
    print(f"  Actual time:       PerCh={perch_gemm_us:.1f}μs, PG={pg_gemm_us:.1f}μs")
    print(f"  PerCh efficiency:  {min(perch_bw_time, compute_time)/perch_gemm_us*100:.0f}% of roofline")
    print(f"  PG efficiency:     {min(pg_bw_time, compute_time)/pg_gemm_us*100:.0f}% of roofline")

    overhead_us = pg_gemm_us - perch_gemm_us
    overhead_per_group = overhead_us / ng
    print(f"\n  ── Per-group overhead ──")
    print(f"  Total overhead:    {overhead_us:.1f}μs")
    print(f"  Per-group overhead: {overhead_per_group:.1f}μs/group")
    print(f"  Approximate breakdown:")
    print(f"    __syncthreads × 2:   ~{ng*2*0.01:.1f}μs (negligible)")
    print(f"    Pipeline startup/drain × {ng}: dominant factor")
    print(f"    Scale load+apply × {ng}: moderate")
    print()
