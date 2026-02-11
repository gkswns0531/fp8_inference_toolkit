#!/usr/bin/env python3
"""
Per-channel vs Per-group comparison at M=700.
Same total compute, different quantization granularity.
"""
import torch
import sys
sys.path.insert(0, '.')
import int4_native_tc_ops as ops


def cuda_time(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16
    gs = 128
    M = 700

    projs = [
        ("qkv_proj",  2560, 3584),
        ("o_proj",    2560, 2560),
        ("gate_proj", 2560, 9728),
        ("up_proj",   2560, 9728),
        ("down_proj", 9728, 2560),
    ]

    print(f"{'='*100}")
    print(f"  Per-channel vs Per-group comparison (M={M})")
    print(f"{'='*100}\n")

    header = (f"  {'Layer':<12} {'K':>5} {'N':>5}  "
              f"{'BF16':>8} {'PerCh(q)':>9} {'PerCh(g)':>9} {'PerCh':>8}  "
              f"{'PerGrp(q)':>10} {'PerGrp(g)':>10} {'PerGrp':>8}  "
              f"{'Ch/BF16':>7} {'Grp/BF16':>8}")
    print(header)
    print("-" * len(header))

    tot_bf16 = 0; tot_ch = 0; tot_grp = 0
    tot_ch_q = 0; tot_ch_g = 0; tot_grp_q = 0; tot_grp_g = 0

    for name, K, N in projs:
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)

        # ── BF16 baseline ──
        t_bf16 = cuda_time(lambda: torch.mm(x, w.t()))

        # ── Per-channel (INT4-naive style): single CUTLASS GEMM ──
        w_packed_ch, w_scale_ch = ops.static_int4_weight_quant(w)
        out_ch = torch.empty(M, N, dtype=dtype, device=device)

        def quant_ch():
            return ops.dynamic_int4_quant(x)
        t_ch_q = cuda_time(quant_ch)

        xp_ch, xs_ch = quant_ch()
        def gemm_ch():
            ops.cutlass_int4_scaled_mm(xp_ch, w_packed_ch, xs_ch, w_scale_ch, out_ch, M, N, K)
        t_ch_g = cuda_time(gemm_ch)
        t_ch = t_ch_q + t_ch_g

        # ── Per-group (Enhanced-PG style): fused grouped GEMM ──
        num_groups = K // gs
        w_packed_grp, w_scale_grp = ops.static_int4_weight_quant_grouped(w, gs)
        wcs = []
        for g in range(num_groups):
            wcs.append(ops.unpack_int4_to_int8(w_packed_grp[g], gs).float().sum(dim=1))
        w_col_sum = torch.stack(wcs)
        out_grp = torch.empty(M, N, dtype=dtype, device=device)

        def quant_grp():
            return ops.dynamic_int4_quant_asymmetric_clipped_grouped(x, gs, 0.95)
        t_grp_q = cuda_time(quant_grp)

        xp_grp, xs_grp, azp_grp = quant_grp()
        def gemm_grp():
            ops.cutlass_int4_fused_grouped_gemm_azp(
                xp_grp, w_packed_grp, xs_grp, w_scale_grp,
                out_grp, azp_grp, w_col_sum, M, N, gs, num_groups)
        t_grp_g = cuda_time(gemm_grp)
        t_grp = t_grp_q + t_grp_g

        tot_bf16 += t_bf16
        tot_ch += t_ch; tot_ch_q += t_ch_q; tot_ch_g += t_ch_g
        tot_grp += t_grp; tot_grp_q += t_grp_q; tot_grp_g += t_grp_g

        print(f"  {name:<12} {K:>5} {N:>5}  "
              f"{t_bf16:>8.4f} {t_ch_q:>9.4f} {t_ch_g:>9.4f} {t_ch:>8.4f}  "
              f"{t_grp_q:>10.4f} {t_grp_g:>10.4f} {t_grp:>8.4f}  "
              f"{t_ch/t_bf16:>6.2f}x {t_grp/t_bf16:>7.2f}x")

    print()
    print(f"  {'Per-layer total':<24} "
          f"{tot_bf16:>8.4f} {tot_ch_q:>9.4f} {tot_ch_g:>9.4f} {tot_ch:>8.4f}  "
          f"{tot_grp_q:>10.4f} {tot_grp_g:>10.4f} {tot_grp:>8.4f}")
    print(f"  {'×36 layers (ms)':<24} "
          f"{tot_bf16*36:>8.2f} {tot_ch_q*36:>9.2f} {tot_ch_g*36:>9.2f} {tot_ch*36:>8.2f}  "
          f"{tot_grp_q*36:>10.2f} {tot_grp_g*36:>10.2f} {tot_grp*36:>8.2f}")

    print(f"\n{'='*100}")
    print(f"  Summary (×36 layers)")
    print(f"{'='*100}")
    print(f"  BF16 cuBLAS:            {tot_bf16*36:>8.2f} ms")
    print(f"  Per-channel INT4:       {tot_ch*36:>8.2f} ms  (quant: {tot_ch_q*36:.2f}, GEMM: {tot_ch_g*36:.2f})")
    print(f"  Per-group Enhanced-PG:  {tot_grp*36:>8.2f} ms  (quant: {tot_grp_q*36:.2f}, GEMM: {tot_grp_g*36:.2f})")
    print(f"\n  Per-channel vs BF16:    {tot_ch/tot_bf16:.2f}x")
    print(f"  Per-group vs BF16:      {tot_grp/tot_bf16:.2f}x")
    print(f"  Per-group overhead vs per-channel: +{(tot_grp-tot_ch)*36:.2f} ms")
    print(f"    - Quant overhead:     +{(tot_grp_q-tot_ch_q)*36:.2f} ms")
    print(f"    - GEMM overhead:      +{(tot_grp_g-tot_ch_g)*36:.2f} ms")


if __name__ == "__main__":
    main()
