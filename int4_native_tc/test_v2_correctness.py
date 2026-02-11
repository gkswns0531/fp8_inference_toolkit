#!/usr/bin/env python3
"""Quick correctness test: V2 kernel output vs V1 kernel reference."""
import torch
import sys
sys.path.insert(0, '.')
import int4_native_tc_ops as ops


def test_correctness():
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16
    gs = 128

    for M in [10, 100, 700]:
        for K, N, label in [(2560, 9728, "gate_proj"), (2560, 3584, "qkv_proj")]:
            num_groups = K // gs

            x = torch.randn(M, K, dtype=dtype, device=device)
            w = torch.randn(N, K, dtype=dtype, device=device)

            # Quantize
            w_packed, w_scale = ops.static_int4_weight_quant_grouped(w, gs)
            xp, xs, azp = ops.dynamic_int4_quant_asymmetric_clipped_grouped(x, gs, 0.95)

            # w_col_sum
            wcs = []
            for g in range(num_groups):
                wcs.append(ops.unpack_int4_to_int8(w_packed[g], gs).float().sum(dim=1))
            w_col_sum = torch.stack(wcs)

            # Run the fused GEMM (dispatches to V2 for M>32, V1 for M<=32)
            out = torch.empty(M, N, dtype=dtype, device=device)
            ops.cutlass_int4_fused_grouped_gemm_azp(
                xp, w_packed, xs, w_scale, out, azp, w_col_sum,
                M, N, gs, num_groups)

            # Reference: per-group loop via old CUTLASS
            out_ref = torch.empty(M, N, dtype=dtype, device=device)
            ops.cutlass_int4_scaled_mm_azp_grouped(
                xp, w_packed, xs, w_scale, out_ref, azp, w_col_sum,
                M, N, gs, num_groups)

            # Compare
            cos = torch.nn.functional.cosine_similarity(
                out.float().flatten().unsqueeze(0),
                out_ref.float().flatten().unsqueeze(0))
            max_diff = (out.float() - out_ref.float()).abs().max().item()
            rel_diff = max_diff / out_ref.float().abs().max().item()

            status = "PASS" if cos > 0.999 else "FAIL"
            print(f"  [{status}] M={M:>4}, {label:<12} (K={K}, N={N}): "
                  f"cosine={cos.item():.6f}, max_diff={max_diff:.4f}, rel={rel_diff:.6f}")

    # Also test symmetric (non-AZP) kernel
    print("\n  Symmetric kernel:")
    M, K, N = 700, 2560, 9728
    num_groups = K // gs
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    w_packed, w_scale = ops.static_int4_weight_quant_grouped(w, gs)
    xp, xs = ops.dynamic_int4_quant_clipped_grouped(x, gs, 0.95)

    out = torch.empty(M, N, dtype=dtype, device=device)
    ops.cutlass_int4_fused_grouped_gemm(
        xp, w_packed, xs, w_scale, out, M, N, gs, num_groups)

    out_ref = torch.empty(M, N, dtype=dtype, device=device)
    ops.cutlass_int4_scaled_mm_grouped(
        xp, w_packed, xs, w_scale, out_ref, M, N, gs, num_groups)

    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten().unsqueeze(0),
        out_ref.float().flatten().unsqueeze(0))
    status = "PASS" if cos > 0.999 else "FAIL"
    print(f"  [{status}] M={M}, gate_proj sym: cosine={cos.item():.6f}")


if __name__ == "__main__":
    print("V2 kernel correctness test:")
    test_correctness()
    print("\nDone!")
