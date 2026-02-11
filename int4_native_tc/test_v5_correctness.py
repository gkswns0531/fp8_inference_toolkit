#!/usr/bin/env python3
"""Correctness test: V5 kernel output vs V4 kernel reference."""
import torch
import sys
sys.path.insert(0, '.')
import int4_native_tc_ops as ops


def test_v5_symmetric():
    """Test V5 symmetric (non-AZP) kernel against V4 reference."""
    print("  V5 Symmetric kernel tests:")
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16
    gs = 128

    for M in [64, 100, 700]:
        for K, N, label in [
            (2560, 9728, "gate_proj"),    # num_groups=20, divisible by 4
            (2560, 3584, "qkv_proj"),     # num_groups=20, divisible by 4
            (9728, 2560, "down_proj"),    # num_groups=76, divisible by 4
        ]:
            num_groups = K // gs

            x = torch.randn(M, K, dtype=dtype, device=device)
            w = torch.randn(N, K, dtype=dtype, device=device)

            # Quantize
            w_packed, w_scale = ops.static_int4_weight_quant_grouped(w, gs)
            xp, xs = ops.dynamic_int4_quant_clipped_grouped(x, gs, 0.95)

            # V5 output
            out_v5 = torch.empty(M, N, dtype=dtype, device=device)
            ops.cutlass_int4_fused_grouped_gemm_v5(
                xp, w_packed, xs, w_scale, out_v5, M, N, gs, num_groups)

            # V4 reference
            out_v4 = torch.empty(M, N, dtype=dtype, device=device)
            ops.cutlass_int4_fused_grouped_gemm(
                xp, w_packed, xs, w_scale, out_v4, M, N, gs, num_groups)

            # Compare
            cos = torch.nn.functional.cosine_similarity(
                out_v5.float().flatten().unsqueeze(0),
                out_v4.float().flatten().unsqueeze(0))
            max_diff = (out_v5.float() - out_v4.float()).abs().max().item()
            rel_diff = max_diff / (out_v4.float().abs().max().item() + 1e-8)

            gpl = 4 if num_groups % 4 == 0 else 2
            status = "PASS" if cos > 0.9999 else "FAIL"
            print(f"    [{status}] M={M:>4}, {label:<12} (K={K}, N={N}, G={num_groups}, GPL={gpl}): "
                  f"cosine={cos.item():.6f}, max_diff={max_diff:.6f}, rel={rel_diff:.6f}")


def test_v5_azp():
    """Test V5 AZP kernel against V4 AZP reference."""
    print("\n  V5 AZP kernel tests:")
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16
    gs = 128

    for M in [64, 100, 700]:
        for K, N, label in [
            (2560, 9728, "gate_proj"),
            (2560, 3584, "qkv_proj"),
        ]:
            num_groups = K // gs

            x = torch.randn(M, K, dtype=dtype, device=device)
            w = torch.randn(N, K, dtype=dtype, device=device)

            # Quantize with asymmetric
            w_packed, w_scale = ops.static_int4_weight_quant_grouped(w, gs)
            xp, xs, azp = ops.dynamic_int4_quant_asymmetric_clipped_grouped(x, gs, 0.95)

            # w_col_sum
            wcs = []
            for g in range(num_groups):
                wcs.append(ops.unpack_int4_to_int8(w_packed[g], gs).float().sum(dim=1))
            w_col_sum = torch.stack(wcs)

            # V5 AZP output
            out_v5 = torch.empty(M, N, dtype=dtype, device=device)
            ops.cutlass_int4_fused_grouped_gemm_v5_azp(
                xp, w_packed, xs, w_scale, out_v5, azp, w_col_sum,
                M, N, gs, num_groups)

            # V4 AZP reference
            out_v4 = torch.empty(M, N, dtype=dtype, device=device)
            ops.cutlass_int4_fused_grouped_gemm_azp(
                xp, w_packed, xs, w_scale, out_v4, azp, w_col_sum,
                M, N, gs, num_groups)

            # Compare
            cos = torch.nn.functional.cosine_similarity(
                out_v5.float().flatten().unsqueeze(0),
                out_v4.float().flatten().unsqueeze(0))
            max_diff = (out_v5.float() - out_v4.float()).abs().max().item()
            rel_diff = max_diff / (out_v4.float().abs().max().item() + 1e-8)

            gpl = 4 if num_groups % 4 == 0 else 2
            status = "PASS" if cos > 0.9999 else "FAIL"
            print(f"    [{status}] M={M:>4}, {label:<12} (K={K}, N={N}, G={num_groups}, GPL={gpl}): "
                  f"cosine={cos.item():.6f}, max_diff={max_diff:.6f}, rel={rel_diff:.6f}")


def test_v5_small_m():
    """Test V5 with small M (should fallback to V1 kernel)."""
    print("\n  V5 Small-M fallback tests:")
    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.bfloat16
    gs = 128

    for M in [1, 10, 32]:
        K, N = 2560, 9728
        num_groups = K // gs

        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)

        w_packed, w_scale = ops.static_int4_weight_quant_grouped(w, gs)
        xp, xs = ops.dynamic_int4_quant_clipped_grouped(x, gs, 0.95)

        out_v5 = torch.empty(M, N, dtype=dtype, device=device)
        ops.cutlass_int4_fused_grouped_gemm_v5(
            xp, w_packed, xs, w_scale, out_v5, M, N, gs, num_groups)

        out_v4 = torch.empty(M, N, dtype=dtype, device=device)
        ops.cutlass_int4_fused_grouped_gemm(
            xp, w_packed, xs, w_scale, out_v4, M, N, gs, num_groups)

        cos = torch.nn.functional.cosine_similarity(
            out_v5.float().flatten().unsqueeze(0),
            out_v4.float().flatten().unsqueeze(0))
        status = "PASS" if cos > 0.9999 else "FAIL"
        print(f"    [{status}] M={M:>4}, gate_proj (K={K}, N={N}): cosine={cos.item():.6f}")


if __name__ == "__main__":
    print("V5 multi-group K-loop kernel correctness test:")
    test_v5_symmetric()
    test_v5_azp()
    test_v5_small_m()
    print("\nDone!")
