#!/usr/bin/env python3
"""
W4A8-FP8 Concept Validation Benchmark

Concept:
  - Weight: INT4 per-group (g128) stored in memory (4x compression)
  - Runtime dequant: INT4 → FP8 (e4m3)
  - Activation: BF16 → FP8 (per-token scale)
  - GEMM: torch._scaled_mm (cuBLAS FP8, no accumulator flush!)

Compare accuracy and speed:
  1. BF16 baseline (torch.mm)
  2. Standard FP8 (BF16→FP8 direct, torch._scaled_mm)
  3. W4A4 per-group V8 (INT4×INT4, our kernel)
  4. W4A8-FP8 (INT4 weight→FP8 dequant + FP8 GEMM) ★ NEW

Usage:
    python benchmark_w4a8fp8.py
"""

import torch
import torch.nn.functional as F
import time
import numpy as np

torch.manual_seed(42)


def quantize_to_fp8_per_tensor(x: torch.Tensor):
    """Quantize tensor to FP8 e4m3 with per-tensor scale."""
    amax = x.float().abs().max().clamp(min=1e-12)
    scale = amax / 448.0
    x_fp8 = (x.float() / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


def quantize_to_fp8_per_row(x: torch.Tensor):
    """Quantize tensor to FP8 e4m3 with per-row scale."""
    amax = x.float().abs().amax(dim=1).clamp(min=1e-12)  # [M]
    scale = amax / 448.0
    x_fp8 = (x.float() / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


def quantize_to_fp8_per_col(x: torch.Tensor):
    """Quantize tensor to FP8 e4m3 with per-column scale."""
    amax = x.float().abs().amax(dim=1).clamp(min=1e-12)  # [N] (row of weight = column of result)
    scale = amax / 448.0
    x_fp8 = (x.float() / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def mae_relative(a: torch.Tensor, b: torch.Tensor) -> float:
    ref_norm = b.float().abs().mean().item()
    return (a.float() - b.float()).abs().mean().item() / max(ref_norm, 1e-10) * 100


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  W4A8-FP8 Concept Validation")
print("=" * 70)

# Check torch._scaled_mm support
print(f"\nGPU: {torch.cuda.get_device_name()}")
print(f"SM: {torch.cuda.get_device_capability()}")
print(f"PyTorch: {torch.__version__}")

# Test shapes (Qwen3-4B linear layers)
SHAPES = [
    ("qkv_proj", 2560, 6144),
    ("gate_up_proj", 2560, 19456),
]

BATCH_SIZES = [1, 16, 500]
GROUP_SIZE = 128

# Import our INT4 ops
try:
    import int4_native_tc as ops
    HAS_INT4_TC = True
    print("int4_native_tc: loaded")
except ImportError as e:
    HAS_INT4_TC = False
    print(f"int4_native_tc: NOT available ({e})")


print("\n" + "=" * 70)
print("  Part 1: torch._scaled_mm Scale Format Test")
print("=" * 70)

# Test what scale formats work on this GPU
M_test, K_test, N_test = 16, 256, 128
a_test = torch.randn(M_test, K_test, dtype=torch.bfloat16, device='cuda')
w_test = torch.randn(N_test, K_test, dtype=torch.bfloat16, device='cuda')
ref_test = torch.mm(a_test, w_test.T)

a_fp8, a_s = quantize_to_fp8_per_tensor(a_test)
w_fp8, w_s = quantize_to_fp8_per_tensor(w_test)

# Test per-tensor scales (should always work)
try:
    out = torch._scaled_mm(
        a_fp8, w_fp8.T,
        scale_a=a_s.unsqueeze(0).unsqueeze(0),
        scale_b=w_s.unsqueeze(0).unsqueeze(0),
        out_dtype=torch.bfloat16,
    )
    cos = cosine_sim(out, ref_test)
    print(f"  Per-tensor scales: OK (CosSim={cos:.6f})")
    SCALE_MODE = "per_tensor"
except Exception as e:
    print(f"  Per-tensor scales: FAILED ({e})")
    SCALE_MODE = None

# Test per-row/per-col scales
try:
    a_fp8_r, a_s_r = quantize_to_fp8_per_row(a_test)
    w_fp8_c, w_s_c = quantize_to_fp8_per_col(w_test)
    out = torch._scaled_mm(
        a_fp8_r, w_fp8_c.T,
        scale_a=a_s_r.unsqueeze(1),       # [M, 1]
        scale_b=w_s_c.unsqueeze(0),       # [1, N]
        out_dtype=torch.bfloat16,
    )
    cos = cosine_sim(out, ref_test)
    print(f"  Per-row/col scales: OK (CosSim={cos:.6f})")
    SCALE_MODE = "per_row_col"
except Exception as e:
    print(f"  Per-row/col scales: FAILED ({e})")

print(f"\n  Using scale mode: {SCALE_MODE}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Part 2: Accuracy Comparison")
print("=" * 70)

for layer_name, K, N in SHAPES:
    print(f"\n{'─'*60}")
    print(f"  Layer: {layer_name} (K={K}, N={N})")
    print(f"{'─'*60}")

    # Generate realistic weight and activation
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') * 0.02
    num_groups = K // GROUP_SIZE

    for M in BATCH_SIZES:
        act_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

        # 1. BF16 baseline
        ref = torch.mm(act_bf16, weight_bf16.T)

        # 2. Standard FP8 (BF16 → FP8 direct)
        if SCALE_MODE == "per_row_col":
            a_fp8, a_scale = quantize_to_fp8_per_row(act_bf16)
            w_fp8, w_scale = quantize_to_fp8_per_col(weight_bf16)
            fp8_out = torch._scaled_mm(
                a_fp8, w_fp8.T,
                scale_a=a_scale.unsqueeze(1),
                scale_b=w_scale.unsqueeze(0),
                out_dtype=torch.bfloat16,
            )
        else:
            a_fp8, a_scale = quantize_to_fp8_per_tensor(act_bf16)
            w_fp8, w_scale = quantize_to_fp8_per_tensor(weight_bf16)
            fp8_out = torch._scaled_mm(
                a_fp8, w_fp8.T,
                scale_a=a_scale.unsqueeze(0).unsqueeze(0),
                scale_b=w_scale.unsqueeze(0).unsqueeze(0),
                out_dtype=torch.bfloat16,
            )
        fp8_cos = cosine_sim(fp8_out, ref)
        fp8_mae = mae_relative(fp8_out, ref)

        # 3. W4A4 per-group (our kernel)
        w4a4_cos, w4a4_mae = 0.0, 0.0
        if HAS_INT4_TC:
            w_packed, w_scale_g = ops.static_int4_weight_quant_grouped(weight_bf16, GROUP_SIZE)
            x_packed, x_scale_g = ops.dynamic_int4_quant_grouped(act_bf16, GROUP_SIZE)
            w4a4_out = torch.empty(M, N, dtype=act_bf16.dtype, device='cuda')
            ops.cutlass_int4_fused_grouped_gemm(
                x_packed, w_packed, x_scale_g, w_scale_g,
                w4a4_out, M, N, GROUP_SIZE, num_groups)
            w4a4_cos = cosine_sim(w4a4_out, ref)
            w4a4_mae = mae_relative(w4a4_out, ref)

        # 4. W4A8-FP8: INT4 weight dequant → FP8, FP8 activation, torch._scaled_mm
        #    Step A: Quantize weight to INT4 per-group
        if HAS_INT4_TC:
            w_packed_g, w_scale_g = ops.static_int4_weight_quant_grouped(weight_bf16, GROUP_SIZE)
            #    Step B: Dequant INT4 → BF16 (simulates runtime dequant)
            w_dequant = ops.dequant_int4_grouped_to_bf16(w_packed_g, w_scale_g, GROUP_SIZE)
        else:
            # Pure PyTorch fallback for INT4 per-group quantization
            w_fp32 = weight_bf16.float()
            w_reshaped = w_fp32.reshape(N, num_groups, GROUP_SIZE)
            g_amax = w_reshaped.abs().amax(dim=2).clamp(min=1e-12)  # [N, num_groups]
            g_scale = g_amax / 7.0  # symmetric INT4 range [-8, 7]
            w_int4 = torch.clamp(torch.round(w_reshaped / g_scale.unsqueeze(2)), -8, 7)
            w_dequant = (w_int4 * g_scale.unsqueeze(2)).reshape(N, K).to(torch.bfloat16)

        #    Step C: Requantize dequanted weight to FP8
        if SCALE_MODE == "per_row_col":
            w_fp8_d, w_scale_d = quantize_to_fp8_per_col(w_dequant)
            a_fp8_d, a_scale_d = quantize_to_fp8_per_row(act_bf16)
            w4a8fp8_out = torch._scaled_mm(
                a_fp8_d, w_fp8_d.T,
                scale_a=a_scale_d.unsqueeze(1),
                scale_b=w_scale_d.unsqueeze(0),
                out_dtype=torch.bfloat16,
            )
        else:
            w_fp8_d, w_scale_d = quantize_to_fp8_per_tensor(w_dequant)
            a_fp8_d, a_scale_d = quantize_to_fp8_per_tensor(act_bf16)
            w4a8fp8_out = torch._scaled_mm(
                a_fp8_d, w_fp8_d.T,
                scale_a=a_scale_d.unsqueeze(0).unsqueeze(0),
                scale_b=w_scale_d.unsqueeze(0).unsqueeze(0),
                out_dtype=torch.bfloat16,
            )
        w4a8fp8_cos = cosine_sim(w4a8fp8_out, ref)
        w4a8fp8_mae = mae_relative(w4a8fp8_out, ref)

        print(f"\n  M={M:>4}:")
        print(f"    {'Method':<22} {'CosSim':>10} {'MAE/ref%':>10}")
        print(f"    {'─'*44}")
        print(f"    {'FP8 (direct)':.<22} {fp8_cos:>10.6f} {fp8_mae:>9.2f}%")
        print(f"    {'W4A8-FP8 (ours)':.<22} {w4a8fp8_cos:>10.6f} {w4a8fp8_mae:>9.2f}%")
        if HAS_INT4_TC:
            print(f"    {'W4A4-PG V8':.<22} {w4a4_cos:>10.6f} {w4a4_mae:>9.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Part 3: Latency Comparison")
print("=" * 70)

K, N = 2560, 6144
M = 500
WARMUP = 10
RUNS = 50

weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') * 0.02
act_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
num_groups = K // GROUP_SIZE

# Pre-compute everything that can be pre-computed
# FP8 direct
if SCALE_MODE == "per_row_col":
    w_fp8_direct, w_scale_direct = quantize_to_fp8_per_col(weight_bf16)
else:
    w_fp8_direct, w_scale_direct = quantize_to_fp8_per_tensor(weight_bf16)

# W4A8-FP8: pre-quantize weight to INT4, pre-dequant to FP8
if HAS_INT4_TC:
    w_packed_g, w_scale_g = ops.static_int4_weight_quant_grouped(weight_bf16, GROUP_SIZE)
    w_dequant = ops.dequant_int4_grouped_to_bf16(w_packed_g, w_scale_g, GROUP_SIZE)
else:
    w_fp32 = weight_bf16.float()
    w_reshaped = w_fp32.reshape(N, num_groups, GROUP_SIZE)
    g_amax = w_reshaped.abs().amax(dim=2).clamp(min=1e-12)
    g_scale = g_amax / 7.0
    w_int4 = torch.clamp(torch.round(w_reshaped / g_scale.unsqueeze(2)), -8, 7)
    w_dequant = (w_int4 * g_scale.unsqueeze(2)).reshape(N, K).to(torch.bfloat16)

if SCALE_MODE == "per_row_col":
    w_fp8_w4a8, w_scale_w4a8 = quantize_to_fp8_per_col(w_dequant)
else:
    w_fp8_w4a8, w_scale_w4a8 = quantize_to_fp8_per_tensor(w_dequant)

# W4A4 pre-compute
if HAS_INT4_TC:
    w_packed_v8, w_scale_v8 = ops.static_int4_weight_quant_grouped(weight_bf16, GROUP_SIZE)

# Pre-allocate output
out_buf = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')


def bench(name, fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds
    avg = np.mean(times)
    std = np.std(times)
    print(f"  {name:<35} {avg:>8.0f} us  (std={std:.0f})")
    return avg


print(f"\n  Shape: M={M}, K={K}, N={N}")
print(f"  Warmup={WARMUP}, Runs={RUNS}")
print()

# BF16
bench("BF16 (torch.mm)", lambda: torch.mm(act_bf16, weight_bf16.T))

# FP8 direct (pre-quantized weight, runtime activation quant + GEMM)
def fp8_direct():
    if SCALE_MODE == "per_row_col":
        a_fp8, a_s = quantize_to_fp8_per_row(act_bf16)
        return torch._scaled_mm(a_fp8, w_fp8_direct.T,
                                scale_a=a_s.unsqueeze(1),
                                scale_b=w_scale_direct.unsqueeze(0),
                                out_dtype=torch.bfloat16)
    else:
        a_fp8, a_s = quantize_to_fp8_per_tensor(act_bf16)
        return torch._scaled_mm(a_fp8, w_fp8_direct.T,
                                scale_a=a_s.unsqueeze(0).unsqueeze(0),
                                scale_b=w_scale_direct.unsqueeze(0).unsqueeze(0),
                                out_dtype=torch.bfloat16)

bench("FP8 direct (quant+GEMM)", fp8_direct)

# W4A8-FP8: pre-dequanted weight as FP8 (same speed as FP8 direct)
def w4a8fp8_predequant():
    if SCALE_MODE == "per_row_col":
        a_fp8, a_s = quantize_to_fp8_per_row(act_bf16)
        return torch._scaled_mm(a_fp8, w_fp8_w4a8.T,
                                scale_a=a_s.unsqueeze(1),
                                scale_b=w_scale_w4a8.unsqueeze(0),
                                out_dtype=torch.bfloat16)
    else:
        a_fp8, a_s = quantize_to_fp8_per_tensor(act_bf16)
        return torch._scaled_mm(a_fp8, w_fp8_w4a8.T,
                                scale_a=a_s.unsqueeze(0).unsqueeze(0),
                                scale_b=w_scale_w4a8.unsqueeze(0).unsqueeze(0),
                                out_dtype=torch.bfloat16)

bench("W4A8-FP8 pre-dequant (quant+GEMM)", w4a8fp8_predequant)

# W4A8-FP8: runtime dequant (INT4→FP8 every forward pass)
def w4a8fp8_runtime_dequant():
    # Dequant INT4 → BF16 → FP8 (PyTorch ops, slow but correct)
    if HAS_INT4_TC:
        w_deq = ops.dequant_int4_grouped_to_bf16(w_packed_g, w_scale_g, GROUP_SIZE)
    else:
        w_deq = w_dequant  # pre-computed for non-TC path

    if SCALE_MODE == "per_row_col":
        w_fp8, w_s = quantize_to_fp8_per_col(w_deq)
        a_fp8, a_s = quantize_to_fp8_per_row(act_bf16)
        return torch._scaled_mm(a_fp8, w_fp8.T,
                                scale_a=a_s.unsqueeze(1),
                                scale_b=w_s.unsqueeze(0),
                                out_dtype=torch.bfloat16)
    else:
        w_fp8, w_s = quantize_to_fp8_per_tensor(w_deq)
        a_fp8, a_s = quantize_to_fp8_per_tensor(act_bf16)
        return torch._scaled_mm(a_fp8, w_fp8.T,
                                scale_a=a_s.unsqueeze(0).unsqueeze(0),
                                scale_b=w_s.unsqueeze(0).unsqueeze(0),
                                out_dtype=torch.bfloat16)

bench("W4A8-FP8 runtime dequant (full)", w4a8fp8_runtime_dequant)

# W4A4 per-group V8
if HAS_INT4_TC:
    def w4a4_v8():
        x_packed, x_scale = ops.dynamic_int4_quant_grouped(act_bf16, GROUP_SIZE)
        ops.cutlass_int4_fused_grouped_gemm(
            x_packed, w_packed_v8, x_scale, w_scale_v8,
            out_buf, M, N, GROUP_SIZE, num_groups)
        return out_buf

    bench("W4A4 PG-V8 (INT4×INT4, ours)", w4a4_v8)


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Part 4: Memory Footprint Comparison")
print("=" * 70)

for layer_name, K, N in SHAPES:
    bf16_bytes = N * K * 2
    fp8_bytes = N * K * 1  # FP8 weight + scale
    int4_bytes = N * K // 2 + (K // GROUP_SIZE) * N * 4  # packed + group scales

    print(f"\n  {layer_name} (K={K}, N={N}):")
    print(f"    BF16:     {bf16_bytes / 1e6:>6.1f} MB  (1.0x)")
    print(f"    FP8:      {fp8_bytes / 1e6:>6.1f} MB  ({bf16_bytes/fp8_bytes:.1f}x compression)")
    print(f"    INT4 PG:  {int4_bytes / 1e6:>6.1f} MB  ({bf16_bytes/int4_bytes:.1f}x compression)")


print("\n" + "=" * 70)
print("  Summary")
print("=" * 70)
print("""
  W4A8-FP8 approach:
    Weight: INT4 per-group (4x compression) → runtime dequant to FP8
    Activation: BF16 → FP8 per-token quantization
    GEMM: torch._scaled_mm (cuBLAS FP8, NO accumulator flush!)

  vs W4A4 (current):
    Weight: INT4 per-group (4x compression)
    Activation: BF16 → INT4 per-group quantization
    GEMM: INT4×INT4 MMA → INT32, accumulator flush every 2 MMA iterations

  Key advantage: FP8 GEMM has no per-group accumulator flush overhead.
  Key trade-off: FP8 activation (8-bit, per-token) vs INT4 activation (4-bit, per-group).
""")
