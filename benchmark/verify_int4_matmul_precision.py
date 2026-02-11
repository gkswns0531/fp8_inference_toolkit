#!/usr/bin/env python3
"""
INT4×INT4 native matmul vs INT4→BF16 dequant matmul 정밀도 비교

두 경로의 정보 손실 차이를 수학적으로 검증:
1. Native: INT4 × INT4 → INT32 누적 → scale 적용
2. Dequant: (INT4 × scale) → BF16, BF16 × BF16 → FP32 누적
"""

import torch
import numpy as np

torch.manual_seed(42)


def int4_native_matmul(A_int4: torch.Tensor, B_int4: torch.Tensor,
                        scale_A: torch.Tensor, scale_B: torch.Tensor) -> torch.Tensor:
    """
    가상의 네이티브 INT4×INT4 경로 시뮬레이션.
    INT32에서 정확한 정수 연산 후 scale 적용.
    """
    # INT32로 exact 정수 matmul
    A_int32 = A_int4.to(torch.int32)
    B_int32 = B_int4.to(torch.int32)
    result_int32 = torch.matmul(A_int32, B_int32.T)  # exact integer

    # scale 적용 (FP32에서)
    result_fp32 = result_int32.float() * scale_A.unsqueeze(1) * scale_B.unsqueeze(0)
    return result_fp32


def int4_dequant_matmul(A_int4: torch.Tensor, B_int4: torch.Tensor,
                         scale_A: torch.Tensor, scale_B: torch.Tensor) -> torch.Tensor:
    """
    현재 실제 사용되는 dequant 경로 시뮬레이션.
    INT4 → BF16 dequant 후 BF16 matmul, FP32 누적.
    """
    # INT4 × scale → BF16 (여기서 rounding 발생 가능)
    A_bf16 = (A_int4.float() * scale_A.unsqueeze(1)).to(torch.bfloat16)
    B_bf16 = (B_int4.float() * scale_B.unsqueeze(1)).to(torch.bfloat16)

    # BF16 × BF16 → FP32 누적
    result_fp32 = torch.matmul(A_bf16.float(), B_bf16.float().T)
    return result_fp32


def int4_dequant_fp16_matmul(A_int4: torch.Tensor, B_int4: torch.Tensor,
                              scale_A: torch.Tensor, scale_B: torch.Tensor) -> torch.Tensor:
    """FP16 dequant 경로 (Marlin 기본)."""
    A_fp16 = (A_int4.float() * scale_A.unsqueeze(1)).to(torch.float16)
    B_fp16 = (B_int4.float() * scale_B.unsqueeze(1)).to(torch.float16)
    result_fp32 = torch.matmul(A_fp16.float(), B_fp16.float().T)
    return result_fp32


def int4_dequant_fp32_matmul(A_int4: torch.Tensor, B_int4: torch.Tensor,
                              scale_A: torch.Tensor, scale_B: torch.Tensor) -> torch.Tensor:
    """FP32 dequant 경로 (이론적 최고 정밀도 기준선)."""
    A_fp32 = A_int4.float() * scale_A.unsqueeze(1)
    B_fp32 = B_int4.float() * scale_B.unsqueeze(1)
    result_fp32 = torch.matmul(A_fp32, B_fp32.T)
    return result_fp32


def analyze_diff(name: str, result: torch.Tensor, reference: torch.Tensor):
    """두 결과의 차이 분석."""
    diff = (result - reference).abs()
    ref_abs = reference.abs()
    rel_diff = diff / (ref_abs + 1e-10)

    print(f"\n  [{name}]")
    print(f"    Absolute diff  - mean: {diff.mean():.10f}, max: {diff.max():.10f}")
    print(f"    Relative diff  - mean: {rel_diff.mean():.8f}, max: {rel_diff.max():.8f}")
    print(f"    Exact matches  - {(diff == 0).sum().item()} / {diff.numel()} "
          f"({(diff == 0).sum().item()/diff.numel()*100:.1f}%)")


print("=" * 70)
print("  INT4×INT4 Native vs Dequant Matmul 정밀도 검증")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# Test 1: 작은 행렬 (직접 확인 가능)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  Test 1: Small Matrix (M=4, K=8, N=4)")
print("─" * 70)

M, K, N = 4, 8, 4

# INT4 값 생성 (signed: -8 ~ 7)
A_int4 = torch.randint(-8, 8, (M, K), dtype=torch.int8)
B_int4 = torch.randint(-8, 8, (N, K), dtype=torch.int8)

# 임의의 scale (per-row)
scale_A = torch.randn(M).abs() * 0.1
scale_B = torch.randn(N).abs() * 0.1

print(f"\n  A_int4 sample: {A_int4[0].tolist()}")
print(f"  B_int4 sample: {B_int4[0].tolist()}")
print(f"  scale_A: {scale_A.tolist()}")
print(f"  scale_B: {scale_B.tolist()}")

# 4가지 경로 계산
native = int4_native_matmul(A_int4, B_int4, scale_A, scale_B)
dq_bf16 = int4_dequant_matmul(A_int4, B_int4, scale_A, scale_B)
dq_fp16 = int4_dequant_fp16_matmul(A_int4, B_int4, scale_A, scale_B)
dq_fp32 = int4_dequant_fp32_matmul(A_int4, B_int4, scale_A, scale_B)

print(f"\n  Native result[0]: {native[0].tolist()}")
print(f"  BF16 dq result[0]: {dq_bf16[0].tolist()}")
print(f"  FP16 dq result[0]: {dq_fp16[0].tolist()}")
print(f"  FP32 dq result[0]: {dq_fp32[0].tolist()}")

print("\n  === vs FP32 Dequant (이론적 최고 정밀도) ===")
analyze_diff("Native INT4×INT4", native, dq_fp32)
analyze_diff("BF16 Dequant", dq_bf16, dq_fp32)
analyze_diff("FP16 Dequant", dq_fp16, dq_fp32)

print("\n  === Native vs Dequant 직접 비교 ===")
analyze_diff("Native vs BF16 Dequant", native, dq_bf16)
analyze_diff("Native vs FP16 Dequant", native, dq_fp16)

# ─────────────────────────────────────────────────────────────────────
# Test 2: 실제 LLM 크기 (hidden_dim=2048)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  Test 2: LLM-scale Matrix (M=16, K=2048, N=2048)")
print("─" * 70)

M, K, N = 16, 2048, 2048

A_int4 = torch.randint(-8, 8, (M, K), dtype=torch.int8)
B_int4 = torch.randint(-8, 8, (N, K), dtype=torch.int8)

# Per-channel scale (실제 양자화에서 사용하는 방식)
scale_A = torch.randn(M).abs() * 0.05
scale_B = torch.randn(N).abs() * 0.05

native = int4_native_matmul(A_int4, B_int4, scale_A, scale_B)
dq_bf16 = int4_dequant_matmul(A_int4, B_int4, scale_A, scale_B)
dq_fp16 = int4_dequant_fp16_matmul(A_int4, B_int4, scale_A, scale_B)
dq_fp32 = int4_dequant_fp32_matmul(A_int4, B_int4, scale_A, scale_B)

print("\n  === vs FP32 Dequant (이론적 최고 정밀도) ===")
analyze_diff("Native INT4×INT4", native, dq_fp32)
analyze_diff("BF16 Dequant", dq_bf16, dq_fp32)
analyze_diff("FP16 Dequant", dq_fp16, dq_fp32)

print("\n  === Native vs Dequant 직접 비교 ===")
analyze_diff("Native vs BF16 Dequant", native, dq_bf16)
analyze_diff("Native vs FP16 Dequant", native, dq_fp16)

# ─────────────────────────────────────────────────────────────────────
# Test 3: Scale이 2의 거듭제곱일 때 (BF16 exact case)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  Test 3: Power-of-2 Scales (M=16, K=2048, N=2048)")
print("  (BF16/FP16에서 정확히 표현 가능한 scale)")
print("─" * 70)

# 2의 거듭제곱 scale → dequant가 exact
scale_A_pow2 = (2.0 ** torch.randint(-4, 0, (M,)).float())
scale_B_pow2 = (2.0 ** torch.randint(-4, 0, (N,)).float())

native_p2 = int4_native_matmul(A_int4, B_int4, scale_A_pow2, scale_B_pow2)
dq_bf16_p2 = int4_dequant_matmul(A_int4, B_int4, scale_A_pow2, scale_B_pow2)
dq_fp16_p2 = int4_dequant_fp16_matmul(A_int4, B_int4, scale_A_pow2, scale_B_pow2)
dq_fp32_p2 = int4_dequant_fp32_matmul(A_int4, B_int4, scale_A_pow2, scale_B_pow2)

print("\n  === vs FP32 Dequant ===")
analyze_diff("Native INT4×INT4", native_p2, dq_fp32_p2)
analyze_diff("BF16 Dequant", dq_bf16_p2, dq_fp32_p2)
analyze_diff("FP16 Dequant", dq_fp16_p2, dq_fp32_p2)

print("\n  === Native vs Dequant 직접 비교 ===")
analyze_diff("Native vs BF16 Dequant", native_p2, dq_bf16_p2)
analyze_diff("Native vs FP16 Dequant", native_p2, dq_fp16_p2)

# ─────────────────────────────────────────────────────────────────────
# Test 4: BF16 rounding 분석
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  Test 4: Dequant 단계의 BF16 rounding 분석")
print("─" * 70)

# 단일 값 dequant 정밀도
print("\n  INT4 값 × scale → BF16 rounding 예시:")
test_scale = torch.tensor(0.1)
for v in range(-8, 8):
    exact = v * 0.1
    bf16_val = torch.tensor(v * 0.1, dtype=torch.bfloat16).item()
    fp16_val = torch.tensor(v * 0.1, dtype=torch.float16).item()
    print(f"    {v:3d} × 0.1 = {exact:8.4f} | BF16: {bf16_val:8.4f} (err={abs(exact-bf16_val):.6f})"
          f" | FP16: {fp16_val:8.4f} (err={abs(exact-fp16_val):.6f})")

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print("""
  1. INT4→BF16 dequant 자체는 거의 무손실 (BF16이 INT4의 16개 값을 표현 가능)
  2. Scale 적용 시 BF16 rounding이 발생 → 이것이 native와의 유일한 차이
  3. Scale이 2의 거듭제곱이면 dequant가 exact → native와 완전 동일
  4. 임의 scale에서도 relative error는 ~1e-3 수준 (BF16 machine epsilon)
  5. FP16 dequant가 BF16보다 정밀 (mantissa 10bit vs 7bit)
  6. 이 오차는 INT4 양자화 자체의 오차 대비 무시할 수준
""")
