"""
INT4 Native Tensor Core Extension

Provides:
  - cutlass_int4_gemm:        INT4x INT4 GEMM -> INT32
  - cutlass_int4_scaled_mm:   INT4x INT4 GEMM with fused EVT scale -> BF16/FP16
  - cutlass_w4a8_scaled_mm:   INT8x INT8 GEMM with fused EVT scale -> BF16/FP16
                              (W4A8: weight pre-unpacked from INT4 to INT8)
  - dynamic_int4_quant:       BF16/FP16 -> INT4 per-row dynamic quantization
  - dynamic_int8_quant:       BF16/FP16 -> INT8 per-row dynamic quantization
  - static_int4_weight_quant: BF16/FP16 -> INT4 per-channel weight quantization
  - unpack_int4_to_int8:      INT4 packed uint8 [N,K/2] -> INT8 [N,K]
  - dequant_int4_to_bf16:     INT4 packed -> BF16 dequantization (for W4A16)
  - dequant_int4_to_fp16:     INT4 packed -> FP16 dequantization (for W4A16)

Per-group quantization (for Enhanced per-group configs):
  - static_int4_weight_quant_grouped:  [N,K] -> [G,N,gs/2] + [G,N] scales
  - dynamic_int4_quant_grouped:        [M,K] -> [G,M,gs/2] + [G,M] scales
  - dynamic_int4_quant_clipped_grouped: Same with outlier clipping (clip_ratio)
  - dynamic_int4_quant_asymmetric_grouped: Asymmetric INT4 [0,15] -> packed + scale + azp_adj
  - dynamic_int4_quant_asymmetric_clipped_grouped: Fused clip + asymmetric INT4 + optional smooth_scale
  - dynamic_int8_quant_grouped:        [M,K] -> [G,M,gs] + [G,M] scales
  - dynamic_int8_quant_asymmetric_grouped: [M,K] -> [G,M,gs] + scales + azp

Per-group GEMM (C++ level loop, CUTLASS per-group, FP32 accumulation):
  - cutlass_int4_scaled_mm_grouped:      Per-group INT4xINT4 -> BF16/FP16
  - cutlass_int4_scaled_mm_azp_grouped:  Per-group INT4xINT4 + AZP correction
  - cutlass_w4a8_scaled_mm_grouped:      Per-group INT8xINT8 -> BF16/FP16 + AZP

V5 multi-group K-loop fused GEMM (reduced __syncthreads via multi-group SMEM loading):
  - cutlass_int4_fused_grouped_gemm_v5:      Symmetric multi-group fused GEMM -> BF16
  - cutlass_int4_fused_grouped_gemm_v5_azp:  Multi-group fused GEMM + AZP correction -> BF16

V7: 128×128 tile, 3-stage pipeline, 256 threads (8 warps), load-after-compute:
  - cutlass_int4_fused_grouped_gemm_v7:      V7 fused GEMM -> BF16
  - cutlass_int4_fused_grouped_gemm_v7_azp:  V7 fused GEMM + AZP correction -> BF16

Dequant-GEMM approach (dequant INT4 to BF16, then cuBLAS BF16 GEMM):
  - cutlass_int4_dequant_gemm_grouped:  Dequant INT4 activations -> BF16 + cuBLAS BF16 GEMM
  - dequant_int4_grouped_to_bf16:       Per-group INT4 -> BF16 with fused scale
  - unpack_int4_grouped_to_int8_contiguous: Per-group INT4 activation -> contiguous INT8
  - unpack_int4_grouped_to_int8_weight:     Per-group INT4 weight -> contiguous INT8

V6 QServe-style progressive INT8 GEMM (INT4 unpacked to INT8, fused per-group scale):
  - progressive_int4_gemm_grouped:  [M,K] int8 x [N,K] int8 -> [M,N] bf16 with per-group scales
"""

from int4_native_tc_ops import (  # type: ignore[import]
    cutlass_int4_gemm,
    cutlass_int4_scaled_mm,
    cutlass_w4a8_scaled_mm,
    cutlass_w4a16_mm_grouped,
    cutlass_int4_scaled_mm_grouped,
    cutlass_int4_scaled_mm_azp_grouped,
    cutlass_w4a8_scaled_mm_grouped,
    cutlass_int4_fused_grouped_gemm,
    cutlass_int4_fused_grouped_gemm_azp,
    cutlass_int4_fused_grouped_gemm_v5,
    cutlass_int4_fused_grouped_gemm_v5_azp,
    cutlass_int4_fused_grouped_gemm_v7,
    cutlass_int4_fused_grouped_gemm_v7_azp,
    cutlass_int4_fused_grouped_gemm_v8,
    cutlass_int4_fused_grouped_gemm_v9,
    dynamic_int4_quant,
    dynamic_int8_quant,
    static_int4_weight_quant,
    unpack_int4_to_int8,
    dequant_int4_to_bf16,
    dequant_int4_to_fp16,
    static_int4_weight_quant_grouped,
    dynamic_int4_quant_grouped,
    dynamic_int4_quant_clipped_grouped,
    dynamic_int4_quant_asymmetric_grouped,
    dynamic_int4_quant_asymmetric_clipped_grouped,
    dynamic_int8_quant_grouped,
    dynamic_int8_quant_asymmetric_grouped,
    cutlass_int4_dequant_gemm_grouped,
    dequant_int4_grouped_to_bf16,
    unpack_int4_grouped_to_int8_contiguous,
    unpack_int4_grouped_to_int8_weight,
    progressive_int4_gemm_grouped,
    dequant_int4_grouped_to_fp8,
    dequant_int4_grouped_to_fp8_with_scales,
)

__all__ = [
    "cutlass_int4_gemm",
    "cutlass_int4_scaled_mm",
    "cutlass_w4a8_scaled_mm",
    "cutlass_w4a16_mm_grouped",
    "cutlass_int4_scaled_mm_grouped",
    "cutlass_int4_scaled_mm_azp_grouped",
    "cutlass_w4a8_scaled_mm_grouped",
    "cutlass_int4_fused_grouped_gemm",
    "cutlass_int4_fused_grouped_gemm_azp",
    "cutlass_int4_fused_grouped_gemm_v5",
    "cutlass_int4_fused_grouped_gemm_v5_azp",
    "cutlass_int4_fused_grouped_gemm_v7",
    "cutlass_int4_fused_grouped_gemm_v7_azp",
    "cutlass_int4_fused_grouped_gemm_v8",
    "cutlass_int4_fused_grouped_gemm_v9",
    "dynamic_int4_quant",
    "dynamic_int8_quant",
    "static_int4_weight_quant",
    "unpack_int4_to_int8",
    "dequant_int4_to_bf16",
    "dequant_int4_to_fp16",
    "static_int4_weight_quant_grouped",
    "dynamic_int4_quant_grouped",
    "dynamic_int4_quant_clipped_grouped",
    "dynamic_int4_quant_asymmetric_grouped",
    "dynamic_int4_quant_asymmetric_clipped_grouped",
    "dynamic_int8_quant_grouped",
    "dynamic_int8_quant_asymmetric_grouped",
    "cutlass_int4_dequant_gemm_grouped",
    "dequant_int4_grouped_to_bf16",
    "unpack_int4_grouped_to_int8_contiguous",
    "unpack_int4_grouped_to_int8_weight",
    "progressive_int4_gemm_grouped",
    "dequant_int4_grouped_to_fp8",
    "dequant_int4_grouped_to_fp8_with_scales",
]
