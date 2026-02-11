/*
 * PyTorch C++ Extension bindings for INT4 Native Tensor Core ops.
 */

#include <torch/extension.h>

// GEMM ops (int4_gemm.cu)
torch::Tensor cutlass_int4_gemm(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    int64_t M, int64_t N, int64_t K);

void cutlass_int4_scaled_mm(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t K);

void cutlass_w4a8_scaled_mm(
    torch::Tensor a,
    torch::Tensor b_int8,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t K);

// Per-group GEMM ops (int4_gemm.cu)
void cutlass_w4a16_mm_grouped(
    torch::Tensor x,
    torch::Tensor w_packed,
    torch::Tensor w_scale,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

void cutlass_int4_scaled_mm_grouped(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

void cutlass_w4a8_scaled_mm_grouped(
    torch::Tensor a_int8,
    torch::Tensor b_int8,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    torch::Tensor azp,
    torch::Tensor w_col_sum,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

// Quantization ops (int4_quant.cu)
std::tuple<torch::Tensor, torch::Tensor> dynamic_int4_quant(torch::Tensor input);
std::tuple<torch::Tensor, torch::Tensor> dynamic_int8_quant(torch::Tensor input);
std::tuple<torch::Tensor, torch::Tensor> static_int4_weight_quant(torch::Tensor weight);
torch::Tensor unpack_int4_to_int8(torch::Tensor packed, int64_t K);
torch::Tensor dequant_int4_to_bf16(torch::Tensor packed, torch::Tensor scale, int64_t K);
torch::Tensor dequant_int4_to_fp16(torch::Tensor packed, torch::Tensor scale, int64_t K);

// Per-group quantization ops (int4_quant.cu)
std::tuple<torch::Tensor, torch::Tensor> static_int4_weight_quant_grouped(
    torch::Tensor weight, int64_t group_size);
std::tuple<torch::Tensor, torch::Tensor> dynamic_int4_quant_grouped(
    torch::Tensor input, int64_t group_size);
std::tuple<torch::Tensor, torch::Tensor> dynamic_int4_quant_clipped_grouped(
    torch::Tensor input, int64_t group_size, double clip_ratio);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    dynamic_int4_quant_asymmetric_grouped(
    torch::Tensor input, int64_t group_size);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    dynamic_int4_quant_asymmetric_clipped_grouped(
    torch::Tensor input, int64_t group_size, double clip_ratio,
    std::optional<torch::Tensor> smooth_scale);
std::tuple<torch::Tensor, torch::Tensor> dynamic_int8_quant_grouped(
    torch::Tensor input, int64_t group_size);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    dynamic_int8_quant_asymmetric_grouped(
    torch::Tensor input, int64_t group_size);

// Per-group GEMM with AZP correction (int4_gemm.cu)
void cutlass_int4_scaled_mm_azp_grouped(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    torch::Tensor azp_adj,
    torch::Tensor w_col_sum,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

// Fused grouped GEMM ops (int4_gemm.cu) - single kernel launch for all groups
void cutlass_int4_fused_grouped_gemm(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

void cutlass_int4_fused_grouped_gemm_azp(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    torch::Tensor azp_adj,
    torch::Tensor w_col_sum,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

// V9: V8's 16-byte loads + 3-stage pipeline (int4_gemm.cu)
void cutlass_int4_fused_grouped_gemm_v9(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

// V8: V4 tile with 16-byte vectorized cp.async (int4_gemm.cu)
void cutlass_int4_fused_grouped_gemm_v8(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

// V7: 128×128 tile, 3-stage pipeline, 256 threads (int4_gemm.cu)
void cutlass_int4_fused_grouped_gemm_v7(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

void cutlass_int4_fused_grouped_gemm_v7_azp(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    torch::Tensor azp_adj,
    torch::Tensor w_col_sum,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

// V5 multi-group K-loop fused grouped GEMM ops (int4_gemm.cu)
void cutlass_int4_fused_grouped_gemm_v5(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

void cutlass_int4_fused_grouped_gemm_v5_azp(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    torch::Tensor azp_adj,
    torch::Tensor w_col_sum,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups);

// Dequant-GEMM: dequant INT4 activations to BF16 + cuBLAS BF16 GEMM (int4_gemm.cu)
void cutlass_int4_dequant_gemm_grouped(
    torch::Tensor a_packed,
    torch::Tensor b_bf16,
    torch::Tensor scale_a,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t K,
    int64_t group_size, int64_t num_groups);

// Per-group INT4 → BF16 dequant with fused scale (int4_quant.cu)
torch::Tensor dequant_int4_grouped_to_bf16(
    torch::Tensor packed, torch::Tensor scale, int64_t group_size);

// Per-group INT4 → contiguous INT8 unpacking (int4_quant.cu)
torch::Tensor unpack_int4_grouped_to_int8_contiguous(
    torch::Tensor packed, int64_t group_size);

// Per-group INT4 weight → contiguous INT8 unpacking (int4_quant.cu)
torch::Tensor unpack_int4_grouped_to_int8_weight(
    torch::Tensor packed, int64_t group_size);

// W4A8-FP8: Per-group INT4 weight → FP8 dequant (int4_quant.cu)
std::tuple<torch::Tensor, torch::Tensor> dequant_int4_grouped_to_fp8(
    torch::Tensor packed, torch::Tensor scale, int64_t group_size);
void dequant_int4_grouped_to_fp8_with_scales(
    torch::Tensor packed, torch::Tensor group_scales,
    torch::Tensor channel_scales, torch::Tensor fp8_out,
    int64_t group_size);

// V6 QServe-style progressive INT8 GEMM (int4_gemm.cu)
void progressive_int4_gemm_grouped(
    torch::Tensor a_int8,
    torch::Tensor w_int8,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t K,
    int64_t group_size, int64_t num_groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Original GEMM ops
    m.def("cutlass_int4_gemm", &cutlass_int4_gemm,
          "CUTLASS INT4x INT4 GEMM -> INT32");
    m.def("cutlass_int4_scaled_mm", &cutlass_int4_scaled_mm,
          "CUTLASS INT4x INT4 scaled GEMM with EVT epilogue -> BF16/FP16");
    m.def("cutlass_w4a8_scaled_mm", &cutlass_w4a8_scaled_mm,
          "CUTLASS INT8x INT8 scaled GEMM with EVT epilogue -> BF16/FP16");

    // Per-group GEMM ops
    m.def("cutlass_w4a16_mm_grouped", &cutlass_w4a16_mm_grouped,
          "Per-group W4A16: INT4 weight dequant -> BF16/FP16 GEMM with FP32 accumulation");
    m.def("cutlass_int4_scaled_mm_grouped", &cutlass_int4_scaled_mm_grouped,
          "Per-group INT4x INT4 GEMM with FP32 accumulation -> BF16/FP16");
    m.def("cutlass_w4a8_scaled_mm_grouped", &cutlass_w4a8_scaled_mm_grouped,
          "Per-group INT8x INT8 GEMM with FP32 accumulation + optional AZP -> BF16/FP16");

    // Original quantization ops
    m.def("dynamic_int4_quant", &dynamic_int4_quant,
          "Dynamic per-row INT4 activation quantization");
    m.def("dynamic_int8_quant", &dynamic_int8_quant,
          "Dynamic per-row INT8 activation quantization");
    m.def("static_int4_weight_quant", &static_int4_weight_quant,
          "Static per-channel INT4 weight quantization");
    m.def("unpack_int4_to_int8", &unpack_int4_to_int8,
          "Unpack INT4 packed uint8 [N, K/2] -> INT8 [N, K]");
    m.def("dequant_int4_to_bf16", &dequant_int4_to_bf16,
          "INT4 packed -> BF16 dequantization");
    m.def("dequant_int4_to_fp16", &dequant_int4_to_fp16,
          "INT4 packed -> FP16 dequantization");

    // Per-group quantization ops
    m.def("static_int4_weight_quant_grouped", &static_int4_weight_quant_grouped,
          "Static per-group INT4 weight quantization [N,K] -> [G,N,gs/2] + [G,N] scales");
    m.def("dynamic_int4_quant_grouped", &dynamic_int4_quant_grouped,
          "Dynamic per-group INT4 activation quantization [M,K] -> [G,M,gs/2] + [G,M] scales");
    m.def("dynamic_int4_quant_clipped_grouped", &dynamic_int4_quant_clipped_grouped,
          "Dynamic per-group INT4 activation quantization with outlier clipping");
    m.def("dynamic_int4_quant_asymmetric_grouped", &dynamic_int4_quant_asymmetric_grouped,
          "Dynamic per-group asymmetric INT4 activation quantization [M,K] -> [G,M,gs/2] + scales + azp_adj");
    m.def("dynamic_int4_quant_asymmetric_clipped_grouped",
          &dynamic_int4_quant_asymmetric_clipped_grouped,
          "Asymmetric INT4 quant with fused clipping and optional smooth_scale",
          py::arg("input"), py::arg("group_size"), py::arg("clip_ratio"),
          py::arg("smooth_scale") = py::none());
    m.def("dynamic_int8_quant_grouped", &dynamic_int8_quant_grouped,
          "Dynamic per-group INT8 activation quantization [M,K] -> [G,M,gs] + [G,M] scales");
    m.def("dynamic_int8_quant_asymmetric_grouped", &dynamic_int8_quant_asymmetric_grouped,
          "Dynamic per-group asymmetric INT8 activation quantization [M,K] -> [G,M,gs] + scales + azp");

    // Per-group GEMM with AZP correction
    m.def("cutlass_int4_scaled_mm_azp_grouped", &cutlass_int4_scaled_mm_azp_grouped,
          "Per-group INT4xINT4 GEMM with AZP correction for asymmetric activation");

    // Fused grouped GEMM ops (single kernel launch for all groups)
    m.def("cutlass_int4_fused_grouped_gemm", &cutlass_int4_fused_grouped_gemm,
          "Fused per-group INT4xINT4 GEMM: single kernel launch -> BF16");
    m.def("cutlass_int4_fused_grouped_gemm_azp", &cutlass_int4_fused_grouped_gemm_azp,
          "Fused per-group INT4xINT4 GEMM with AZP correction: single kernel launch -> BF16");

    // V9: V8's 16-byte loads + 3-stage pipeline
    m.def("cutlass_int4_fused_grouped_gemm_v9", &cutlass_int4_fused_grouped_gemm_v9,
          "V9 fused INT4xINT4 GEMM: 16-byte loads + 3-stage pipeline -> BF16");

    // V8: V4 tile with 16-byte vectorized cp.async
    m.def("cutlass_int4_fused_grouped_gemm_v8", &cutlass_int4_fused_grouped_gemm_v8,
          "V8 fused INT4xINT4 GEMM: V4 tile with 16-byte vectorized loads -> BF16");

    // V7: 128×128 tile, 3-stage pipeline, 256 threads
    m.def("cutlass_int4_fused_grouped_gemm_v7", &cutlass_int4_fused_grouped_gemm_v7,
          "V7 fused INT4xINT4 GEMM: 128x128 tile, 3-stage pipeline, 256 threads -> BF16");
    m.def("cutlass_int4_fused_grouped_gemm_v7_azp", &cutlass_int4_fused_grouped_gemm_v7_azp,
          "V7 fused INT4xINT4 GEMM with AZP: 128x128 tile, 3-stage pipeline -> BF16");

    // V5 multi-group K-loop fused grouped GEMM ops
    m.def("cutlass_int4_fused_grouped_gemm_v5", &cutlass_int4_fused_grouped_gemm_v5,
          "V5 multi-group fused INT4xINT4 GEMM: reduced syncs via multi-group SMEM loading -> BF16");
    m.def("cutlass_int4_fused_grouped_gemm_v5_azp", &cutlass_int4_fused_grouped_gemm_v5_azp,
          "V5 multi-group fused INT4xINT4 GEMM with AZP: reduced syncs -> BF16");

    // Dequant-GEMM approach: dequant INT4 to BF16 + cuBLAS BF16 GEMM
    m.def("cutlass_int4_dequant_gemm_grouped", &cutlass_int4_dequant_gemm_grouped,
          "Dequant INT4 activations to BF16 + single cuBLAS BF16 GEMM (no per-group sync)");

    // Per-group dequant/unpack utilities
    m.def("dequant_int4_grouped_to_bf16", &dequant_int4_grouped_to_bf16,
          "Per-group INT4 → BF16 dequant with fused scale: [G,M,gs/2]+[G,M] → [M,K] bf16");
    m.def("unpack_int4_grouped_to_int8_contiguous", &unpack_int4_grouped_to_int8_contiguous,
          "Per-group INT4 activation → contiguous INT8: [G,M,gs/2] → [M,K] int8");
    m.def("unpack_int4_grouped_to_int8_weight", &unpack_int4_grouped_to_int8_weight,
          "Per-group INT4 weight → contiguous INT8: [G,N,gs/2] → [N,K] int8");

    // V6 QServe-style progressive INT8 GEMM
    m.def("progressive_int4_gemm_grouped", &progressive_int4_gemm_grouped,
          "V6 QServe-style progressive INT8 GEMM: unpack INT4->INT8, fused per-group scale -> BF16");

    // W4A8-FP8: Per-group INT4 weight → FP8 dequant
    m.def("dequant_int4_grouped_to_fp8", &dequant_int4_grouped_to_fp8,
          "Per-group INT4 → FP8 dequant: [G,N,gs/2]+[G,N] → [N,K] fp8_e4m3 + [N] channel_scales");
    m.def("dequant_int4_grouped_to_fp8_with_scales", &dequant_int4_grouped_to_fp8_with_scales,
          "Per-group INT4 → FP8 dequant with precomputed channel_scales (runtime-only pass)");
}
