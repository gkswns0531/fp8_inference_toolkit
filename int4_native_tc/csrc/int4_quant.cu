/*
 * INT4 Quantization/Dequantization CUDA Kernels
 *
 * 1. dynamic_int4_quant: BF16/FP16 -> INT4 per-row symmetric quantization
 *    - For W4A4 activation quantization
 *    - Uses vectorized loads (vec8 for BF16/FP16 = 128-bit) for throughput
 *    - Output: packed uint8 [M, K/2] + scale [M] float
 *
 * 2. static_int4_weight_quant: BF16 -> INT4 per-channel symmetric quantization
 *    - For weight quantization (both W4A4 and W4A16)
 *    - Output: packed uint8 [N, K/2] + scale [N] float
 *
 * 3. dequant_int4_to_bf16: INT4 packed -> BF16 dequantization
 *    - For W4A16 inference (dequant weights -> BF16 GEMM)
 *
 * Packing convention: two int4 values per uint8 byte
 *   byte = (high_nibble << 4) | (low_nibble & 0x0F)
 *   where low_nibble = element at even index, high_nibble = element at odd index
 *   INT4 range: [-8, 7] stored as 2's complement in 4 bits
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

struct CubMaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

static inline __device__ int8_t float_to_int4_rn(float x) {
    // Round to nearest, clamp to [-8, 7]
    float rounded = rintf(x);
    rounded = fminf(fmaxf(rounded, -8.0f), 7.0f);
    return static_cast<int8_t>(rounded);
}

// Pack two int4 values into one uint8:
//   low nibble  = val0 (even index)
//   high nibble = val1 (odd index)
static inline __device__ uint8_t pack_int4x2(int8_t val0, int8_t val1) {
    return static_cast<uint8_t>((val1 & 0x0F) << 4 | (val0 & 0x0F));
}

// Unpack uint8 into two int4 values (sign-extend)
static inline __device__ void unpack_int4x2(uint8_t packed, int8_t& val0, int8_t& val1) {
    // Low nibble (even index)
    val0 = static_cast<int8_t>(packed << 4) >> 4;  // sign-extend
    // High nibble (odd index)
    val1 = static_cast<int8_t>(packed & 0xF0) >> 4; // sign-extend
}

// ─────────────────────────────────────────────────────────────────────────────
// Vectorization support
//
// Aligned vector type for coalesced memory access.
// For BF16/FP16: vec8 = 8 * 2 bytes = 128-bit load (single LDG.128)
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t, int N>
struct __align__(sizeof(scalar_t) * N) vec_n_t {
    scalar_t val[N];
};

// VEC_ELEMS: number of elements per vector load.
// For BF16/FP16 (2 bytes): 128 bits / 16 bits = 8 elements per LDG.128
// For FP32 (4 bytes): 128 bits / 32 bits = 4 elements per LDG.128
template <typename scalar_t>
constexpr int vec_elems() {
    return 16 / sizeof(scalar_t);  // 128-bit / element_size
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Dynamic per-row INT4 activation quantization (vectorized)
//
// Input:  [M, K] BF16/FP16
// Output: [M, K/2] uint8 (packed INT4), [M] float (scales)
//
// Algorithm:
//   1. Per-row absmax via vectorized loads + CUB BlockReduce
//   2. scale = max(absmax / 7.0, min_scaling_factor)
//   3. q = clamp(round(x / scale), -8, 7)
//   4. Pack pairs into uint8 with vectorized stores
//
// Launch: grid=(M,), block=(256,)
// ─────────────────────────────────────────────────────────────────────────────

// Minimum scaling factor to prevent underflow (same concept as FP8 quant)
static constexpr float kMinScalingFactor = 1.0f / (7.0f * 512.0f);

template <typename scalar_t>
__global__ void dynamic_int4_quant_kernel(
    const scalar_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scale_out,
    const int K)
{
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int64_t row = blockIdx.x;

    const scalar_t* row_in = input + row * K;
    uint8_t* row_out = output + row * (K / 2);

    // Phase 1: find per-row absmax using vectorized loads
    constexpr int VEC = vec_elems<scalar_t>();
    using vec_t = vec_n_t<scalar_t, VEC>;

    float thread_max = 0.0f;
    const int num_vec = K / VEC;
    const auto* v_in = reinterpret_cast<const vec_t*>(row_in);

    for (int i = tid; i < num_vec; i += stride) {
        vec_t v = v_in[i];
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            thread_max = fmaxf(thread_max, fabsf(static_cast<float>(v.val[j])));
        }
    }

    // Handle tail elements (when K is not divisible by VEC)
    for (int i = num_vec * VEC + tid; i < K; i += stride) {
        thread_max = fmaxf(thread_max, fabsf(static_cast<float>(row_in[i])));
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

    __shared__ float absmax;
    if (tid == 0) {
        absmax = block_max;
        // Apply min_scaling_factor to prevent underflow (FP8 pattern)
        scale_out[row] = fmaxf(block_max / 7.0f, kMinScalingFactor);
    }
    __syncthreads();

    float inv_s = (absmax == 0.0f) ? 0.0f : 7.0f / absmax;

    // Phase 2: quantize and pack pairs using vectorized loads
    // Load VEC elements at a time, quantize, and pack into VEC/2 bytes
    // Then store packed bytes efficiently
    const int half_vec = VEC / 2;
    const int num_vec_packs = K / VEC;

    for (int i = tid; i < num_vec_packs; i += stride) {
        vec_t v = v_in[i];
        uint8_t packed[VEC / 2];

        #pragma unroll
        for (int j = 0; j < half_vec; j++) {
            float v0 = static_cast<float>(v.val[j * 2]) * inv_s;
            float v1 = static_cast<float>(v.val[j * 2 + 1]) * inv_s;
            int8_t q0 = float_to_int4_rn(v0);
            int8_t q1 = float_to_int4_rn(v1);
            packed[j] = pack_int4x2(q0, q1);
        }

        // Store half_vec bytes at once
        // For BF16/FP16: VEC=8, half_vec=4 -> store 4 bytes as uint32
        if constexpr (half_vec == 4) {
            *reinterpret_cast<uint32_t*>(row_out + i * half_vec) =
                *reinterpret_cast<uint32_t*>(packed);
        } else if constexpr (half_vec == 2) {
            *reinterpret_cast<uint16_t*>(row_out + i * half_vec) =
                *reinterpret_cast<uint16_t*>(packed);
        } else {
            #pragma unroll
            for (int j = 0; j < half_vec; j++) {
                row_out[i * half_vec + j] = packed[j];
            }
        }
    }

    // Handle tail elements (when K is not divisible by VEC)
    int tail_start = num_vec_packs * VEC;
    int tail_half_start = tail_start / 2;
    int tail_pairs = (K - tail_start) / 2;
    for (int i = tid; i < tail_pairs; i += stride) {
        int idx0 = tail_start + i * 2;
        int idx1 = tail_start + i * 2 + 1;

        float v0 = static_cast<float>(row_in[idx0]) * inv_s;
        float v1 = static_cast<float>(row_in[idx1]) * inv_s;

        int8_t q0 = float_to_int4_rn(v0);
        int8_t q1 = float_to_int4_rn(v1);

        row_out[tail_half_start + i] = pack_int4x2(q0, q1);
    }
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_int4_quant(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % 2 == 0, "K must be even for INT4 packing");

    auto packed = torch::empty({M, K / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    auto scale = torch::empty({M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    int block = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "dynamic_int4_quant", [&] {
            dynamic_int4_quant_kernel<scalar_t><<<M, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                packed.data_ptr<uint8_t>(),
                scale.data_ptr<float>(),
                K);
        });

    return {packed, scale};
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Static per-channel INT4 weight quantization
//
// Input:  [N, K] BF16/FP16/FP32 (weight matrix, output_features x input_features)
// Output: [N, K/2] uint8 (packed INT4), [N] float (per-channel scales)
//
// Launch: grid=(N,), block=(256,)
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void static_int4_weight_quant_kernel(
    const scalar_t* __restrict__ weight,
    uint8_t* __restrict__ output,
    float* __restrict__ scale_out,
    const int K)
{
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int64_t row = blockIdx.x;  // channel index

    const scalar_t* row_in = weight + row * K;
    uint8_t* row_out = output + row * (K / 2);

    // Phase 1: per-channel absmax
    float thread_max = 0.0f;
    for (int i = tid; i < K; i += stride) {
        float v = fabsf(static_cast<float>(row_in[i]));
        thread_max = fmaxf(thread_max, v);
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

    __shared__ float absmax;
    if (tid == 0) {
        absmax = block_max;
        scale_out[row] = (block_max == 0.0f) ? 1.0f : block_max / 7.0f;
    }
    __syncthreads();

    float inv_s = (absmax == 0.0f) ? 0.0f : 7.0f / absmax;

    // Phase 2: quantize and pack
    int half_K = K / 2;
    for (int i = tid; i < half_K; i += stride) {
        int idx0 = i * 2;
        int idx1 = i * 2 + 1;

        float v0 = static_cast<float>(row_in[idx0]) * inv_s;
        float v1 = static_cast<float>(row_in[idx1]) * inv_s;

        int8_t q0 = float_to_int4_rn(v0);
        int8_t q1 = float_to_int4_rn(v1);

        row_out[i] = pack_int4x2(q0, q1);
    }
}

std::tuple<torch::Tensor, torch::Tensor> static_int4_weight_quant(torch::Tensor weight) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N, K]");

    int N = weight.size(0);
    int K = weight.size(1);
    TORCH_CHECK(K % 2 == 0, "K must be even for INT4 packing");

    auto packed = torch::empty({N, K / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(weight.device()));
    auto scale = torch::empty({N},
        torch::TensorOptions().dtype(torch::kFloat32).device(weight.device()));

    auto stream = at::cuda::getCurrentCUDAStream(weight.get_device());
    int block = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        weight.scalar_type(), "static_int4_weight_quant", [&] {
            static_int4_weight_quant_kernel<scalar_t><<<N, block, 0, stream>>>(
                weight.data_ptr<scalar_t>(),
                packed.data_ptr<uint8_t>(),
                scale.data_ptr<float>(),
                K);
        });

    return {packed, scale};
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: INT4 -> BF16 dequantization (for W4A16 inference)
//
// Input:  [N, K/2] uint8 (packed INT4 weights), [N] float (per-channel scales)
// Output: [N, K] BF16 (dequantized weights)
//
// dequant[n, k] = int4_val[n, k] * scale[n]
//
// Launch: grid=(N,), block=(256,)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void dequant_int4_to_bf16_kernel(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    const int K)
{
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int64_t row = blockIdx.x;

    const uint8_t* row_in = packed + row * (K / 2);
    __nv_bfloat16* row_out = output + row * K;
    float s = scale[row];

    int half_K = K / 2;
    for (int i = tid; i < half_K; i += stride) {
        int8_t v0, v1;
        unpack_int4x2(row_in[i], v0, v1);

        row_out[i * 2]     = __float2bfloat16(static_cast<float>(v0) * s);
        row_out[i * 2 + 1] = __float2bfloat16(static_cast<float>(v1) * s);
    }
}

torch::Tensor dequant_int4_to_bf16(
    torch::Tensor packed,
    torch::Tensor scale,
    int64_t K)
{
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA tensor");

    int N = packed.size(0);
    TORCH_CHECK(packed.size(1) == K / 2, "packed dim 1 must be K/2");

    auto output = torch::empty({N, K},
        torch::TensorOptions().dtype(torch::kBFloat16).device(packed.device()));

    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    int block = 256;

    dequant_int4_to_bf16_kernel<<<N, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        K);

    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: INT4 -> FP16 dequantization (for W4A16 with FP16 models)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void dequant_int4_to_fp16_kernel(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scale,
    __half* __restrict__ output,
    const int K)
{
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int64_t row = blockIdx.x;

    const uint8_t* row_in = packed + row * (K / 2);
    __half* row_out = output + row * K;
    float s = scale[row];

    int half_K = K / 2;
    for (int i = tid; i < half_K; i += stride) {
        int8_t v0, v1;
        unpack_int4x2(row_in[i], v0, v1);

        row_out[i * 2]     = __float2half(static_cast<float>(v0) * s);
        row_out[i * 2 + 1] = __float2half(static_cast<float>(v1) * s);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: Dynamic per-row INT8 activation quantization (vectorized)
//
// Input:  [M, K] BF16/FP16
// Output: [M, K] int8 (NOT packed, 1 byte per value), [M] float (scales)
//
// Algorithm:
//   1. Per-row absmax via vectorized loads + CUB BlockReduce
//   2. scale = max(absmax / 127.0, min_scaling_factor_int8)
//   3. q = clamp(round(x / scale), -128, 127)
//   4. Store as int8 (no packing needed)
//
// Launch: grid=(M,), block=(256,)
// ─────────────────────────────────────────────────────────────────────────────

static constexpr float kMinScalingFactorInt8 = 1.0f / (127.0f * 512.0f);

static inline __device__ int8_t float_to_int8_rn(float x) {
    float rounded = rintf(x);
    rounded = fminf(fmaxf(rounded, -128.0f), 127.0f);
    return static_cast<int8_t>(rounded);
}

template <typename scalar_t>
__global__ void dynamic_int8_quant_kernel(
    const scalar_t* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale_out,
    const int K)
{
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int64_t row = blockIdx.x;

    const scalar_t* row_in = input + row * K;
    int8_t* row_out = output + row * K;

    // Phase 1: find per-row absmax using vectorized loads
    constexpr int VEC = vec_elems<scalar_t>();
    using vec_t = vec_n_t<scalar_t, VEC>;

    float thread_max = 0.0f;
    const int num_vec = K / VEC;
    const auto* v_in = reinterpret_cast<const vec_t*>(row_in);

    for (int i = tid; i < num_vec; i += stride) {
        vec_t v = v_in[i];
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            thread_max = fmaxf(thread_max, fabsf(static_cast<float>(v.val[j])));
        }
    }

    // Handle tail elements
    for (int i = num_vec * VEC + tid; i < K; i += stride) {
        thread_max = fmaxf(thread_max, fabsf(static_cast<float>(row_in[i])));
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

    __shared__ float absmax;
    if (tid == 0) {
        absmax = block_max;
        scale_out[row] = fmaxf(block_max / 127.0f, kMinScalingFactorInt8);
    }
    __syncthreads();

    float inv_s = (absmax == 0.0f) ? 0.0f : 127.0f / absmax;

    // Phase 2: quantize and store int8 using vectorized loads
    for (int i = tid; i < num_vec; i += stride) {
        vec_t v = v_in[i];
        int8_t quantized[VEC];

        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            quantized[j] = float_to_int8_rn(static_cast<float>(v.val[j]) * inv_s);
        }

        // Store VEC int8 values at once (VEC=8 for BF16 → 8 bytes = 64-bit store)
        if constexpr (VEC == 8) {
            *reinterpret_cast<int64_t*>(row_out + i * VEC) =
                *reinterpret_cast<int64_t*>(quantized);
        } else if constexpr (VEC == 4) {
            *reinterpret_cast<int32_t*>(row_out + i * VEC) =
                *reinterpret_cast<int32_t*>(quantized);
        } else {
            #pragma unroll
            for (int j = 0; j < VEC; j++) {
                row_out[i * VEC + j] = quantized[j];
            }
        }
    }

    // Handle tail elements
    for (int i = num_vec * VEC + tid; i < K; i += stride) {
        float v = static_cast<float>(row_in[i]) * inv_s;
        row_out[i] = float_to_int8_rn(v);
    }
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_int8_quant(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);

    auto quantized = torch::empty({M, K},
        torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
    auto scale = torch::empty({M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    int block = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "dynamic_int8_quant", [&] {
            dynamic_int8_quant_kernel<scalar_t><<<M, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                quantized.data_ptr<int8_t>(),
                scale.data_ptr<float>(),
                K);
        });

    return {quantized, scale};
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 6: INT4 → INT8 unpacking (for W4A8 weight preparation)
//
// Input:  [N, K/2] uint8 (packed INT4 weights)
// Output: [N, K] int8 (unpacked, sign-extended)
//
// Each uint8 contains 2 INT4 values:
//   low nibble  = even index element
//   high nibble = odd index element
// These are sign-extended to int8 range [-8, 7].
//
// Launch: grid=(N,), block=(256,)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void unpack_int4_to_int8_kernel(
    const uint8_t* __restrict__ packed,
    int8_t* __restrict__ output,
    const int K)
{
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int64_t row = blockIdx.x;

    const uint8_t* row_in = packed + row * (K / 2);
    int8_t* row_out = output + row * K;

    int half_K = K / 2;

    // Process 4 packed bytes at a time (= 8 INT4 values) for coalesced access
    int vec4_count = half_K / 4;
    for (int i = tid; i < vec4_count; i += stride) {
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(row_in + i * 4);
        int8_t unpacked[8];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t byte = (packed4 >> (j * 8)) & 0xFF;
            int8_t v0, v1;
            unpack_int4x2(byte, v0, v1);
            unpacked[j * 2] = v0;
            unpacked[j * 2 + 1] = v1;
        }
        *reinterpret_cast<int64_t*>(row_out + i * 8) =
            *reinterpret_cast<int64_t*>(unpacked);
    }

    // Handle tail bytes
    for (int i = vec4_count * 4 + tid; i < half_K; i += stride) {
        int8_t v0, v1;
        unpack_int4x2(row_in[i], v0, v1);
        row_out[i * 2] = v0;
        row_out[i * 2 + 1] = v1;
    }
}

torch::Tensor unpack_int4_to_int8(
    torch::Tensor packed,
    int64_t K)
{
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 2, "packed must be 2D [N, K/2]");

    int N = packed.size(0);
    TORCH_CHECK(packed.size(1) == K / 2, "packed dim 1 must be K/2");

    auto output = torch::empty({N, K},
        torch::TensorOptions().dtype(torch::kInt8).device(packed.device()));

    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    int block = 256;

    unpack_int4_to_int8_kernel<<<N, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        K);

    return output;
}

torch::Tensor dequant_int4_to_fp16(
    torch::Tensor packed,
    torch::Tensor scale,
    int64_t K)
{
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");

    int N = packed.size(0);
    TORCH_CHECK(packed.size(1) == K / 2, "packed dim 1 must be K/2");

    auto output = torch::empty({N, K},
        torch::TensorOptions().dtype(torch::kFloat16).device(packed.device()));

    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    int block = 256;

    dequant_int4_to_fp16_kernel<<<N, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(output.data_ptr()),
        K);

    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-group quantization kernels
//
// All per-group kernels process the K dimension in groups of `group_size`.
// Grid: (rows, num_groups) where rows = N for weights, M for activations.
// Each thread block handles one (row, group) pair → group_size elements.
//
// Output layout: [num_groups, rows, ...] so that group g's data is contiguous,
// enabling direct pointer arithmetic for per-group CUTLASS GEMM calls.
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7: Static per-group INT4 weight quantization
//
// Input:  [N, K] BF16/FP16/FP32 weight, group_size
// Output: [num_groups, N, gs/2] uint8 packed, [num_groups, N] float scales
//
// Grid: (N, num_groups), Block: (256)
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void static_int4_weight_quant_grouped_kernel(
    const scalar_t* __restrict__ weight,
    uint8_t* __restrict__ output,
    float* __restrict__ scale_out,
    const int K, const int group_size)
{
    const int n = blockIdx.x;
    const int g = blockIdx.y;
    const int N_total = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs = group_size;

    const scalar_t* slice = weight + n * K + g * gs;
    uint8_t* out_slice = output + (g * N_total + n) * (gs / 2);
    float* scale_ptr = scale_out + g * N_total + n;

    // Phase 1: per-group absmax
    float thread_max = 0.0f;
    for (int i = tid; i < gs; i += stride) {
        thread_max = fmaxf(thread_max, fabsf(static_cast<float>(slice[i])));
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

    __shared__ float absmax;
    if (tid == 0) {
        absmax = block_max;
        *scale_ptr = (block_max == 0.0f) ? 1.0f : block_max / 7.0f;
    }
    __syncthreads();

    float inv_s = (absmax == 0.0f) ? 0.0f : 7.0f / absmax;

    // Phase 2: quantize and pack
    int half_gs = gs / 2;
    for (int i = tid; i < half_gs; i += stride) {
        float v0 = static_cast<float>(slice[i * 2]) * inv_s;
        float v1 = static_cast<float>(slice[i * 2 + 1]) * inv_s;
        int8_t q0 = float_to_int4_rn(v0);
        int8_t q1 = float_to_int4_rn(v1);
        out_slice[i] = pack_int4x2(q0, q1);
    }
}

std::tuple<torch::Tensor, torch::Tensor> static_int4_weight_quant_grouped(
    torch::Tensor weight, int64_t group_size)
{
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N, K]");

    int N = weight.size(0);
    int K = weight.size(1);
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    int num_groups = K / group_size;
    int gs = static_cast<int>(group_size);

    auto packed = torch::empty({num_groups, N, gs / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(weight.device()));
    auto scale = torch::empty({num_groups, N},
        torch::TensorOptions().dtype(torch::kFloat32).device(weight.device()));

    auto stream = at::cuda::getCurrentCUDAStream(weight.get_device());
    dim3 grid(N, num_groups);
    int block = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        weight.scalar_type(), "static_int4_weight_quant_grouped", [&] {
            static_int4_weight_quant_grouped_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                weight.data_ptr<scalar_t>(),
                packed.data_ptr<uint8_t>(),
                scale.data_ptr<float>(),
                K, gs);
        });

    return {packed, scale};
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 8: Dynamic per-group INT4 activation quantization
//
// Input:  [M, K] BF16/FP16, group_size
// Output: [num_groups, M, gs/2] uint8 packed, [num_groups, M] float scales
//
// Grid: (M, num_groups), Block: (256)
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void dynamic_int4_quant_grouped_kernel(
    const scalar_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scale_out,
    const int K, const int group_size)
{
    const int m = blockIdx.x;
    const int g = blockIdx.y;
    const int M_total = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs = group_size;

    const scalar_t* slice = input + m * K + g * gs;
    uint8_t* out_slice = output + (g * M_total + m) * (gs / 2);
    float* scale_ptr = scale_out + g * M_total + m;

    // Phase 1: per-group absmax
    float thread_max = 0.0f;
    for (int i = tid; i < gs; i += stride) {
        thread_max = fmaxf(thread_max, fabsf(static_cast<float>(slice[i])));
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

    __shared__ float absmax;
    if (tid == 0) {
        absmax = block_max;
        *scale_ptr = fmaxf(block_max / 7.0f, kMinScalingFactor);
    }
    __syncthreads();

    float inv_s = (absmax == 0.0f) ? 0.0f : 7.0f / absmax;

    // Phase 2: quantize and pack
    int half_gs = gs / 2;
    for (int i = tid; i < half_gs; i += stride) {
        float v0 = static_cast<float>(slice[i * 2]) * inv_s;
        float v1 = static_cast<float>(slice[i * 2 + 1]) * inv_s;
        int8_t q0 = float_to_int4_rn(v0);
        int8_t q1 = float_to_int4_rn(v1);
        out_slice[i] = pack_int4x2(q0, q1);
    }
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_int4_quant_grouped(
    torch::Tensor input, int64_t group_size)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    int num_groups = K / group_size;
    int gs = static_cast<int>(group_size);

    auto packed = torch::empty({num_groups, M, gs / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    auto scale = torch::empty({num_groups, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    dim3 grid(M, num_groups);
    int block = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "dynamic_int4_quant_grouped", [&] {
            dynamic_int4_quant_grouped_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                packed.data_ptr<uint8_t>(),
                scale.data_ptr<float>(),
                K, gs);
        });

    return {packed, scale};
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 8b: Dynamic per-group INT4 activation quantization with outlier clipping
//
// Same as Kernel 8 but applies clip_ratio to absmax before computing scale.
// Values beyond clip_ratio * absmax get saturated to ±7/±8, but majority of
// values get better quantization resolution. clip_ratio in (0, 1].
//
// Input:  [M, K] BF16/FP16, group_size, clip_ratio
// Output: [num_groups, M, gs/2] uint8 packed, [num_groups, M] float scales
//
// Grid: (M, num_groups), Block: (256)
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void dynamic_int4_quant_clipped_grouped_kernel(
    const scalar_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scale_out,
    const int K, const int group_size, const float clip_ratio)
{
    const int m = blockIdx.x;
    const int g = blockIdx.y;
    const int M_total = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs = group_size;

    const scalar_t* slice = input + m * K + g * gs;
    uint8_t* out_slice = output + (g * M_total + m) * (gs / 2);
    float* scale_ptr = scale_out + g * M_total + m;

    // Phase 1: per-group absmax
    float thread_max = 0.0f;
    for (int i = tid; i < gs; i += stride) {
        thread_max = fmaxf(thread_max, fabsf(static_cast<float>(slice[i])));
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

    // Apply clip_ratio: use clipped_max instead of raw absmax
    __shared__ float clipped_max;
    if (tid == 0) {
        float cm = block_max * clip_ratio;
        clipped_max = cm;
        *scale_ptr = fmaxf(cm / 7.0f, kMinScalingFactor);
    }
    __syncthreads();

    float inv_s = (clipped_max == 0.0f) ? 0.0f : 7.0f / clipped_max;

    // Phase 2: quantize and pack (values beyond clip range get clamped by float_to_int4_rn)
    int half_gs = gs / 2;
    for (int i = tid; i < half_gs; i += stride) {
        float v0 = static_cast<float>(slice[i * 2]) * inv_s;
        float v1 = static_cast<float>(slice[i * 2 + 1]) * inv_s;
        int8_t q0 = float_to_int4_rn(v0);
        int8_t q1 = float_to_int4_rn(v1);
        out_slice[i] = pack_int4x2(q0, q1);
    }
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_int4_quant_clipped_grouped(
    torch::Tensor input, int64_t group_size, double clip_ratio)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
    TORCH_CHECK(clip_ratio > 0.0 && clip_ratio <= 1.0,
                "clip_ratio must be in (0, 1]");

    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    int num_groups = K / group_size;
    int gs = static_cast<int>(group_size);

    auto packed = torch::empty({num_groups, M, gs / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    auto scale = torch::empty({num_groups, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    dim3 grid(M, num_groups);
    int block = 256;
    float cr = static_cast<float>(clip_ratio);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "dynamic_int4_quant_clipped_grouped", [&] {
            dynamic_int4_quant_clipped_grouped_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                packed.data_ptr<uint8_t>(),
                scale.data_ptr<float>(),
                K, gs, cr);
        });

    return {packed, scale};
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 9: Dynamic per-group INT8 activation quantization (symmetric)
//
// Input:  [M, K] BF16/FP16, group_size
// Output: [num_groups, M, gs] int8, [num_groups, M] float scales
//
// Grid: (M, num_groups), Block: (256)
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void dynamic_int8_quant_grouped_kernel(
    const scalar_t* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale_out,
    const int K, const int group_size)
{
    const int m = blockIdx.x;
    const int g = blockIdx.y;
    const int M_total = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs = group_size;

    const scalar_t* slice = input + m * K + g * gs;
    int8_t* out_slice = output + (g * M_total + m) * gs;
    float* scale_ptr = scale_out + g * M_total + m;

    // Phase 1: per-group absmax
    float thread_max = 0.0f;
    for (int i = tid; i < gs; i += stride) {
        thread_max = fmaxf(thread_max, fabsf(static_cast<float>(slice[i])));
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

    __shared__ float absmax;
    if (tid == 0) {
        absmax = block_max;
        *scale_ptr = fmaxf(block_max / 127.0f, kMinScalingFactorInt8);
    }
    __syncthreads();

    float inv_s = (absmax == 0.0f) ? 0.0f : 127.0f / absmax;

    // Phase 2: quantize
    for (int i = tid; i < gs; i += stride) {
        float v = static_cast<float>(slice[i]) * inv_s;
        out_slice[i] = float_to_int8_rn(v);
    }
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_int8_quant_grouped(
    torch::Tensor input, int64_t group_size)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");

    int num_groups = K / group_size;
    int gs = static_cast<int>(group_size);

    auto quantized = torch::empty({num_groups, M, gs},
        torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
    auto scale = torch::empty({num_groups, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    dim3 grid(M, num_groups);
    int block = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "dynamic_int8_quant_grouped", [&] {
            dynamic_int8_quant_grouped_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                quantized.data_ptr<int8_t>(),
                scale.data_ptr<float>(),
                K, gs);
        });

    return {quantized, scale};
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 10: Dynamic per-group asymmetric INT8 activation quantization
//
// Uses full [0,255] range for positively-biased distributions (post-softmax,
// post-SiLU). Output is signed int8 (shifted by -128) for compatibility with
// INT8 MMA instructions. The effective_azp is the zero-point in signed space.
//
// Input:  [M, K] BF16/FP16, group_size
// Output: [num_groups, M, gs] int8 (signed, shifted from uint8 by -128)
//         [num_groups, M] float scales
//         [num_groups, M] float effective_azp (azp - 128, in signed int8 space)
//
// Grid: (M, num_groups), Block: (256)
// ─────────────────────────────────────────────────────────────────────────────

struct CubMinOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fminf(a, b);
    }
};

template <typename scalar_t>
__global__ void dynamic_int8_quant_asymmetric_grouped_kernel(
    const scalar_t* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale_out,
    float* __restrict__ azp_out,
    const int K, const int group_size)
{
    const int m = blockIdx.x;
    const int g = blockIdx.y;
    const int M_total = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs = group_size;

    const scalar_t* slice = input + m * K + g * gs;
    int8_t* out_slice = output + (g * M_total + m) * gs;
    float* scale_ptr = scale_out + g * M_total + m;
    float* azp_ptr = azp_out + g * M_total + m;

    // Phase 1: find min and max
    float thread_min = INFINITY;
    float thread_max = -INFINITY;
    for (int i = tid; i < gs; i += stride) {
        float v = static_cast<float>(slice[i]);
        thread_min = fminf(thread_min, v);
        thread_max = fmaxf(thread_max, v);
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;

    float block_min = BlockReduce(tmp).Reduce(thread_min, CubMinOp{}, blockDim.x);
    __shared__ float s_min;
    if (tid == 0) s_min = block_min;
    __syncthreads();

    float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

    __shared__ float s_scale, s_azp_unsigned;
    if (tid == 0) {
        float range = block_max - s_min;
        float scale = fmaxf(range / 255.0f, 1e-10f);
        float azp_unsigned = rintf(-s_min / scale);  // unsigned zero point [0, 255]
        azp_unsigned = fminf(fmaxf(azp_unsigned, 0.0f), 255.0f);

        s_scale = scale;
        s_azp_unsigned = azp_unsigned;
        *scale_ptr = scale;
        *azp_ptr = azp_unsigned - 128.0f;  // effective_azp in signed space
    }
    __syncthreads();

    float inv_scale = 1.0f / s_scale;
    float azp_u = s_azp_unsigned;

    // Phase 2: quantize to uint8 range [0,255], then shift to int8 [-128,127]
    for (int i = tid; i < gs; i += stride) {
        float v = static_cast<float>(slice[i]);
        float q = rintf(v * inv_scale + azp_u);
        q = fminf(fmaxf(q, 0.0f), 255.0f);
        // Shift from unsigned [0,255] to signed [-128,127] for INT8 MMA
        out_slice[i] = static_cast<int8_t>(static_cast<int>(q) - 128);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dynamic_int8_quant_asymmetric_grouped(
    torch::Tensor input, int64_t group_size)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");

    int num_groups = K / group_size;
    int gs = static_cast<int>(group_size);

    auto quantized = torch::empty({num_groups, M, gs},
        torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
    auto scale = torch::empty({num_groups, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    auto azp = torch::empty({num_groups, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    dim3 grid(M, num_groups);
    int block = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "dynamic_int8_quant_asymmetric_grouped", [&] {
            dynamic_int8_quant_asymmetric_grouped_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                quantized.data_ptr<int8_t>(),
                scale.data_ptr<float>(),
                azp.data_ptr<float>(),
                K, gs);
        });

    return {quantized, scale, azp};
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 11: Dynamic per-group asymmetric INT4 activation quantization
//
// Maps activations to unsigned [0, 15] range with zero point, then shifts
// to signed [-8, 7] for compatibility with INT4 MMA instructions.
// Returns azp_adj = (8 - azp) for post-GEMM correction.
//
// Correction formula after CUTLASS INT4 GEMM per group:
//   Y_correct = Y_gemm + scale_a * scale_b * azp_adj * w_col_sum
// where w_col_sum[n] = sum of signed INT4 weight values for output neuron n.
//
// Input:  [M, K] BF16/FP16, group_size
// Output: [num_groups, M, gs/2] uint8 packed (signed INT4, shifted)
//         [num_groups, M] float scales
//         [num_groups, M] float azp_adj  (= 8 - zero_point)
//
// Grid: (M, num_groups), Block: (256)
// ─────────────────────────────────────────────────────────────────────────────

static inline __device__ uint8_t pack_uint4x2(uint8_t val0, uint8_t val1) {
    // Pack two unsigned 4-bit values [0,15] shifted to signed [-8,7]
    int8_t s0 = static_cast<int8_t>(val0) - 8;
    int8_t s1 = static_cast<int8_t>(val1) - 8;
    return static_cast<uint8_t>((s1 & 0x0F) << 4 | (s0 & 0x0F));
}

template <typename scalar_t>
__global__ void dynamic_int4_quant_asymmetric_grouped_kernel(
    const scalar_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scale_out,
    float* __restrict__ azp_adj_out,
    const int K, const int group_size)
{
    const int m = blockIdx.x;
    const int g = blockIdx.y;
    const int M_total = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs = group_size;

    const scalar_t* slice = input + m * K + g * gs;
    uint8_t* out_slice = output + (g * M_total + m) * (gs / 2);
    float* scale_ptr = scale_out + g * M_total + m;
    float* azp_adj_ptr = azp_adj_out + g * M_total + m;

    // Phase 1: find min and max per group
    float thread_min = INFINITY;
    float thread_max = -INFINITY;
    for (int i = tid; i < gs; i += stride) {
        float v = static_cast<float>(slice[i]);
        thread_min = fminf(thread_min, v);
        thread_max = fmaxf(thread_max, v);
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;

    float block_min = BlockReduce(tmp).Reduce(thread_min, CubMinOp{}, blockDim.x);
    __shared__ float s_min;
    if (tid == 0) s_min = block_min;
    __syncthreads();

    float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

    __shared__ float s_scale, s_azp;
    if (tid == 0) {
        float range = block_max - s_min;
        float scale = fmaxf(range / 15.0f, kMinScalingFactor);
        float azp = rintf(-s_min / scale);  // unsigned zero point [0, 15]
        azp = fminf(fmaxf(azp, 0.0f), 15.0f);

        s_scale = scale;
        s_azp = azp;
        *scale_ptr = scale;
        *azp_adj_ptr = 8.0f - azp;  // correction factor for GEMM
    }
    __syncthreads();

    float inv_scale = 1.0f / s_scale;
    float azp_val = s_azp;

    // Phase 2: quantize to unsigned [0,15], shift to signed [-8,7], pack
    int half_gs = gs / 2;
    for (int i = tid; i < half_gs; i += stride) {
        float v0 = static_cast<float>(slice[i * 2]);
        float v1 = static_cast<float>(slice[i * 2 + 1]);

        float q0 = rintf(v0 * inv_scale + azp_val);
        float q1 = rintf(v1 * inv_scale + azp_val);
        q0 = fminf(fmaxf(q0, 0.0f), 15.0f);
        q1 = fminf(fmaxf(q1, 0.0f), 15.0f);

        // Pack as unsigned then shift to signed for INT4 MMA
        out_slice[i] = pack_uint4x2(
            static_cast<uint8_t>(q0), static_cast<uint8_t>(q1));
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dynamic_int4_quant_asymmetric_grouped(
    torch::Tensor input, int64_t group_size)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    int num_groups = K / group_size;
    int gs = static_cast<int>(group_size);

    auto packed = torch::empty({num_groups, M, gs / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    auto scale = torch::empty({num_groups, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    auto azp_adj = torch::empty({num_groups, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    dim3 grid(M, num_groups);
    int block = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "dynamic_int4_quant_asymmetric_grouped", [&] {
            dynamic_int4_quant_asymmetric_grouped_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                packed.data_ptr<uint8_t>(),
                scale.data_ptr<float>(),
                azp_adj.data_ptr<float>(),
                K, gs);
        });

    return {packed, scale, azp_adj};
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 11b: Dynamic per-group asymmetric INT4 quant with fused clipping
//             and optional smooth_scale division
//
// Fuses three operations into a single CUDA kernel:
//   1. (Optional) smooth_scale division: x /= smooth_scale
//   2. Outlier clipping: clamp per-group range by clip_ratio
//   3. Asymmetric INT4 quantization with zero-point
//
// This eliminates 5+ Python-level kernel launches per layer:
//   reshape + abs + amax + clamp + reshape (clipping) → fused into quant kernel
//   smooth_scale division (elementwise) → fused into quant kernel
//
// Input:  [M, K] BF16/FP16, group_size, clip_ratio, optional smooth_scale [K]
// Output: [num_groups, M, gs/2] uint8 packed (signed INT4, shifted)
//         [num_groups, M] float scales
//         [num_groups, M] float azp_adj  (= 8 - zero_point)
//
// Grid: (M, num_groups), Block: (256)
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void dynamic_int4_quant_asymmetric_clipped_grouped_kernel(
    const scalar_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scale_out,
    float* __restrict__ azp_adj_out,
    const int K, const int group_size, const float clip_ratio,
    const scalar_t* __restrict__ smooth_scale)  // nullable
{
    const int m = blockIdx.x;
    const int g = blockIdx.y;
    const int M_total = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs = group_size;

    const scalar_t* slice = input + m * K + g * gs;
    const scalar_t* smooth_slice = (smooth_scale != nullptr)
        ? smooth_scale + g * gs : nullptr;
    uint8_t* out_slice = output + (g * M_total + m) * (gs / 2);
    float* scale_ptr = scale_out + g * M_total + m;
    float* azp_adj_ptr = azp_adj_out + g * M_total + m;

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;

    // Phase 1: find absmax per group (with smooth_scale applied) for symmetric clip
    float thread_absmax = 0.0f;
    for (int i = tid; i < gs; i += stride) {
        float v = static_cast<float>(slice[i]);
        if (smooth_slice != nullptr) {
            v /= static_cast<float>(smooth_slice[i]);
        }
        thread_absmax = fmaxf(thread_absmax, fabsf(v));
    }

    float block_absmax = BlockReduce(tmp).Reduce(thread_absmax, CubMaxOp{}, blockDim.x);
    __shared__ float s_clip_threshold;
    if (tid == 0) {
        s_clip_threshold = block_absmax * clip_ratio;
    }
    __syncthreads();

    float clip_thresh = s_clip_threshold;

    // Phase 2: find min/max of clipped values (matching Python: clamp(-threshold, threshold))
    float thread_min = INFINITY;
    float thread_max = -INFINITY;
    for (int i = tid; i < gs; i += stride) {
        float v = static_cast<float>(slice[i]);
        if (smooth_slice != nullptr) {
            v /= static_cast<float>(smooth_slice[i]);
        }
        v = fminf(fmaxf(v, -clip_thresh), clip_thresh);
        thread_min = fminf(thread_min, v);
        thread_max = fmaxf(thread_max, v);
    }

    float block_min = BlockReduce(tmp).Reduce(thread_min, CubMinOp{}, blockDim.x);
    __shared__ float s_min;
    if (tid == 0) s_min = block_min;
    __syncthreads();

    float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

    // Phase 2b: compute scale and azp from clipped min/max
    __shared__ float s_scale, s_azp;
    if (tid == 0) {
        float range = block_max - s_min;
        float scale = fmaxf(range / 15.0f, kMinScalingFactor);
        float azp = rintf(-s_min / scale);  // unsigned zero point [0, 15]
        azp = fminf(fmaxf(azp, 0.0f), 15.0f);

        s_scale = scale;
        s_azp = azp;
        *scale_ptr = scale;
        *azp_adj_ptr = 8.0f - azp;  // correction factor for GEMM
    }
    __syncthreads();

    float inv_scale = 1.0f / s_scale;
    float azp_val = s_azp;

    // Phase 3: clip, quantize to unsigned [0,15], shift to signed [-8,7], pack
    int half_gs = gs / 2;
    for (int i = tid; i < half_gs; i += stride) {
        float v0 = static_cast<float>(slice[i * 2]);
        float v1 = static_cast<float>(slice[i * 2 + 1]);

        if (smooth_slice != nullptr) {
            v0 /= static_cast<float>(smooth_slice[i * 2]);
            v1 /= static_cast<float>(smooth_slice[i * 2 + 1]);
        }

        // Apply symmetric clipping (same as Python clamp(-threshold, threshold))
        v0 = fminf(fmaxf(v0, -clip_thresh), clip_thresh);
        v1 = fminf(fmaxf(v1, -clip_thresh), clip_thresh);

        float q0 = rintf(v0 * inv_scale + azp_val);
        float q1 = rintf(v1 * inv_scale + azp_val);
        q0 = fminf(fmaxf(q0, 0.0f), 15.0f);
        q1 = fminf(fmaxf(q1, 0.0f), 15.0f);

        // Pack as unsigned then shift to signed for INT4 MMA
        out_slice[i] = pack_uint4x2(
            static_cast<uint8_t>(q0), static_cast<uint8_t>(q1));
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dynamic_int4_quant_asymmetric_clipped_grouped(
    torch::Tensor input, int64_t group_size, double clip_ratio,
    std::optional<torch::Tensor> smooth_scale)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
    TORCH_CHECK(clip_ratio > 0.0 && clip_ratio <= 1.0,
                "clip_ratio must be in (0, 1]");

    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    int num_groups = K / group_size;
    int gs = static_cast<int>(group_size);
    float cr = static_cast<float>(clip_ratio);

    auto packed = torch::empty({num_groups, M, gs / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    auto scale = torch::empty({num_groups, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    auto azp_adj = torch::empty({num_groups, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    dim3 grid(M, num_groups);
    int block = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "dynamic_int4_quant_asymmetric_clipped_grouped", [&] {
            const scalar_t* smooth_ptr = nullptr;
            if (smooth_scale.has_value()) {
                TORCH_CHECK(smooth_scale->is_cuda(), "smooth_scale must be CUDA tensor");
                TORCH_CHECK(smooth_scale->numel() == K,
                            "smooth_scale must have K elements");
                smooth_ptr = smooth_scale->data_ptr<scalar_t>();
            }
            dynamic_int4_quant_asymmetric_clipped_grouped_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                packed.data_ptr<uint8_t>(),
                scale.data_ptr<float>(),
                azp_adj.data_ptr<float>(),
                K, gs, cr,
                smooth_ptr);
        });

    return {packed, scale, azp_adj};
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 12: Per-group INT4 → BF16 dequantization with fused scale
//
// Dequantizes per-group packed INT4 data to a contiguous BF16 matrix,
// applying per-group scales during the unpack. This is the key preprocessing
// step for the dequant-GEMM approach (Option D) which avoids per-group
// barrier synchronizations by converting everything to BF16 first, then
// running a single full-K BF16 GEMM.
//
// Input:  packed [num_groups, rows, gs/2] uint8 (per-group INT4 packed)
//         scale  [num_groups, rows] float (per-group per-row scales)
// Output: out    [rows, K] bf16, where K = num_groups * group_size
//
// Memory layout transformation:
//   packed[g, m, k_local/2] with scale[g, m]
//   → out[m, g * gs + k_local] = unpack(packed[g, m, k_local/2]) * scale[g, m]
//
// This reassembles the group-fragmented data into a single contiguous matrix
// suitable for cuBLAS BF16 GEMM.
//
// Grid: (rows,), Block: (256,)
// Each block handles one row, iterating over all groups and elements.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void dequant_int4_grouped_to_bf16_kernel(
    const uint8_t* __restrict__ packed,  // [ng, rows, gs/2]
    const float* __restrict__ scale,     // [ng, rows]
    __nv_bfloat16* __restrict__ out,     // [rows, K]  K = ng * gs
    const int rows, const int gs, const int ng)
{
    const int m = blockIdx.x;  // row index
    if (m >= rows) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs_half = gs / 2;
    const int K = ng * gs;

    // Process all groups for this row
    for (int g = 0; g < ng; g++) {
        float s = scale[g * rows + m];

        // Pointer to this group's packed data for row m
        const uint8_t* group_packed = packed + ((int64_t)g * rows + m) * gs_half;

        // Output offset: row m, starting at column g * gs
        __nv_bfloat16* out_row = out + (int64_t)m * K + g * gs;

        // Each thread unpacks multiple bytes (2 INT4 values per byte)
        // Use uint32_t loads for efficiency (4 bytes = 8 INT4 values at a time)
        const int vec4_count = gs_half / 4;
        for (int i = tid; i < vec4_count; i += stride) {
            uint32_t packed4 = *reinterpret_cast<const uint32_t*>(group_packed + i * 4);

            // Unpack 4 bytes → 8 bf16 values
            __nv_bfloat16 vals[8];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t byte = (packed4 >> (j * 8)) & 0xFF;
                int8_t v0, v1;
                // Low nibble (even index) - sign extend
                v0 = static_cast<int8_t>(byte << 4) >> 4;
                // High nibble (odd index) - sign extend
                v1 = static_cast<int8_t>(byte & 0xF0) >> 4;
                vals[j * 2]     = __float2bfloat16(static_cast<float>(v0) * s);
                vals[j * 2 + 1] = __float2bfloat16(static_cast<float>(v1) * s);
            }

            // Store 8 bf16 values (16 bytes) at once
            *reinterpret_cast<uint4*>(out_row + i * 8) =
                *reinterpret_cast<uint4*>(vals);
        }

        // Handle tail bytes (when gs_half is not divisible by 4)
        for (int i = vec4_count * 4 + tid; i < gs_half; i += stride) {
            int8_t v0, v1;
            unpack_int4x2(group_packed[i], v0, v1);
            out_row[i * 2]     = __float2bfloat16(static_cast<float>(v0) * s);
            out_row[i * 2 + 1] = __float2bfloat16(static_cast<float>(v1) * s);
        }
    }
}

torch::Tensor dequant_int4_grouped_to_bf16(
    torch::Tensor packed,   // [num_groups, rows, gs/2] uint8
    torch::Tensor scale,    // [num_groups, rows] float
    int64_t group_size)
{
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 3, "packed must be 3D [num_groups, rows, gs/2]");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA tensor");
    TORCH_CHECK(scale.dim() == 2, "scale must be 2D [num_groups, rows]");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    int num_groups = packed.size(0);
    int rows = packed.size(1);
    int gs = static_cast<int>(group_size);
    int K = num_groups * gs;

    TORCH_CHECK(packed.size(2) == gs / 2,
                "packed dim 2 must be gs/2, got ", packed.size(2),
                " expected ", gs / 2);
    TORCH_CHECK(scale.size(0) == num_groups && scale.size(1) == rows,
                "scale shape must be [num_groups, rows]");

    auto output = torch::empty({rows, K},
        torch::TensorOptions().dtype(torch::kBFloat16).device(packed.device()));

    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    int block = 256;

    dequant_int4_grouped_to_bf16_kernel<<<rows, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        rows, gs, num_groups);

    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 13: Per-group INT4 activation → contiguous INT8 unpacking
//
// Unpacks per-group packed INT4 activations [num_groups, M, gs/2] into
// a contiguous INT8 matrix [M, K] where K = num_groups * group_size.
// This is the activation preprocessing step for the QServe-style progressive
// INT8 GEMM approach, where INT4 values are sign-extended to INT8 (no scale
// application) and laid out contiguously for efficient multi-group K-tiles.
//
// Memory layout transformation:
//   packed[g, m, k_local/2] → out[m, g * gs + k_local]
//   Each INT4 value in [-8, 7] is sign-extended to INT8 [-8, 7].
//
// Grid: (M,), Block: (256,)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void unpack_int4_grouped_to_int8_contiguous_kernel(
    const uint8_t* __restrict__ packed,  // [ng, M, gs/2]
    int8_t* __restrict__ out,            // [M, K]  K = ng * gs
    const int M, const int gs, const int ng)
{
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs_half = gs / 2;
    const int K = ng * gs;

    for (int g = 0; g < ng; g++) {
        // Pointer to this group's packed data for row m
        const uint8_t* group_packed = packed + ((int64_t)g * M + m) * gs_half;

        // Output offset: row m, starting at column g * gs
        int8_t* out_row = out + (int64_t)m * K + g * gs;

        // Use uint32_t loads for efficiency (4 bytes = 8 INT4 values)
        const int vec4_count = gs_half / 4;
        for (int i = tid; i < vec4_count; i += stride) {
            uint32_t packed4 = *reinterpret_cast<const uint32_t*>(group_packed + i * 4);
            int8_t unpacked[8];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t byte = (packed4 >> (j * 8)) & 0xFF;
                // Low nibble (even index) - sign extend
                unpacked[j * 2]     = static_cast<int8_t>(byte << 4) >> 4;
                // High nibble (odd index) - sign extend
                unpacked[j * 2 + 1] = static_cast<int8_t>(byte & 0xF0) >> 4;
            }
            // Store 8 int8 values (8 bytes = 64-bit store)
            *reinterpret_cast<int64_t*>(out_row + i * 8) =
                *reinterpret_cast<int64_t*>(unpacked);
        }

        // Handle tail bytes
        for (int i = vec4_count * 4 + tid; i < gs_half; i += stride) {
            int8_t v0, v1;
            unpack_int4x2(group_packed[i], v0, v1);
            out_row[i * 2]     = v0;
            out_row[i * 2 + 1] = v1;
        }
    }
}

torch::Tensor unpack_int4_grouped_to_int8_contiguous(
    torch::Tensor packed,   // [num_groups, M, gs/2] uint8
    int64_t group_size)
{
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 3, "packed must be 3D [num_groups, rows, gs/2]");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    int num_groups = packed.size(0);
    int M = packed.size(1);
    int gs = static_cast<int>(group_size);
    int K = num_groups * gs;

    TORCH_CHECK(packed.size(2) == gs / 2,
                "packed dim 2 must be gs/2, got ", packed.size(2),
                " expected ", gs / 2);

    auto output = torch::empty({M, K},
        torch::TensorOptions().dtype(torch::kInt8).device(packed.device()));

    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    int block = 256;

    unpack_int4_grouped_to_int8_contiguous_kernel<<<M, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        M, gs, num_groups);

    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 14: Per-group INT4 weight → contiguous INT8 unpacking
//
// Unpacks per-group packed INT4 weights [num_groups, N, gs/2] into
// a contiguous INT8 matrix [N, K] where K = num_groups * group_size.
// This is the weight preprocessing step (done once at model load time) for
// the QServe-style progressive INT8 GEMM approach.
//
// Memory layout transformation:
//   packed[g, n, k_local/2] → out[n, g * gs + k_local]
//   Each INT4 value in [-8, 7] is sign-extended to INT8 [-8, 7].
//
// Grid: (N,), Block: (256,)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void unpack_int4_grouped_to_int8_weight_kernel(
    const uint8_t* __restrict__ packed,  // [ng, N, gs/2]
    int8_t* __restrict__ out,            // [N, K]  K = ng * gs
    const int N, const int gs, const int ng)
{
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs_half = gs / 2;
    const int K = ng * gs;

    for (int g = 0; g < ng; g++) {
        const uint8_t* group_packed = packed + ((int64_t)g * N + n) * gs_half;
        int8_t* out_row = out + (int64_t)n * K + g * gs;

        const int vec4_count = gs_half / 4;
        for (int i = tid; i < vec4_count; i += stride) {
            uint32_t packed4 = *reinterpret_cast<const uint32_t*>(group_packed + i * 4);
            int8_t unpacked[8];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t byte = (packed4 >> (j * 8)) & 0xFF;
                unpacked[j * 2]     = static_cast<int8_t>(byte << 4) >> 4;
                unpacked[j * 2 + 1] = static_cast<int8_t>(byte & 0xF0) >> 4;
            }
            *reinterpret_cast<int64_t*>(out_row + i * 8) =
                *reinterpret_cast<int64_t*>(unpacked);
        }

        for (int i = vec4_count * 4 + tid; i < gs_half; i += stride) {
            int8_t v0, v1;
            unpack_int4x2(group_packed[i], v0, v1);
            out_row[i * 2]     = v0;
            out_row[i * 2 + 1] = v1;
        }
    }
}

torch::Tensor unpack_int4_grouped_to_int8_weight(
    torch::Tensor packed,   // [num_groups, N, gs/2] uint8
    int64_t group_size)
{
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 3, "packed must be 3D [num_groups, N, gs/2]");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    int num_groups = packed.size(0);
    int N = packed.size(1);
    int gs = static_cast<int>(group_size);
    int K = num_groups * gs;

    TORCH_CHECK(packed.size(2) == gs / 2,
                "packed dim 2 must be gs/2, got ", packed.size(2),
                " expected ", gs / 2);

    auto output = torch::empty({N, K},
        torch::TensorOptions().dtype(torch::kInt8).device(packed.device()));

    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    int block = 256;

    unpack_int4_grouped_to_int8_weight_kernel<<<N, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        N, gs, num_groups);

    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 15: Per-group INT4 weight → FP8 (e4m3) dequantization
//
// Converts INT4 per-group packed weights to FP8 e4m3 format for use with
// FP8 GEMM (torch._scaled_mm). This enables W4A8-FP8 inference:
//   - Weight stored as INT4 per-group (4x compression)
//   - Runtime dequant to FP8
//   - FP8 × FP8 GEMM (cuBLAS, no accumulator flush!)
//
// Two-pass approach:
//   Pass 1 (compute_channel_amax_kernel): Find per-channel (per-row) max abs
//          value of dequanted weights → channel_amax[N]
//   Pass 2 (dequant_int4_grouped_to_fp8_kernel): Dequant INT4 → FP8 using
//          combined group_scale / channel_scale
//
// Input:  packed [ng, N, gs/2] uint8, group_scales [ng, N] float32
// Output: fp8_out [N, K] float8_e4m3fn, channel_scales [N] float32
//
// Grid: (N,), Block: (256,)
// ─────────────────────────────────────────────────────────────────────────────

// Pass 1: Compute per-channel absolute max of dequanted INT4 values
__global__ void compute_channel_amax_kernel(
    const uint8_t* __restrict__ packed,       // [ng, N, gs/2]
    const float* __restrict__ group_scales,   // [ng, N]
    float* __restrict__ channel_amax,         // [N]
    const int N, const int gs, const int ng)
{
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs_half = gs / 2;

    float local_max = 0.0f;

    for (int g = 0; g < ng; g++) {
        float s = fabsf(group_scales[g * N + n]);
        const uint8_t* group_packed = packed + ((int64_t)g * N + n) * gs_half;

        // Each thread processes multiple bytes
        for (int i = tid; i < gs_half; i += stride) {
            uint8_t byte = group_packed[i];
            int8_t v0 = static_cast<int8_t>(byte << 4) >> 4;
            int8_t v1 = static_cast<int8_t>(byte & 0xF0) >> 4;
            float abs0 = fabsf(static_cast<float>(v0)) * s;
            float abs1 = fabsf(static_cast<float>(v1)) * s;
            local_max = fmaxf(local_max, fmaxf(abs0, abs1));
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }

    // Block-level reduction via shared memory
    __shared__ float warp_maxes[8];  // up to 8 warps (256 threads)
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_maxes[warp_id] = local_max;
    }
    __syncthreads();

    if (tid == 0) {
        float block_max = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; i++) {
            block_max = fmaxf(block_max, warp_maxes[i]);
        }
        // FP8 e4m3 max representable value = 448.0
        // channel_scale = amax / 448.0, clamped to avoid division by zero
        channel_amax[n] = fmaxf(block_max / 448.0f, 1e-12f);
    }
}

// Pass 2: Dequant INT4 → FP8 using precomputed channel scales
__global__ void dequant_int4_grouped_to_fp8_kernel(
    const uint8_t* __restrict__ packed,        // [ng, N, gs/2]
    const float* __restrict__ group_scales,    // [ng, N]
    const float* __restrict__ channel_scales,  // [N] (= amax / 448)
    uint8_t* __restrict__ fp8_out,             // [N, K] as raw uint8 for FP8
    const int N, const int gs, const int ng)
{
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gs_half = gs / 2;
    const int K = ng * gs;

    float ch_scale_inv = 1.0f / channel_scales[n];  // pre-divide

    for (int g = 0; g < ng; g++) {
        float g_scale = group_scales[g * N + n];
        float combined = g_scale * ch_scale_inv;  // group_scale / channel_scale

        const uint8_t* group_packed = packed + ((int64_t)g * N + n) * gs_half;
        uint8_t* out_row = fp8_out + (int64_t)n * K + g * gs;

        // Vectorized: process 4 bytes (8 INT4 values) at a time
        const int vec4_count = gs_half / 4;
        for (int i = tid; i < vec4_count; i += stride) {
            uint32_t packed4 = *reinterpret_cast<const uint32_t*>(group_packed + i * 4);

            uint8_t fp8_vals[8];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t byte = (packed4 >> (j * 8)) & 0xFF;
                int8_t v0 = static_cast<int8_t>(byte << 4) >> 4;
                int8_t v1 = static_cast<int8_t>(byte & 0xF0) >> 4;

                float f0 = static_cast<float>(v0) * combined;
                float f1 = static_cast<float>(v1) * combined;

                // Clamp to FP8 e4m3 range [-448, 448] and convert
                f0 = fminf(fmaxf(f0, -448.0f), 448.0f);
                f1 = fminf(fmaxf(f1, -448.0f), 448.0f);

                // Convert float → FP8 e4m3fn using CUDA intrinsic
                fp8_vals[j * 2]     = __nv_cvt_float_to_fp8(f0, __NV_SATFINITE, __NV_E4M3);
                fp8_vals[j * 2 + 1] = __nv_cvt_float_to_fp8(f1, __NV_SATFINITE, __NV_E4M3);
            }

            // Store 8 FP8 values (8 bytes) at once
            *reinterpret_cast<uint2*>(out_row + i * 8) =
                *reinterpret_cast<uint2*>(fp8_vals);
        }

        // Handle tail bytes
        for (int i = vec4_count * 4 + tid; i < gs_half; i += stride) {
            int8_t v0, v1;
            unpack_int4x2(group_packed[i], v0, v1);
            float f0 = static_cast<float>(v0) * combined;
            float f1 = static_cast<float>(v1) * combined;
            f0 = fminf(fmaxf(f0, -448.0f), 448.0f);
            f1 = fminf(fmaxf(f1, -448.0f), 448.0f);
            out_row[i * 2]     = __nv_cvt_float_to_fp8(f0, __NV_SATFINITE, __NV_E4M3);
            out_row[i * 2 + 1] = __nv_cvt_float_to_fp8(f1, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

// C++ wrapper: Pass 2 only — dequant with precomputed channel_scales
// Use this at runtime when channel_scales have been computed at load time.
void dequant_int4_grouped_to_fp8_with_scales(
    torch::Tensor packed,          // [num_groups, N, gs/2] uint8
    torch::Tensor group_scales,    // [num_groups, N] float
    torch::Tensor channel_scales,  // [N] float (precomputed)
    torch::Tensor fp8_out,         // [N, K] float8_e4m3fn (pre-allocated)
    int64_t group_size)
{
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8);
    TORCH_CHECK(packed.dim() == 3);

    int num_groups = packed.size(0);
    int N = packed.size(1);
    int gs = static_cast<int>(group_size);

    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    int block = 256;

    dequant_int4_grouped_to_fp8_kernel<<<N, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        group_scales.data_ptr<float>(),
        channel_scales.data_ptr<float>(),
        reinterpret_cast<uint8_t*>(fp8_out.data_ptr()),
        N, gs, num_groups);
}

// C++ wrapper: two-pass INT4 per-group → FP8 per-channel dequant
std::tuple<torch::Tensor, torch::Tensor> dequant_int4_grouped_to_fp8(
    torch::Tensor packed,   // [num_groups, N, gs/2] uint8
    torch::Tensor scale,    // [num_groups, N] float
    int64_t group_size)
{
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 3, "packed must be 3D [num_groups, N, gs/2]");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA tensor");
    TORCH_CHECK(scale.dim() == 2, "scale must be 2D [num_groups, N]");

    int num_groups = packed.size(0);
    int N = packed.size(1);
    int gs = static_cast<int>(group_size);
    int K = num_groups * gs;

    TORCH_CHECK(packed.size(2) == gs / 2,
                "packed dim 2 must be gs/2, got ", packed.size(2));
    TORCH_CHECK(scale.size(0) == num_groups && scale.size(1) == N,
                "scale shape must be [num_groups, N]");

    // Output: FP8 weight [N, K] and per-channel scales [N]
    auto fp8_out = torch::empty({N, K},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(packed.device()));
    auto channel_scales = torch::empty({N},
        torch::TensorOptions().dtype(torch::kFloat32).device(packed.device()));

    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    int block = 256;

    // Pass 1: Compute channel amax → channel_scales
    compute_channel_amax_kernel<<<N, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        channel_scales.data_ptr<float>(),
        N, gs, num_groups);

    // Pass 2: Dequant INT4 → FP8
    dequant_int4_grouped_to_fp8_kernel<<<N, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        channel_scales.data_ptr<float>(),
        reinterpret_cast<uint8_t*>(fp8_out.data_ptr()),
        N, gs, num_groups);

    return {fp8_out, channel_scales};
}
