/*
 * CUTLASS INT4x INT4 Tensor Core GEMM kernel with EVT ScaledEpilogue
 *
 * Uses CUTLASS 2.x GemmUniversalAdapter + Epilogue Visitor Tree (EVT) to fuse
 * scale application into the GEMM epilogue, eliminating the separate
 * apply_scales_kernel and its M*N memory round-trip.
 *
 * Target instruction: mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite
 * Compatible with SM80+ (Ampere, Ada Lovelace L4 SM89).
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// clang-format off
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/integer_subbyte.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c2x.hpp"
// clang-format on

using namespace cute;

// ─────────────────────────────────────────────────────────────────────────────
// Utility: CUTLASS status check macro
// ─────────────────────────────────────────────────────────────────────────────

#define CUTLASS_CHECK(status)                          \
  {                                                    \
    cutlass::Status error = status;                    \
    TORCH_CHECK(error == cutlass::Status::kSuccess,    \
                cutlassGetStatusString(error));         \
  }

// ─────────────────────────────────────────────────────────────────────────────
// Utility: next power of 2, shared memory query
// ─────────────────────────────────────────────────────────────────────────────

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

static int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return max_shared_mem_per_block_opt_in;
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture guard: restrict compiled kernel to SM80-SM89 (inclusive)
//
// Unlike FP8 (which needs separate SM89 math operators), INT4
// OpMultiplyAddSaturate is identical across SM80-SM89. We use < 900
// to INCLUDE SM89 (Ada Lovelace / L4, __CUDA_ARCH__ = 890).
// ─────────────────────────────────────────────────────────────────────────────

template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[80, 90).\n");
    asm("trap;");
  #endif
#endif
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// ScaledEpilogue: fuses acc * scale_b[n] * scale_a[m] into GEMM epilogue
//
// Directly adapted from vLLM's scaled_mm_epilogues_c2x.hpp.
// Works for any accumulator type (int32_t for INT4/INT8, float for FP8).
// ─────────────────────────────────────────────────────────────────────────────

template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  template <typename T>
  using ColOrScalarLoad =
      cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
          OutputTileThreadMap, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoad =
      cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
          OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename Descriptor, typename T>
  static auto args_from_tensor(torch::Tensor const& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = static_cast<T*>(tensor.data_ptr());
    if constexpr (std::is_same_v<Descriptor, ColOrScalarLoad<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoad<T>>) {
      return Arguments{data_ptr, tensor.numel() != 1};
    } else {
      return Arguments{data_ptr};
    }
  }
};

template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// cutlass_2x_int4_gemm: CUTLASS 2.x GEMM template for INT4 with EVT
//
// Pattern from vLLM's cutlass_2x_gemm, adapted for int4b_t:
//   - ElementAcc = int32_t
//   - Operator = OpMultiplyAddSaturate
//   - InstructionShape K = 64 (INT4 MMA k-dimension)
// ─────────────────────────────────────────────────────────────────────────────

template <typename Arch, template <typename> typename ArchGuard,
          typename ElementD_,
          template <typename, typename> typename Epilogue_,
          typename TileShape, typename WarpShape,
          int32_t MainLoopStages>
struct cutlass_2x_int4_gemm {
  using ElementAB = cutlass::int4b_t;
  using ElementD = ElementD_;
  using ElementAcc = int32_t;
  using Operator = cutlass::arch::OpMultiplyAddSaturate;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          TileShape, WarpShape, float, 4, 1 /* epilogue stages */
          >;

  using Epilogue = Epilogue_<ElementD, OutputTileThreadMap>;
  using EVTCompute = typename Epilogue::EVTCompute;

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  // 128 bits / 4 bits per element = 32 elements alignment
  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 4;

  using RowMajor = typename cutlass::layout::RowMajor;
  using ColumnMajor = typename cutlass::layout::ColumnMajor;

  // clang-format off
  using KernelType =
    ArchGuard<typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementAB, RowMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      ElementAB, ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      float, cutlass::layout::RowMajor, AlignmentCD,
      ElementAcc, float, cutlass::arch::OpClassTensorOp,
      Arch,
      TileShape, WarpShape, InstructionShape,
      EVTD,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      MainLoopStages, Operator,
      1 /* epilogue stages */
      >::GemmKernel>;
  // clang-format on

  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

// ─────────────────────────────────────────────────────────────────────────────
// Tile-shape configurations for different M ranges (SM80 INT4)
//
// INT4 InstructionShape is always GemmShape<16, 8, 64>.
// WarpShape K = 128 (matches existing working config and CUTLASS defaults
// for INT4 in default_gemm_configuration.h).
// ─────────────────────────────────────────────────────────────────────────────

template <typename OutType, template <typename, typename> typename Epilogue>
struct sm80_int4_config_default {
  // M in (128, inf) and M in (64, 128] with N >= 8192
  using TileShape = cutlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using Cutlass2xGemm =
      cutlass_2x_int4_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89,
                           OutType, Epilogue, TileShape, WarpShape, 3>;
};

template <typename OutType, template <typename, typename> typename Epilogue>
struct sm80_int4_config_M64 {
  // M in (32, 64] and M in (64, 128] with N < 8192
  using TileShape = cutlass::gemm::GemmShape<64, 128, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using Cutlass2xGemm =
      cutlass_2x_int4_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89,
                           OutType, Epilogue, TileShape, WarpShape, 3>;
};

template <typename OutType, template <typename, typename> typename Epilogue>
struct sm80_int4_config_M32 {
  // M in (16, 32]
  using TileShape = cutlass::gemm::GemmShape<32, 64, 128>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 128>;
  using Cutlass2xGemm =
      cutlass_2x_int4_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89,
                           OutType, Epilogue, TileShape, WarpShape, 3>;
};

template <typename OutType, template <typename, typename> typename Epilogue>
struct sm80_int4_config_M16 {
  // M in [1, 16]
  using TileShape = cutlass::gemm::GemmShape<16, 64, 128>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 128>;
  using Cutlass2xGemm =
      cutlass_2x_int4_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89,
                           OutType, Epilogue, TileShape, WarpShape, 3>;
};

// ─────────────────────────────────────────────────────────────────────────────
// cutlass_2x_int8_gemm: INT8 × INT8 GEMM with EVT ScaledEpilogue
//
// Both A (activation) and B (weight) are int8_t. Weight is pre-unpacked from
// INT4 to INT8 at model load time. Uses INT8×INT8 MMA (m16n8k32).
//
// For W4A8: activation is dynamically quantized to INT8, weight is pre-unpacked
// from INT4→INT8 at load time. VRAM: weight is INT8 (2× BF16 savings).
// ─────────────────────────────────────────────────────────────────────────────

template <typename Arch, template <typename> typename ArchGuard,
          typename ElementD_,
          template <typename, typename> typename Epilogue_,
          typename TileShape, typename WarpShape,
          int32_t MainLoopStages>
struct cutlass_2x_int8_gemm {
  using ElementAB = int8_t;
  using ElementD = ElementD_;
  using ElementAcc = int32_t;
  using Operator = cutlass::arch::OpMultiplyAddSaturate;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          TileShape, WarpShape, float, 4, 1>;

  using Epilogue = Epilogue_<ElementD, OutputTileThreadMap>;
  using EVTCompute = typename Epilogue::EVTCompute;

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  // 128 bits / 8 bits per element = 16 elements alignment
  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 4;

  using RowMajor = typename cutlass::layout::RowMajor;
  using ColumnMajor = typename cutlass::layout::ColumnMajor;

  // clang-format off
  using KernelType =
    ArchGuard<typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementAB, RowMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      ElementAB, ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      float, cutlass::layout::RowMajor, AlignmentCD,
      ElementAcc, float, cutlass::arch::OpClassTensorOp,
      Arch,
      TileShape, WarpShape, InstructionShape,
      EVTD,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      MainLoopStages, Operator,
      1 /* epilogue stages */
      >::GemmKernel>;
  // clang-format on

  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

// ─────────────────────────────────────────────────────────────────────────────
// Tile-shape configurations for INT8×INT8 GEMM on SM80
//
// InstructionShape = GemmShape<16, 8, 32> (INT8×INT8 MMA)
// WarpShape K = 64 (2× InstructionShape K)
// ─────────────────────────────────────────────────────────────────────────────

template <typename OutType, template <typename, typename> typename Epilogue>
struct sm80_int8_config_default {
  using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using Cutlass2xGemm =
      cutlass_2x_int8_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89,
                            OutType, Epilogue, TileShape, WarpShape, 3>;
};

template <typename OutType, template <typename, typename> typename Epilogue>
struct sm80_int8_config_M64 {
  using TileShape = cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using Cutlass2xGemm =
      cutlass_2x_int8_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89,
                            OutType, Epilogue, TileShape, WarpShape, 3>;
};

template <typename OutType, template <typename, typename> typename Epilogue>
struct sm80_int8_config_M32 {
  using TileShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using Cutlass2xGemm =
      cutlass_2x_int8_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89,
                            OutType, Epilogue, TileShape, WarpShape, 3>;
};

template <typename OutType, template <typename, typename> typename Epilogue>
struct sm80_int8_config_M16 {
  using TileShape = cutlass::gemm::GemmShape<16, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 64>;
  using Cutlass2xGemm =
      cutlass_2x_int8_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89,
                            OutType, Epilogue, TileShape, WarpShape, 3>;
};

// ─────────────────────────────────────────────────────────────────────────────
// cutlass_int8_gemm_caller: launches INT8×INT8 CUTLASS GEMM with EVT
// ─────────────────────────────────────────────────────────────────────────────

template <typename Gemm, typename... EpilogueArgs>
inline void cutlass_int8_gemm_caller(
    torch::Tensor& out,
    torch::Tensor const& a,  // [M, K] int8
    torch::Tensor const& b,  // [N, K] int8
    int64_t M, int64_t N, int64_t K,
    EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  cutlass::gemm::GemmCoord problem_size{(int)M, (int)N, (int)K};

  int64_t lda = K;
  int64_t ldb = K;
  int64_t ldc = N;

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  using Epilogue = typename Gemm::Epilogue;
  auto evt_args =
      Epilogue::prepare_args(std::forward<EpilogueArgs>(epilogue_params)...);

  typename Gemm::EVTD::Arguments epilogue_args{
      evt_args,
      d_args,
  };

  typename Gemm::Op::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
      problem_size,
      1,
      epilogue_args,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0, 0, 0, 0,
      lda, ldb, ldc, ldc};

  typename Gemm::Op gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

// ─────────────────────────────────────────────────────────────────────────────
// fallback_cutlass_int8_gemm_caller: shared memory fallback for INT8
// ─────────────────────────────────────────────────────────────────────────────

template <typename Gemm, typename FallbackGemm, typename... EpilogueArgs>
inline void fallback_cutlass_int8_gemm_caller(
    torch::Tensor& out,
    torch::Tensor const& a, torch::Tensor const& b,
    int64_t M, int64_t N, int64_t K,
    EpilogueArgs&&... args) {
  static const int max_shared_mem_per_block_opt_in =
      get_cuda_max_shared_memory_per_block_opt_in(0);

  size_t const gemm_shared_mem_size =
      sizeof(typename Gemm::KernelType::SharedStorage);
  size_t const fallback_gemm_shared_mem_size =
      sizeof(typename FallbackGemm::KernelType::SharedStorage);

  if (gemm_shared_mem_size <= (size_t)max_shared_mem_per_block_opt_in) {
    return cutlass_int8_gemm_caller<Gemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  } else {
    TORCH_CHECK(fallback_gemm_shared_mem_size <=
                (size_t)max_shared_mem_per_block_opt_in);
    return cutlass_int8_gemm_caller<FallbackGemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// INT8×INT8 M-range dispatch
// ─────────────────────────────────────────────────────────────────────────────

template <typename OutType, template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_int8_gemm_sm80_dispatch(
    torch::Tensor& out,
    torch::Tensor const& a, torch::Tensor const& b,
    int64_t M, int64_t N, int64_t K,
    EpilogueArgs&&... args) {

  using Cutlass2xGemmDefault =
      typename sm80_int8_config_default<OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM64 =
      typename sm80_int8_config_M64<OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM32 =
      typename sm80_int8_config_M32<OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM16 =
      typename sm80_int8_config_M16<OutType, Epilogue>::Cutlass2xGemm;

  using FallbackGemm = Cutlass2xGemmM32;

  uint32_t const m = static_cast<uint32_t>(M);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_pow_2(m));

  if (mp2 <= 16) {
    return fallback_cutlass_int8_gemm_caller<Cutlass2xGemmM16, FallbackGemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 32) {
    return fallback_cutlass_int8_gemm_caller<Cutlass2xGemmM32, FallbackGemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 64) {
    return fallback_cutlass_int8_gemm_caller<Cutlass2xGemmM64, FallbackGemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    uint32_t const n = static_cast<uint32_t>(N);
    if (n < 8192) {
      return fallback_cutlass_int8_gemm_caller<Cutlass2xGemmM64, FallbackGemm>(
          out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
    } else {
      return fallback_cutlass_int8_gemm_caller<Cutlass2xGemmDefault, FallbackGemm>(
          out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
    }
  } else {
    return fallback_cutlass_int8_gemm_caller<Cutlass2xGemmDefault, FallbackGemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// cutlass_gemm_caller: launches CUTLASS GEMM with EVT epilogue
// Adapted from vLLM's cutlass_gemm_caller for INT4 packed tensors.
//
// Key difference from vLLM: our INT4 tensors are packed as uint8 with
// shape [M, K/2] and [N, K/2]. We pass logical K to CUTLASS and raw
// data pointers. CUTLASS int4b_t handles the sub-byte addressing.
// ─────────────────────────────────────────────────────────────────────────────

template <typename Gemm, typename... EpilogueArgs>
inline void cutlass_int4_gemm_caller(
    torch::Tensor& out,
    torch::Tensor const& a_packed,  // [M, K/2] uint8
    torch::Tensor const& b_packed,  // [N, K/2] uint8
    int64_t M, int64_t N, int64_t K,
    EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  cutlass::gemm::GemmCoord problem_size{(int)M, (int)N, (int)K};

  // Leading dimensions in elements (not bytes):
  // A: [M, K] row-major, lda = K
  // B: [N, K] col-major, ldb = K
  // C: [M, N] row-major, ldc = N
  int64_t lda = K;
  int64_t ldb = K;
  int64_t ldc = N;

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a_ptr = static_cast<ElementAB const*>(a_packed.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b_packed.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  using Epilogue = typename Gemm::Epilogue;
  auto evt_args =
      Epilogue::prepare_args(std::forward<EpilogueArgs>(epilogue_params)...);

  typename Gemm::EVTD::Arguments epilogue_args{
      evt_args,
      d_args,
  };

  typename Gemm::Op::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
      problem_size,
      1,  // batch count
      epilogue_args,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0, 0, 0, 0,
      lda, ldb, ldc, ldc};

  typename Gemm::Op gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a_packed.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device());

  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

// ─────────────────────────────────────────────────────────────────────────────
// fallback_cutlass_gemm_caller: selects between primary and fallback kernel
// based on shared memory availability (same pattern as vLLM)
// ─────────────────────────────────────────────────────────────────────────────

template <typename Gemm, typename FallbackGemm, typename... EpilogueArgs>
inline void fallback_cutlass_int4_gemm_caller(
    torch::Tensor& out,
    torch::Tensor const& a, torch::Tensor const& b,
    int64_t M, int64_t N, int64_t K,
    EpilogueArgs&&... args) {
  static const int max_shared_mem_per_block_opt_in =
      get_cuda_max_shared_memory_per_block_opt_in(0);

  size_t const gemm_shared_mem_size =
      sizeof(typename Gemm::KernelType::SharedStorage);
  size_t const fallback_gemm_shared_mem_size =
      sizeof(typename FallbackGemm::KernelType::SharedStorage);

  if (gemm_shared_mem_size <= (size_t)max_shared_mem_per_block_opt_in) {
    return cutlass_int4_gemm_caller<Gemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  } else {
    TORCH_CHECK(fallback_gemm_shared_mem_size <=
                (size_t)max_shared_mem_per_block_opt_in);
    return cutlass_int4_gemm_caller<FallbackGemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// M-range dispatch: select tile configuration based on problem M dimension
// ─────────────────────────────────────────────────────────────────────────────

template <typename OutType, template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_int4_gemm_sm80_dispatch(
    torch::Tensor& out,
    torch::Tensor const& a, torch::Tensor const& b,
    int64_t M, int64_t N, int64_t K,
    EpilogueArgs&&... args) {

  using Cutlass2xGemmDefault =
      typename sm80_int4_config_default<OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM64 =
      typename sm80_int4_config_M64<OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM32 =
      typename sm80_int4_config_M32<OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM16 =
      typename sm80_int4_config_M16<OutType, Epilogue>::Cutlass2xGemm;

  // Fallback: M32 config has moderate shared memory requirements
  using FallbackGemm = Cutlass2xGemmM32;

  uint32_t const m = static_cast<uint32_t>(M);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_pow_2(m));

  if (mp2 <= 16) {
    return fallback_cutlass_int4_gemm_caller<Cutlass2xGemmM16, FallbackGemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 32) {
    return fallback_cutlass_int4_gemm_caller<Cutlass2xGemmM32, FallbackGemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 64) {
    return fallback_cutlass_int4_gemm_caller<Cutlass2xGemmM64, FallbackGemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    uint32_t const n = static_cast<uint32_t>(N);
    bool const small_n = n < 8192;
    if (small_n) {
      return fallback_cutlass_int4_gemm_caller<Cutlass2xGemmM64, FallbackGemm>(
          out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
    } else {
      return fallback_cutlass_int4_gemm_caller<Cutlass2xGemmDefault, FallbackGemm>(
          out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
    }
  } else {
    return fallback_cutlass_int4_gemm_caller<Cutlass2xGemmDefault, FallbackGemm>(
        out, a, b, M, N, K, std::forward<EpilogueArgs>(args)...);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_gemm (retained for testing/debugging)
//
// Raw INT4xINT4 GEMM -> INT32 output (no scales)
// Uses the simpler device::Gemm API without EVT.
// ─────────────────────────────────────────────────────────────────────────────

using ElementA = cutlass::int4b_t;
using ElementB = cutlass::int4b_t;
using ElementAccumulator = int32_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using EpilogueOp_INT32 = cutlass::epilogue::thread::LinearCombinationClamp<
    int32_t, 128 / cutlass::sizeof_bits<int32_t>::value,
    int32_t, float>;

using Gemm_Large_INT32 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB,
    int32_t, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp_INT32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3, kAlignmentA, kAlignmentB, false,
    cutlass::arch::OpMultiplyAddSaturate>;

using Gemm_Medium_INT32 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB,
    int32_t, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp_INT32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3, kAlignmentA, kAlignmentB, false,
    cutlass::arch::OpMultiplyAddSaturate>;

using Gemm_Small_INT32 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB,
    int32_t, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp_INT32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3, kAlignmentA, kAlignmentB, false,
    cutlass::arch::OpMultiplyAddSaturate>;

template <typename GemmOp, typename ElementOut>
static void launch_gemm(
    const void* a_ptr, const void* b_ptr, void* c_ptr,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename GemmOp::Arguments args(
        problem_size,
        {static_cast<const ElementA*>(a_ptr), K},
        {static_cast<const ElementB*>(b_ptr), K},
        {static_cast<const ElementOut*>(c_ptr), N},
        {static_cast<ElementOut*>(c_ptr), N},
        {alpha, beta}
    );

    GemmOp gemm_op;

    size_t smem_size = sizeof(typename GemmOp::GemmKernel::SharedStorage);
    if (smem_size >= (48 << 10)) {
        cudaFuncSetAttribute(
            cutlass::Kernel<typename GemmOp::GemmKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
    }

    cutlass::Status status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS INT4 GEMM cannot implement: ",
                cutlass::cutlassGetStatusString(status));

    status = gemm_op.initialize(args, nullptr, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS INT4 GEMM init failed: ",
                cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS INT4 GEMM run failed: ",
                cutlass::cutlassGetStatusString(status));
}

torch::Tensor cutlass_int4_gemm(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    int64_t M, int64_t N, int64_t K)
{
    TORCH_CHECK(a_packed.is_cuda(), "a_packed must be CUDA tensor");
    TORCH_CHECK(b_packed.is_cuda(), "b_packed must be CUDA tensor");
    TORCH_CHECK(a_packed.dtype() == torch::kUInt8, "a_packed must be uint8");
    TORCH_CHECK(b_packed.dtype() == torch::kUInt8, "b_packed must be uint8");
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64 for INT4 alignment");

    auto out = torch::empty({M, N}, torch::TensorOptions()
        .dtype(torch::kInt32).device(a_packed.device()));

    auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device());
    static const int max_smem = get_cuda_max_shared_memory_per_block_opt_in(0);

    if (M <= 16) {
        launch_gemm<Gemm_Small_INT32, int32_t>(
            a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
            M, N, K, 1.0f, 0.0f, stream);
    } else if (M <= 256 ||
               sizeof(typename Gemm_Large_INT32::GemmKernel::SharedStorage) > (size_t)max_smem) {
        launch_gemm<Gemm_Medium_INT32, int32_t>(
            a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
            M, N, K, 1.0f, 0.0f, stream);
    } else {
        launch_gemm<Gemm_Large_INT32, int32_t>(
            a_packed.data_ptr(), b_packed.data_ptr(), out.data_ptr(),
            M, N, K, 1.0f, 0.0f, stream);
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_scaled_mm
//
// Computes: out[m,n] = (sum_k A_int4[m,k] * B_int4[n,k]) * scale_a[m] * scale_b[n]
//
// Scale application is FUSED into the GEMM epilogue via EVT.
// Output is directly in BF16 — no separate dtype conversion needed.
//
// Inputs:
//   a_packed: [M, K/2] uint8
//   b_packed: [N, K/2] uint8
//   scale_a:  [M] float  (per-row activation scale)
//   scale_b:  [N] float  (per-channel weight scale)
//   out:      [M, N] bf16 (pre-allocated output)
//   M, N, K
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_int4_scaled_mm(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t K)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64 for INT4 alignment");
    TORCH_CHECK(a_packed.dtype() == torch::kUInt8, "a_packed must be uint8");
    TORCH_CHECK(b_packed.dtype() == torch::kUInt8, "b_packed must be uint8");

    if (out.dtype() == torch::kBFloat16) {
        cutlass_int4_gemm_sm80_dispatch<cutlass::bfloat16_t, ScaledEpilogue>(
            out, a_packed, b_packed, M, N, K, scale_a, scale_b);
    } else if (out.dtype() == torch::kFloat16) {
        cutlass_int4_gemm_sm80_dispatch<cutlass::half_t, ScaledEpilogue>(
            out, a_packed, b_packed, M, N, K, scale_a, scale_b);
    } else {
        TORCH_CHECK(false, "cutlass_int4_scaled_mm: unsupported output dtype, "
                    "must be bfloat16 or float16");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_w4a8_scaled_mm
//
// W4A8: INT8 activation × INT4 weight → BF16/FP16 output with fused scaling
//
// Computes: out[m,n] = (sum_k A_int8[m,k] * B_int4[n,k]) * scale_a[m] * scale_b[n]
//
// Inputs:
//   a:         [M, K] int8   (activation, per-row INT8 quantized)
//   b_packed:  [N, K/2] uint8 (weight, packed INT4)
//   scale_a:   [M] float     (per-row activation scale)
//   scale_b:   [N] float     (per-channel weight scale)
//   out:       [M, N] bf16/fp16 (pre-allocated output)
//   M, N, K
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_w4a8_scaled_mm(
    torch::Tensor a,
    torch::Tensor b_int8,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t K)
{
    TORCH_CHECK(a.is_cuda() && b_int8.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for INT8 alignment");
    TORCH_CHECK(a.dtype() == torch::kInt8, "a must be int8");
    TORCH_CHECK(b_int8.dtype() == torch::kInt8, "b_int8 must be int8");

    if (out.dtype() == torch::kBFloat16) {
        cutlass_int8_gemm_sm80_dispatch<cutlass::bfloat16_t, ScaledEpilogue>(
            out, a, b_int8, M, N, K, scale_a, scale_b);
    } else if (out.dtype() == torch::kFloat16) {
        cutlass_int8_gemm_sm80_dispatch<cutlass::half_t, ScaledEpilogue>(
            out, a, b_int8, M, N, K, scale_a, scale_b);
    } else {
        TORCH_CHECK(false, "cutlass_w4a8_scaled_mm: unsupported output dtype, "
                    "must be bfloat16 or float16");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-group GEMM: C++ level loop over groups with CUTLASS per-group GEMM
//
// Instead of Python-level loops, the group iteration happens in C++ with
// direct calls to CUTLASS dispatch functions. This eliminates Python overhead
// while maintaining CUTLASS Tensor Core MMA for each group's computation.
//
// For K=2560, group_size=128: 20 CUTLASS kernel launches, each with K=128.
// Results are accumulated in FP32 for numerical stability.
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// cutlass_int4_scaled_mm_grouped
//
// Per-group INT4×INT4 GEMM with FP32 accumulation.
//
// Computes: out[m,n] = sum_g (sum_{k in g} a_int4[m,k] * b_int4[n,k])
//                      * scale_a[g,m] * scale_b[g,n]
//
// Inputs:
//   a_packed: [num_groups, M, gs/2] uint8
//   b_packed: [num_groups, N, gs/2] uint8
//   scale_a:  [num_groups, M] float
//   scale_b:  [num_groups, N] float
//   out:      [M, N] bf16/fp16
//   M, N, group_size, num_groups
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_int4_scaled_mm_grouped(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size % 64 == 0, "group_size must be multiple of 64");
    TORCH_CHECK(a_packed.dim() == 3 && a_packed.size(0) == num_groups,
                "a_packed must be [num_groups, M, gs/2]");
    TORCH_CHECK(b_packed.dim() == 3 && b_packed.size(0) == num_groups,
                "b_packed must be [num_groups, N, gs/2]");

    auto device = a_packed.device();
    bool is_bf16 = (out.dtype() == torch::kBFloat16);

    // FP32 accumulator for numerical stability across groups
    auto acc = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // BF16/FP16 temporary for each group's CUTLASS output
    auto temp = torch::empty({M, N},
        torch::TensorOptions().dtype(out.dtype()).device(device));

    for (int64_t g = 0; g < num_groups; g++) {
        // Tensor views for group g (contiguous since dim-0 select on contiguous)
        auto a_g = a_packed.select(0, g);   // [M, gs/2]
        auto b_g = b_packed.select(0, g);   // [N, gs/2]
        auto sa_g = scale_a.select(0, g);   // [M]
        auto sb_g = scale_b.select(0, g);   // [N]

        // CUTLASS INT4×INT4 GEMM for this group
        if (is_bf16) {
            cutlass_int4_gemm_sm80_dispatch<cutlass::bfloat16_t, ScaledEpilogue>(
                temp, a_g, b_g, M, N, group_size, sa_g, sb_g);
        } else {
            cutlass_int4_gemm_sm80_dispatch<cutlass::half_t, ScaledEpilogue>(
                temp, a_g, b_g, M, N, group_size, sa_g, sb_g);
        }

        // Accumulate in FP32 (PyTorch handles bf16→fp32 promotion)
        acc.add_(temp);
    }

    // Convert FP32 accumulator → output dtype
    out.copy_(acc.to(out.dtype()));
}

// ─────────────────────────────────────────────────────────────────────────────
// cutlass_int4_scaled_mm_azp_grouped
//
// Per-group INT4×INT4 GEMM with FP32 accumulation and AZP correction for
// asymmetric activation quantization.
//
// Same as cutlass_int4_scaled_mm_grouped but with zero-point correction:
//   Y_correct[m,n] = Y_gemm[m,n] + scale_a[g,m] * scale_b[g,n]
//                    * azp_adj[g,m] * w_col_sum[g,n]
//
// where azp_adj = (8 - zero_point) and w_col_sum = sum of INT4 weight values.
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_int4_scaled_mm_azp_grouped(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    torch::Tensor azp_adj,
    torch::Tensor w_col_sum,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size % 64 == 0, "group_size must be multiple of 64");
    TORCH_CHECK(a_packed.dim() == 3 && a_packed.size(0) == num_groups,
                "a_packed must be [num_groups, M, gs/2]");
    TORCH_CHECK(b_packed.dim() == 3 && b_packed.size(0) == num_groups,
                "b_packed must be [num_groups, N, gs/2]");
    TORCH_CHECK(azp_adj.dim() == 2 && azp_adj.size(0) == num_groups,
                "azp_adj must be [num_groups, M]");
    TORCH_CHECK(w_col_sum.dim() == 2 && w_col_sum.size(0) == num_groups,
                "w_col_sum must be [num_groups, N]");

    auto device = a_packed.device();
    bool is_bf16 = (out.dtype() == torch::kBFloat16);

    // FP32 accumulator
    auto acc = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    auto temp = torch::empty({M, N},
        torch::TensorOptions().dtype(out.dtype()).device(device));

    for (int64_t g = 0; g < num_groups; g++) {
        auto a_g = a_packed.select(0, g);   // [M, gs/2]
        auto b_g = b_packed.select(0, g);   // [N, gs/2]
        auto sa_g = scale_a.select(0, g);   // [M]
        auto sb_g = scale_b.select(0, g);   // [N]

        // CUTLASS INT4×INT4 GEMM
        if (is_bf16) {
            cutlass_int4_gemm_sm80_dispatch<cutlass::bfloat16_t, ScaledEpilogue>(
                temp, a_g, b_g, M, N, group_size, sa_g, sb_g);
        } else {
            cutlass_int4_gemm_sm80_dispatch<cutlass::half_t, ScaledEpilogue>(
                temp, a_g, b_g, M, N, group_size, sa_g, sb_g);
        }

        acc.add_(temp);

        // AZP correction: Y += scale_a * scale_b * azp_adj * w_col_sum
        // = outer(azp_adj * scale_a, w_col_sum * scale_b)
        auto azp_g = azp_adj.select(0, g);       // [M]
        auto wcs_g = w_col_sum.select(0, g);     // [N]
        auto u = (azp_g * sa_g).unsqueeze(1);    // [M, 1]
        auto v = (wcs_g * sb_g).unsqueeze(0);    // [1, N]
        acc.add_(u * v);
    }

    out.copy_(acc.to(out.dtype()));
}

// ─────────────────────────────────────────────────────────────────────────────
// cutlass_w4a8_scaled_mm_grouped
//
// Per-group INT8×INT8 GEMM with FP32 accumulation and optional AZP correction.
//
// Computes (symmetric):
//   out[m,n] = sum_g (sum_{k in g} a_int8[m,k] * b_int8[n,k])
//              * scale_a[g,m] * scale_b[g,n]
//
// Computes (asymmetric, when azp and w_col_sum provided):
//   out[m,n] = sum_g [(sum_{k in g} a_int8[m,k] * b_int8[n,k])
//              * scale_a[g,m] * scale_b[g,n]
//              - azp[g,m] * scale_a[g,m] * w_col_sum[g,n]]
//
// Inputs:
//   a_int8:     [num_groups, M, gs] int8
//   b_int8:     [num_groups, N, gs] int8
//   scale_a:    [num_groups, M] float
//   scale_b:    [num_groups, N] float
//   out:        [M, N] bf16/fp16
//   azp:        [num_groups, M] float (empty tensor if symmetric)
//   w_col_sum:  [num_groups, N] float (empty tensor if symmetric)
//   M, N, group_size, num_groups
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_w4a8_scaled_mm_grouped(
    torch::Tensor a_int8,
    torch::Tensor b_int8,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    torch::Tensor azp,
    torch::Tensor w_col_sum,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_int8.is_cuda() && b_int8.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size % 32 == 0, "group_size must be multiple of 32");
    TORCH_CHECK(a_int8.dim() == 3 && a_int8.size(0) == num_groups,
                "a_int8 must be [num_groups, M, gs]");
    TORCH_CHECK(b_int8.dim() == 3 && b_int8.size(0) == num_groups,
                "b_int8 must be [num_groups, N, gs]");
    TORCH_CHECK(a_int8.dtype() == torch::kInt8, "a_int8 must be int8");
    TORCH_CHECK(b_int8.dtype() == torch::kInt8, "b_int8 must be int8");

    auto device = a_int8.device();
    bool is_bf16 = (out.dtype() == torch::kBFloat16);
    bool use_asymmetric = (azp.numel() > 0 && w_col_sum.numel() > 0);

    // FP32 accumulator
    auto acc = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Temporary for each group's CUTLASS output
    auto temp = torch::empty({M, N},
        torch::TensorOptions().dtype(out.dtype()).device(device));

    for (int64_t g = 0; g < num_groups; g++) {
        auto a_g = a_int8.select(0, g);     // [M, gs]
        auto b_g = b_int8.select(0, g);     // [N, gs]
        auto sa_g = scale_a.select(0, g);   // [M]
        auto sb_g = scale_b.select(0, g);   // [N]

        // INT8×INT8 CUTLASS GEMM for this group
        if (is_bf16) {
            cutlass_int8_gemm_sm80_dispatch<cutlass::bfloat16_t, ScaledEpilogue>(
                temp, a_g, b_g, M, N, group_size, sa_g, sb_g);
        } else {
            cutlass_int8_gemm_sm80_dispatch<cutlass::half_t, ScaledEpilogue>(
                temp, a_g, b_g, M, N, group_size, sa_g, sb_g);
        }

        // Accumulate GEMM result
        acc.add_(temp);

        // AZP correction for asymmetric quantization
        if (use_asymmetric) {
            auto azp_g = azp.select(0, g);         // [M]
            auto wcs_g = w_col_sum.select(0, g);   // [N]

            // correction[m,n] = azp[g,m] * scale_a[g,m] * w_col_sum[g,n]
            auto azp_scaled = (azp_g * sa_g).unsqueeze(1);  // [M, 1]
            auto correction = azp_scaled * wcs_g.unsqueeze(0);  // [M, N]
            acc.sub_(correction);
        }
    }

    // Convert FP32 → output dtype
    out.copy_(acc.to(out.dtype()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-group W4A16: INT4 weight dequant → BF16/FP16 GEMM
//
// For post-activation layers (o_proj, down_proj) where activation stays in
// BF16/FP16 (no activation quantization). Weight is per-group INT4 quantized
// for 4x memory reduction.
//
// Per group:
//   1. Dequant INT4 weight → BF16: w_bf16[N, gs] = unpack(w_packed[N, gs/2]) * scale[N]
//   2. BF16 GEMM: partial[M, N] = x[M, gs] @ w_bf16[N, gs].T
//   3. Accumulate partial results in FP32
//
// This preserves full BF16 activation precision while still benefiting from
// INT4 weight storage. Only weight quantization error is introduced.
//
// Inputs:
//   x:        [M, K] bf16/fp16 (raw activation, NOT quantized)
//   w_packed: [num_groups, N, gs/2] uint8 (per-group INT4 packed weight)
//   w_scale:  [num_groups, N] float (per-group weight scales)
//   out:      [M, N] bf16/fp16 (pre-allocated output)
//   M, N, group_size, num_groups
// ─────────────────────────────────────────────────────────────────────────────

// Dequant functions defined in int4_quant.cu
extern torch::Tensor dequant_int4_to_bf16(
    torch::Tensor packed, torch::Tensor scale, int64_t K);
extern torch::Tensor dequant_int4_to_fp16(
    torch::Tensor packed, torch::Tensor scale, int64_t K);

void cutlass_w4a16_mm_grouped(
    torch::Tensor x,
    torch::Tensor w_packed,
    torch::Tensor w_scale,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(w_scale.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");
    TORCH_CHECK(w_packed.dim() == 3 && w_packed.size(0) == num_groups,
                "w_packed must be [num_groups, N, gs/2]");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == M,
                "x must be [M, K]");

    auto device = x.device();
    bool is_bf16 = (out.dtype() == torch::kBFloat16);

    // FP32 accumulator for numerical stability across groups
    auto acc = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    for (int64_t g = 0; g < num_groups; g++) {
        // 1. Extract activation slice: x[:, g*gs:(g+1)*gs] → [M, gs]
        auto x_g = x.narrow(1, g * group_size, group_size).contiguous();

        // 2. Dequant weight for this group: [N, gs/2] → [N, gs] bf16/fp16
        auto w_packed_g = w_packed.select(0, g);   // [N, gs/2]
        auto w_scale_g = w_scale.select(0, g);     // [N]

        torch::Tensor w_dequant_g;
        if (is_bf16) {
            w_dequant_g = dequant_int4_to_bf16(w_packed_g, w_scale_g, group_size);
        } else {
            w_dequant_g = dequant_int4_to_fp16(w_packed_g, w_scale_g, group_size);
        }
        // w_dequant_g: [N, gs] bf16/fp16

        // 3. BF16/FP16 GEMM: [M, gs] @ [gs, N] → [M, N]
        auto partial = torch::matmul(x_g, w_dequant_g.t());

        // 4. Accumulate in FP32
        acc.add_(partial);
    }

    // Convert FP32 → output dtype
    out.copy_(acc.to(out.dtype()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused Grouped INT4×INT4 GEMM Kernel
//
// Fuses the entire group loop (num_groups CUTLASS launches + num_groups add_)
// into a single CUDA kernel launch. Each thread block computes a TILE_M×TILE_N
// output tile, iterating over all groups internally with FP32 register
// accumulation.
//
// Uses MMA PTX instruction: mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
// For K=128 (group_size), each group requires 2 MMA iterations (128/64=2).
//
// Thread block: 128 threads = 4 warps
//   Warp layout: 2×2 (2 in M direction, 2 in N direction)
//   Each warp computes 16×16 output (2 MMAs in N: 16×8 × 2)
//   Thread block tile: 32×32
//
// Shared memory per group iteration:
//   x tile: TILE_M × (gs/2) bytes for packed INT4
//   w tile: TILE_N × (gs/2) bytes for packed INT4
//   scales: (TILE_M + TILE_N) × sizeof(float)
//
// Performance: 1 kernel launch instead of 40 (20 GEMM + 20 add_)
// ─────────────────────────────────────────────────────────────────────────────

// Template constants
static constexpr int FUSED_TILE_M = 32;
static constexpr int FUSED_TILE_N = 32;
static constexpr int FUSED_WARPS = 4;
static constexpr int FUSED_THREADS = FUSED_WARPS * 32;
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 64;

// ─── Shared memory staging for coalesced loads ───
// Each group iteration loads x_tile[TILE_M][gs/2] and w_tile[TILE_N][gs/2]
// into shared memory for proper MMA fragment loading.

// ─── Fused kernel: symmetric (no AZP) ───

__global__ void __launch_bounds__(128)
fused_int4_grouped_gemm_kernel(
    const uint8_t* __restrict__ x_packed,   // [num_groups, M, gs/2]
    const uint8_t* __restrict__ w_packed,   // [num_groups, N, gs/2]
    const float* __restrict__ scale_x,      // [num_groups, M]
    const float* __restrict__ scale_w,      // [num_groups, N]
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const int M, const int N,
    const int num_groups, const int gs)
{
    // Thread block produces TILE_M × TILE_N output tile
    const int tile_m = blockIdx.x * FUSED_TILE_M;
    const int tile_n = blockIdx.y * FUSED_TILE_N;
    const int tid = threadIdx.x;

    // Warp ID and lane ID
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Warp layout: 2×2, each warp handles 16×16 output
    const int warp_m = warp_id / 2;  // 0 or 1
    const int warp_n = warp_id % 2;  // 0 or 1
    const int warp_row_start = warp_m * MMA_M;
    const int warp_col_start = warp_n * MMA_N * 2;  // 2 MMAs in N direction

    // FP32 accumulator (across all groups)
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    const int gs_half = gs / 2;
    const int mma_k_iters = gs / MMA_K;

    // Dynamic shared memory for A and B tiles + scales
    // Layout: [TILE_M * gs_half] for A | [TILE_N * gs_half] for B | scales
    extern __shared__ uint8_t smem[];
    uint8_t* smem_a = smem;
    uint8_t* smem_b = smem + FUSED_TILE_M * gs_half;
    float* smem_sx = reinterpret_cast<float*>(smem_b + FUSED_TILE_N * gs_half);
    float* smem_sw = smem_sx + FUSED_TILE_M;

    // ─── Group loop ───
    for (int g = 0; g < num_groups; g++) {
        const uint8_t* x_g = x_packed + (int64_t)g * M * gs_half;
        const uint8_t* w_g = w_packed + (int64_t)g * N * gs_half;
        const float* sx_g = scale_x + (int64_t)g * M;
        const float* sw_g = scale_w + (int64_t)g * N;

        // ─── Cooperative load of A tile [TILE_M, gs/2] into shared memory ───
        {
            int total_bytes_a = FUSED_TILE_M * gs_half;
            // Use uint32 loads for efficiency (4 bytes at a time)
            int total_words_a = total_bytes_a / 4;
            for (int i = tid; i < total_words_a; i += FUSED_THREADS) {
                int row = (i * 4) / gs_half;
                int col = (i * 4) % gs_half;
                int global_row = tile_m + row;
                uint32_t val = 0;
                if (global_row < M) {
                    val = *reinterpret_cast<const uint32_t*>(
                        x_g + (int64_t)global_row * gs_half + col);
                }
                *reinterpret_cast<uint32_t*>(smem_a + row * gs_half + col) = val;
            }
        }

        // ─── Cooperative load of B tile [TILE_N, gs/2] into shared memory ───
        {
            int total_bytes_b = FUSED_TILE_N * gs_half;
            int total_words_b = total_bytes_b / 4;
            for (int i = tid; i < total_words_b; i += FUSED_THREADS) {
                int row = (i * 4) / gs_half;
                int col = (i * 4) % gs_half;
                int global_row = tile_n + row;
                uint32_t val = 0;
                if (global_row < N) {
                    val = *reinterpret_cast<const uint32_t*>(
                        w_g + (int64_t)global_row * gs_half + col);
                }
                *reinterpret_cast<uint32_t*>(smem_b + row * gs_half + col) = val;
            }
        }

        // ─── Load scales into shared memory ───
        if (tid < FUSED_TILE_M) {
            int global_row = tile_m + tid;
            smem_sx[tid] = (global_row < M) ? sx_g[global_row] : 0.f;
        }
        if (tid < FUSED_TILE_N) {
            int global_col = tile_n + tid;
            smem_sw[tid] = (global_col < N) ? sw_g[global_col] : 0.f;
        }

        __syncthreads();

        // ─── MMA computation from shared memory ───
        int32_t mma_acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

        for (int ki = 0; ki < mma_k_iters; ki++) {
            int k_byte_offset = ki * (MMA_K / 2);  // 32 bytes per K iteration

            // Load A fragment from shared memory
            // MMA m16n8k64 fragment layout (empirically verified):
            //   Lane L: row R = L/4 (0-7), K-group G = L%4 (0-3)
            //   reg[0] = A_u32[R       * (gs_half/4) + G*2 + 0]  (K = G*16+0..7)
            //   reg[1] = A_u32[(R + 8) * (gs_half/4) + G*2 + 0]  (K = G*16+0..7)
            //   reg[2] = A_u32[R       * (gs_half/4) + G*2 + 1]  (K = G*16+8..15)
            //   reg[3] = A_u32[(R + 8) * (gs_half/4) + G*2 + 1]  (K = G*16+8..15)
            uint32_t a_frag[4];
            {
                int a_R = lane_id / 4;   // row within MMA tile (0-7)
                int a_G = lane_id % 4;   // K-group (0-3)
                int a_row0 = warp_row_start + a_R;
                int a_row1 = warp_row_start + a_R + 8;
                int a_u32_cols = gs_half / 4;  // uint32 elements per row
                int k_u32_offset = k_byte_offset / 4;
                const uint32_t* a_u32 = reinterpret_cast<const uint32_t*>(smem_a);
                a_frag[0] = a_u32[a_row0 * a_u32_cols + k_u32_offset + a_G * 2 + 0];
                a_frag[1] = a_u32[a_row1 * a_u32_cols + k_u32_offset + a_G * 2 + 0];
                a_frag[2] = a_u32[a_row0 * a_u32_cols + k_u32_offset + a_G * 2 + 1];
                a_frag[3] = a_u32[a_row1 * a_u32_cols + k_u32_offset + a_G * 2 + 1];
            }

            // Two MMA operations in N direction (16×8 × 2 = 16×16)
            for (int ni = 0; ni < 2; ni++) {
                // Load B fragment from shared memory
                // MMA m16n8k64 fragment layout (empirically verified):
                //   Lane L: col C = L/4 (0-7), K-group G = L%4 (0-3)
                //   reg[0] = B_u32[C * (gs_half/4) + G*2 + 0]  (K = G*16+0..7)
                //   reg[1] = B_u32[C * (gs_half/4) + G*2 + 1]  (K = G*16+8..15)
                uint32_t b_frag[2];
                {
                    int b_C = lane_id / 4;   // col within B (0-7)
                    int b_G = lane_id % 4;   // K-group (0-3)
                    int b_smem_row = warp_col_start + ni * MMA_N + b_C;
                    int b_u32_cols = gs_half / 4;
                    int k_u32_offset_b = k_byte_offset / 4;
                    const uint32_t* b_u32 = reinterpret_cast<const uint32_t*>(smem_b);
                    b_frag[0] = b_u32[b_smem_row * b_u32_cols + k_u32_offset_b + b_G * 2 + 0];
                    b_frag[1] = b_u32[b_smem_row * b_u32_cols + k_u32_offset_b + b_G * 2 + 1];
                }

                // Execute MMA
#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
                asm volatile(
                    "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=r"(mma_acc[ni][0]), "=r"(mma_acc[ni][1]),
                      "=r"(mma_acc[ni][2]), "=r"(mma_acc[ni][3])
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]),
                      "r"(mma_acc[ni][0]), "r"(mma_acc[ni][1]),
                      "r"(mma_acc[ni][2]), "r"(mma_acc[ni][3]));
#endif
            }
        } // end K-loop

        // Apply per-group scales and accumulate into FP32
        // MMA output mapping (m16n8k64, C/D matrix):
        //   Thread t, fragment element fi:
        //     D[0] → row = t/4,     col = (t%4)*2
        //     D[1] → row = t/4,     col = (t%4)*2 + 1
        //     D[2] → row = t/4 + 8, col = (t%4)*2
        //     D[3] → row = t/4 + 8, col = (t%4)*2 + 1
        for (int ni = 0; ni < 2; ni++) {
            for (int fi = 0; fi < 4; fi++) {
                int local_m = warp_row_start + (lane_id / 4) + (fi >= 2 ? 8 : 0);
                int local_n = warp_col_start + ni * MMA_N + (lane_id % 4) * 2 + (fi % 2);

                float sx = smem_sx[local_m];
                float sw = smem_sw[local_n];
                acc[ni][fi] += static_cast<float>(mma_acc[ni][fi]) * sx * sw;
            }
        }

        __syncthreads();  // Before next group's shared memory loads
    } // end group loop

    // Write FP32 → BF16 output
    for (int ni = 0; ni < 2; ni++) {
        for (int fi = 0; fi < 4; fi++) {
            int local_m = warp_row_start + (lane_id / 4) + (fi >= 2 ? 8 : 0);
            int local_n = warp_col_start + ni * MMA_N + (lane_id % 4) * 2 + (fi % 2);

            int global_m = tile_m + local_m;
            int global_n = tile_n + local_n;

            if (global_m < M && global_n < N) {
                out[global_m * N + global_n] = __float2bfloat16(acc[ni][fi]);
            }
        }
    }
}

// ─── Fused kernel: asymmetric with AZP correction ───

__global__ void __launch_bounds__(128)
fused_int4_grouped_gemm_azp_kernel(
    const uint8_t* __restrict__ x_packed,   // [num_groups, M, gs/2]
    const uint8_t* __restrict__ w_packed,   // [num_groups, N, gs/2]
    const float* __restrict__ scale_x,      // [num_groups, M]
    const float* __restrict__ scale_w,      // [num_groups, N]
    const float* __restrict__ azp_adj,      // [num_groups, M]
    const float* __restrict__ w_col_sum,    // [num_groups, N]
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const int M, const int N,
    const int num_groups, const int gs)
{
    const int tile_m = blockIdx.x * FUSED_TILE_M;
    const int tile_n = blockIdx.y * FUSED_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int warp_row_start = warp_m * MMA_M;
    const int warp_col_start = warp_n * MMA_N * 2;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    const int gs_half = gs / 2;
    const int mma_k_iters = gs / MMA_K;

    // Shared memory layout: A tile | B tile | scale_x | scale_w | azp | wcs
    extern __shared__ uint8_t smem_azp[];
    uint8_t* smem_a = smem_azp;
    uint8_t* smem_b = smem_a + FUSED_TILE_M * gs_half;
    float* smem_sx = reinterpret_cast<float*>(smem_b + FUSED_TILE_N * gs_half);
    float* smem_sw = smem_sx + FUSED_TILE_M;
    float* smem_azp_v = smem_sw + FUSED_TILE_N;
    float* smem_wcs = smem_azp_v + FUSED_TILE_M;

    for (int g = 0; g < num_groups; g++) {
        const uint8_t* x_g = x_packed + (int64_t)g * M * gs_half;
        const uint8_t* w_g = w_packed + (int64_t)g * N * gs_half;
        const float* sx_g = scale_x + (int64_t)g * M;
        const float* sw_g = scale_w + (int64_t)g * N;
        const float* azp_g = azp_adj + (int64_t)g * M;
        const float* wcs_g = w_col_sum + (int64_t)g * N;

        // Cooperative load A tile
        {
            int total_words = (FUSED_TILE_M * gs_half) / 4;
            for (int i = tid; i < total_words; i += FUSED_THREADS) {
                int row = (i * 4) / gs_half;
                int col = (i * 4) % gs_half;
                int global_row = tile_m + row;
                uint32_t val = 0;
                if (global_row < M) {
                    val = *reinterpret_cast<const uint32_t*>(
                        x_g + (int64_t)global_row * gs_half + col);
                }
                *reinterpret_cast<uint32_t*>(smem_a + row * gs_half + col) = val;
            }
        }

        // Cooperative load B tile
        {
            int total_words = (FUSED_TILE_N * gs_half) / 4;
            for (int i = tid; i < total_words; i += FUSED_THREADS) {
                int row = (i * 4) / gs_half;
                int col = (i * 4) % gs_half;
                int global_row = tile_n + row;
                uint32_t val = 0;
                if (global_row < N) {
                    val = *reinterpret_cast<const uint32_t*>(
                        w_g + (int64_t)global_row * gs_half + col);
                }
                *reinterpret_cast<uint32_t*>(smem_b + row * gs_half + col) = val;
            }
        }

        // Load scales and AZP data
        if (tid < FUSED_TILE_M) {
            int r = tile_m + tid;
            smem_sx[tid] = (r < M) ? sx_g[r] : 0.f;
            smem_azp_v[tid] = (r < M) ? azp_g[r] : 0.f;
        }
        if (tid < FUSED_TILE_N) {
            int c = tile_n + tid;
            smem_sw[tid] = (c < N) ? sw_g[c] : 0.f;
            smem_wcs[tid] = (c < N) ? wcs_g[c] : 0.f;
        }

        __syncthreads();

        // MMA computation
        int32_t mma_acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

        for (int ki = 0; ki < mma_k_iters; ki++) {
            int k_byte_offset = ki * (MMA_K / 2);

            // A fragment: Lane L → row R=L/4, K-group G=L%4
            uint32_t a_frag[4];
            {
                int a_R = lane_id / 4;
                int a_G = lane_id % 4;
                int a_row0 = warp_row_start + a_R;
                int a_row1 = warp_row_start + a_R + 8;
                int a_u32_cols = gs_half / 4;
                int k_u32_off = k_byte_offset / 4;
                const uint32_t* a_u32 = reinterpret_cast<const uint32_t*>(smem_a);
                a_frag[0] = a_u32[a_row0 * a_u32_cols + k_u32_off + a_G * 2 + 0];
                a_frag[1] = a_u32[a_row1 * a_u32_cols + k_u32_off + a_G * 2 + 0];
                a_frag[2] = a_u32[a_row0 * a_u32_cols + k_u32_off + a_G * 2 + 1];
                a_frag[3] = a_u32[a_row1 * a_u32_cols + k_u32_off + a_G * 2 + 1];
            }

            for (int ni = 0; ni < 2; ni++) {
                // B fragment: Lane L → col C=L/4, K-group G=L%4
                uint32_t b_frag[2];
                {
                    int b_C = lane_id / 4;
                    int b_G = lane_id % 4;
                    int b_smem_row = warp_col_start + ni * MMA_N + b_C;
                    int b_u32_cols = gs_half / 4;
                    int k_u32_off = k_byte_offset / 4;
                    const uint32_t* b_u32 = reinterpret_cast<const uint32_t*>(smem_b);
                    b_frag[0] = b_u32[b_smem_row * b_u32_cols + k_u32_off + b_G * 2 + 0];
                    b_frag[1] = b_u32[b_smem_row * b_u32_cols + k_u32_off + b_G * 2 + 1];
                }

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
                asm volatile(
                    "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=r"(mma_acc[ni][0]), "=r"(mma_acc[ni][1]),
                      "=r"(mma_acc[ni][2]), "=r"(mma_acc[ni][3])
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]),
                      "r"(mma_acc[ni][0]), "r"(mma_acc[ni][1]),
                      "r"(mma_acc[ni][2]), "r"(mma_acc[ni][3]));
#endif
            }
        }

        // Apply scales + AZP correction
        for (int ni = 0; ni < 2; ni++) {
            for (int fi = 0; fi < 4; fi++) {
                int local_m = warp_row_start + (lane_id / 4) + (fi >= 2 ? 8 : 0);
                int local_n = warp_col_start + ni * MMA_N + (lane_id % 4) * 2 + (fi % 2);

                float sx = smem_sx[local_m];
                float sw = smem_sw[local_n];
                float gemm_val = static_cast<float>(mma_acc[ni][fi]) * sx * sw;
                float azp_corr = smem_azp_v[local_m] * sx * smem_wcs[local_n] * sw;
                acc[ni][fi] += gemm_val + azp_corr;
            }
        }

        __syncthreads();
    }

    // Write output
    for (int ni = 0; ni < 2; ni++) {
        for (int fi = 0; fi < 4; fi++) {
            int local_m = warp_row_start + (lane_id / 4) + (fi >= 2 ? 8 : 0);
            int local_n = warp_col_start + ni * MMA_N + (lane_id % 4) * 2 + (fi % 2);

            int global_m = tile_m + local_m;
            int global_n = tile_n + local_n;

            if (global_m < M && global_n < N) {
                out[global_m * N + global_n] = __float2bfloat16(acc[ni][fi]);
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// V2 Optimized fused INT4 grouped GEMM kernel
//
// Key optimizations over the V1 32×32 kernel:
// 1. Tile size: 64×128 (8x more output per block → 8x fewer blocks)
// 2. Bank-conflict-free shared memory via 4-byte row padding
// 3. Efficient scale pre-loading and factored AZP computation
//
// Thread block: 128 threads (4 warps), 2×2 warp layout
// Warp tile: 32×64 → 16 MMA (m16n8k64) per K-iteration per warp
// Registers: 64 int32 MMA acc + 64 float FP32 acc + ~24 misc ≈ 152
// ═════════════════════════════════════════════════════════════════════════════

static constexpr int V2_TILE_M = 64;
static constexpr int V2_TILE_N = 128;
static constexpr int V2_THREADS = 128;  // 4 warps

template <bool USE_AZP, int GS = 128>
__global__ void __launch_bounds__(128)
fused_int4_grouped_gemm_v2_kernel(
    const uint8_t* __restrict__ x_packed,   // [num_groups, M, gs/2]
    const uint8_t* __restrict__ w_packed,   // [num_groups, N, gs/2]
    const float* __restrict__ scale_x,      // [num_groups, M]
    const float* __restrict__ scale_w,      // [num_groups, N]
    const float* __restrict__ azp_adj,      // [num_groups, M] (USE_AZP only)
    const float* __restrict__ w_col_sum,    // [num_groups, N] (USE_AZP only)
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const int M, const int N,
    const int num_groups)
{
    // Compile-time constants derived from group_size template parameter
    static constexpr int gs = GS;
    static constexpr int gs_half = GS / 2;
    static constexpr int mma_k_iters = GS / MMA_K;
    static constexpr int A_STRIDE = gs_half + 4;
    static constexpr int B_STRIDE = gs_half + 4;

    const int tile_m = blockIdx.x * V2_TILE_M;
    const int tile_n = blockIdx.y * V2_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // 2×2 warp layout: each warp handles 32×64 output
    const int warp_m = warp_id / 2;  // 0 or 1
    const int warp_n = warp_id % 2;  // 0 or 1
    const int warp_row_start = warp_m * 32;
    const int warp_col_start = warp_n * 64;

    // Shared memory layout
    extern __shared__ uint8_t smem_v2[];
    uint8_t* smem_a = smem_v2;
    uint8_t* smem_b = smem_a + V2_TILE_M * A_STRIDE;
    float* smem_sx = reinterpret_cast<float*>(smem_b + V2_TILE_N * B_STRIDE);
    float* smem_sw = smem_sx + V2_TILE_M;
    // AZP pointers (only meaningful when USE_AZP=true)
    float* smem_azp_val = smem_sw + V2_TILE_N;
    float* smem_wcs_val = smem_azp_val + V2_TILE_M;

    // FP32 accumulator: [2 mma_m_iter][8 mma_n_iter][4 frag] = 64 floats
    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            #pragma unroll
            for (int fi = 0; fi < 4; fi++)
                acc[mi][ni][fi] = 0.f;

    // Lane decomposition for MMA fragments (constant across groups)
    const int frag_row = lane_id / 4;   // 0..7
    const int frag_grp = lane_id % 4;   // 0..3
    static constexpr int a_u32_stride = A_STRIDE / 4;
    static constexpr int b_u32_stride = B_STRIDE / 4;

    // ═══ Group loop ═══
    for (int g = 0; g < num_groups; g++) {
        const uint8_t* x_g = x_packed + (int64_t)g * M * gs_half;
        const uint8_t* w_g = w_packed + (int64_t)g * N * gs_half;
        const float* sx_g = scale_x + (int64_t)g * M;
        const float* sw_g = scale_w + (int64_t)g * N;

        // ── Load A tile [64, gs_half] → padded shared memory ──
        {
            const int words_per_row = gs_half / 4;
            const int total_words = V2_TILE_M * words_per_row;
            for (int i = tid; i < total_words; i += V2_THREADS) {
                int row = i / words_per_row;
                int col_w = i % words_per_row;
                int global_row = tile_m + row;
                uint32_t val = 0;
                if (global_row < M) {
                    val = *reinterpret_cast<const uint32_t*>(
                        x_g + (int64_t)global_row * gs_half + col_w * 4);
                }
                *reinterpret_cast<uint32_t*>(
                    smem_a + row * A_STRIDE + col_w * 4) = val;
            }
        }

        // ── Load B tile [128, gs_half] → padded shared memory ──
        {
            const int words_per_row = gs_half / 4;
            const int total_words = V2_TILE_N * words_per_row;
            for (int i = tid; i < total_words; i += V2_THREADS) {
                int row = i / words_per_row;
                int col_w = i % words_per_row;
                int global_row = tile_n + row;
                uint32_t val = 0;
                if (global_row < N) {
                    val = *reinterpret_cast<const uint32_t*>(
                        w_g + (int64_t)global_row * gs_half + col_w * 4);
                }
                *reinterpret_cast<uint32_t*>(
                    smem_b + row * B_STRIDE + col_w * 4) = val;
            }
        }

        // ── Load scales ──
        for (int i = tid; i < V2_TILE_M; i += V2_THREADS) {
            int r = tile_m + i;
            smem_sx[i] = (r < M) ? sx_g[r] : 0.f;
            if constexpr (USE_AZP) {
                const float* azp_g = azp_adj + (int64_t)g * M;
                smem_azp_val[i] = (r < M) ? azp_g[r] : 0.f;
            }
        }
        for (int i = tid; i < V2_TILE_N; i += V2_THREADS) {
            int c = tile_n + i;
            smem_sw[i] = (c < N) ? sw_g[c] : 0.f;
            if constexpr (USE_AZP) {
                const float* wcs_g = w_col_sum + (int64_t)g * N;
                smem_wcs_val[i] = (c < N) ? wcs_g[c] : 0.f;
            }
        }

        __syncthreads();

        // ── MMA computation ──
        int32_t mma_acc[2][8][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 8; ni++)
                #pragma unroll
                for (int fi = 0; fi < 4; fi++)
                    mma_acc[mi][ni][fi] = 0;

        const uint32_t* a_u32 = reinterpret_cast<const uint32_t*>(smem_a);
        const uint32_t* b_u32 = reinterpret_cast<const uint32_t*>(smem_b);

        for (int ki = 0; ki < mma_k_iters; ki++) {
            const int k_u32_off = ki * (MMA_K / 2) / 4;  // 8 uint32 per K-iter

            // Load A fragments for both M-positions (2 × m16)
            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int row0 = warp_row_start + mi * MMA_M + frag_row;
                int row1 = row0 + 8;
                int base0 = row0 * a_u32_stride + k_u32_off + frag_grp * 2;
                int base1 = row1 * a_u32_stride + k_u32_off + frag_grp * 2;
                a_frag[mi][0] = a_u32[base0];
                a_frag[mi][1] = a_u32[base1];
                a_frag[mi][2] = a_u32[base0 + 1];
                a_frag[mi][3] = a_u32[base1 + 1];
            }

            // MMA for all M×N positions
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int b_row = warp_col_start + ni * MMA_N + frag_row;
                    int b_base = b_row * b_u32_stride + k_u32_off + frag_grp * 2;
                    uint32_t b0 = b_u32[b_base];
                    uint32_t b1 = b_u32[b_base + 1];

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
                    asm volatile(
                        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(mma_acc[mi][ni][0]), "=r"(mma_acc[mi][ni][1]),
                          "=r"(mma_acc[mi][ni][2]), "=r"(mma_acc[mi][ni][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b0), "r"(b1),
                          "r"(mma_acc[mi][ni][0]), "r"(mma_acc[mi][ni][1]),
                          "r"(mma_acc[mi][ni][2]), "r"(mma_acc[mi][ni][3]));
#endif
                }
            }
        } // end K-loop

        // ── Apply per-group scales and accumulate into FP32 ──
        // MMA output: D[0]→(row0,col0), D[1]→(row0,col1), D[2]→(row1,col0), D[3]→(row1,col1)
        // where row0=lane/4, row1=row0+8, col0=(lane%4)*2, col1=col0+1
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m0 = warp_row_start + mi * MMA_M + frag_row;
            int m1 = m0 + 8;
            float sx0 = smem_sx[m0];
            float sx1 = smem_sx[m1];
            float azp0 = 0.f, azp1 = 0.f;
            if constexpr (USE_AZP) {
                azp0 = smem_azp_val[m0];
                azp1 = smem_azp_val[m1];
            }

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                int n0 = warp_col_start + ni * MMA_N + frag_grp * 2;
                float sw0 = smem_sw[n0];
                float sw1 = smem_sw[n0 + 1];

                if constexpr (USE_AZP) {
                    float wcs0 = smem_wcs_val[n0];
                    float wcs1 = smem_wcs_val[n0 + 1];
                    // Factored: sx * sw * (int32 + azp * wcs)
                    acc[mi][ni][0] += (static_cast<float>(mma_acc[mi][ni][0]) + azp0 * wcs0) * sx0 * sw0;
                    acc[mi][ni][1] += (static_cast<float>(mma_acc[mi][ni][1]) + azp0 * wcs1) * sx0 * sw1;
                    acc[mi][ni][2] += (static_cast<float>(mma_acc[mi][ni][2]) + azp1 * wcs0) * sx1 * sw0;
                    acc[mi][ni][3] += (static_cast<float>(mma_acc[mi][ni][3]) + azp1 * wcs1) * sx1 * sw1;
                } else {
                    acc[mi][ni][0] += static_cast<float>(mma_acc[mi][ni][0]) * sx0 * sw0;
                    acc[mi][ni][1] += static_cast<float>(mma_acc[mi][ni][1]) * sx0 * sw1;
                    acc[mi][ni][2] += static_cast<float>(mma_acc[mi][ni][2]) * sx1 * sw0;
                    acc[mi][ni][3] += static_cast<float>(mma_acc[mi][ni][3]) * sx1 * sw1;
                }
            }
        }

        __syncthreads();  // before next group's shared memory loads
    } // end group loop

    // ── Write FP32 → BF16 output ──
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int gm0 = tile_m + warp_row_start + mi * MMA_M + frag_row;
            int gm1 = gm0 + 8;
            int gn0 = tile_n + warp_col_start + ni * MMA_N + frag_grp * 2;
            int gn1 = gn0 + 1;

            if (gm0 < M && gn0 < N) out[gm0 * N + gn0] = __float2bfloat16(acc[mi][ni][0]);
            if (gm0 < M && gn1 < N) out[gm0 * N + gn1] = __float2bfloat16(acc[mi][ni][1]);
            if (gm1 < M && gn0 < N) out[gm1 * N + gn0] = __float2bfloat16(acc[mi][ni][2]);
            if (gm1 < M && gn1 < N) out[gm1 * N + gn1] = __float2bfloat16(acc[mi][ni][3]);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// V3 Double-buffered fused INT4 grouped GEMM kernel with cp.async prefetch
//
// Key optimizations over V2:
// 1. Double-buffered shared memory: 2 sets of A/B/scale tiles
// 2. cp.async for non-blocking GMEM→SMEM prefetch (SM80+)
// 3. Overlapped load (group g+1) with compute (group g)
//
// Same tile size as V2: 64×128, 128 threads (4 warps), 2×2 warp layout
// Same stride as V2: gs_half + 4 (bank-conflict-free)
// SMEM: ~29KB (2 × 14.5KB per buffer) — within 48KB default limit
// ═════════════════════════════════════════════════════════════════════════════

static constexpr int V3_TILE_M = 64;
static constexpr int V3_TILE_N = 128;
static constexpr int V3_THREADS = 128;

template <bool USE_AZP>
__global__ void __launch_bounds__(128)
fused_int4_grouped_gemm_v3_kernel(
    const uint8_t* __restrict__ x_packed,   // [num_groups, M, gs/2]
    const uint8_t* __restrict__ w_packed,   // [num_groups, N, gs/2]
    const float* __restrict__ scale_x,      // [num_groups, M]
    const float* __restrict__ scale_w,      // [num_groups, N]
    const float* __restrict__ azp_adj,      // [num_groups, M] (USE_AZP only)
    const float* __restrict__ w_col_sum,    // [num_groups, N] (USE_AZP only)
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const int M, const int N,
    const int num_groups, const int gs)
{
    const int tile_m = blockIdx.x * V3_TILE_M;
    const int tile_n = blockIdx.y * V3_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int warp_row_start = warp_m * 32;
    const int warp_col_start = warp_n * 64;

    const int gs_half = gs / 2;
    const int mma_k_iters = gs / MMA_K;
    const int frag_row = lane_id / 4;
    const int frag_grp = lane_id % 4;

    // Stride: gs_half + 4 for bank-conflict-free access (same as V2)
    const int A_STRIDE = gs_half + 4;
    const int B_STRIDE = gs_half + 4;

    // ── Double-buffered shared memory layout ──
    // Each buffer: A[64][A_STRIDE] | B[128][B_STRIDE] | sx[64] | sw[128]
    //             (+ azp[64] | wcs[128] if USE_AZP)
    const int scale_floats = V3_TILE_M + V3_TILE_N;
    const int azp_floats = USE_AZP ? (V3_TILE_M + V3_TILE_N) : 0;
    const int buf_bytes = V3_TILE_M * A_STRIDE + V3_TILE_N * B_STRIDE
                         + (scale_floats + azp_floats) * (int)sizeof(float);

    extern __shared__ uint8_t smem_v3[];

    // FP32 accumulator (persists across all groups)
    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            #pragma unroll
            for (int fi = 0; fi < 4; fi++)
                acc[mi][ni][fi] = 0.f;

    // ── Helper: issue cp.async loads for group g into buffer buf_idx ──
    // Tiles are loaded with 4-byte cp.async (preserves stride alignment)
    auto issue_loads = [&](int g, int buf_idx) __attribute__((always_inline)) {
        uint8_t* base = smem_v3 + buf_idx * buf_bytes;
        uint8_t* sa = base;
        uint8_t* sb = sa + V3_TILE_M * A_STRIDE;
        float* ssx = reinterpret_cast<float*>(sb + V3_TILE_N * B_STRIDE);
        float* ssw = ssx + V3_TILE_M;
        float* sazp = ssw + V3_TILE_N;
        float* swcs = sazp + V3_TILE_M;

        const uint8_t* x_g = x_packed + (int64_t)g * M * gs_half;
        const uint8_t* w_g = w_packed + (int64_t)g * N * gs_half;
        const float* sx_g = scale_x + (int64_t)g * M;
        const float* sw_g = scale_w + (int64_t)g * N;

        // Load A tile [64, gs_half] → padded SMEM via 4-byte cp.async
        {
            const int a_words = gs_half / 4;
            const int total_a = V3_TILE_M * a_words;
            for (int i = tid; i < total_a; i += V3_THREADS) {
                int row = i / a_words;
                int col_w = i % a_words;
                int global_row = tile_m + row;
                uint8_t* dst = sa + row * A_STRIDE + col_w * 4;
                unsigned int dst_smem = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (global_row < M) {
                    const void* src = x_g + (int64_t)global_row * gs_half + col_w * 4;
                    asm volatile(
                        "cp.async.ca.shared.global [%0], [%1], 4;\n"
                        :: "r"(dst_smem), "l"(src));
                } else {
                    *reinterpret_cast<uint32_t*>(dst) = 0u;
                }
            }
        }

        // Load B tile [128, gs_half] → padded SMEM via 4-byte cp.async
        {
            const int b_words = gs_half / 4;
            const int total_b = V3_TILE_N * b_words;
            for (int i = tid; i < total_b; i += V3_THREADS) {
                int row = i / b_words;
                int col_w = i % b_words;
                int global_row = tile_n + row;
                uint8_t* dst = sb + row * B_STRIDE + col_w * 4;
                unsigned int dst_smem = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (global_row < N) {
                    const void* src = w_g + (int64_t)global_row * gs_half + col_w * 4;
                    asm volatile(
                        "cp.async.ca.shared.global [%0], [%1], 4;\n"
                        :: "r"(dst_smem), "l"(src));
                } else {
                    *reinterpret_cast<uint32_t*>(dst) = 0u;
                }
            }
        }

        // Load scales (small, direct store — visible after __syncthreads)
        for (int i = tid; i < V3_TILE_M; i += V3_THREADS) {
            int r = tile_m + i;
            ssx[i] = (r < M) ? sx_g[r] : 0.f;
            if constexpr (USE_AZP) {
                const float* azp_g = azp_adj + (int64_t)g * M;
                sazp[i] = (r < M) ? azp_g[r] : 0.f;
            }
        }
        for (int i = tid; i < V3_TILE_N; i += V3_THREADS) {
            int c = tile_n + i;
            ssw[i] = (c < N) ? sw_g[c] : 0.f;
            if constexpr (USE_AZP) {
                const float* wcs_g = w_col_sum + (int64_t)g * N;
                swcs[i] = (c < N) ? wcs_g[c] : 0.f;
            }
        }

        asm volatile("cp.async.commit_group;\n" ::);
    };

    // ── Prologue: start loading group 0 into buffer 0 ──
    issue_loads(0, 0);

    // ── Main group loop with double buffering ──
    for (int g = 0; g < num_groups; g++) {
        const int cur = g & 1;

        // Wait for current buffer's async copies to complete
        asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
        __syncthreads();

        // Start prefetching next group into the other buffer
        if (g + 1 < num_groups) {
            issue_loads(g + 1, 1 - cur);
        }

        // ── Compute MMA on current buffer ──
        uint8_t* base = smem_v3 + cur * buf_bytes;
        const uint8_t* cur_a = base;
        const uint8_t* cur_b = cur_a + V3_TILE_M * A_STRIDE;
        const float* cur_sx = reinterpret_cast<const float*>(
            cur_b + V3_TILE_N * B_STRIDE);
        const float* cur_sw = cur_sx + V3_TILE_M;
        const float* cur_azp = cur_sw + V3_TILE_N;
        const float* cur_wcs = cur_azp + V3_TILE_M;

        const int a_u32_stride = A_STRIDE / 4;
        const int b_u32_stride = B_STRIDE / 4;
        const uint32_t* a_u32 = reinterpret_cast<const uint32_t*>(cur_a);
        const uint32_t* b_u32 = reinterpret_cast<const uint32_t*>(cur_b);

        int32_t mma_acc[2][8][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 8; ni++)
                #pragma unroll
                for (int fi = 0; fi < 4; fi++)
                    mma_acc[mi][ni][fi] = 0;

        #pragma unroll
        for (int ki = 0; ki < mma_k_iters; ki++) {
            const int k_u32_off = ki * (MMA_K / 2) / 4;

            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int row0 = warp_row_start + mi * MMA_M + frag_row;
                int row1 = row0 + 8;
                int base0 = row0 * a_u32_stride + k_u32_off + frag_grp * 2;
                int base1 = row1 * a_u32_stride + k_u32_off + frag_grp * 2;
                a_frag[mi][0] = a_u32[base0];
                a_frag[mi][1] = a_u32[base1];
                a_frag[mi][2] = a_u32[base0 + 1];
                a_frag[mi][3] = a_u32[base1 + 1];
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int b_row = warp_col_start + ni * MMA_N + frag_row;
                    int b_base = b_row * b_u32_stride + k_u32_off + frag_grp * 2;
                    uint32_t b0 = b_u32[b_base];
                    uint32_t b1 = b_u32[b_base + 1];

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
                    asm volatile(
                        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(mma_acc[mi][ni][0]), "=r"(mma_acc[mi][ni][1]),
                          "=r"(mma_acc[mi][ni][2]), "=r"(mma_acc[mi][ni][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b0), "r"(b1),
                          "r"(mma_acc[mi][ni][0]), "r"(mma_acc[mi][ni][1]),
                          "r"(mma_acc[mi][ni][2]), "r"(mma_acc[mi][ni][3]));
#endif
                }
            }
        } // end K-loop

        // ── Apply per-group scales and accumulate into FP32 ──
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m0 = warp_row_start + mi * MMA_M + frag_row;
            int m1 = m0 + 8;
            float sx0 = cur_sx[m0];
            float sx1 = cur_sx[m1];
            float azp0 = 0.f, azp1 = 0.f;
            if constexpr (USE_AZP) {
                azp0 = cur_azp[m0];
                azp1 = cur_azp[m1];
            }

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                int n0 = warp_col_start + ni * MMA_N + frag_grp * 2;
                float sw0 = cur_sw[n0];
                float sw1 = cur_sw[n0 + 1];

                if constexpr (USE_AZP) {
                    float wcs0 = cur_wcs[n0];
                    float wcs1 = cur_wcs[n0 + 1];
                    acc[mi][ni][0] += (static_cast<float>(mma_acc[mi][ni][0]) + azp0 * wcs0) * sx0 * sw0;
                    acc[mi][ni][1] += (static_cast<float>(mma_acc[mi][ni][1]) + azp0 * wcs1) * sx0 * sw1;
                    acc[mi][ni][2] += (static_cast<float>(mma_acc[mi][ni][2]) + azp1 * wcs0) * sx1 * sw0;
                    acc[mi][ni][3] += (static_cast<float>(mma_acc[mi][ni][3]) + azp1 * wcs1) * sx1 * sw1;
                } else {
                    acc[mi][ni][0] += static_cast<float>(mma_acc[mi][ni][0]) * sx0 * sw0;
                    acc[mi][ni][1] += static_cast<float>(mma_acc[mi][ni][1]) * sx0 * sw1;
                    acc[mi][ni][2] += static_cast<float>(mma_acc[mi][ni][2]) * sx1 * sw0;
                    acc[mi][ni][3] += static_cast<float>(mma_acc[mi][ni][3]) * sx1 * sw1;
                }
            }
        }

        __syncthreads();  // ensure all warps done reading before next overwrites
    } // end group loop

    // ── Write FP32 → BF16 output ──
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int gm0 = tile_m + warp_row_start + mi * MMA_M + frag_row;
            int gm1 = gm0 + 8;
            int gn0 = tile_n + warp_col_start + ni * MMA_N + frag_grp * 2;
            int gn1 = gn0 + 1;

            if (gm0 < M && gn0 < N) out[gm0 * N + gn0] = __float2bfloat16(acc[mi][ni][0]);
            if (gm0 < M && gn1 < N) out[gm0 * N + gn1] = __float2bfloat16(acc[mi][ni][1]);
            if (gm1 < M && gn0 < N) out[gm1 * N + gn0] = __float2bfloat16(acc[mi][ni][2]);
            if (gm1 < M && gn1 < N) out[gm1 * N + gn1] = __float2bfloat16(acc[mi][ni][3]);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// V4 Double-buffered fused INT4 grouped GEMM kernel with cp.async pipeline
//
// Combines V2's constexpr loop unrolling with double-buffered cp.async:
// 1. constexpr GS template → all K-loops fully unrolled (like V2)
// 2. Double-buffered SMEM → prefetch group g+1 while computing group g
// 3. cp.async 4-byte GMEM→SMEM copies (non-blocking, bypasses registers)
//
// V3 failed because it used runtime gs (preventing unrolling).
// V4 fixes this by using the same constexpr approach as V2.
//
// Same tile: 64×128, 128 threads (4 warps), 2×2 warp layout
// Same stride: gs_half+4 (bank-conflict-free)
// SMEM: 2 × ~14.5KB = ~29KB (within 48KB default, 3 blocks/SM at 100KB)
// ═════════════════════════════════════════════════════════════════════════════

template <bool USE_AZP, int GS = 128>
__global__ void __launch_bounds__(128)
fused_int4_grouped_gemm_v4_kernel(
    const uint8_t* __restrict__ x_packed,   // [num_groups, M, gs/2]
    const uint8_t* __restrict__ w_packed,   // [num_groups, N, gs/2]
    const float* __restrict__ scale_x,      // [num_groups, M]
    const float* __restrict__ scale_w,      // [num_groups, N]
    const float* __restrict__ azp_adj,      // [num_groups, M] (USE_AZP only)
    const float* __restrict__ w_col_sum,    // [num_groups, N] (USE_AZP only)
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const int M, const int N,
    const int num_groups)
{
    // ── Compile-time constants (same as V2) ──
    static constexpr int TILE_M = 64;
    static constexpr int TILE_N = 128;
    static constexpr int THREADS = 128;
    static constexpr int gs = GS;
    static constexpr int gs_half = GS / 2;
    static constexpr int mma_k_iters = GS / MMA_K;
    static constexpr int A_STRIDE = gs_half + 4;
    static constexpr int B_STRIDE = gs_half + 4;
    static constexpr int a_u32_stride = A_STRIDE / 4;
    static constexpr int b_u32_stride = B_STRIDE / 4;

    // Per-stage SMEM: A_tile | B_tile | sx | sw [| azp | wcs]
    static constexpr int scale_floats = TILE_M + TILE_N;
    static constexpr int azp_floats = USE_AZP ? (TILE_M + TILE_N) : 0;
    static constexpr int STAGE_BYTES =
        TILE_M * A_STRIDE + TILE_N * B_STRIDE
        + (scale_floats + azp_floats) * (int)sizeof(float);

    // ── Thread/warp decomposition ──
    const int tile_m = blockIdx.x * TILE_M;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int warp_row_start = warp_m * 32;
    const int warp_col_start = warp_n * 64;
    const int frag_row = lane_id / 4;
    const int frag_grp = lane_id % 4;

    // ── Double-buffered shared memory ──
    extern __shared__ uint8_t smem_v4[];

    // ── FP32 accumulator ──
    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            #pragma unroll
            for (int fi = 0; fi < 4; fi++)
                acc[mi][ni][fi] = 0.f;

    // ── Helper: issue cp.async loads for a group into a stage ──
    auto v4_load_stage = [&](int g, int stage) __attribute__((always_inline)) {
        uint8_t* sa = smem_v4 + stage * STAGE_BYTES;
        uint8_t* sb = sa + TILE_M * A_STRIDE;
        float* ssx = reinterpret_cast<float*>(sb + TILE_N * B_STRIDE);
        float* ssw = ssx + TILE_M;

        const uint8_t* xg = x_packed + (int64_t)g * M * gs_half;
        const uint8_t* wg = w_packed + (int64_t)g * N * gs_half;

        // A tile [TILE_M, gs_half] via cp.async 4-byte
        {
            constexpr int a_words = gs_half / 4;
            constexpr int total_a = TILE_M * a_words;
            for (int i = tid; i < total_a; i += THREADS) {
                const int row = i / a_words;
                const int col = i % a_words;
                const int gr = tile_m + row;
                uint8_t* dst = sa + row * A_STRIDE + col * 4;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < M) {
                    const void* src = xg + (int64_t)gr * gs_half + col * 4;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    *reinterpret_cast<uint32_t*>(dst) = 0u;
                }
            }
        }

        // B tile [TILE_N, gs_half] via cp.async 4-byte
        {
            constexpr int b_words = gs_half / 4;
            constexpr int total_b = TILE_N * b_words;
            for (int i = tid; i < total_b; i += THREADS) {
                const int row = i / b_words;
                const int col = i % b_words;
                const int gr = tile_n + row;
                uint8_t* dst = sb + row * B_STRIDE + col * 4;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < N) {
                    const void* src = wg + (int64_t)gr * gs_half + col * 4;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    *reinterpret_cast<uint32_t*>(dst) = 0u;
                }
            }
        }

        // Scales via regular stores (small, visible after __syncthreads)
        const float* sxg = scale_x + (int64_t)g * M;
        const float* swg = scale_w + (int64_t)g * N;
        for (int i = tid; i < TILE_M; i += THREADS) {
            int r = tile_m + i;
            ssx[i] = (r < M) ? sxg[r] : 0.f;
        }
        for (int i = tid; i < TILE_N; i += THREADS) {
            int c = tile_n + i;
            ssw[i] = (c < N) ? swg[c] : 0.f;
        }
        if constexpr (USE_AZP) {
            float* sazp = ssw + TILE_N;
            float* swcs = sazp + TILE_M;
            const float* azpg = azp_adj + (int64_t)g * M;
            const float* wcsg = w_col_sum + (int64_t)g * N;
            for (int i = tid; i < TILE_M; i += THREADS) {
                int r = tile_m + i;
                sazp[i] = (r < M) ? azpg[r] : 0.f;
            }
            for (int i = tid; i < TILE_N; i += THREADS) {
                int c = tile_n + i;
                swcs[i] = (c < N) ? wcsg[c] : 0.f;
            }
        }

        asm volatile("cp.async.commit_group;\n" ::);
    };

    // ══ Prologue: load group 0 into stage 0 ══
    v4_load_stage(0, 0);

    // ══ Main group loop with double buffering ══
    for (int g = 0; g < num_groups; g++) {
        const int cur = g & 1;

        // Prefetch next group into the other stage
        if (g + 1 < num_groups) {
            v4_load_stage(g + 1, 1 - cur);
        }

        // Wait for current stage: allow at most 1 pending (the prefetch)
        if (g + 1 < num_groups) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(1));
        } else {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
        }
        __syncthreads();

        // ── Stage pointers ──
        const uint8_t* cur_base = smem_v4 + cur * STAGE_BYTES;
        const uint32_t* a_u32 = reinterpret_cast<const uint32_t*>(cur_base);
        const uint32_t* b_u32 = reinterpret_cast<const uint32_t*>(
            cur_base + TILE_M * A_STRIDE);
        const float* cur_sx = reinterpret_cast<const float*>(
            cur_base + TILE_M * A_STRIDE + TILE_N * B_STRIDE);
        const float* cur_sw = cur_sx + TILE_M;
        const float* cur_azp = cur_sw + TILE_N;
        const float* cur_wcs = cur_azp + TILE_M;

        // ── MMA computation (identical to V2) ──
        int32_t mma_acc[2][8][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 8; ni++)
                #pragma unroll
                for (int fi = 0; fi < 4; fi++)
                    mma_acc[mi][ni][fi] = 0;

        #pragma unroll
        for (int ki = 0; ki < mma_k_iters; ki++) {
            const int k_u32_off = ki * (MMA_K / 2) / 4;

            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int row0 = warp_row_start + mi * MMA_M + frag_row;
                int row1 = row0 + 8;
                int base0 = row0 * a_u32_stride + k_u32_off + frag_grp * 2;
                int base1 = row1 * a_u32_stride + k_u32_off + frag_grp * 2;
                a_frag[mi][0] = a_u32[base0];
                a_frag[mi][1] = a_u32[base1];
                a_frag[mi][2] = a_u32[base0 + 1];
                a_frag[mi][3] = a_u32[base1 + 1];
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int b_row = warp_col_start + ni * MMA_N + frag_row;
                    int b_base = b_row * b_u32_stride + k_u32_off + frag_grp * 2;
                    uint32_t b0 = b_u32[b_base];
                    uint32_t b1 = b_u32[b_base + 1];

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
                    asm volatile(
                        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(mma_acc[mi][ni][0]), "=r"(mma_acc[mi][ni][1]),
                          "=r"(mma_acc[mi][ni][2]), "=r"(mma_acc[mi][ni][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b0), "r"(b1),
                          "r"(mma_acc[mi][ni][0]), "r"(mma_acc[mi][ni][1]),
                          "r"(mma_acc[mi][ni][2]), "r"(mma_acc[mi][ni][3]));
#endif
                }
            }
        }

        // ── Apply per-group scales and accumulate into FP32 ──
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m0 = warp_row_start + mi * MMA_M + frag_row;
            int m1 = m0 + 8;
            float sx0 = cur_sx[m0];
            float sx1 = cur_sx[m1];
            float azp0 = 0.f, azp1 = 0.f;
            if constexpr (USE_AZP) {
                azp0 = cur_azp[m0];
                azp1 = cur_azp[m1];
            }

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                int n0 = warp_col_start + ni * MMA_N + frag_grp * 2;
                float sw0 = cur_sw[n0];
                float sw1 = cur_sw[n0 + 1];

                if constexpr (USE_AZP) {
                    float wcs0 = cur_wcs[n0];
                    float wcs1 = cur_wcs[n0 + 1];
                    acc[mi][ni][0] += (static_cast<float>(mma_acc[mi][ni][0]) + azp0 * wcs0) * sx0 * sw0;
                    acc[mi][ni][1] += (static_cast<float>(mma_acc[mi][ni][1]) + azp0 * wcs1) * sx0 * sw1;
                    acc[mi][ni][2] += (static_cast<float>(mma_acc[mi][ni][2]) + azp1 * wcs0) * sx1 * sw0;
                    acc[mi][ni][3] += (static_cast<float>(mma_acc[mi][ni][3]) + azp1 * wcs1) * sx1 * sw1;
                } else {
                    acc[mi][ni][0] += static_cast<float>(mma_acc[mi][ni][0]) * sx0 * sw0;
                    acc[mi][ni][1] += static_cast<float>(mma_acc[mi][ni][1]) * sx0 * sw1;
                    acc[mi][ni][2] += static_cast<float>(mma_acc[mi][ni][2]) * sx1 * sw0;
                    acc[mi][ni][3] += static_cast<float>(mma_acc[mi][ni][3]) * sx1 * sw1;
                }
            }
        }

        __syncthreads();  // protect stage reads before next overwrite
    }

    // ── Write FP32 → BF16 output (identical to V2) ──
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int gm0 = tile_m + warp_row_start + mi * MMA_M + frag_row;
            int gm1 = gm0 + 8;
            int gn0 = tile_n + warp_col_start + ni * MMA_N + frag_grp * 2;
            int gn1 = gn0 + 1;

            if (gm0 < M && gn0 < N) out[gm0 * N + gn0] = __float2bfloat16(acc[mi][ni][0]);
            if (gm0 < M && gn1 < N) out[gm0 * N + gn1] = __float2bfloat16(acc[mi][ni][1]);
            if (gm1 < M && gn0 < N) out[gm1 * N + gn0] = __float2bfloat16(acc[mi][ni][2]);
            if (gm1 < M && gn1 < N) out[gm1 * N + gn1] = __float2bfloat16(acc[mi][ni][3]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V5 kernel: Multi-Group K-Loop Fused INT4 Grouped GEMM
//
// Optimization over V4: Loads GROUPS_PER_LOAD groups' worth of activation and
// weight data into each SMEM stage, processing them sequentially within a
// single sync-protected window. This reduces __syncthreads() calls by
// GROUPS_PER_LOAD× compared to V4.
//
// V4 sync count: 2 × num_groups (e.g., 40 for num_groups=20)
// V5 sync count: 2 × ceil(num_groups / GROUPS_PER_LOAD)
//   GROUPS_PER_LOAD=2: 20 syncs (2× reduction)
//   GROUPS_PER_LOAD=4: 10 syncs (4× reduction)
//
// SMEM per stage (GROUPS_PER_LOAD=2, no AZP):
//   A: TILE_M × (2×gs_half + 4) = 64 × 132 =  8448 bytes
//   B: TILE_N × (2×gs_half + 4) = 128 × 132 = 16896 bytes
//   scales: (TILE_M + TILE_N) × 2 × 4 =       1536 bytes
//   Total per stage: 26880 bytes
//   Double buffered: 53760 bytes (~53 KB, within SM80 limit)
//
// SMEM per stage (GROUPS_PER_LOAD=4, no AZP):
//   A: 64 × 260 = 16640, B: 128 × 260 = 33280, scales: 3072
//   Total per stage: 52992, double buffered: 105984 (~104 KB, within SM80 limit)
// ─────────────────────────────────────────────────────────────────────────────

template <bool USE_AZP, int GS = 128, int GROUPS_PER_LOAD = 2>
__global__ void __launch_bounds__(128)
fused_int4_grouped_gemm_v5_kernel(
    const uint8_t* __restrict__ x_packed,   // [num_groups, M, gs/2]
    const uint8_t* __restrict__ w_packed,   // [num_groups, N, gs/2]
    const float* __restrict__ scale_x,      // [num_groups, M]
    const float* __restrict__ scale_w,      // [num_groups, N]
    const float* __restrict__ azp_adj,      // [num_groups, M] (USE_AZP only)
    const float* __restrict__ w_col_sum,    // [num_groups, N] (USE_AZP only)
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const int M, const int N,
    const int num_groups)
{
    // ── Compile-time constants ──
    static constexpr int TILE_M = 64;
    static constexpr int TILE_N = 128;
    static constexpr int THREADS = 128;
    static constexpr int gs = GS;
    static constexpr int gs_half = GS / 2;
    static constexpr int mma_k_iters = GS / MMA_K;  // MMA iterations per group

    // Multi-group SMEM layout: A and B tiles hold GROUPS_PER_LOAD groups
    // contiguously along K dimension, with 4-byte padding at the end.
    static constexpr int A_K_BYTES = GROUPS_PER_LOAD * gs_half;
    static constexpr int A_STRIDE = A_K_BYTES + 4;  // 4-byte pad for bank conflicts
    static constexpr int B_K_BYTES = GROUPS_PER_LOAD * gs_half;
    static constexpr int B_STRIDE = B_K_BYTES + 4;
    static constexpr int a_u32_stride = A_STRIDE / 4;
    static constexpr int b_u32_stride = B_STRIDE / 4;

    // Scale arrays: GROUPS_PER_LOAD sets of scales per stage
    static constexpr int scale_floats = GROUPS_PER_LOAD * (TILE_M + TILE_N);
    static constexpr int azp_floats = USE_AZP ? GROUPS_PER_LOAD * (TILE_M + TILE_N) : 0;
    static constexpr int STAGE_BYTES =
        TILE_M * A_STRIDE + TILE_N * B_STRIDE
        + (scale_floats + azp_floats) * (int)sizeof(float);

    // ── Thread/warp decomposition (same as V4) ──
    const int tile_m = blockIdx.x * TILE_M;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int warp_row_start = warp_m * 32;
    const int warp_col_start = warp_n * 64;
    const int frag_row = lane_id / 4;
    const int frag_grp = lane_id % 4;

    // ── Double-buffered shared memory ──
    extern __shared__ uint8_t smem_v5[];

    // ── FP32 accumulator for final output ──
    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            #pragma unroll
            for (int fi = 0; fi < 4; fi++)
                acc[mi][ni][fi] = 0.f;

    // ── Helper: issue cp.async loads for GROUPS_PER_LOAD groups into a stage ──
    auto v5_load_stage = [&](int g_base, int chunk_size, int stage) __attribute__((always_inline)) {
        uint8_t* sa = smem_v5 + stage * STAGE_BYTES;
        uint8_t* sb = sa + TILE_M * A_STRIDE;
        float* ssx_base = reinterpret_cast<float*>(sb + TILE_N * B_STRIDE);

        // Load A tiles for each group in the chunk, packed contiguously in K
        #pragma unroll
        for (int gl = 0; gl < GROUPS_PER_LOAD; gl++) {
            int g = g_base + gl;
            if (g >= g_base + chunk_size) break;  // handle remainder chunk

            const uint8_t* xg = x_packed + (int64_t)g * M * gs_half;
            constexpr int a_words_per_group = gs_half / 4;
            constexpr int total_a = TILE_M * a_words_per_group;

            for (int i = tid; i < total_a; i += THREADS) {
                const int row = i / a_words_per_group;
                const int col = i % a_words_per_group;
                const int gr = tile_m + row;
                // Offset into multi-group K dimension: gl * gs_half bytes, col * 4 bytes
                uint8_t* dst = sa + row * A_STRIDE + gl * gs_half + col * 4;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < M) {
                    const void* src = xg + (int64_t)gr * gs_half + col * 4;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    *reinterpret_cast<uint32_t*>(dst) = 0u;
                }
            }
        }

        // Load B tiles for each group in the chunk
        #pragma unroll
        for (int gl = 0; gl < GROUPS_PER_LOAD; gl++) {
            int g = g_base + gl;
            if (g >= g_base + chunk_size) break;

            const uint8_t* wg = w_packed + (int64_t)g * N * gs_half;
            constexpr int b_words_per_group = gs_half / 4;
            constexpr int total_b = TILE_N * b_words_per_group;

            for (int i = tid; i < total_b; i += THREADS) {
                const int row = i / b_words_per_group;
                const int col = i % b_words_per_group;
                const int gr = tile_n + row;
                uint8_t* dst = sb + row * B_STRIDE + gl * gs_half + col * 4;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < N) {
                    const void* src = wg + (int64_t)gr * gs_half + col * 4;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    *reinterpret_cast<uint32_t*>(dst) = 0u;
                }
            }
        }

        // Load scales for each group in the chunk (regular stores)
        #pragma unroll
        for (int gl = 0; gl < GROUPS_PER_LOAD; gl++) {
            int g = g_base + gl;
            if (g >= g_base + chunk_size) break;

            float* ssx = ssx_base + gl * (TILE_M + TILE_N);
            float* ssw = ssx + TILE_M;
            const float* sxg = scale_x + (int64_t)g * M;
            const float* swg = scale_w + (int64_t)g * N;

            for (int i = tid; i < TILE_M; i += THREADS) {
                int r = tile_m + i;
                ssx[i] = (r < M) ? sxg[r] : 0.f;
            }
            for (int i = tid; i < TILE_N; i += THREADS) {
                int c = tile_n + i;
                ssw[i] = (c < N) ? swg[c] : 0.f;
            }

            if constexpr (USE_AZP) {
                float* sazp_base = ssx_base + scale_floats;
                float* sazp = sazp_base + gl * (TILE_M + TILE_N);
                float* swcs = sazp + TILE_M;
                const float* azpg = azp_adj + (int64_t)g * M;
                const float* wcsg = w_col_sum + (int64_t)g * N;
                for (int i = tid; i < TILE_M; i += THREADS) {
                    int r = tile_m + i;
                    sazp[i] = (r < M) ? azpg[r] : 0.f;
                }
                for (int i = tid; i < TILE_N; i += THREADS) {
                    int c = tile_n + i;
                    swcs[i] = (c < N) ? wcsg[c] : 0.f;
                }
            }
        }

        asm volatile("cp.async.commit_group;\n" ::);
    };

    // ── Number of chunks ──
    const int num_chunks = (num_groups + GROUPS_PER_LOAD - 1) / GROUPS_PER_LOAD;

    // ══ Prologue: load first chunk into stage 0 ══
    {
        int first_chunk_size = (GROUPS_PER_LOAD <= num_groups) ? GROUPS_PER_LOAD : num_groups;
        v5_load_stage(0, first_chunk_size, 0);
    }

    // ══ Main chunk loop with double buffering ══
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int cur = chunk & 1;
        const int g_base = chunk * GROUPS_PER_LOAD;
        const int chunk_size = ((g_base + GROUPS_PER_LOAD) <= num_groups)
                             ? GROUPS_PER_LOAD
                             : (num_groups - g_base);

        // Prefetch next chunk into the other stage
        if (chunk + 1 < num_chunks) {
            int next_g_base = (chunk + 1) * GROUPS_PER_LOAD;
            int next_chunk_size = ((next_g_base + GROUPS_PER_LOAD) <= num_groups)
                                ? GROUPS_PER_LOAD
                                : (num_groups - next_g_base);
            v5_load_stage(next_g_base, next_chunk_size, 1 - cur);
        }

        // Wait for current stage
        if (chunk + 1 < num_chunks) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(1));
        } else {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
        }
        __syncthreads();

        // ── Stage pointers for current chunk ──
        const uint8_t* cur_base = smem_v5 + cur * STAGE_BYTES;
        const uint32_t* a_u32 = reinterpret_cast<const uint32_t*>(cur_base);
        const uint32_t* b_u32 = reinterpret_cast<const uint32_t*>(
            cur_base + TILE_M * A_STRIDE);
        const float* scales_base = reinterpret_cast<const float*>(
            cur_base + TILE_M * A_STRIDE + TILE_N * B_STRIDE);
        const float* azp_base_ptr = scales_base + scale_floats;  // only used when USE_AZP

        // ── Process each group within this chunk sequentially ──
        // No __syncthreads needed between groups within a chunk since all
        // data is already resident in SMEM from the single load.
        for (int gl = 0; gl < chunk_size; gl++) {
            // Pointers to this group's scales within the SMEM stage
            const float* cur_sx = scales_base + gl * (TILE_M + TILE_N);
            const float* cur_sw = cur_sx + TILE_M;
            const float* cur_azp = nullptr;
            const float* cur_wcs = nullptr;
            if constexpr (USE_AZP) {
                cur_azp = azp_base_ptr + gl * (TILE_M + TILE_N);
                cur_wcs = cur_azp + TILE_M;
            }

            // K offset within multi-group SMEM tile for this group
            const int k_u32_group_off = gl * (gs_half / 4);

            // ── MMA computation for this group ──
            int32_t mma_acc[2][8][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 8; ni++)
                    #pragma unroll
                    for (int fi = 0; fi < 4; fi++)
                        mma_acc[mi][ni][fi] = 0;

            #pragma unroll
            for (int ki = 0; ki < mma_k_iters; ki++) {
                const int k_u32_off = k_u32_group_off + ki * (MMA_K / 2) / 4;

                uint32_t a_frag[2][4];
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    int row0 = warp_row_start + mi * MMA_M + frag_row;
                    int row1 = row0 + 8;
                    int base0 = row0 * a_u32_stride + k_u32_off + frag_grp * 2;
                    int base1 = row1 * a_u32_stride + k_u32_off + frag_grp * 2;
                    a_frag[mi][0] = a_u32[base0];
                    a_frag[mi][1] = a_u32[base1];
                    a_frag[mi][2] = a_u32[base0 + 1];
                    a_frag[mi][3] = a_u32[base1 + 1];
                }

                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    #pragma unroll
                    for (int ni = 0; ni < 8; ni++) {
                        int b_row = warp_col_start + ni * MMA_N + frag_row;
                        int b_base = b_row * b_u32_stride + k_u32_off + frag_grp * 2;
                        uint32_t b0 = b_u32[b_base];
                        uint32_t b1 = b_u32[b_base + 1];

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
                        asm volatile(
                            "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                            : "=r"(mma_acc[mi][ni][0]), "=r"(mma_acc[mi][ni][1]),
                              "=r"(mma_acc[mi][ni][2]), "=r"(mma_acc[mi][ni][3])
                            : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                              "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                              "r"(b0), "r"(b1),
                              "r"(mma_acc[mi][ni][0]), "r"(mma_acc[mi][ni][1]),
                              "r"(mma_acc[mi][ni][2]), "r"(mma_acc[mi][ni][3]));
#endif
                    }
                }
            }

            // ── Apply per-group scales and accumulate into FP32 ──
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int m0 = warp_row_start + mi * MMA_M + frag_row;
                int m1 = m0 + 8;
                float sx0 = cur_sx[m0];
                float sx1 = cur_sx[m1];
                float azp0 = 0.f, azp1 = 0.f;
                if constexpr (USE_AZP) {
                    azp0 = cur_azp[m0];
                    azp1 = cur_azp[m1];
                }

                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int n0 = warp_col_start + ni * MMA_N + frag_grp * 2;
                    float sw0 = cur_sw[n0];
                    float sw1 = cur_sw[n0 + 1];

                    if constexpr (USE_AZP) {
                        float wcs0 = cur_wcs[n0];
                        float wcs1 = cur_wcs[n0 + 1];
                        acc[mi][ni][0] += (static_cast<float>(mma_acc[mi][ni][0]) + azp0 * wcs0) * sx0 * sw0;
                        acc[mi][ni][1] += (static_cast<float>(mma_acc[mi][ni][1]) + azp0 * wcs1) * sx0 * sw1;
                        acc[mi][ni][2] += (static_cast<float>(mma_acc[mi][ni][2]) + azp1 * wcs0) * sx1 * sw0;
                        acc[mi][ni][3] += (static_cast<float>(mma_acc[mi][ni][3]) + azp1 * wcs1) * sx1 * sw1;
                    } else {
                        acc[mi][ni][0] += static_cast<float>(mma_acc[mi][ni][0]) * sx0 * sw0;
                        acc[mi][ni][1] += static_cast<float>(mma_acc[mi][ni][1]) * sx0 * sw1;
                        acc[mi][ni][2] += static_cast<float>(mma_acc[mi][ni][2]) * sx1 * sw0;
                        acc[mi][ni][3] += static_cast<float>(mma_acc[mi][ni][3]) * sx1 * sw1;
                    }
                }
            }
        }  // end gl loop within chunk

        __syncthreads();  // protect stage reads before next chunk overwrites
    }  // end chunk loop

    // ── Write FP32 → BF16 output (identical to V4) ──
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int gm0 = tile_m + warp_row_start + mi * MMA_M + frag_row;
            int gm1 = gm0 + 8;
            int gn0 = tile_n + warp_col_start + ni * MMA_N + frag_grp * 2;
            int gn1 = gn0 + 1;

            if (gm0 < M && gn0 < N) out[gm0 * N + gn0] = __float2bfloat16(acc[mi][ni][0]);
            if (gm0 < M && gn1 < N) out[gm0 * N + gn1] = __float2bfloat16(acc[mi][ni][1]);
            if (gm1 < M && gn0 < N) out[gm1 * N + gn0] = __float2bfloat16(acc[mi][ni][2]);
            if (gm1 < M && gn1 < N) out[gm1 * N + gn1] = __float2bfloat16(acc[mi][ni][3]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8 kernel: V4 tile (64×128) with 16-byte vectorized cp.async loads
//
// Same architecture as V4 but with optimized memory access:
//   - 16-byte cp.async loads (vs V4's 4-byte) → 4× fewer load instructions
//   - SMEM stride = 80 (multiple of 16 for alignment, bank-conflict-free)
//   - Same TILE_M=64, TILE_N=128, 128 threads, 2-stage pipeline
//   - Maintains 2 blocks/SM occupancy (SMEM: 2×16896 = 33.8KB < 99KB)
//
// SMEM per stage: A[64×80] + B[128×80] + scales = 16128 bytes
// ─────────────────────────────────────────────────────────────────────────────

template <bool USE_AZP, int GS = 128>
__global__ void __launch_bounds__(128)
fused_int4_grouped_gemm_v8_kernel(
    const uint8_t* __restrict__ x_packed,   // [num_groups, M, gs/2]
    const uint8_t* __restrict__ w_packed,   // [num_groups, N, gs/2]
    const float* __restrict__ scale_x,      // [num_groups, M]
    const float* __restrict__ scale_w,      // [num_groups, N]
    const float* __restrict__ azp_adj,      // [num_groups, M] (USE_AZP only)
    const float* __restrict__ w_col_sum,    // [num_groups, N] (USE_AZP only)
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const int M, const int N,
    const int num_groups)
{
    // ── Compile-time constants ──
    static constexpr int TILE_M = 64;
    static constexpr int TILE_N = 128;
    static constexpr int THREADS = 128;
    static constexpr int gs_half = GS / 2;           // = 64
    static constexpr int mma_k_iters = GS / MMA_K;   // = 2
    // SMEM stride: 80 bytes = 16-byte aligned, bank-conflict-free (20 % 32 ≠ 0)
    static constexpr int A_STRIDE = 80;
    static constexpr int B_STRIDE = 80;
    static constexpr int a_u32_stride = A_STRIDE / 4;  // = 20
    static constexpr int b_u32_stride = B_STRIDE / 4;  // = 20

    static constexpr int scale_floats = TILE_M + TILE_N;
    static constexpr int azp_floats = USE_AZP ? (TILE_M + TILE_N) : 0;
    static constexpr int STAGE_BYTES =
        TILE_M * A_STRIDE + TILE_N * B_STRIDE
        + (scale_floats + azp_floats) * (int)sizeof(float);

    // ── 16-byte load constants ──
    static constexpr int VECTORS_PER_ROW = gs_half / 16;  // = 4 (64 bytes / 16)

    // ── Thread/warp decomposition (same as V4) ──
    const int tile_m = blockIdx.x * TILE_M;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int warp_row_start = warp_m * 32;
    const int warp_col_start = warp_n * 64;
    const int frag_row = lane_id / 4;
    const int frag_grp = lane_id % 4;

    extern __shared__ uint8_t smem_v8[];

    // ── FP32 accumulator ──
    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            #pragma unroll
            for (int fi = 0; fi < 4; fi++)
                acc[mi][ni][fi] = 0.f;

    // ── Helper: issue cp.async loads for a group into a stage ──
    auto v8_load_stage = [&](int g, int stage) __attribute__((always_inline)) {
        uint8_t* sa = smem_v8 + stage * STAGE_BYTES;
        uint8_t* sb = sa + TILE_M * A_STRIDE;
        float* ssx = reinterpret_cast<float*>(sb + TILE_N * B_STRIDE);
        float* ssw = ssx + TILE_M;

        const uint8_t* xg = x_packed + (int64_t)g * M * gs_half;
        const uint8_t* wg = w_packed + (int64_t)g * N * gs_half;

        // A tile [TILE_M=64, gs_half=64] via cp.async 16-byte
        // 64 rows × 4 vectors = 256 total, 128 threads → 2 loads/thread
        {
            constexpr int total_a = TILE_M * VECTORS_PER_ROW;  // = 256
            for (int i = tid; i < total_a; i += THREADS) {
                const int row = i / VECTORS_PER_ROW;
                const int col = i % VECTORS_PER_ROW;
                const int gr = tile_m + row;
                uint8_t* dst = sa + row * A_STRIDE + col * 16;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < M) {
                    const void* src = xg + (int64_t)gr * gs_half + col * 16;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    // Zero-fill 16 bytes
                    reinterpret_cast<uint4*>(dst)[0] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        // B tile [TILE_N=128, gs_half=64] via cp.async 16-byte
        // 128 rows × 4 vectors = 512 total, 128 threads → 4 loads/thread
        {
            constexpr int total_b = TILE_N * VECTORS_PER_ROW;  // = 512
            for (int i = tid; i < total_b; i += THREADS) {
                const int row = i / VECTORS_PER_ROW;
                const int col = i % VECTORS_PER_ROW;
                const int gr = tile_n + row;
                uint8_t* dst = sb + row * B_STRIDE + col * 16;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < N) {
                    const void* src = wg + (int64_t)gr * gs_half + col * 16;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    reinterpret_cast<uint4*>(dst)[0] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        // Scales via regular stores
        const float* sxg = scale_x + (int64_t)g * M;
        const float* swg = scale_w + (int64_t)g * N;
        for (int i = tid; i < TILE_M; i += THREADS) {
            int r = tile_m + i;
            ssx[i] = (r < M) ? sxg[r] : 0.f;
        }
        for (int i = tid; i < TILE_N; i += THREADS) {
            int c = tile_n + i;
            ssw[i] = (c < N) ? swg[c] : 0.f;
        }
        if constexpr (USE_AZP) {
            float* sazp = ssw + TILE_N;
            float* swcs = sazp + TILE_M;
            const float* azpg = azp_adj + (int64_t)g * M;
            const float* wcsg = w_col_sum + (int64_t)g * N;
            for (int i = tid; i < TILE_M; i += THREADS) {
                int r = tile_m + i;
                sazp[i] = (r < M) ? azpg[r] : 0.f;
            }
            for (int i = tid; i < TILE_N; i += THREADS) {
                int c = tile_n + i;
                swcs[i] = (c < N) ? wcsg[c] : 0.f;
            }
        }

        asm volatile("cp.async.commit_group;\n" ::);
    };

    // ══ Prologue: load group 0 into stage 0 ══
    v8_load_stage(0, 0);

    // ══ Main group loop (same structure as V4) ══
    for (int g = 0; g < num_groups; g++) {
        const int cur = g & 1;

        if (g + 1 < num_groups) {
            v8_load_stage(g + 1, 1 - cur);
        }

        if (g + 1 < num_groups) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(1));
        } else {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
        }
        __syncthreads();

        // ── Stage pointers ──
        const uint8_t* cur_base = smem_v8 + cur * STAGE_BYTES;
        const uint32_t* a_u32 = reinterpret_cast<const uint32_t*>(cur_base);
        const uint32_t* b_u32 = reinterpret_cast<const uint32_t*>(
            cur_base + TILE_M * A_STRIDE);
        const float* cur_sx = reinterpret_cast<const float*>(
            cur_base + TILE_M * A_STRIDE + TILE_N * B_STRIDE);
        const float* cur_sw = cur_sx + TILE_M;
        const float* cur_azp = cur_sw + TILE_N;
        const float* cur_wcs = cur_azp + TILE_M;

        // ── MMA computation (identical to V4) ──
        int32_t mma_acc[2][8][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 8; ni++)
                #pragma unroll
                for (int fi = 0; fi < 4; fi++)
                    mma_acc[mi][ni][fi] = 0;

        #pragma unroll
        for (int ki = 0; ki < mma_k_iters; ki++) {
            const int k_u32_off = ki * (MMA_K / 2) / 4;

            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int row0 = warp_row_start + mi * MMA_M + frag_row;
                int row1 = row0 + 8;
                int base0 = row0 * a_u32_stride + k_u32_off + frag_grp * 2;
                int base1 = row1 * a_u32_stride + k_u32_off + frag_grp * 2;
                a_frag[mi][0] = a_u32[base0];
                a_frag[mi][1] = a_u32[base1];
                a_frag[mi][2] = a_u32[base0 + 1];
                a_frag[mi][3] = a_u32[base1 + 1];
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int b_row = warp_col_start + ni * MMA_N + frag_row;
                    int b_base = b_row * b_u32_stride + k_u32_off + frag_grp * 2;
                    uint32_t b0 = b_u32[b_base];
                    uint32_t b1 = b_u32[b_base + 1];

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
                    asm volatile(
                        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(mma_acc[mi][ni][0]), "=r"(mma_acc[mi][ni][1]),
                          "=r"(mma_acc[mi][ni][2]), "=r"(mma_acc[mi][ni][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b0), "r"(b1),
                          "r"(mma_acc[mi][ni][0]), "r"(mma_acc[mi][ni][1]),
                          "r"(mma_acc[mi][ni][2]), "r"(mma_acc[mi][ni][3]));
#endif
                }
            }
        }

        // ── Apply per-group scales (identical to V4) ──
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m0 = warp_row_start + mi * MMA_M + frag_row;
            int m1 = m0 + 8;
            float sx0 = cur_sx[m0];
            float sx1 = cur_sx[m1];
            float azp0 = 0.f, azp1 = 0.f;
            if constexpr (USE_AZP) {
                azp0 = cur_azp[m0];
                azp1 = cur_azp[m1];
            }

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                int n0 = warp_col_start + ni * MMA_N + frag_grp * 2;
                float sw0 = cur_sw[n0];
                float sw1 = cur_sw[n0 + 1];

                if constexpr (USE_AZP) {
                    float wcs0 = cur_wcs[n0];
                    float wcs1 = cur_wcs[n0 + 1];
                    acc[mi][ni][0] += (static_cast<float>(mma_acc[mi][ni][0]) + azp0 * wcs0) * sx0 * sw0;
                    acc[mi][ni][1] += (static_cast<float>(mma_acc[mi][ni][1]) + azp0 * wcs1) * sx0 * sw1;
                    acc[mi][ni][2] += (static_cast<float>(mma_acc[mi][ni][2]) + azp1 * wcs0) * sx1 * sw0;
                    acc[mi][ni][3] += (static_cast<float>(mma_acc[mi][ni][3]) + azp1 * wcs1) * sx1 * sw1;
                } else {
                    acc[mi][ni][0] += static_cast<float>(mma_acc[mi][ni][0]) * sx0 * sw0;
                    acc[mi][ni][1] += static_cast<float>(mma_acc[mi][ni][1]) * sx0 * sw1;
                    acc[mi][ni][2] += static_cast<float>(mma_acc[mi][ni][2]) * sx1 * sw0;
                    acc[mi][ni][3] += static_cast<float>(mma_acc[mi][ni][3]) * sx1 * sw1;
                }
            }
        }

        __syncthreads();
    }

    // ── Write FP32 → BF16 output ──
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int gm0 = tile_m + warp_row_start + mi * MMA_M + frag_row;
            int gm1 = gm0 + 8;
            int gn0 = tile_n + warp_col_start + ni * MMA_N + frag_grp * 2;
            int gn1 = gn0 + 1;

            if (gm0 < M && gn0 < N) out[gm0 * N + gn0] = __float2bfloat16(acc[mi][ni][0]);
            if (gm0 < M && gn1 < N) out[gm0 * N + gn1] = __float2bfloat16(acc[mi][ni][1]);
            if (gm1 < M && gn0 < N) out[gm1 * N + gn0] = __float2bfloat16(acc[mi][ni][2]);
            if (gm1 < M && gn1 < N) out[gm1 * N + gn1] = __float2bfloat16(acc[mi][ni][3]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V9 kernel: V8's 16-byte loads + 3-stage pipeline + load-after-compute
//
// Combines V8's optimized memory access with V7's pipeline pattern:
//   - 16-byte cp.async loads (4× fewer load instructions than V4)
//   - SMEM stride = 80 (16-byte aligned, bank-conflict-free)
//   - 3-stage triple buffering → 2 outstanding prefetches
//   - load-after-compute → only 1 __syncthreads per group (vs V4's 2)
//   - Same TILE_M=64, TILE_N=128, 128 threads (V4 tile size)
//
// SMEM: 3 × 16128 = 48384 bytes = 47.25KB (under 48KB default, no opt-in!)
// Maintains 2 blocks/SM occupancy: 2 × 48384 = 96768 < 101376 (99KB)
// ─────────────────────────────────────────────────────────────────────────────

template <bool USE_AZP, int GS = 128>
__global__ void __launch_bounds__(128)
fused_int4_grouped_gemm_v9_kernel(
    const uint8_t* __restrict__ x_packed,   // [num_groups, M, gs/2]
    const uint8_t* __restrict__ w_packed,   // [num_groups, N, gs/2]
    const float* __restrict__ scale_x,      // [num_groups, M]
    const float* __restrict__ scale_w,      // [num_groups, N]
    const float* __restrict__ azp_adj,      // [num_groups, M] (USE_AZP only)
    const float* __restrict__ w_col_sum,    // [num_groups, N] (USE_AZP only)
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const int M, const int N,
    const int num_groups)
{
    // ── Compile-time constants ──
    static constexpr int TILE_M = 64;
    static constexpr int TILE_N = 128;
    static constexpr int THREADS = 128;
    static constexpr int STAGES = 3;
    static constexpr int gs_half = GS / 2;           // = 64
    static constexpr int mma_k_iters = GS / MMA_K;   // = 2
    static constexpr int A_STRIDE = 80;               // 16-byte aligned
    static constexpr int B_STRIDE = 80;
    static constexpr int a_u32_stride = A_STRIDE / 4; // = 20
    static constexpr int b_u32_stride = B_STRIDE / 4; // = 20
    static constexpr int VECTORS_PER_ROW = gs_half / 16;  // = 4

    static constexpr int scale_floats = TILE_M + TILE_N;
    static constexpr int azp_floats = USE_AZP ? (TILE_M + TILE_N) : 0;
    static constexpr int STAGE_BYTES =
        TILE_M * A_STRIDE + TILE_N * B_STRIDE
        + (scale_floats + azp_floats) * (int)sizeof(float);

    // ── Thread/warp decomposition (same as V4) ──
    const int tile_m = blockIdx.x * TILE_M;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int warp_row_start = warp_m * 32;
    const int warp_col_start = warp_n * 64;
    const int frag_row = lane_id / 4;
    const int frag_grp = lane_id % 4;

    // ── Triple-buffered shared memory ──
    extern __shared__ uint8_t smem_v9[];

    // ── FP32 accumulator ──
    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            #pragma unroll
            for (int fi = 0; fi < 4; fi++)
                acc[mi][ni][fi] = 0.f;

    // ── Helper: issue 16-byte cp.async loads for a group into a stage ──
    auto v9_load_stage = [&](int g, int stage) __attribute__((always_inline)) {
        uint8_t* sa = smem_v9 + stage * STAGE_BYTES;
        uint8_t* sb = sa + TILE_M * A_STRIDE;
        float* ssx = reinterpret_cast<float*>(sb + TILE_N * B_STRIDE);
        float* ssw = ssx + TILE_M;

        const uint8_t* xg = x_packed + (int64_t)g * M * gs_half;
        const uint8_t* wg = w_packed + (int64_t)g * N * gs_half;

        // A tile [64, 64] via cp.async 16-byte: 256 ops / 128 threads = 2/thread
        {
            constexpr int total_a = TILE_M * VECTORS_PER_ROW;  // = 256
            for (int i = tid; i < total_a; i += THREADS) {
                const int row = i / VECTORS_PER_ROW;
                const int col = i % VECTORS_PER_ROW;
                const int gr = tile_m + row;
                uint8_t* dst = sa + row * A_STRIDE + col * 16;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < M) {
                    const void* src = xg + (int64_t)gr * gs_half + col * 16;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    reinterpret_cast<uint4*>(dst)[0] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        // B tile [128, 64] via cp.async 16-byte: 512 ops / 128 threads = 4/thread
        {
            constexpr int total_b = TILE_N * VECTORS_PER_ROW;  // = 512
            for (int i = tid; i < total_b; i += THREADS) {
                const int row = i / VECTORS_PER_ROW;
                const int col = i % VECTORS_PER_ROW;
                const int gr = tile_n + row;
                uint8_t* dst = sb + row * B_STRIDE + col * 16;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < N) {
                    const void* src = wg + (int64_t)gr * gs_half + col * 16;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    reinterpret_cast<uint4*>(dst)[0] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        // Scales via regular stores
        const float* sxg = scale_x + (int64_t)g * M;
        const float* swg = scale_w + (int64_t)g * N;
        for (int i = tid; i < TILE_M; i += THREADS) {
            int r = tile_m + i;
            ssx[i] = (r < M) ? sxg[r] : 0.f;
        }
        for (int i = tid; i < TILE_N; i += THREADS) {
            int c = tile_n + i;
            ssw[i] = (c < N) ? swg[c] : 0.f;
        }
        if constexpr (USE_AZP) {
            float* sazp = ssw + TILE_N;
            float* swcs = sazp + TILE_M;
            const float* azpg = azp_adj + (int64_t)g * M;
            const float* wcsg = w_col_sum + (int64_t)g * N;
            for (int i = tid; i < TILE_M; i += THREADS) {
                int r = tile_m + i;
                sazp[i] = (r < M) ? azpg[r] : 0.f;
            }
            for (int i = tid; i < TILE_N; i += THREADS) {
                int c = tile_n + i;
                swcs[i] = (c < N) ? wcsg[c] : 0.f;
            }
        }

        asm volatile("cp.async.commit_group;\n" ::);
    };

    // ══ Prologue: load groups 0 and 1 into stages 0 and 1 ══
    v9_load_stage(0, 0);
    if (num_groups > 1) {
        v9_load_stage(1, 1);
    }

    // ══ Main group loop with triple buffering ══
    // At each iteration g:
    //   - Wait for stage g%3 to be ready
    //   - Single __syncthreads (protects load + prev reads)
    //   - Compute MMA on stage g%3
    //   - Apply per-group scales
    //   - Prefetch group g+2 into stage (g+2)%3
    // No WAR hazard: (g+2)%3 ≠ g%3 for all g (triple buffer property)
    for (int g = 0; g < num_groups; g++) {
        // Wait for current group's data
        if (g + 1 < num_groups) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(1));
        } else {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
        }
        __syncthreads();  // single sync per group

        // ── Stage pointers ──
        const int cur = g % STAGES;
        const uint8_t* cur_base = smem_v9 + cur * STAGE_BYTES;
        const uint32_t* a_u32 = reinterpret_cast<const uint32_t*>(cur_base);
        const uint32_t* b_u32 = reinterpret_cast<const uint32_t*>(
            cur_base + TILE_M * A_STRIDE);
        const float* cur_sx = reinterpret_cast<const float*>(
            cur_base + TILE_M * A_STRIDE + TILE_N * B_STRIDE);
        const float* cur_sw = cur_sx + TILE_M;
        const float* cur_azp = cur_sw + TILE_N;
        const float* cur_wcs = cur_azp + TILE_M;

        // ── MMA computation ──
        int32_t mma_acc[2][8][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 8; ni++)
                #pragma unroll
                for (int fi = 0; fi < 4; fi++)
                    mma_acc[mi][ni][fi] = 0;

        #pragma unroll
        for (int ki = 0; ki < mma_k_iters; ki++) {
            const int k_u32_off = ki * (MMA_K / 2) / 4;

            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int row0 = warp_row_start + mi * MMA_M + frag_row;
                int row1 = row0 + 8;
                int base0 = row0 * a_u32_stride + k_u32_off + frag_grp * 2;
                int base1 = row1 * a_u32_stride + k_u32_off + frag_grp * 2;
                a_frag[mi][0] = a_u32[base0];
                a_frag[mi][1] = a_u32[base1];
                a_frag[mi][2] = a_u32[base0 + 1];
                a_frag[mi][3] = a_u32[base1 + 1];
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int b_row = warp_col_start + ni * MMA_N + frag_row;
                    int b_base = b_row * b_u32_stride + k_u32_off + frag_grp * 2;
                    uint32_t b0 = b_u32[b_base];
                    uint32_t b1 = b_u32[b_base + 1];

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
                    asm volatile(
                        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(mma_acc[mi][ni][0]), "=r"(mma_acc[mi][ni][1]),
                          "=r"(mma_acc[mi][ni][2]), "=r"(mma_acc[mi][ni][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b0), "r"(b1),
                          "r"(mma_acc[mi][ni][0]), "r"(mma_acc[mi][ni][1]),
                          "r"(mma_acc[mi][ni][2]), "r"(mma_acc[mi][ni][3]));
#endif
                }
            }
        }

        // ── Apply per-group scales ──
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m0 = warp_row_start + mi * MMA_M + frag_row;
            int m1 = m0 + 8;
            float sx0 = cur_sx[m0];
            float sx1 = cur_sx[m1];
            float azp0 = 0.f, azp1 = 0.f;
            if constexpr (USE_AZP) {
                azp0 = cur_azp[m0];
                azp1 = cur_azp[m1];
            }

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                int n0 = warp_col_start + ni * MMA_N + frag_grp * 2;
                float sw0 = cur_sw[n0];
                float sw1 = cur_sw[n0 + 1];

                if constexpr (USE_AZP) {
                    float wcs0 = cur_wcs[n0];
                    float wcs1 = cur_wcs[n0 + 1];
                    acc[mi][ni][0] += (static_cast<float>(mma_acc[mi][ni][0]) + azp0 * wcs0) * sx0 * sw0;
                    acc[mi][ni][1] += (static_cast<float>(mma_acc[mi][ni][1]) + azp0 * wcs1) * sx0 * sw1;
                    acc[mi][ni][2] += (static_cast<float>(mma_acc[mi][ni][2]) + azp1 * wcs0) * sx1 * sw0;
                    acc[mi][ni][3] += (static_cast<float>(mma_acc[mi][ni][3]) + azp1 * wcs1) * sx1 * sw1;
                } else {
                    acc[mi][ni][0] += static_cast<float>(mma_acc[mi][ni][0]) * sx0 * sw0;
                    acc[mi][ni][1] += static_cast<float>(mma_acc[mi][ni][1]) * sx0 * sw1;
                    acc[mi][ni][2] += static_cast<float>(mma_acc[mi][ni][2]) * sx1 * sw0;
                    acc[mi][ni][3] += static_cast<float>(mma_acc[mi][ni][3]) * sx1 * sw1;
                }
            }
        }

        // ── Prefetch: load group g+2 into stage (g+2) % 3 ──
        if (g + 2 < num_groups) {
            v9_load_stage(g + 2, (g + 2) % STAGES);
        }
    }

    // ── Write FP32 → BF16 output ──
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int gm0 = tile_m + warp_row_start + mi * MMA_M + frag_row;
            int gm1 = gm0 + 8;
            int gn0 = tile_n + warp_col_start + ni * MMA_N + frag_grp * 2;
            int gn1 = gn0 + 1;

            if (gm0 < M && gn0 < N) out[gm0 * N + gn0] = __float2bfloat16(acc[mi][ni][0]);
            if (gm0 < M && gn1 < N) out[gm0 * N + gn1] = __float2bfloat16(acc[mi][ni][1]);
            if (gm1 < M && gn0 < N) out[gm1 * N + gn0] = __float2bfloat16(acc[mi][ni][2]);
            if (gm1 < M && gn1 < N) out[gm1 * N + gn1] = __float2bfloat16(acc[mi][ni][3]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V7 kernel: 128×128 tile, 3-stage pipeline, 256 threads (8 warps)
//
// Key optimizations vs V4 (64×128, 2-stage, 128 threads):
//   - 128×128 tile → 2× compute per load, better compute/memory ratio
//   - 3-stage cp.async pipeline → 2 outstanding prefetches
//   - load-after-compute pattern → only 1 __syncthreads per group (vs V4's 2)
//   - 256 threads (8 warps) → more warp-level parallelism for latency hiding
//
// SMEM budget (GS=128, non-AZP):
//   Stage = A[128×68] + B[128×68] + scales[256×4] = 18432 bytes
//   3 stages = 55296 bytes ≈ 54KB (fits L4's 99KB opt-in limit)
// ─────────────────────────────────────────────────────────────────────────────

template <bool USE_AZP, int GS = 128>
__global__ void __launch_bounds__(256)
fused_int4_grouped_gemm_v7_kernel(
    const uint8_t* __restrict__ x_packed,   // [num_groups, M, gs/2]
    const uint8_t* __restrict__ w_packed,   // [num_groups, N, gs/2]
    const float* __restrict__ scale_x,      // [num_groups, M]
    const float* __restrict__ scale_w,      // [num_groups, N]
    const float* __restrict__ azp_adj,      // [num_groups, M] (USE_AZP only)
    const float* __restrict__ w_col_sum,    // [num_groups, N] (USE_AZP only)
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const int M, const int N,
    const int num_groups)
{
    // ── Compile-time constants ──
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int THREADS = 256;
    static constexpr int STAGES = 3;
    static constexpr int gs = GS;
    static constexpr int gs_half = GS / 2;
    static constexpr int mma_k_iters = GS / MMA_K;  // = 2 for GS=128
    static constexpr int A_STRIDE = gs_half + 4;     // padding for bank conflicts
    static constexpr int B_STRIDE = gs_half + 4;
    static constexpr int a_u32_stride = A_STRIDE / 4;
    static constexpr int b_u32_stride = B_STRIDE / 4;

    // Per-stage SMEM: A_tile | B_tile | sx | sw [| azp | wcs]
    static constexpr int scale_floats = TILE_M + TILE_N;
    static constexpr int azp_floats = USE_AZP ? (TILE_M + TILE_N) : 0;
    static constexpr int STAGE_BYTES =
        TILE_M * A_STRIDE + TILE_N * B_STRIDE
        + (scale_floats + azp_floats) * (int)sizeof(float);

    // ── Thread/warp decomposition: 8 warps in 4M × 2N ──
    const int tile_m = blockIdx.x * TILE_M;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;       // 0..3
    const int warp_n = warp_id % 2;       // 0..1
    const int warp_row_start = warp_m * 32;
    const int warp_col_start = warp_n * 64;
    const int frag_row = lane_id / 4;
    const int frag_grp = lane_id % 4;

    // ── Triple-buffered shared memory ──
    extern __shared__ uint8_t smem_v7[];

    // ── FP32 accumulator (per-thread: 2×8×4 = 64 values) ──
    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            #pragma unroll
            for (int fi = 0; fi < 4; fi++)
                acc[mi][ni][fi] = 0.f;

    // ── Helper: issue cp.async loads for a group into a SMEM stage ──
    auto v7_load_stage = [&](int g, int stage) __attribute__((always_inline)) {
        uint8_t* sa = smem_v7 + stage * STAGE_BYTES;
        uint8_t* sb = sa + TILE_M * A_STRIDE;
        float* ssx = reinterpret_cast<float*>(sb + TILE_N * B_STRIDE);
        float* ssw = ssx + TILE_M;

        const uint8_t* xg = x_packed + (int64_t)g * M * gs_half;
        const uint8_t* wg = w_packed + (int64_t)g * N * gs_half;

        // A tile [TILE_M=128, gs_half=64] via cp.async 4-byte
        {
            constexpr int a_words = gs_half / 4;   // = 16
            constexpr int total_a = TILE_M * a_words;  // = 2048
            for (int i = tid; i < total_a; i += THREADS) {
                const int row = i / a_words;
                const int col = i % a_words;
                const int gr = tile_m + row;
                uint8_t* dst = sa + row * A_STRIDE + col * 4;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < M) {
                    const void* src = xg + (int64_t)gr * gs_half + col * 4;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    *reinterpret_cast<uint32_t*>(dst) = 0u;
                }
            }
        }

        // B tile [TILE_N=128, gs_half=64] via cp.async 4-byte
        {
            constexpr int b_words = gs_half / 4;   // = 16
            constexpr int total_b = TILE_N * b_words;  // = 2048
            for (int i = tid; i < total_b; i += THREADS) {
                const int row = i / b_words;
                const int col = i % b_words;
                const int gr = tile_n + row;
                uint8_t* dst = sb + row * B_STRIDE + col * 4;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < N) {
                    const void* src = wg + (int64_t)gr * gs_half + col * 4;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    *reinterpret_cast<uint32_t*>(dst) = 0u;
                }
            }
        }

        // Scales via regular stores (small, visible after __syncthreads)
        const float* sxg = scale_x + (int64_t)g * M;
        const float* swg = scale_w + (int64_t)g * N;
        for (int i = tid; i < TILE_M; i += THREADS) {
            int r = tile_m + i;
            ssx[i] = (r < M) ? sxg[r] : 0.f;
        }
        for (int i = tid; i < TILE_N; i += THREADS) {
            int c = tile_n + i;
            ssw[i] = (c < N) ? swg[c] : 0.f;
        }
        if constexpr (USE_AZP) {
            float* sazp = ssw + TILE_N;
            float* swcs = sazp + TILE_M;
            const float* azpg = azp_adj + (int64_t)g * M;
            const float* wcsg = w_col_sum + (int64_t)g * N;
            for (int i = tid; i < TILE_M; i += THREADS) {
                int r = tile_m + i;
                sazp[i] = (r < M) ? azpg[r] : 0.f;
            }
            for (int i = tid; i < TILE_N; i += THREADS) {
                int c = tile_n + i;
                swcs[i] = (c < N) ? wcsg[c] : 0.f;
            }
        }

        asm volatile("cp.async.commit_group;\n" ::);
    };

    // ══ Prologue: load groups 0 and 1 into stages 0 and 1 ══
    v7_load_stage(0, 0);
    if (num_groups > 1) {
        v7_load_stage(1, 1);
    }

    // ══ Main group loop with triple buffering ══
    // Pipeline invariant: at each iteration g, stages g%3 (current) and
    // (g+1)%3 (next) are loaded/in-flight. We compute on g%3 and prefetch
    // g+2 into (g+2)%3. The 3 stages are always distinct (g%3, (g+1)%3,
    // (g+2)%3), so there are no WAR hazards.
    for (int g = 0; g < num_groups; g++) {
        // Wait for current group's data to arrive
        if (g + 1 < num_groups) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(1));
        } else {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
        }
        __syncthreads();  // single sync: protects both load completion AND
                          // previous iteration's SMEM reads

        // ── Stage pointers ──
        const int cur = g % STAGES;
        const uint8_t* cur_base = smem_v7 + cur * STAGE_BYTES;
        const uint32_t* a_u32 = reinterpret_cast<const uint32_t*>(cur_base);
        const uint32_t* b_u32 = reinterpret_cast<const uint32_t*>(
            cur_base + TILE_M * A_STRIDE);
        const float* cur_sx = reinterpret_cast<const float*>(
            cur_base + TILE_M * A_STRIDE + TILE_N * B_STRIDE);
        const float* cur_sw = cur_sx + TILE_M;
        const float* cur_azp = cur_sw + TILE_N;
        const float* cur_wcs = cur_azp + TILE_M;

        // ── MMA computation ──
        int32_t mma_acc[2][8][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 8; ni++)
                #pragma unroll
                for (int fi = 0; fi < 4; fi++)
                    mma_acc[mi][ni][fi] = 0;

        #pragma unroll
        for (int ki = 0; ki < mma_k_iters; ki++) {
            const int k_u32_off = ki * (MMA_K / 2) / 4;

            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int row0 = warp_row_start + mi * MMA_M + frag_row;
                int row1 = row0 + 8;
                int base0 = row0 * a_u32_stride + k_u32_off + frag_grp * 2;
                int base1 = row1 * a_u32_stride + k_u32_off + frag_grp * 2;
                a_frag[mi][0] = a_u32[base0];
                a_frag[mi][1] = a_u32[base1];
                a_frag[mi][2] = a_u32[base0 + 1];
                a_frag[mi][3] = a_u32[base1 + 1];
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int b_row = warp_col_start + ni * MMA_N + frag_row;
                    int b_base = b_row * b_u32_stride + k_u32_off + frag_grp * 2;
                    uint32_t b0 = b_u32[b_base];
                    uint32_t b1 = b_u32[b_base + 1];

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
                    asm volatile(
                        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(mma_acc[mi][ni][0]), "=r"(mma_acc[mi][ni][1]),
                          "=r"(mma_acc[mi][ni][2]), "=r"(mma_acc[mi][ni][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b0), "r"(b1),
                          "r"(mma_acc[mi][ni][0]), "r"(mma_acc[mi][ni][1]),
                          "r"(mma_acc[mi][ni][2]), "r"(mma_acc[mi][ni][3]));
#endif
                }
            }
        }

        // ── Apply per-group scales and accumulate into FP32 ──
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m0 = warp_row_start + mi * MMA_M + frag_row;
            int m1 = m0 + 8;
            float sx0 = cur_sx[m0];
            float sx1 = cur_sx[m1];
            float azp0 = 0.f, azp1 = 0.f;
            if constexpr (USE_AZP) {
                azp0 = cur_azp[m0];
                azp1 = cur_azp[m1];
            }

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                int n0 = warp_col_start + ni * MMA_N + frag_grp * 2;
                float sw0 = cur_sw[n0];
                float sw1 = cur_sw[n0 + 1];

                if constexpr (USE_AZP) {
                    float wcs0 = cur_wcs[n0];
                    float wcs1 = cur_wcs[n0 + 1];
                    acc[mi][ni][0] += (static_cast<float>(mma_acc[mi][ni][0]) + azp0 * wcs0) * sx0 * sw0;
                    acc[mi][ni][1] += (static_cast<float>(mma_acc[mi][ni][1]) + azp0 * wcs1) * sx0 * sw1;
                    acc[mi][ni][2] += (static_cast<float>(mma_acc[mi][ni][2]) + azp1 * wcs0) * sx1 * sw0;
                    acc[mi][ni][3] += (static_cast<float>(mma_acc[mi][ni][3]) + azp1 * wcs1) * sx1 * sw1;
                } else {
                    acc[mi][ni][0] += static_cast<float>(mma_acc[mi][ni][0]) * sx0 * sw0;
                    acc[mi][ni][1] += static_cast<float>(mma_acc[mi][ni][1]) * sx0 * sw1;
                    acc[mi][ni][2] += static_cast<float>(mma_acc[mi][ni][2]) * sx1 * sw0;
                    acc[mi][ni][3] += static_cast<float>(mma_acc[mi][ni][3]) * sx1 * sw1;
                }
            }
        }

        // ── Prefetch: load group g+2 into stage (g+2) % 3 ──
        // Safe because (g+2)%3 ≠ g%3, and __syncthreads at top of next
        // iteration ensures all reads from stage (g+1)%3 complete before
        // we overwrite it at iteration g+1.
        if (g + 2 < num_groups) {
            v7_load_stage(g + 2, (g + 2) % STAGES);
        }
    }

    // ── Write FP32 → BF16 output ──
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int gm0 = tile_m + warp_row_start + mi * MMA_M + frag_row;
            int gm1 = gm0 + 8;
            int gn0 = tile_n + warp_col_start + ni * MMA_N + frag_grp * 2;
            int gn1 = gn0 + 1;

            if (gm0 < M && gn0 < N) out[gm0 * N + gn0] = __float2bfloat16(acc[mi][ni][0]);
            if (gm0 < M && gn1 < N) out[gm0 * N + gn1] = __float2bfloat16(acc[mi][ni][1]);
            if (gm1 < M && gn0 < N) out[gm1 * N + gn0] = __float2bfloat16(acc[mi][ni][2]);
            if (gm1 < M && gn1 < N) out[gm1 * N + gn1] = __float2bfloat16(acc[mi][ni][3]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_fused_grouped_gemm_v5
//
// V5 multi-group K-loop fused GEMM. Loads GROUPS_PER_LOAD groups per SMEM
// stage to reduce __syncthreads() by GROUPS_PER_LOAD×.
// ─────────────────────────────────────────────────────────────────────────────

// Helper: compute V5 SMEM stage size for given GPL and USE_AZP
template <bool USE_AZP>
static inline size_t v5_smem_size(int gpl) {
    constexpr int V5_TILE_M = 64;
    constexpr int V5_TILE_N = 128;
    constexpr int gs128_half = 64;
    int a_stride = gpl * gs128_half + 4;
    int b_stride = gpl * gs128_half + 4;
    int scale_flt = gpl * (V5_TILE_M + V5_TILE_N);
    int azp_flt = USE_AZP ? gpl * (V5_TILE_M + V5_TILE_N) : 0;
    int stage = V5_TILE_M * a_stride + V5_TILE_N * b_stride
              + (scale_flt + azp_flt) * (int)sizeof(float);
    return 2 * stage;  // double buffered
}

// Helper: try to set max dynamic SMEM. Returns true if successful.
template <typename KernelFunc>
static inline bool v5_try_set_smem(KernelFunc func, size_t smem_size) {
    if (smem_size <= 48 * 1024) return true;  // default limit, always works
    cudaError_t err = cudaFuncSetAttribute(
        func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
        cudaGetLastError();  // clear the error
        return false;
    }
    return true;
}

void cutlass_int4_fused_grouped_gemm_v5(
    torch::Tensor a_packed,     // [num_groups, M, gs/2] uint8
    torch::Tensor b_packed,     // [num_groups, N, gs/2] uint8
    torch::Tensor scale_a,      // [num_groups, M] float
    torch::Tensor scale_b,      // [num_groups, N] float
    torch::Tensor out,          // [M, N] bf16
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size == 128,
                "V5 kernel currently only supports group_size=128");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,
                "cutlass_int4_fused_grouped_gemm_v5 currently supports BF16 output only");

    auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device());

    if (M > 32) {
        constexpr int V5_TILE_M = 64;
        constexpr int V5_TILE_N = 128;
        constexpr int V5_THREADS = 128;
        dim3 grid((M + V5_TILE_M - 1) / V5_TILE_M,
                  (N + V5_TILE_N - 1) / V5_TILE_N);
        dim3 block(V5_THREADS);

        // Try GPL=4 first if num_groups is divisible and SMEM fits the GPU
        bool use_gpl4 = false;
        if (num_groups % 4 == 0) {
            size_t smem4 = v5_smem_size<false>(4);
            use_gpl4 = v5_try_set_smem(
                fused_int4_grouped_gemm_v5_kernel<false, 128, 4>, smem4);
            if (use_gpl4) {
                fused_int4_grouped_gemm_v5_kernel<false, 128, 4>
                    <<<grid, block, smem4, stream>>>(
                    a_packed.data_ptr<uint8_t>(),
                    b_packed.data_ptr<uint8_t>(),
                    scale_a.data_ptr<float>(),
                    scale_b.data_ptr<float>(),
                    nullptr, nullptr,
                    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                    static_cast<int>(M), static_cast<int>(N),
                    static_cast<int>(num_groups));
            }
        }

        if (!use_gpl4) {
            // Fallback to GPL=2 (52.5 KB SMEM, fits all SM80+ GPUs)
            size_t smem2 = v5_smem_size<false>(2);
            v5_try_set_smem(
                fused_int4_grouped_gemm_v5_kernel<false, 128, 2>, smem2);
            fused_int4_grouped_gemm_v5_kernel<false, 128, 2>
                <<<grid, block, smem2, stream>>>(
                a_packed.data_ptr<uint8_t>(),
                b_packed.data_ptr<uint8_t>(),
                scale_a.data_ptr<float>(),
                scale_b.data_ptr<float>(),
                nullptr, nullptr,
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                static_cast<int>(M), static_cast<int>(N),
                static_cast<int>(num_groups));
        }
    } else {
        // Small M: fall back to V1 kernel (32×32 tiles)
        int gs_half = static_cast<int>(group_size / 2);
        dim3 grid((M + FUSED_TILE_M - 1) / FUSED_TILE_M,
                  (N + FUSED_TILE_N - 1) / FUSED_TILE_N);
        dim3 block(FUSED_THREADS);
        size_t smem_size = FUSED_TILE_M * gs_half + FUSED_TILE_N * gs_half
                         + (FUSED_TILE_M + FUSED_TILE_N) * sizeof(float);
        fused_int4_grouped_gemm_kernel<<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups), static_cast<int>(group_size));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_fused_grouped_gemm_v5_azp
//
// V5 multi-group K-loop fused GEMM with AZP correction.
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_int4_fused_grouped_gemm_v5_azp(
    torch::Tensor a_packed,     // [num_groups, M, gs/2] uint8
    torch::Tensor b_packed,     // [num_groups, N, gs/2] uint8
    torch::Tensor scale_a,      // [num_groups, M] float
    torch::Tensor scale_b,      // [num_groups, N] float
    torch::Tensor out,          // [M, N] bf16
    torch::Tensor azp_adj,      // [num_groups, M] float
    torch::Tensor w_col_sum,    // [num_groups, N] float
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size == 128,
                "V5 kernel currently only supports group_size=128");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,
                "cutlass_int4_fused_grouped_gemm_v5_azp currently supports BF16 output only");

    auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device());

    if (M > 32) {
        constexpr int V5_TILE_M = 64;
        constexpr int V5_TILE_N = 128;
        constexpr int V5_THREADS = 128;
        dim3 grid((M + V5_TILE_M - 1) / V5_TILE_M,
                  (N + V5_TILE_N - 1) / V5_TILE_N);
        dim3 block(V5_THREADS);

        // Try GPL=4 first if num_groups is divisible and SMEM fits the GPU
        bool use_gpl4 = false;
        if (num_groups % 4 == 0) {
            size_t smem4 = v5_smem_size<true>(4);
            use_gpl4 = v5_try_set_smem(
                fused_int4_grouped_gemm_v5_kernel<true, 128, 4>, smem4);
            if (use_gpl4) {
                fused_int4_grouped_gemm_v5_kernel<true, 128, 4>
                    <<<grid, block, smem4, stream>>>(
                    a_packed.data_ptr<uint8_t>(),
                    b_packed.data_ptr<uint8_t>(),
                    scale_a.data_ptr<float>(),
                    scale_b.data_ptr<float>(),
                    azp_adj.data_ptr<float>(),
                    w_col_sum.data_ptr<float>(),
                    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                    static_cast<int>(M), static_cast<int>(N),
                    static_cast<int>(num_groups));
            }
        }

        if (!use_gpl4) {
            // Fallback to GPL=2 (55.5 KB SMEM with AZP, fits all SM80+ GPUs)
            size_t smem2 = v5_smem_size<true>(2);
            v5_try_set_smem(
                fused_int4_grouped_gemm_v5_kernel<true, 128, 2>, smem2);
            fused_int4_grouped_gemm_v5_kernel<true, 128, 2>
                <<<grid, block, smem2, stream>>>(
                a_packed.data_ptr<uint8_t>(),
                b_packed.data_ptr<uint8_t>(),
                scale_a.data_ptr<float>(),
                scale_b.data_ptr<float>(),
                azp_adj.data_ptr<float>(),
                w_col_sum.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                static_cast<int>(M), static_cast<int>(N),
                static_cast<int>(num_groups));
        }
    } else {
        // Small M: fall back to V1 AZP kernel (32×32 tiles)
        int gs_half = static_cast<int>(group_size / 2);
        dim3 grid((M + FUSED_TILE_M - 1) / FUSED_TILE_M,
                  (N + FUSED_TILE_N - 1) / FUSED_TILE_N);
        dim3 block(FUSED_THREADS);
        size_t smem_size = FUSED_TILE_M * gs_half + FUSED_TILE_N * gs_half
                         + (FUSED_TILE_M * 2 + FUSED_TILE_N * 2) * sizeof(float);
        fused_int4_grouped_gemm_azp_kernel<<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            azp_adj.data_ptr<float>(),
            w_col_sum.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups), static_cast<int>(group_size));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_fused_grouped_gemm_v9
//
// V9: V8's 16-byte loads + 3-stage pipeline + load-after-compute.
// SMEM: 3 × 16128 = 48384 bytes (under 48KB default, no opt-in needed).
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_int4_fused_grouped_gemm_v9(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size == 128,
                "V9 kernel currently only supports group_size=128");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,
                "cutlass_int4_fused_grouped_gemm_v9 currently supports BF16 output only");

    auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device());

    if (M > 32) {
        constexpr int V9_TILE_M = 64;
        constexpr int V9_TILE_N = 128;
        constexpr int V9_THREADS = 128;
        constexpr int V9_A_STRIDE = 80;
        constexpr int V9_B_STRIDE = 80;
        constexpr int scale_flt = V9_TILE_M + V9_TILE_N;
        constexpr int stage_bytes = V9_TILE_M * V9_A_STRIDE + V9_TILE_N * V9_B_STRIDE
                                  + scale_flt * (int)sizeof(float);
        size_t smem_size = 3 * stage_bytes;  // triple buffer (48384 bytes)

        // Maximize SMEM carveout to allow 2 blocks/SM with 48KB SMEM each
        auto kernel_fn = fused_int4_grouped_gemm_v9_kernel<false, 128>;
        cudaFuncSetAttribute(kernel_fn,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        cudaFuncSetAttribute(kernel_fn,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);

        dim3 grid((M + V9_TILE_M - 1) / V9_TILE_M,
                  (N + V9_TILE_N - 1) / V9_TILE_N);
        dim3 block(V9_THREADS);

        kernel_fn<<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            nullptr, nullptr,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups));
    } else {
        int gs_half = static_cast<int>(group_size / 2);
        dim3 grid((M + FUSED_TILE_M - 1) / FUSED_TILE_M,
                  (N + FUSED_TILE_N - 1) / FUSED_TILE_N);
        dim3 block(FUSED_THREADS);
        size_t smem_size = FUSED_TILE_M * gs_half + FUSED_TILE_N * gs_half
                         + (FUSED_TILE_M + FUSED_TILE_N) * sizeof(float);
        fused_int4_grouped_gemm_kernel<<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups), static_cast<int>(group_size));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_fused_grouped_gemm_v8
//
// V8: V4 tile (64×128) with 16-byte vectorized cp.async loads.
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_int4_fused_grouped_gemm_v8(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor out,
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size == 128,
                "V8 kernel currently only supports group_size=128");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,
                "cutlass_int4_fused_grouped_gemm_v8 currently supports BF16 output only");

    auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device());

    if (M > 32) {
        constexpr int V8_TILE_M = 64;
        constexpr int V8_TILE_N = 128;
        constexpr int V8_THREADS = 128;
        constexpr int V8_A_STRIDE = 80;
        constexpr int V8_B_STRIDE = 80;
        constexpr int scale_flt = V8_TILE_M + V8_TILE_N;
        constexpr int stage_bytes = V8_TILE_M * V8_A_STRIDE + V8_TILE_N * V8_B_STRIDE
                                  + scale_flt * (int)sizeof(float);
        size_t smem_size = 2 * stage_bytes;  // double buffer

        dim3 grid((M + V8_TILE_M - 1) / V8_TILE_M,
                  (N + V8_TILE_N - 1) / V8_TILE_N);
        dim3 block(V8_THREADS);

        fused_int4_grouped_gemm_v8_kernel<false, 128><<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            nullptr, nullptr,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups));
    } else {
        int gs_half = static_cast<int>(group_size / 2);
        dim3 grid((M + FUSED_TILE_M - 1) / FUSED_TILE_M,
                  (N + FUSED_TILE_N - 1) / FUSED_TILE_N);
        dim3 block(FUSED_THREADS);
        size_t smem_size = FUSED_TILE_M * gs_half + FUSED_TILE_N * gs_half
                         + (FUSED_TILE_M + FUSED_TILE_N) * sizeof(float);
        fused_int4_grouped_gemm_kernel<<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups), static_cast<int>(group_size));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_fused_grouped_gemm_v7
//
// V7: 128×128 tile, 3-stage pipeline, 256 threads.
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_int4_fused_grouped_gemm_v7(
    torch::Tensor a_packed,     // [num_groups, M, gs/2] uint8
    torch::Tensor b_packed,     // [num_groups, N, gs/2] uint8
    torch::Tensor scale_a,      // [num_groups, M] float
    torch::Tensor scale_b,      // [num_groups, N] float
    torch::Tensor out,          // [M, N] bf16
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size == 128,
                "V7 kernel currently only supports group_size=128");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,
                "cutlass_int4_fused_grouped_gemm_v7 currently supports BF16 output only");

    auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device());

    if (M > 32) {
        constexpr int V7_TILE_M = 128;
        constexpr int V7_TILE_N = 128;
        constexpr int V7_THREADS = 256;
        constexpr int gs128_half = 64;
        constexpr int a_stride = gs128_half + 4;
        constexpr int b_stride = gs128_half + 4;
        constexpr int scale_flt = V7_TILE_M + V7_TILE_N;
        constexpr int stage_bytes = V7_TILE_M * a_stride + V7_TILE_N * b_stride
                                  + scale_flt * (int)sizeof(float);
        size_t smem_size = 3 * stage_bytes;  // triple buffer

        dim3 grid((M + V7_TILE_M - 1) / V7_TILE_M,
                  (N + V7_TILE_N - 1) / V7_TILE_N);
        dim3 block(V7_THREADS);

        // Opt-in for extended shared memory (55KB > 48KB default)
        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(
                fused_int4_grouped_gemm_v7_kernel<false, 128>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size);
        }

        fused_int4_grouped_gemm_v7_kernel<false, 128><<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            nullptr, nullptr,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups));
    } else {
        // Small M: fall back to V1 kernel (32×32 tiles)
        int gs_half = static_cast<int>(group_size / 2);
        dim3 grid((M + FUSED_TILE_M - 1) / FUSED_TILE_M,
                  (N + FUSED_TILE_N - 1) / FUSED_TILE_N);
        dim3 block(FUSED_THREADS);
        size_t smem_size = FUSED_TILE_M * gs_half + FUSED_TILE_N * gs_half
                         + (FUSED_TILE_M + FUSED_TILE_N) * sizeof(float);
        fused_int4_grouped_gemm_kernel<<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups), static_cast<int>(group_size));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_fused_grouped_gemm_v7_azp
//
// V7 with AZP correction: 128×128 tile, 3-stage pipeline, 256 threads.
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_int4_fused_grouped_gemm_v7_azp(
    torch::Tensor a_packed,     // [num_groups, M, gs/2] uint8
    torch::Tensor b_packed,     // [num_groups, N, gs/2] uint8
    torch::Tensor scale_a,      // [num_groups, M] float
    torch::Tensor scale_b,      // [num_groups, N] float
    torch::Tensor out,          // [M, N] bf16
    torch::Tensor azp_adj,      // [num_groups, M] float
    torch::Tensor w_col_sum,    // [num_groups, N] float
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size == 128,
                "V7 kernel currently only supports group_size=128");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,
                "cutlass_int4_fused_grouped_gemm_v7_azp currently supports BF16 output only");

    auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device());

    if (M > 32) {
        constexpr int V7_TILE_M = 128;
        constexpr int V7_TILE_N = 128;
        constexpr int V7_THREADS = 256;
        constexpr int gs128_half = 64;
        constexpr int a_stride = gs128_half + 4;
        constexpr int b_stride = gs128_half + 4;
        constexpr int scale_flt = V7_TILE_M + V7_TILE_N;
        constexpr int azp_flt = V7_TILE_M + V7_TILE_N;
        constexpr int stage_bytes = V7_TILE_M * a_stride + V7_TILE_N * b_stride
                                  + (scale_flt + azp_flt) * (int)sizeof(float);
        size_t smem_size = 3 * stage_bytes;  // triple buffer

        dim3 grid((M + V7_TILE_M - 1) / V7_TILE_M,
                  (N + V7_TILE_N - 1) / V7_TILE_N);
        dim3 block(V7_THREADS);

        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(
                fused_int4_grouped_gemm_v7_kernel<true, 128>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size);
        }

        fused_int4_grouped_gemm_v7_kernel<true, 128><<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            azp_adj.data_ptr<float>(),
            w_col_sum.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups));
    } else {
        // Small M: fall back to V1 AZP kernel (32×32 tiles)
        int gs_half = static_cast<int>(group_size / 2);
        dim3 grid((M + FUSED_TILE_M - 1) / FUSED_TILE_M,
                  (N + FUSED_TILE_N - 1) / FUSED_TILE_N);
        dim3 block(FUSED_THREADS);
        size_t smem_size = FUSED_TILE_M * gs_half + FUSED_TILE_N * gs_half
                         + (FUSED_TILE_M * 2 + FUSED_TILE_N * 2) * sizeof(float);
        fused_int4_grouped_gemm_azp_kernel<<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            azp_adj.data_ptr<float>(),
            w_col_sum.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups), static_cast<int>(group_size));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_fused_grouped_gemm
//
// Fused per-group INT4×INT4 GEMM: single kernel launch for all groups.
// Replaces cutlass_int4_scaled_mm_grouped (20 CUTLASS + 20 add_ → 1 kernel).
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_int4_fused_grouped_gemm(
    torch::Tensor a_packed,     // [num_groups, M, gs/2] uint8
    torch::Tensor b_packed,     // [num_groups, N, gs/2] uint8
    torch::Tensor scale_a,      // [num_groups, M] float
    torch::Tensor scale_b,      // [num_groups, N] float
    torch::Tensor out,          // [M, N] bf16
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size % 64 == 0, "group_size must be multiple of 64");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,
                "cutlass_int4_fused_grouped_gemm currently supports BF16 output only");

    auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device());
    int gs_half = static_cast<int>(group_size / 2);

    if (M > 32) {
        // V4 kernel: 64×128 tiles, double-buffered cp.async pipeline
        TORCH_CHECK(group_size == 128,
                    "V4 kernel currently only supports group_size=128");
        constexpr int V4_TILE_M = 64;
        constexpr int V4_TILE_N = 128;
        constexpr int V4_THREADS = 128;
        constexpr int gs128_half = 64;
        constexpr int a_stride = gs128_half + 4;
        constexpr int b_stride = gs128_half + 4;
        constexpr int stage_bytes = V4_TILE_M * a_stride + V4_TILE_N * b_stride
                                  + (V4_TILE_M + V4_TILE_N) * (int)sizeof(float);
        size_t smem_size = 2 * stage_bytes;  // double buffer
        dim3 grid((M + V4_TILE_M - 1) / V4_TILE_M,
                  (N + V4_TILE_N - 1) / V4_TILE_N);
        dim3 block(V4_THREADS);
        fused_int4_grouped_gemm_v4_kernel<false, 128><<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            nullptr, nullptr,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups));
    } else {
        // V1 kernel: 32×32 tiles (better M-utilization for small M)
        dim3 grid((M + FUSED_TILE_M - 1) / FUSED_TILE_M,
                  (N + FUSED_TILE_N - 1) / FUSED_TILE_N);
        dim3 block(FUSED_THREADS);
        size_t smem_size = FUSED_TILE_M * gs_half + FUSED_TILE_N * gs_half
                         + (FUSED_TILE_M + FUSED_TILE_N) * sizeof(float);
        fused_int4_grouped_gemm_kernel<<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups), static_cast<int>(group_size));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_fused_grouped_gemm_azp
//
// Fused per-group INT4×INT4 GEMM with AZP correction.
// ─────────────────────────────────────────────────────────────────────────────

void cutlass_int4_fused_grouped_gemm_azp(
    torch::Tensor a_packed,     // [num_groups, M, gs/2] uint8
    torch::Tensor b_packed,     // [num_groups, N, gs/2] uint8
    torch::Tensor scale_a,      // [num_groups, M] float
    torch::Tensor scale_b,      // [num_groups, N] float
    torch::Tensor out,          // [M, N] bf16
    torch::Tensor azp_adj,      // [num_groups, M] float
    torch::Tensor w_col_sum,    // [num_groups, N] float
    int64_t M, int64_t N, int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(group_size % 64 == 0, "group_size must be multiple of 64");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,
                "cutlass_int4_fused_grouped_gemm_azp currently supports BF16 output only");

    auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device());
    int gs_half = static_cast<int>(group_size / 2);

    if (M > 32) {
        // V4 kernel: 64×128 tiles, double-buffered cp.async pipeline
        TORCH_CHECK(group_size == 128,
                    "V4 kernel currently only supports group_size=128");
        constexpr int V4_TILE_M = 64;
        constexpr int V4_TILE_N = 128;
        constexpr int V4_THREADS = 128;
        constexpr int gs128_half = 64;
        constexpr int a_stride = gs128_half + 4;
        constexpr int b_stride = gs128_half + 4;
        constexpr int stage_bytes = V4_TILE_M * a_stride + V4_TILE_N * b_stride
                                  + (V4_TILE_M * 2 + V4_TILE_N * 2) * (int)sizeof(float);
        size_t smem_size = 2 * stage_bytes;  // double buffer
        dim3 grid((M + V4_TILE_M - 1) / V4_TILE_M,
                  (N + V4_TILE_N - 1) / V4_TILE_N);
        dim3 block(V4_THREADS);
        fused_int4_grouped_gemm_v4_kernel<true, 128><<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            azp_adj.data_ptr<float>(),
            w_col_sum.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups));
    } else {
        // V1 kernel: 32×32 tiles (better M-utilization for small M)
        dim3 grid((M + FUSED_TILE_M - 1) / FUSED_TILE_M,
                  (N + FUSED_TILE_N - 1) / FUSED_TILE_N);
        dim3 block(FUSED_THREADS);
        size_t smem_size = FUSED_TILE_M * gs_half + FUSED_TILE_N * gs_half
                         + (FUSED_TILE_M * 2 + FUSED_TILE_N * 2) * sizeof(float);
        fused_int4_grouped_gemm_azp_kernel<<<grid, block, smem_size, stream>>>(
            a_packed.data_ptr<uint8_t>(),
            b_packed.data_ptr<uint8_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            azp_adj.data_ptr<float>(),
            w_col_sum.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int>(M), static_cast<int>(N),
            static_cast<int>(num_groups), static_cast<int>(group_size));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: cutlass_int4_dequant_gemm_grouped
//
// Dequant-GEMM approach for per-group INT4 quantization.
//
// Instead of running per-group INT4 MMA with __syncthreads() between groups
// (which is slow due to 20 groups * 2 syncs = 40 barriers), this approach:
//   1. Dequantizes INT4 activations to BF16 with per-group scales applied
//   2. Runs a single cuBLAS BF16 GEMM over the full K dimension
//
// The weight is pre-dequantized to BF16 at model load time (done once).
//
// This trades INT4 compute savings for:
//   - No per-group synchronization barriers
//   - Full K-dimension cuBLAS BF16 GEMM (highly optimized, pipelined)
//   - Better compute-to-memory ratio per kernel launch
//
// Computes: out[M, N] = dequant_bf16(a_packed, scale_a) @ b_bf16
//
// Inputs:
//   a_packed: [num_groups, M, gs/2] uint8 (per-group INT4 packed activations)
//   b_bf16:   [K, N] bf16 (pre-dequanted weight, K = num_groups * group_size)
//   scale_a:  [num_groups, M] float (per-group activation scales)
//   out:      [M, N] bf16 (pre-allocated output)
//   M, N, K, group_size, num_groups
// ─────────────────────────────────────────────────────────────────────────────

// Dequant function defined in int4_quant.cu
extern torch::Tensor dequant_int4_grouped_to_bf16(
    torch::Tensor packed, torch::Tensor scale, int64_t group_size);

void cutlass_int4_dequant_gemm_grouped(
    torch::Tensor a_packed,     // [num_groups, M, gs/2] uint8
    torch::Tensor b_bf16,       // [K, N] bf16
    torch::Tensor scale_a,      // [num_groups, M] float
    torch::Tensor out,          // [M, N] bf16
    int64_t M, int64_t N, int64_t K,
    int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_packed.is_cuda(), "a_packed must be CUDA tensor");
    TORCH_CHECK(b_bf16.is_cuda(), "b_bf16 must be CUDA tensor");
    TORCH_CHECK(scale_a.is_cuda(), "scale_a must be CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA tensor");

    TORCH_CHECK(a_packed.dtype() == torch::kUInt8, "a_packed must be uint8");
    TORCH_CHECK(b_bf16.dtype() == torch::kBFloat16, "b_bf16 must be bf16");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bf16");

    TORCH_CHECK(a_packed.dim() == 3, "a_packed must be [num_groups, M, gs/2]");
    TORCH_CHECK(a_packed.size(0) == num_groups, "a_packed dim 0 must be num_groups");
    TORCH_CHECK(a_packed.size(1) == M, "a_packed dim 1 must be M");
    TORCH_CHECK(a_packed.size(2) == group_size / 2,
                "a_packed dim 2 must be group_size/2");

    TORCH_CHECK(b_bf16.dim() == 2, "b_bf16 must be [K, N]");
    TORCH_CHECK(b_bf16.size(0) == K, "b_bf16 dim 0 must be K");
    TORCH_CHECK(b_bf16.size(1) == N, "b_bf16 dim 1 must be N");

    TORCH_CHECK(K == num_groups * group_size,
                "K must equal num_groups * group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    // Step 1: Dequant INT4 activations → BF16 [M, K]
    // Fuses unpack + scale application for all groups into contiguous BF16
    auto a_bf16 = dequant_int4_grouped_to_bf16(a_packed, scale_a, group_size);

    // Step 2: cuBLAS BF16 GEMM: [M, K] @ [K, N] → [M, N]
    // torch::mm dispatches to cuBLAS which is highly optimized for BF16
    auto result = torch::mm(a_bf16, b_bf16);

    // Copy result to pre-allocated output
    out.copy_(result);
}

// ═════════════════════════════════════════════════════════════════════════════
// V6 QServe-Style Progressive Dequantization INT8 GEMM Kernel
//
// Key insight from QServe (MIT Han Lab): instead of INT4x INT4 MMA with
// per-group barriers, unpack INT4 to INT8 and use INT8x INT8 MMA
// (mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32) with contiguous
// [M, K] / [N, K] layouts. Per-group scales are applied at group boundaries
// within the K-loop, all in registers (no SMEM sync needed for scales).
//
// The contiguous layout enables multi-group K-tiles:
//   TILE_K = GROUPS_PER_LOAD * GS (e.g., 2*128=256 or 4*128=512)
//   Only 2 __syncthreads() per K-tile (load + protect), vs. 2 per group in V4.
//
// Data flow:
//   1. At model load: unpack_int4_grouped_to_int8_weight: [ng,N,gs/2] -> [N,K] int8
//   2. At runtime:    unpack_int4_grouped_to_int8_contiguous: [ng,M,gs/2] -> [M,K] int8
//   3. This kernel:   [M,K] int8 x [N,K] int8 -> [M,N] bf16 with per-group scales
//
// Sync count comparison (num_groups=20):
//   V4:  2 x 20 = 40 syncs
//   V5:  2 x ceil(20/2) = 20 syncs (or 10 with GPL=4)
//   V6:  2 x ceil(20/GPL) syncs (same as V5, but INT8 MMA + contiguous layout)
//
// INT8 MMA advantages:
//   - m16n8k32 processes 32 INT8 values per MMA (vs. 64 INT4 per INT4 MMA)
//   - 4 MMA iterations per group (GS=128, K=32 per iter) vs. 2 (GS=128, K=64)
//   - More compute per sync cycle -> better latency hiding
//   - INT8 data is contiguous -> simpler SMEM layout, better coalescing
//
// Tile: 64x128, 128 threads (4 warps), 2x2 warp layout
// Each warp: 32x64 output (2 m16 x 8 n8 = 16 MMA ops per K-iter)
// SMEM per stage: A[64][TILE_K+4] + B[128][TILE_K+4] + scales[GPL*(64+128)*4]
// ═════════════════════════════════════════════════════════════════════════════

// INT8 MMA constants
static constexpr int I8_MMA_M = 16;
static constexpr int I8_MMA_N = 8;
static constexpr int I8_MMA_K = 32;  // INT8 MMA K dimension

template <int GS = 128, int GROUPS_PER_LOAD = 2>
__global__ void __launch_bounds__(128)
progressive_int8_grouped_gemm_kernel(
    const int8_t* __restrict__ a,         // [M, K] int8 contiguous
    const int8_t* __restrict__ w,         // [N, K] int8 contiguous
    const float* __restrict__ scale_a,    // [ng, M] float
    const float* __restrict__ scale_b,    // [ng, N] float
    __nv_bfloat16* __restrict__ out,      // [M, N] bf16
    const int M, const int N, const int K, const int ng)
{
    // ── Compile-time constants ──
    static constexpr int TILE_M = 64;
    static constexpr int TILE_N = 128;
    static constexpr int THREADS = 128;
    static constexpr int gs = GS;

    // Multi-group K-tile: process GROUPS_PER_LOAD groups per SMEM load
    static constexpr int TILE_K = GROUPS_PER_LOAD * GS;
    static constexpr int mma_k_iters_per_group = GS / I8_MMA_K;  // 128/32 = 4

    // SMEM strides with 4-byte padding for bank-conflict-free access
    // INT8 data: 1 byte per element, stride in bytes
    static constexpr int A_STRIDE = TILE_K + 4;  // bytes per A row
    static constexpr int B_STRIDE = TILE_K + 4;  // bytes per B row
    static constexpr int a_i32_stride = A_STRIDE / 4;  // int32 elements per A row
    static constexpr int b_i32_stride = B_STRIDE / 4;  // int32 elements per B row

    // Scale arrays: GROUPS_PER_LOAD sets of scales per stage
    static constexpr int scale_floats = GROUPS_PER_LOAD * (TILE_M + TILE_N);
    static constexpr int STAGE_BYTES =
        TILE_M * A_STRIDE + TILE_N * B_STRIDE
        + scale_floats * (int)sizeof(float);

    // ── Thread/warp decomposition (same as V4/V5) ──
    const int tile_m = blockIdx.x * TILE_M;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int warp_row_start = warp_m * 32;
    const int warp_col_start = warp_n * 64;
    const int frag_row = lane_id / 4;  // 0..7
    const int frag_grp = lane_id % 4;  // 0..3

    // ── Double-buffered shared memory ──
    extern __shared__ uint8_t smem_v6[];

    // ── FP32 accumulator for final output ──
    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            #pragma unroll
            for (int fi = 0; fi < 4; fi++)
                acc[mi][ni][fi] = 0.f;

    // ── Number of K-tile chunks ──
    const int num_chunks = (ng + GROUPS_PER_LOAD - 1) / GROUPS_PER_LOAD;

    // ── Helper: issue cp.async loads for GROUPS_PER_LOAD groups into a stage ──
    auto v6_load_stage = [&](int g_base, int chunk_size, int stage) __attribute__((always_inline)) {
        uint8_t* sa = smem_v6 + stage * STAGE_BYTES;
        uint8_t* sb = sa + TILE_M * A_STRIDE;
        float* ssx_base = reinterpret_cast<float*>(sb + TILE_N * B_STRIDE);

        // k_start in the contiguous [M, K] layout
        const int k_start = g_base * gs;
        const int k_len = chunk_size * gs;  // actual K elements to load

        // Load A tile [TILE_M, TILE_K] from a[tile_m:tile_m+TILE_M, k_start:k_start+TILE_K]
        // INT8 data: 1 byte per element, load as uint32 (4 bytes = 4 int8 values)
        {
            constexpr int a_words_per_row = TILE_K / 4;  // uint32 words per row
            constexpr int total_a = TILE_M * a_words_per_row;
            for (int i = tid; i < total_a; i += THREADS) {
                const int row = i / a_words_per_row;
                const int col_w = i % a_words_per_row;
                const int gr = tile_m + row;
                uint8_t* dst = sa + row * A_STRIDE + col_w * 4;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < M && col_w * 4 < k_len) {
                    const void* src = a + (int64_t)gr * K + k_start + col_w * 4;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    *reinterpret_cast<uint32_t*>(dst) = 0u;
                }
            }
        }

        // Load B tile [TILE_N, TILE_K] from w[tile_n:tile_n+TILE_N, k_start:k_start+TILE_K]
        {
            constexpr int b_words_per_row = TILE_K / 4;
            constexpr int total_b = TILE_N * b_words_per_row;
            for (int i = tid; i < total_b; i += THREADS) {
                const int row = i / b_words_per_row;
                const int col_w = i % b_words_per_row;
                const int gr = tile_n + row;
                uint8_t* dst = sb + row * B_STRIDE + col_w * 4;
                const unsigned int ds = static_cast<unsigned int>(
                    __cvta_generic_to_shared(dst));
                if (gr < N && col_w * 4 < k_len) {
                    const void* src = w + (int64_t)gr * K + k_start + col_w * 4;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                        :: "r"(ds), "l"(src));
                } else {
                    *reinterpret_cast<uint32_t*>(dst) = 0u;
                }
            }
        }

        // Load scales for each group in the chunk (regular stores)
        #pragma unroll
        for (int gl = 0; gl < GROUPS_PER_LOAD; gl++) {
            int g = g_base + gl;
            if (g >= g_base + chunk_size) break;

            float* ssx = ssx_base + gl * (TILE_M + TILE_N);
            float* ssw = ssx + TILE_M;
            const float* sxg = scale_a + (int64_t)g * M;
            const float* swg = scale_b + (int64_t)g * N;

            for (int i = tid; i < TILE_M; i += THREADS) {
                int r = tile_m + i;
                ssx[i] = (r < M) ? sxg[r] : 0.f;
            }
            for (int i = tid; i < TILE_N; i += THREADS) {
                int c = tile_n + i;
                ssw[i] = (c < N) ? swg[c] : 0.f;
            }
        }

        asm volatile("cp.async.commit_group;\n" ::);
    };

    // ══ Prologue: load first chunk into stage 0 ══
    {
        int first_chunk_size = (GROUPS_PER_LOAD <= ng) ? GROUPS_PER_LOAD : ng;
        v6_load_stage(0, first_chunk_size, 0);
    }

    // ══ Main chunk loop with double buffering ══
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int cur = chunk & 1;
        const int g_base = chunk * GROUPS_PER_LOAD;
        const int chunk_size = ((g_base + GROUPS_PER_LOAD) <= ng)
                             ? GROUPS_PER_LOAD
                             : (ng - g_base);

        // Prefetch next chunk into the other stage
        if (chunk + 1 < num_chunks) {
            int next_g_base = (chunk + 1) * GROUPS_PER_LOAD;
            int next_chunk_size = ((next_g_base + GROUPS_PER_LOAD) <= ng)
                                ? GROUPS_PER_LOAD
                                : (ng - next_g_base);
            v6_load_stage(next_g_base, next_chunk_size, 1 - cur);
        }

        // Wait for current stage
        if (chunk + 1 < num_chunks) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(1));
        } else {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
        }
        __syncthreads();

        // ── Stage pointers ──
        const uint8_t* cur_base = smem_v6 + cur * STAGE_BYTES;
        const int32_t* a_i32 = reinterpret_cast<const int32_t*>(cur_base);
        const int32_t* b_i32 = reinterpret_cast<const int32_t*>(
            cur_base + TILE_M * A_STRIDE);
        const float* scales_base = reinterpret_cast<const float*>(
            cur_base + TILE_M * A_STRIDE + TILE_N * B_STRIDE);

        // ── Process each group within this chunk sequentially ──
        // No __syncthreads needed between groups within a chunk since all
        // data is already resident in SMEM from the single load.
        for (int gl = 0; gl < chunk_size; gl++) {
            const float* cur_sx = scales_base + gl * (TILE_M + TILE_N);
            const float* cur_sw = cur_sx + TILE_M;

            // K offset within multi-group SMEM tile for this group (in int32 units)
            // Each int32 holds 4 int8 values, so offset = gl * gs / 4
            const int k_i32_group_off = gl * (gs / 4);

            // ── INT8 MMA computation for this group ──
            // INT8 MMA m16n8k32:
            //   A fragment: 4 x int32 (each int32 = 4 int8 values, total 16 rows x 32 cols)
            //   B fragment: 2 x int32 (each int32 = 4 int8 values, total 8 cols x 32 K)
            //   D fragment: 4 x int32 (m16n8 output)
            //
            // Fragment layout for m16n8k32 (from PTX spec):
            //   Lane L: row R = L/4 (0-7), K-quarter Q = L%4 (0-3)
            //   A: reg[0] = A_i32[R,        Q*2+0]
            //      reg[1] = A_i32[R + 8,    Q*2+0]
            //      reg[2] = A_i32[R,        Q*2+1]
            //      reg[3] = A_i32[R + 8,    Q*2+1]
            //   B: reg[0] = B_i32[R,        Q*2+0]  (R maps to col for B)
            //      reg[1] = B_i32[R,        Q*2+1]
            //   D: D[0]=(R,Q*2), D[1]=(R,Q*2+1), D[2]=(R+8,Q*2), D[3]=(R+8,Q*2+1)

            int32_t mma_acc[2][8][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 8; ni++)
                    #pragma unroll
                    for (int fi = 0; fi < 4; fi++)
                        mma_acc[mi][ni][fi] = 0;

            #pragma unroll
            for (int ki = 0; ki < mma_k_iters_per_group; ki++) {
                // Offset within the SMEM tile for this K iteration
                // Each MMA K iteration processes 32 int8 values = 8 int32 words
                const int k_i32_off = k_i32_group_off + ki * (I8_MMA_K / 4);

                // Load A fragments for both M-positions (2 x m16)
                int32_t a_frag[2][4];
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    int row0 = warp_row_start + mi * I8_MMA_M + frag_row;
                    int row1 = row0 + 8;
                    int base0 = row0 * a_i32_stride + k_i32_off + frag_grp * 2;
                    int base1 = row1 * a_i32_stride + k_i32_off + frag_grp * 2;
                    a_frag[mi][0] = a_i32[base0];
                    a_frag[mi][1] = a_i32[base1];
                    a_frag[mi][2] = a_i32[base0 + 1];
                    a_frag[mi][3] = a_i32[base1 + 1];
                }

                // MMA for all M x N positions
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    #pragma unroll
                    for (int ni = 0; ni < 8; ni++) {
                        int b_row = warp_col_start + ni * I8_MMA_N + frag_row;
                        int b_base = b_row * b_i32_stride + k_i32_off + frag_grp * 2;
                        int32_t b0 = b_i32[b_base];
                        int32_t b1 = b_i32[b_base + 1];

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                            : "=r"(mma_acc[mi][ni][0]), "=r"(mma_acc[mi][ni][1]),
                              "=r"(mma_acc[mi][ni][2]), "=r"(mma_acc[mi][ni][3])
                            : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                              "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                              "r"(b0), "r"(b1),
                              "r"(mma_acc[mi][ni][0]), "r"(mma_acc[mi][ni][1]),
                              "r"(mma_acc[mi][ni][2]), "r"(mma_acc[mi][ni][3]));
#endif
                    }
                }
            } // end K-loop for this group

            // ── Apply per-group scales and accumulate into FP32 ──
            // MMA output mapping (m16n8k32, identical to m16n8k64):
            //   D[0] -> (row0, col0), D[1] -> (row0, col1)
            //   D[2] -> (row1, col0), D[3] -> (row1, col1)
            //   where row0 = lane/4, row1 = row0+8, col0 = (lane%4)*2, col1 = col0+1
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int m0 = warp_row_start + mi * I8_MMA_M + frag_row;
                int m1 = m0 + 8;
                float sx0 = cur_sx[m0];
                float sx1 = cur_sx[m1];

                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int n0 = warp_col_start + ni * I8_MMA_N + frag_grp * 2;
                    float sw0 = cur_sw[n0];
                    float sw1 = cur_sw[n0 + 1];

                    acc[mi][ni][0] += static_cast<float>(mma_acc[mi][ni][0]) * sx0 * sw0;
                    acc[mi][ni][1] += static_cast<float>(mma_acc[mi][ni][1]) * sx0 * sw1;
                    acc[mi][ni][2] += static_cast<float>(mma_acc[mi][ni][2]) * sx1 * sw0;
                    acc[mi][ni][3] += static_cast<float>(mma_acc[mi][ni][3]) * sx1 * sw1;
                }
            }
        }  // end gl loop within chunk

        __syncthreads();  // protect stage reads before next chunk overwrites
    }  // end chunk loop

    // ── Write FP32 -> BF16 output ──
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int gm0 = tile_m + warp_row_start + mi * I8_MMA_M + frag_row;
            int gm1 = gm0 + 8;
            int gn0 = tile_n + warp_col_start + ni * I8_MMA_N + frag_grp * 2;
            int gn1 = gn0 + 1;

            if (gm0 < M && gn0 < N) out[gm0 * N + gn0] = __float2bfloat16(acc[mi][ni][0]);
            if (gm0 < M && gn1 < N) out[gm0 * N + gn1] = __float2bfloat16(acc[mi][ni][1]);
            if (gm1 < M && gn0 < N) out[gm1 * N + gn0] = __float2bfloat16(acc[mi][ni][2]);
            if (gm1 < M && gn1 < N) out[gm1 * N + gn1] = __float2bfloat16(acc[mi][ni][3]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: progressive_int4_gemm_grouped
//
// QServe-style progressive dequantization INT8 GEMM for per-group INT4 data.
// Takes pre-unpacked INT8 data and runs the fused INT8 GEMM with per-group
// scale application.
//
// Inputs:
//   a_int8:    [M, K] int8 (activations, unpacked from per-group INT4)
//   w_int8:    [N, K] int8 (weights, pre-unpacked at model load time)
//   scale_a:   [num_groups, M] float (per-group activation scales)
//   scale_b:   [num_groups, N] float (per-group weight scales)
//   out:       [M, N] bf16 (pre-allocated output)
//   M, N, K, group_size, num_groups
// ─────────────────────────────────────────────────────────────────────────────

void progressive_int4_gemm_grouped(
    torch::Tensor a_int8,       // [M, K] int8
    torch::Tensor w_int8,       // [N, K] int8
    torch::Tensor scale_a,      // [ng, M] float
    torch::Tensor scale_b,      // [ng, N] float
    torch::Tensor out,          // [M, N] bf16
    int64_t M, int64_t N, int64_t K,
    int64_t group_size, int64_t num_groups)
{
    TORCH_CHECK(a_int8.is_cuda() && w_int8.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "Scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");
    TORCH_CHECK(a_int8.dtype() == torch::kInt8, "a_int8 must be int8");
    TORCH_CHECK(w_int8.dtype() == torch::kInt8, "w_int8 must be int8");
    TORCH_CHECK(group_size == 128,
                "Progressive INT8 GEMM currently only supports group_size=128");
    TORCH_CHECK(K == num_groups * group_size,
                "K must equal num_groups * group_size");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,
                "progressive_int4_gemm_grouped currently supports BF16 output only");
    TORCH_CHECK(a_int8.dim() == 2 && a_int8.size(0) == M && a_int8.size(1) == K,
                "a_int8 must be [M, K]");
    TORCH_CHECK(w_int8.dim() == 2 && w_int8.size(0) == N && w_int8.size(1) == K,
                "w_int8 must be [N, K]");

    auto stream = at::cuda::getCurrentCUDAStream(a_int8.get_device());

    constexpr int V6_TILE_M = 64;
    constexpr int V6_TILE_N = 128;
    constexpr int V6_THREADS = 128;
    dim3 grid((M + V6_TILE_M - 1) / V6_TILE_M,
              (N + V6_TILE_N - 1) / V6_TILE_N);
    dim3 block(V6_THREADS);

    int ng = static_cast<int>(num_groups);

    // Query hardware shared memory limit to select the largest GPL that fits.
    // INT8 data is 1 byte per element (vs 0.5 for INT4 packed), so SMEM
    // budgets are 2x larger than the INT4 GEMM kernels.  On L4 (SM89) the
    // max opt-in SMEM is ~100 KB, so GPL=4 (204 KB) and often GPL=2 (103 KB)
    // will exceed the limit.  We compute the double-buffered requirement for
    // each candidate GPL and fall back to the largest one that fits.
    int device = a_int8.get_device();
    int hw_smem_limit = get_cuda_max_shared_memory_per_block_opt_in(device);

    // Helper: compute double-buffered SMEM bytes for a given GPL value.
    auto smem_for_gpl = [](int gpl) -> size_t {
        int tile_k   = gpl * 128;
        int a_stride = tile_k + 4;
        int b_stride = tile_k + 4;
        int scale_flt = gpl * (V6_TILE_M + V6_TILE_N);
        int stage_bytes = V6_TILE_M * a_stride
                        + V6_TILE_N * b_stride
                        + scale_flt * (int)sizeof(float);
        return static_cast<size_t>(2) * stage_bytes;
    };

    // Select the largest GPL in {4, 2, 1} where:
    //   (1) ng is divisible by GPL, and
    //   (2) the double-buffered SMEM fits within the hardware limit.
    int chosen_gpl = 1;  // GPL=1 always fits (52 KB)
    if (ng % 4 == 0 && static_cast<int>(smem_for_gpl(4)) <= hw_smem_limit) {
        chosen_gpl = 4;
    } else if (ng % 2 == 0 && static_cast<int>(smem_for_gpl(2)) <= hw_smem_limit) {
        chosen_gpl = 2;
    }

    // Macro to avoid duplicating the launch code for each template instantiation.
    #define LAUNCH_V6_KERNEL(GPL_VAL)                                         \
    do {                                                                       \
        size_t smem_size = smem_for_gpl(GPL_VAL);                             \
        if (smem_size > 48 * 1024) {                                          \
            cudaFuncSetAttribute(                                             \
                progressive_int8_grouped_gemm_kernel<128, GPL_VAL>,           \
                cudaFuncAttributeMaxDynamicSharedMemorySize,                  \
                smem_size);                                                   \
        }                                                                     \
        progressive_int8_grouped_gemm_kernel<128, GPL_VAL>                    \
            <<<grid, block, smem_size, stream>>>(                             \
            a_int8.data_ptr<int8_t>(),                                        \
            w_int8.data_ptr<int8_t>(),                                        \
            scale_a.data_ptr<float>(),                                        \
            scale_b.data_ptr<float>(),                                        \
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),                 \
            static_cast<int>(M), static_cast<int>(N),                         \
            static_cast<int>(K), ng);                                         \
    } while (0)

    if (chosen_gpl == 4) {
        LAUNCH_V6_KERNEL(4);
    } else if (chosen_gpl == 2) {
        LAUNCH_V6_KERNEL(2);
    } else {
        LAUNCH_V6_KERNEL(1);
    }

    #undef LAUNCH_V6_KERNEL
}
