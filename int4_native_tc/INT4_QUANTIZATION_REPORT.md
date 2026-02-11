# INT4 Quantization for vLLM: Implementation Report

## Project Overview

INT4 weight quantization을 vLLM에 통합하여 Transformer 모델의 VRAM 사용량과 inference 속도를 개선하는 프로젝트.
**대상 모델**: Qwen/Qwen3-Embedding-4B (36 layers, hidden=2560, intermediate=9728)
**대상 GPU**: NVIDIA L4 (SM89, 23 GiB VRAM)

---

## 1. 구현된 Quantization Methods

총 4개의 양자화 방식을 vLLM quantization config로 구현:

| Method | CLI Flag | Weight | Activation | GEMM Kernel |
|--------|---------|--------|-----------|-------------|
| **W4A4 (INT4-TC)** | `--quantization int4-tc` | INT4 packed | INT4 dynamic | CUTLASS s4xs4 TC |
| **W4A16 (Marlin)** | `--quantization w4a16-int4tc` | INT4 Marlin layout | BF16 (no quant) | Marlin fused dequant+BF16 TC |
| **W4A8-Fused** | `--quantization w4a8-fused-int4tc` | INT4 Marlin layout | INT8 dynamic | Marlin fused dequant+INT8 TC |
| **W4A4A8-Mixed** | `--quantization w4a4a8-mixed-int4tc` | INT4 (both formats) | INT4 or INT8 per layer | CUTLASS s4xs4 / Marlin s8xs8 |

---

## 2. Architecture & Key Technical Decisions

### 2.1 Hardware Constraints (SM80/SM89)

SM80 (Ampere) / SM89 (Ada Lovelace) MMA instruction set:

```
Available:
  mma.sync.m16n8k64.s32.s4.s4.s32   (INT4xINT4)  -- 64 elements/instruction
  mma.sync.m16n8k32.s32.s8.s8.s32   (INT8xINT8)  -- 32 elements/instruction
  mma.sync.m16n8k16.f32.bf16.bf16   (BF16xBF16)  -- 16 elements/instruction

NOT available:
  mma.sync.*.s32.s4.s8.s32           (INT4xINT8)  -- does NOT exist
```

**Mixed-input INT4xINT8 TC instruction은 하드웨어에 존재하지 않음.**
해결 방법: INT4를 register level에서 INT8로 upcast 후 INT8xINT8 MMA 사용.

### 2.2 Two GEMM Backends

#### CUTLASS 2.x INT4xINT4 TC (W4A4 path)
- `mma.sync.m16n8k64.s32.s4.s4.s32` instruction 사용
- Fused EVT (Epilogue Visitor Tree) epilogue: `(float)acc * scale_a[m] * scale_b[n] -> bf16`
- Custom CUDA extension: `int4_native_tc`
- Weight: packed uint8 `[N, K/2]`, Activation: packed uint8 `[M, K/2]`

#### Marlin Fused Dequant GEMM (W4A16, W4A8 paths)
- vLLM built-in Marlin kernel 활용
- INT4 weight in Marlin tile layout -> register에서 dequant -> TC MMA
- **W4A16**: INT4->BF16 dequant, `mma.sync.m16n8k16.f32.bf16.bf16`
- **W4A8**: INT4->INT8 dequant, `mma.sync.m16n8k32.s32.s8.s8.s32`
- Weight: Marlin-repacked int32, Scale: Marlin-permuted

### 2.3 `is_a_8bit` Parameter (Critical)

Marlin kernel의 INT8 activation path는 BF16 path와 **다른 weight permutation 및 scale permutation**을 사용:

```python
# W4A16 (BF16 activation):
w_marlin = ops.gptq_marlin_repack(q_gptq, perm, K, N, 4)  # default
s_marlin = marlin_permute_scales(s, K, N, group_size=-1)    # default

# W4A8 (INT8 activation):
w_marlin = ops.gptq_marlin_repack(q_gptq, perm, K, N, 4, is_a_8bit=True)  # INT8 layout
s_marlin = marlin_permute_scales(s, K, N, group_size=-1, is_a_8bit=True)    # INT8 layout
```

**Without `is_a_8bit=True`**: cosine similarity ~0.025 (garbage output)
**With `is_a_8bit=True`**: cosine similarity ~0.989 (correct)

이유: INT8 MMA (`m16n8k32`)와 BF16 MMA (`m16n8k16`)의 register layout이 다르므로,
weight tile을 다른 순서로 배치해야 TC instruction의 operand와 올바르게 매칭됨.

---

## 3. Implementation Details

### 3.1 W4A4 (INT4-TC) — `w4a4_int4tc.py`

**파일**: `vllm/model_executor/layers/quantization/w4a4_int4tc.py` (182 lines)

```
Weight preparation (model load):
  BF16 [N, K] → static_int4_weight_quant() → packed uint8 [N, K/2] + scale [N]

Inference:
  BF16 [M, K] → dynamic_int4_quant() → packed uint8 [M, K/2] + scale [M]
  GEMM: cutlass_int4_scaled_mm(a_packed, w_packed, a_scale, w_scale, out, M, N, K)
  Output: BF16 [M, N]
```

**INT4 Symmetric Quantization (per-channel)**:
```
scale[n] = max(|w[n, :]|) / 7.0
q_signed[n, k] = round(w[n, k] / scale[n]), clamp to [-8, 7]
packed: 2 signed int4 values per uint8 byte (low nibble + high nibble)
```

**Key CUDA kernels** (in `int4_native_tc` extension):
- `static_int4_weight_quant`: BF16 weight -> INT4 packed (per-channel, static)
- `dynamic_int4_quant`: BF16 activation -> INT4 packed (per-row, dynamic)
- `cutlass_int4_scaled_mm`: INT4xINT4 GEMM with fused scale epilogue

### 3.2 W4A16-Marlin — `w4a16_int4tc.py`

**파일**: `vllm/model_executor/layers/quantization/w4a16_int4tc.py` (297 lines)

```
Weight preparation (model load):
  BF16 [N, K]
  → pad K to 128, N to 64 (Marlin alignment)
  → _quantize_w4_symmetric() → unsigned int32 [N, K] + scale [N]
  → _pack_to_gptq_format() → GPTQ int32 [K//8, N]
  → ops.gptq_marlin_repack() → Marlin tile layout
  → marlin_permute_scales() → Marlin scale layout

Inference:
  BF16 [M, K] → (no activation quant) → pad K if needed
  GEMM: ops.marlin_gemm(x_bf16, ..., wtype=uint4b8, ...)
  Output: BF16 [M, N]
```

**Marlin Weight Conversion Pipeline**:
```
Step 1: _quantize_w4_symmetric()
  - Per-channel absmax / 7.0 → scale
  - Round + clamp [-8, 7] → signed INT4
  - signed + 8 → unsigned [0, 15] (Marlin dequant subtracts 8 internally)

Step 2: _pack_to_gptq_format()
  - Transpose [N, K] → [K, N]
  - Pack 8 INT4 values along K into one int32
  - nibble[0] | (nibble[1]<<4) | ... | (nibble[7]<<28)
  - Output: [K//8, N] int32

Step 3: gptq_marlin_repack()
  - GPTQ tile layout → Marlin tile layout
  - Optimized for coalesced GMEM access + bank-conflict-free SMEM

Step 4: marlin_permute_scales()
  - Per-channel scale → Marlin kernel's expected scale layout
  - group_size=-1 (per-channel)
```

### 3.3 W4A8-Fused — `w4a8_fused_int4tc.py`

**파일**: `vllm/model_executor/layers/quantization/w4a8_fused_int4tc.py` (263 lines)

```
Weight preparation: same as W4A16-Marlin BUT with is_a_8bit=True
  → Different Marlin tile permutation for INT8 MMA register layout

Inference:
  BF16 [M, K] → int4_native_tc.dynamic_int8_quant() → INT8 [M, K] + a_scales [M]
  GEMM: ops.marlin_gemm(x_int8, ..., a_scales=a_scales, wtype=uint4b8, ...)
  Marlin internally: INT4 weight → register dequant → INT8 → INT8xINT8 TC MMA
  Output: BF16 [M, N]
```

**Key difference from W4A16**:
- Activation은 INT8로 dynamic quantization (per-row absmax / 127.0)
- Marlin GEMM에 `a_scales` parameter 전달
- Weight/scale permutation에 `is_a_8bit=True` 필수
- Marlin kernel 내부에서 INT32 accumulator → float → scale_a * scale_b → BF16

**VRAM 비교**:
```
W4A8-Preunpack: INT4 weight → model load시 INT8로 unpack → INT8 저장 (1 byte/elem)
W4A8-Fused:     INT4 weight → INT4 그대로 저장 (0.5 byte/elem) → 50% VRAM 절약
```

### 3.4 W4A4A8-Mixed — `w4a4a8_mixed_int4tc.py`

**파일**: `vllm/model_executor/layers/quantization/w4a4a8_mixed_int4tc.py` (325 lines)

**핵심 아이디어**: Transformer layer별로 activation quantization precision을 다르게 적용.

```python
INT8_ACTIVATION_LAYERS = {"o_proj", "down_proj"}

# Layer dispatch in get_quant_method():
if layer_name in INT8_ACTIVATION_LAYERS:
    return W4A8LinearMethod(self)   # INT4 weight × INT8 activation (Marlin)
else:
    return W4A4LinearMethod(self)   # INT4 weight × INT4 activation (CUTLASS)
```

**Motivation (논문 기반)**:
- **q/k/v/gate/up_proj**: LayerNorm 직후 입력 → 깨끗한 분포 → INT4 activation OK
- **o_proj**: softmax 출력 후 → channel-wise magnitude variation → INT8 필요
- **down_proj**: SiLU(gate)*up 후 → heavy-tailed 분포 (2-4x dynamic range 확장) → INT8 필요

**참조 논문**: QServe (ICML 2024), Atom (MLSys 2024), SmoothQuant, GPTQ, AWQ

**두 가지 linear method 구현**:
- `W4A4LinearMethod`: CUTLASS INT4×INT4 TC (same as W4A4)
- `W4A8LinearMethod`: Marlin INT4×INT8 TC (same as W4A8-Fused)

---

## 4. vLLM Integration

### 4.1 Registration

`vllm/model_executor/layers/quantization/__init__.py` 수정:

1. `QuantizationMethods` Literal type에 4개 method 추가
2. `get_quantization_config()`의 `method_to_config` dict에 lazy import 추가

```python
QuantizationMethods = Literal[
    ...,  # existing methods
    "int4-tc",
    "w4a16-int4tc",
    "w4a8-fused-int4tc",
    "w4a4a8-mixed-int4tc",
]
```

### 4.2 Usage

```bash
# W4A4 (INT4×INT4, fastest but lowest quality)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-4B \
    --convert embed \
    --quantization int4-tc \
    --enforce-eager

# W4A16-Marlin (INT4 weight × BF16 activation, best quality)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-4B \
    --convert embed \
    --quantization w4a16-int4tc \
    --enforce-eager

# W4A8-Fused (INT4 weight × INT8 activation, good tradeoff)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-4B \
    --convert embed \
    --quantization w4a8-fused-int4tc \
    --enforce-eager

# W4A4A8-Mixed (INT4×INT4 most layers + INT4×INT8 for o/down_proj)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-4B \
    --convert embed \
    --quantization w4a4a8-mixed-int4tc \
    --enforce-eager
```

**Note**: `--enforce-eager` 필수. `int4_native_tc`는 pybind11 기반 C++ 확장이므로
`torch.compile` (torch.dynamo)가 trace할 수 없음. `torch.compiler.allow_in_graph`
등록을 통해 해결 가능하나 현재 미구현.

### 4.3 Dependencies

```
int4_native_tc     (custom CUDA extension, pybind11)
  - cutlass_int4_gemm, cutlass_int4_scaled_mm (INT4×INT4 CUTLASS GEMM)
  - cutlass_w4a8_scaled_mm (INT8×INT8 CUTLASS GEMM)
  - dynamic_int4_quant, dynamic_int8_quant (CUDA quantization kernels)
  - static_int4_weight_quant (weight quantization)
  - unpack_int4_to_int8, dequant_int4_to_bf16/fp16 (utility)

vllm._custom_ops   (vLLM built-in)
  - gptq_marlin_repack (GPTQ → Marlin weight layout conversion)
  - marlin_gemm (Marlin fused dequant+GEMM kernel)

vllm.model_executor.layers.quantization.utils.marlin_utils
  - marlin_permute_scales (scale layout conversion)
  - GPTQ_MARLIN_MIN_THREAD_K=128, MIN_THREAD_N=64, MAX_PARALLEL=16
```

---

## 5. Unit Test Results

**파일**: `fp8_inference_toolkit/int4_native_tc/test_correctness.py` (1372 lines, 24 tests)

All 24 tests PASSED on NVIDIA L4.

### Core tests (1-16): INT4 infrastructure
| Test | Description | Result |
|------|------------|--------|
| 1 | Dynamic INT4 activation quantization | PASSED |
| 2 | Static INT4 weight quantization | PASSED |
| 3 | CUTLASS INT4×INT4 GEMM (small) | PASSED |
| 4 | CUTLASS INT4×INT4 GEMM (LLM-size 2560×2560) | PASSED |
| 5 | Scaled GEMM end-to-end | PASSED |
| 6 | W4A16 dequant + BF16 GEMM | PASSED |
| 7 | W4A16 full path (quant+dequant+GEMM) | PASSED |
| 8 | W4A4 full path | PASSED |
| 9 | W4A16 Marlin weight conversion | PASSED |
| 10 | W4A16 Marlin GEMM | PASSED |
| 11 | W4A16 Marlin quality (cosine) | PASSED |
| 12 | Dynamic INT8 quantization | PASSED |
| 13 | W4A8 scaled GEMM (pre-unpack) | PASSED |
| 14 | W4A8 full path | PASSED |
| 15 | W4A8 quality | PASSED |
| 16 | Unpack INT4→INT8 | PASSED |

### W4A8-Fused tests (17-20): Marlin INT4×INT8
| Test | Description | Result |
|------|------------|--------|
| 17 | Marlin INT8 GEMM basic | PASSED (mean_cos=0.9890) |
| 18 | Fused full path | PASSED (mean_cos=0.9889) |
| 19 | Fused vs Pre-unpack match | PASSED (cos=0.9891) |
| 20 | Fused quality (LLM-size) | PASSED (mean_cos=0.9890) |

### W4A4A8-Mixed tests (21-24)
| Test | Description | Result |
|------|------------|--------|
| 21 | MLP block simulation (gate/up=INT4, down=INT8) | PASSED (mean_cos=0.945) |
| 22 | Quality comparison W4A4 vs Mixed vs W4A8 | PASSED (W4A4=0.9146, Mixed=0.9444, W4A8=0.9655) |
| 23 | Layer dispatch verification | PASSED (5 INT4 layers, 2 INT8 layers) |
| 24 | down_proj INT4 vs INT8 sensitivity | PASSED (INT4=0.937, INT8=0.988, +0.051) |

### Key observations from unit tests:
- Single-layer cosine: W4A4=0.91, W4A8=0.99, W4A16=0.99
- Test 24 confirms: post-SiLU input에서 INT8 activation이 INT4보다 +0.051 cosine 개선
- Marlin INT8 path (`is_a_8bit=True`)와 pre-unpack INT8 path의 결과가 거의 동일

---

## 6. GEMM Micro-Benchmark Results

**파일**: `fp8_inference_toolkit/benchmark/benchmark_int4_native_gemm.py` (437 lines)

Qwen3-Embedding-4B의 실제 GEMM dimensions에 대해 각 quantization method의 latency 측정.

### 6.1 Full Forward Pass Estimate (36 layers × 5 GEMMs, ms)

| M | BF16 | FP8 | W4A16 | W4A8 | W4A8-Fused | W4A4 | W4A16-Marlin | W4A4A8-Mixed |
|---|------|-----|-------|------|-----------|------|-------------|-------------|
| 1 | 5.0 | 5.5 | 7.5 | 5.1 | 6.5 | 5.2 | 4.2 | 5.6 |
| 16 | 6.5 | 7.0 | 9.3 | 6.2 | 8.8 | 5.9 | 6.1 | 6.6 |
| 64 | 8.7 | 9.2 | 13.3 | 7.7 | 13.3 | 6.3 | 8.5 | 8.5 |
| 256 | 20.7 | 14.2 | 27.1 | 13.4 | 21.2 | 9.6 | 15.2 | 13.0 |
| 512 | 36.5 | 24.6 | 50.9 | 23.2 | 30.2 | 15.3 | 27.2 | 23.0 |
| 1024 | 67.3 | 45.1 | 98.6 | 42.9 | 48.3 | 27.0 | 50.8 | 38.3 |

### 6.2 Benchmark Key Findings
- **W4A4 (INT4×INT4 CUTLASS)**: 최저 latency (BF16 대비 2x+ 빠름), 하지만 품질 문제
- **W4A16-Marlin**: M=1에서 최저 latency (4.2ms), memory-bound regime에서 가장 유리
- **W4A8-Fused (Marlin)**: M이 커질수록 W4A16-Marlin보다 빠름 (INT8 TC 2x throughput)
- **W4A4A8-Mixed**: W4A4에 가까운 latency + 부분적 품질 개선

---

## 7. Real Model Test Results (vLLM Server)

**모델**: Qwen/Qwen3-Embedding-4B (36 layers)
**GPU**: NVIDIA L4, 23 GiB VRAM
**Task**: Embedding (10 diverse test sentences, Korean/English)
**비교 기준**: BF16 baseline embedding 대비 cosine similarity

### 7.1 Per-sentence Cosine Similarity

| Sentence | W4A16-Marlin | W4A8-Fused | W4A4A8-Mixed |
|----------|-------------|-----------|-------------|
| "The quick brown fox..." | 0.780 | 0.769 | 0.362 |
| "Machine learning models..." | 0.897 | 0.883 | 0.495 |
| "서울은 대한민국의 수도입니다." | 0.828 | 0.807 | 0.214 |
| "Quantum computing..." | 0.815 | 0.833 | 0.407 |
| "The transformer architecture..." | 0.875 | 0.863 | 0.458 |
| "Python is a popular..." | 0.858 | 0.834 | 0.346 |
| "Climate change..." | 0.770 | 0.772 | 0.286 |
| "CUDA enables..." | 0.846 | 0.820 | 0.317 |
| "The mitochondria..." | 0.820 | 0.798 | 0.357 |
| "Retrieval-augmented..." | 0.890 | 0.881 | 0.548 |

### 7.2 Summary

| Method | Weight | Activation | Mean Cosine | Model VRAM | Usability |
|--------|--------|-----------|-------------|-----------|-----------|
| **BF16** | BF16 | BF16 | **1.000** | 7.55 GiB | Baseline |
| **W4A16-Marlin** | INT4 | BF16 | **0.838** | ~2.5 GiB | Good |
| **W4A8-Fused** | INT4 | INT8 | **0.826** | 2.45 GiB | Good |
| **W4A4A8-Mixed** | INT4 | INT4/INT8 | **0.379** | 2.45 GiB | Not usable |

### 7.3 Analysis

**BF16 consistency check**: compiled vs eager mode cosine ~0.9999 (negligible difference)

**Unit test vs Real model gap**:
- Unit test (single layer, random matrix): W4A8 cosine ~0.989
- Real model (36 layers, real weights): W4A8 cosine ~0.826
- Gap caused by: (1) error accumulation across 36 layers (2) real weight distributions have outliers

**W4A4A8-Mixed quality failure**:
- INT4 activation (16 quantization levels) is fundamentally too aggressive
- Single-layer cosine ~0.91 compounds across 36 layers → ~0.38 at model level
- Even "precision-insensitive" layers (post-LayerNorm) accumulate significant error
- Selective INT8 on o_proj/down_proj recovers some quality but not enough

**W4A16 vs W4A8 comparison**:
- Very similar quality (0.838 vs 0.826)
- INT8 activation quantization loses very little vs BF16 activation
- W4A8 potentially faster at large batch sizes (INT8 TC 2x throughput)

---

## 8. File Index

### vLLM Quantization Configs
| File | Lines | Description |
|------|-------|-------------|
| `vllm/.../quantization/w4a4_int4tc.py` | 182 | W4A4: INT4 weight × INT4 activation, CUTLASS TC |
| `vllm/.../quantization/w4a16_int4tc.py` | 297 | W4A16: INT4 weight × BF16 activation, Marlin |
| `vllm/.../quantization/w4a8_fused_int4tc.py` | 263 | W4A8-Fused: INT4 weight × INT8 activation, Marlin |
| `vllm/.../quantization/w4a4a8_mixed_int4tc.py` | 325 | W4A4A8-Mixed: per-layer INT4/INT8 dispatch |
| `vllm/.../quantization/__init__.py` | (modified) | Registration: QuantizationMethods + method_to_config |

### CUDA Extension
| File | Description |
|------|-------------|
| `fp8_inference_toolkit/int4_native_tc/` | INT4 Native TC extension package |
| `int4_native_tc/__init__.py` | Python bindings for 9 CUDA kernels |
| `int4_native_tc/csrc/int4_gemm.cu` | CUTLASS INT4×INT4 and INT8×INT8 GEMM kernels |
| `int4_native_tc/csrc/int4_quant.cu` | INT4/INT8 quantization kernels |
| `int4_native_tc/csrc/bindings.cpp` | pybind11 bindings |

### Tests & Benchmarks
| File | Lines | Description |
|------|-------|-------------|
| `int4_native_tc/test_correctness.py` | 1372 | 24 correctness tests |
| `benchmark/benchmark_int4_native_gemm.py` | 437 | GEMM micro-benchmark (8 methods, 5 layer shapes) |

### Utility
| File | Description |
|------|-------------|
| `launch_mixed_server.py` | vLLM server launcher with custom quant registration |

---

## 9. Shared Utility Functions

`w4a16_int4tc.py`에서 정의되어 W4A8-Fused와 W4A4A8-Mixed에서 import하여 재사용:

```python
from vllm.model_executor.layers.quantization.w4a16_int4tc import (
    _quantize_w4_symmetric,   # BF16 → INT4 per-channel symmetric quant
    _pack_to_gptq_format,     # INT4 unsigned → GPTQ int32 packed format
)
```

---

## 10. Known Issues & Limitations

### 10.1 `torch.compile` Incompatibility
- `int4_native_tc`는 pybind11 C++ extension으로, `torch.dynamo`가 trace 불가
- vLLM v1은 기본적으로 `torch.compile` 활성화 → `--enforce-eager` 필수
- 해결: `torch.compiler.allow_in_graph()` 등록 또는 PyTorch custom op으로 재등록

### 10.2 INT4 Activation Quality
- INT4 activation (16 levels)은 single-layer에서는 cosine ~0.91이지만
- 36-layer 모델에서는 error accumulation으로 cosine ~0.38까지 하락
- 실용적으로 INT4 activation은 deep transformer에서 사용 불가

### 10.3 Marlin Alignment Requirements
- K must be divisible by 128 (`GPTQ_MARLIN_MIN_THREAD_K`)
- N must be divisible by 64 (`GPTQ_MARLIN_MIN_THREAD_N`)
- Padding이 필요한 경우 weight와 activation 모두 동일하게 padding

### 10.4 GPU Compute Capability
- W4A4 (CUTLASS INT4 TC): SM80+ (Ampere, Ada Lovelace, Hopper)
- W4A16/W4A8 (Marlin): SM75+ (Turing, Ampere, Ada Lovelace, Hopper)
- FP8 GEMM: SM89+ (Ada Lovelace, Hopper)

---

## 11. Conclusions & Recommendations

### Best Method by Use Case

| Use Case | Recommended Method | Reason |
|----------|-------------------|--------|
| **최대 품질 + VRAM 절약** | W4A16-Marlin | Cosine 0.838, 2.5 GiB, 안정적 |
| **속도-품질 균형** | W4A8-Fused | Cosine 0.826, 2.45 GiB, large batch에서 빠름 |
| **최대 속도 (품질 무관)** | W4A4 (INT4-TC) | 최저 latency, but cosine < 0.2 at model level |
| **실험적 mixed** | W4A4A8-Mixed | Cosine 0.379, 연구용 |

### Key Takeaway (Phase 1)

1. **INT4 weight quantization은 안전**: W4A16 (0.838), W4A8 (0.826) 모두 실용적 품질
2. **INT8 activation은 BF16에 근접**: W4A16 vs W4A8 차이 1.2%p만
3. **INT4 activation은 실모델에서 사용 불가**: 단일 layer 0.91 → 36 layers 0.38
4. **W4A8-Fused가 최적 tradeoff**:
   - VRAM: BF16 대비 67% 절약 (7.55 → 2.45 GiB)
   - 품질: Cosine 0.826 (실용적 수준)
   - 속도: INT8 TC MMA 2x throughput → compute-bound에서 유리
   - Weight bandwidth: INT4 (0.5 byte/elem) → memory-bound에서 유리

---

# Phase 2: Enhanced Per-Group Quantization

## 12. Phase 2 개요 및 목표

Phase 1에서 INT4 activation quantization이 실모델에서 사용 불가(cosine 0.38~0.49)한 핵심 원인 분석:

| 문제 | 원인 | Phase 2 해결 |
|------|------|-------------|
| **Per-channel weight quant** | K=2560개 값에 scale 1개 → 조대한 양자화 | Per-group g128: 128개 값당 scale 1개 |
| **Per-row activation quant** | K=2560개 값에 scale 1개 → outlier 1개가 전체 지배 | Per-group g128: 128개 값당 scale 1개 |
| **INT4×INT8 post-activation** | INT8 activation quantization 오차 존재 | INT4×BF16: activation 양자화 완전 제거 |
| **Marlin 의존성** | Marlin은 per-group activation quant 미지원 | Marlin 완전 제거, CUTLASS 통일 |

**목표**: W4A4A8-Mixed의 cosine 0.485 → 0.80+ 달성

---

## 13. Enhanced-PG Architecture (`w4a8-enhanced-int4tc`)

### 13.1 Layer-wise Quantization Strategy

```
Transformer Layer:
  ┌─ LayerNorm ─┬─ q_proj ──┐
  │             ├─ k_proj ──┤  INT4×INT4 (CUTLASS TC)
  │             └─ v_proj ──┘  Per-group g128 weight + activation
  │
  ├─ Attention ─── o_proj ────  INT4×BF16 (dequant + cuBLAS)
  │                             Per-group g128 weight only
  │
  ├─ LayerNorm ─┬─ gate_proj ─┐
  │             └─ up_proj ───┘  INT4×INT4 (CUTLASS TC)
  │                              Per-group g128 weight + activation
  │
  └─ SiLU(gate)×up ─ down_proj ─  INT4×BF16 (dequant + cuBLAS)
                                   Per-group g128 weight only
```

**Post-activation layers (o_proj, down_proj)에 BF16 activation을 사용하는 이유**:
- o_proj 입력: softmax 출력 후 → channel-wise magnitude variation 극심
- down_proj 입력: SiLU(gate)×up 후 → heavy-tailed 분포, activation spike
- 이 레이어들은 activation quantization 오차에 매우 민감
- BF16 activation = **zero activation precision loss**

### 13.2 Per-Group Quantization Loop

Marlin을 제거하고, **C++ level group loop + CUTLASS per-group GEMM** 구조 채택:

```
INT4×INT4 path (q/k/v/gate/up_proj):
  for g in 0..num_groups:
    x_packed[g], x_scale[g] = dynamic_int4_quant(x[:, g*128:(g+1)*128])
    partial[g] = CUTLASS_INT4_TC(x_packed[g], w_packed[g], x_scale[g], w_scale[g])
    acc_fp32 += partial[g]
  out = acc_fp32.to(bf16)

INT4×BF16 path (o_proj, down_proj):
  for g in 0..num_groups:
    w_bf16[g] = dequant_int4_to_bf16(w_packed[g], w_scale[g])
    partial[g] = cuBLAS_GEMM(x[:, g*128:(g+1)*128], w_bf16[g].T)
    acc_fp32 += partial[g]
  out = acc_fp32.to(bf16)
```

**FP32 누적**: 20개 group의 BF16 부분합을 BF16으로 직접 합산하면 정밀도 손실 발생.
FP32로 누적 후 최종 변환하여 이를 방지.

### 13.3 Marlin 완전 제거

Phase 1에서 사용한 Marlin 의존성을 모두 제거:

```python
# 제거된 import:
from vllm.model_executor.layers.quantization.utils.marlin_utils import (...)
from vllm.model_executor.layers.quantization.w4a16_int4tc import (...)
from vllm.scalar_type import scalar_types

# 대체: 모든 GEMM을 CUTLASS 또는 cuBLAS로 처리
import int4_native_tc  # custom CUDA extension
```

---

## 14. 구현된 기법 상세

### 14.1 Per-Group Weight Quantization

```
기존 (Phase 1): Per-channel
  weight [N, 2560] → scale 1개/channel → 2560개 값에 1개 scale

Enhanced (Phase 2): Per-group g128
  weight [N, 2560] → 20 groups × 128 values → group당 scale 1개
  → 양자화 해상도 20배 향상
```

**CUDA Kernel**: `static_int4_weight_quant_grouped`
- Grid: `(N, num_groups)`, Block: `(256)`
- 각 thread block이 (output_channel, group) 1개 처리
- Output: `[num_groups, N, gs/2]` packed + `[num_groups, N]` scales

### 14.2 Per-Group Activation Quantization

```
기존 (Phase 1): Per-row
  activation [M, 2560] → scale 1개/row → 2560개 값에 1개 scale
  → outlier 1개가 전체 행의 scale을 지배

Enhanced (Phase 2): Per-group g128
  activation [M, 2560] → 20 groups × 128 values → group당 scale 1개
  → outlier 영향이 128개 값으로 제한
```

**CUDA Kernel**: `dynamic_int4_quant_grouped`
- Grid: `(M, num_groups)`, Block: `(256)`
- 런타임 동적 양자화 (activation은 입력마다 분포가 다름)
- Output: `[num_groups, M, gs/2]` packed + `[num_groups, M]` scales

### 14.3 SmoothQuant (테스트 → 역효과 확인)

**이론**: activation outlier를 weight 쪽으로 이전하여 양자화 난이도를 균형화.

```
s_j = (max|X_j|)^α / (max|W_j|)^(1-α)
X̃ = X · diag(s)^(-1)    (activation이 평탄해짐)
W̃ = diag(s) · W          (weight가 난이도 흡수)
```

**실험 결과**:

| Config | Alpha | Calibration | Mean Cosine |
|--------|:-----:|-------------|:-----------:|
| Enhanced-PG (no SQ) | - | - | **0.880** |
| Enhanced-PG+SQ | 0.1 | wikitext | 0.842 |
| Enhanced-PG+SQ | 0.2 | wikitext | 0.840 |
| Enhanced-PG+SQ | 0.3 | wikitext | 0.856 |
| Enhanced-PG+SQ | 0.5 | wikitext | 0.851 |
| Enhanced-PG+SQ | 0.5 | random tokens | 0.867 |

**결론: SmoothQuant는 per-group quantization에서 역효과.**

원인 분석: SmoothQuant는 per-channel absmax 기반으로 smoothing factor를 계산.
이 과정에서 **within-group coefficient of variation이 증가** (0.7547 → 0.7745):
- Smoothing이 채널 간 값 크기를 재분배하면서
- 같은 group 내 값들의 편차가 오히려 커짐
- Per-group scale이 이를 capture하지 못하여 양자화 오차 증가

### 14.4 Outlier Clipping

**아이디어**: absmax의 일정 비율만 사용하여 scale 계산. Outlier를 포화시키는 대신
나머지 값들의 양자화 해상도를 향상.

```
기존: scale = absmax / 7.0
Clipping: scale = absmax * clip_ratio / 7.0
→ absmax 근처 값들은 ±7/±8로 포화되지만, 대다수 값의 해상도 향상
```

**CUDA Kernel**: `dynamic_int4_quant_clipped_grouped`
- `dynamic_int4_quant_grouped`와 동일하되 `clip_ratio` 파라미터 추가
- `clip_ratio` ∈ (0, 1]: 1.0 = no clipping

**실험 결과 (단독 적용)**:

| clip_ratio | Mean Cosine | vs Baseline |
|:----------:|:-----------:|:-----------:|
| 1.00 (baseline) | 0.880 | - |
| 0.99 | 0.877 | -0.003 |
| 0.95 | 0.881 | +0.001 |
| 0.90 | 0.878 | -0.003 |
| 0.85 | 0.878 | -0.002 |

**단독 적용 시 효과 미미**: per-group g128이 이미 outlier를 128개 값으로 제한하므로
추가 clipping의 한계 이득이 적음. 단, **Asymmetric과 조합 시 시너지 발생** (14.5절 참조).

### 14.5 Asymmetric INT4 Activation Quantization

**핵심 통찰** (quantization.md Section 2.5):
> "가중치에는 대칭(Symmetric) 양자화를, 활성화에는 비대칭(Asymmetric) 양자화를
> 적용하는 것이 일반적이다. 활성화는 비선형 함수를 거쳐 비대칭적 분포를 보이기 때문."

**Symmetric (기존)**: [-8, 7] 범위, 16 levels, 0 중심 고정
```
scale = absmax / 7.0
q = clamp(round(x / scale), -8, 7)
```

**Asymmetric (신규)**: [0, 15] 범위, 16 levels, 실제 분포에 적응
```
scale = (x_max - x_min) / 15.0
azp = clamp(round(-x_min / scale), 0, 15)    // zero point
q_unsigned = clamp(round(x / scale + azp), 0, 15)
q_signed = q_unsigned - 8                     // INT4 MMA 호환 [-8,7]
```

**INT4 MMA 호환성 문제와 AZP 보정**:

CUTLASS INT4 MMA는 signed INT4 [-8,7]을 기대. Asymmetric 값을 signed로 shift하면
GEMM 결과에 bias가 발생하므로 보정 필요:

```
GEMM 계산:   Y_gemm[m,n] = scale_a[m] · scale_b[n] · Σ_k (a_signed[k] · b_signed[k])
실제 원하는: Y_true[m,n] = scale_a[m] · scale_b[n] · Σ_k ((a_unsigned[k] - azp[m]) · b_signed[k])

보정:
  Y_true = Y_gemm + scale_a[m] · scale_b[n] · (8 - azp[m]) · Σ_k(b_signed[k])
                                                   ↑ azp_adj       ↑ w_col_sum[n]
```

`w_col_sum[g][n]` = group g의 output neuron n에 대한 INT4 weight 값의 합 (사전 계산).

**CUDA Kernels**:
- `dynamic_int4_quant_asymmetric_grouped`: 비대칭 INT4 양자화 + azp_adj 반환
- `cutlass_int4_scaled_mm_azp_grouped`: AZP 보정 포함 INT4×INT4 GEMM

**실험 결과**:

| Config | Mean Cosine | Min | Max | vs Baseline |
|--------|:-----------:|:---:|:---:|:-----------:|
| Enhanced-PG (sym, baseline) | 0.880 | 0.854 | 0.913 | - |
| PG+Asymmetric | 0.898 | 0.869 | 0.914 | +0.018 |
| PG+Asym+Clip(0.99) | 0.900 | 0.862 | 0.915 | +0.020 |
| **PG+Asym+Clip(0.95)** | **0.906** | **0.878** | **0.927** | **+0.026** |

**Asymmetric + Clip(0.95) = 최고 성능 (0.906)**

Clip 단독은 효과 없지만, Asymmetric과 조합하면 시너지 발생:
- Asymmetric은 [min, max] 범위를 사용
- Clip이 outlier를 먼저 제거 → 더 타이트한 [min, max] 범위 → 더 높은 해상도

---

## 15. Phase 2 전체 결과 비교

### 15.1 Cosine Similarity 총괄표

**모델**: Qwen/Qwen3-Embedding-4B, **GPU**: NVIDIA L4
**테스트**: 10개 문장, BF16 baseline 대비 embedding cosine similarity

| Config | Weight Quant | Act Quant | Mean Cosine |
|--------|-------------|-----------|:-----------:|
| BF16 (reference) | BF16 | BF16 | 1.000 |
| W4A16 (g128, Marlin) | INT4 g128 | BF16 | 0.933 |
| **Enhanced-PG+Asym+Clip** | INT4 g128 | INT4 g128 asym+clip | **0.906** |
| Enhanced-PG+Asymmetric | INT4 g128 | INT4 g128 asym | 0.900 |
| Enhanced-PG (sym) | INT4 g128 | INT4 g128 sym | 0.880 |
| Enhanced-PG+SQ (best) | INT4 g128+SQ | INT4 g128 sym | 0.867 |
| W4A8-Fused (Phase 1) | INT4 Marlin | INT8 per-row | 0.826 |
| W4A16 (Phase 1, old test) | INT4 Marlin | BF16 | 0.838 |
| W4A4A8-Mixed (Phase 1) | INT4 per-ch | INT4/INT8 per-row | 0.485 |

### 15.2 Phase 1 → Phase 2 개선 분석

```
Phase 1 (W4A4A8-Mixed):  0.485  ─────────────────────────────────┐
  + Per-group g128:       +0.395  (per-channel → per-group)       │
  + INT4×BF16 post-act:   (포함)  (INT4×INT8 → INT4×BF16)        │  총 +0.421
  = Enhanced-PG:           0.880                                   │
  + Asymmetric INT4:      +0.018  (sym [-8,7] → asym [0,15])     │
  + Clip(0.95)+Asym:      +0.008  (outlier clipping 시너지)       │
  = Enhanced-PG+Asym+Clip: 0.906  ────────────────────────────────┘

남은 Gap:  0.906 → 0.933 (W4A16 Marlin) = 2.7%
           → INT4 activation quantization의 본질적 한계
```

### 15.3 기법별 기여도 요약

| 기법 | 적용 대상 | 효과 | 상태 |
|------|----------|------|------|
| Per-group weight g128 | Weight | +++ (핵심) | ✅ 적용 |
| Per-group activation g128 | Activation | +++ (핵심) | ✅ 적용 |
| INT4×BF16 post-activation | o_proj, down_proj | +++ (핵심) | ✅ 적용 |
| FP32 cross-group accumulation | GEMM | ++ | ✅ 적용 |
| Asymmetric INT4 activation | Activation | ++ (+0.018~0.026) | ✅ 적용 |
| Outlier clipping + Asymmetric | Activation | + (+0.008) | ✅ 적용 |
| SmoothQuant | Weight+Activation | **역효과** (-0.013~0.040) | ❌ 미사용 |
| Outlier clipping (단독) | Activation | 무효 | ❌ 미사용 |

---

## 16. 파일 구조 (Phase 2 추가분)

### 16.1 vLLM Quantization Config

| File | Description |
|------|-------------|
| `vllm/.../quantization/w4a8_enhanced_int4tc.py` | Enhanced-PG: per-group + asymmetric + clipping |

**주요 클래스**:
- `W4A8EnhancedInt4TCConfig`: Config (calibration, alpha, clip_ratio, asymmetric)
- `EnhancedW4A4LinearMethod`: INT4×INT4 path (q/k/v/gate/up_proj)
- `EnhancedW4A16LinearMethod`: INT4×BF16 path (o_proj, down_proj)

**환경변수**:
```bash
INT4TC_CALIBRATION_STATS=calib.pt   # SmoothQuant calibration file
INT4TC_SQ_ALPHA=0.5                 # SmoothQuant alpha (default: 0.5)
INT4TC_CLIP_RATIO=0.95              # Outlier clip ratio (default: 1.0)
INT4TC_ASYMMETRIC=1                 # Enable asymmetric INT4 (default: 0)
```

### 16.2 CUDA Kernels (Phase 2 추가)

| Function | File | Description |
|----------|------|-------------|
| `static_int4_weight_quant_grouped` | int4_quant.cu | Per-group weight quantization |
| `dynamic_int4_quant_grouped` | int4_quant.cu | Per-group symmetric activation quantization |
| `dynamic_int4_quant_clipped_grouped` | int4_quant.cu | Per-group + outlier clipping |
| `dynamic_int4_quant_asymmetric_grouped` | int4_quant.cu | Per-group asymmetric activation quantization |
| `dynamic_int8_quant_grouped` | int4_quant.cu | Per-group INT8 symmetric |
| `dynamic_int8_quant_asymmetric_grouped` | int4_quant.cu | Per-group INT8 asymmetric |
| `cutlass_int4_scaled_mm_grouped` | int4_gemm.cu | Per-group INT4×INT4 GEMM (C++ loop) |
| `cutlass_int4_scaled_mm_azp_grouped` | int4_gemm.cu | Per-group INT4×INT4 GEMM + AZP correction |
| `cutlass_w4a8_scaled_mm_grouped` | int4_gemm.cu | Per-group INT8×INT8 GEMM + optional AZP |
| `cutlass_w4a16_mm_grouped` | int4_gemm.cu | Per-group INT4→BF16 dequant + cuBLAS GEMM |

### 16.3 Calibration Tools

| File | Description |
|------|-------------|
| `calibration/collector.py` | Activation statistics collector (hook-based) |
| `calibration/smoothquant.py` | SmoothQuant factor computation |
| `calibration/run_calibration.py` | CLI tool for calibration data generation |

### 16.4 Measurement Scripts

| File | Description |
|------|-------------|
| `measure_cosine.py` | Full cosine measurement (all configs) |
| `test_features.py` | Feature-specific A/B test (clipping, asymmetric) |

---

## 17. 사용법

### 17.1 기본 사용 (최적 설정)

```bash
# Asymmetric + Clip(0.95) — 현재 최고 품질 (cosine 0.906)
INT4TC_ASYMMETRIC=1 INT4TC_CLIP_RATIO=0.95 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-4B \
    --quantization w4a8-enhanced-int4tc \
    --enforce-eager
```

### 17.2 설정별 사용

```bash
# Per-group only (no asymmetric, no clipping) — cosine 0.880
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-4B \
    --quantization w4a8-enhanced-int4tc \
    --enforce-eager

# Asymmetric only — cosine 0.900
INT4TC_ASYMMETRIC=1 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-4B \
    --quantization w4a8-enhanced-int4tc \
    --enforce-eager
```

---

## 18. 미적용 기법 및 향후 계획

### 18.1 남은 기법

현재 cosine gap: **0.906 → 0.933** (W4A16 Marlin) = **2.7%**

이 gap은 INT4 activation quantization의 정밀도 한계에서 발생.
다음 기법들로 추가 개선 가능:

| 기법 | 예상 효과 | 난이도 | 설명 |
|------|----------|--------|------|
| **GPTQ error compensation** | 높음 | 중 | RTN → Hessian 기반 열 단위 오차 보상. Weight 양자화 품질 직접 개선 |
| **Hadamard rotation (DuQuant/QuIP#)** | 높음 | 중 | Hadamard 행렬로 activation outlier 균등 분산. W4A4 최고 성능 기법 |
| **AWQ salient weight protection** | 중간 | 중 | Activation magnitude로 중요 가중치 식별, per-channel scaling 보호 |

### 18.2 기법별 작동 원리

**GPTQ**: 현재 weight quantization은 RTN (Round-to-Nearest) — 단순 반올림.
GPTQ는 한 가중치를 양자화할 때 Hessian 역행렬을 사용하여 나머지 미양자화 가중치를
조정, 출력 오차를 보상. 4-bit에서 RTN 대비 상당한 품질 차이를 만듦.

**Hadamard rotation**: SmoothQuant와 달리 outlier를 weight로 "이전"하는 것이 아니라,
Hadamard 변환으로 전체 채널에 "균등 분산". Within-group CV를 악화시키지 않으므로
per-group quantization과 궁합이 좋을 것으로 예상.

**AWQ**: Activation magnitude로 돌출(salient) 가중치를 식별 → per-channel scaling
적용으로 양자화 오차 최소화. SmoothQuant와 다른 접근 — 가중치 쪽만 조정.
