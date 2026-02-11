# vLLM 양자화 커널 분석 (v0.15)

vLLM에서 FP8, INT4 양자화 모델의 GEMM 연산이 실제로 어떤 커널과 텐서코어 인스트럭션을 사용하는지 분석한 문서입니다.

> 분석 기준: vLLM 0.15.2rc1, CUDA, NVIDIA GPU (L4/A100/H100)
> 소스코드: `/home/ubuntu/vllm/csrc/quantization/`

---

## 핵심 요약

- **FP8 (W8A8)**: Activation을 FP8로 동적 양자화 → FP8 텐서코어 네이티브 사용 (`e4m3 × e4m3`)
- **INT4 (W4A16)**: INT4 weight를 상위 타입(FP16/BF16/INT8/FP8)으로 dequant → 해당 타입 텐서코어 사용
- **네이티브 INT4 텐서코어(`s4 × s4`)를 사용하는 커널은 vLLM에 없음** (H100 포함 모든 아키텍처)

---

## 1. GEMM 기본 개념

```
Output = A × B

A = Activation (이전 레이어 출력, 입력 텐서)
B = Weight (학습된 가중치 행렬)
```

양자화 표기법:
- **W8A8**: Weight 8bit, Activation 8bit
- **W4A16**: Weight 4bit, Activation 16bit (FP16/BF16)

NVIDIA 텐서코어 MMA 인스트럭션은 **A, B가 같은 타입**이어야 함:
```
mma.sync.aligned.m16n8k16.row.col.{출력}.{A타입}.{B타입}.{누적}
```

따라서 A, B 저장 타입이 다르면 **dequant하여 같은 타입으로 맞춘 후** MMA 실행.

---

## 2. FP8 커널 전수조사

### 2.1 Activation 양자화 과정

모든 FP8 커널은 matmul 전에 activation을 FP8로 양자화:

```python
# ScaledMMLinearKernel.apply_weights()
if x.dtype != fp8_dtype:
    x_2d_q, x_s = self.quant_fp8(x_2d, x_s, x_s_ub)  # BF16/FP16 → FP8

return self.apply_scaled_mm(
    A=x_2d_q,   # Activation (FP8)
    B=w,         # Weight (FP8)
    As=x_s,      # Activation scale
    Bs=w_s,      # Weight scale
)
```

내부적으로 `ops.scaled_fp8_quant()` → `scaled_fp8_conversion<true, fp8_type>()` CUDA 커널 사용.

### 2.2 FP8 커널 목록

#### CUTLASS W8A8 (wgmma 기반, SM90+)

| 커널 파일 | A | B | MMA | Accum | Output | Min SM |
|-----------|---|---|-----|-------|--------|--------|
| `scaled_mm_sm90_fp8.cu` | FP8 e4m3 | FP8 e4m3 | `wgmma` | FP32 | BF16/FP16 | SM90 (H100) |
| `scaled_mm_blockwise_sm90_fp8.cu` | FP8 e4m3 | FP8 e4m3 (128x128 block) | `wgmma` | FP32 | BF16/FP16 | SM90 |
| `scaled_mm_sm100_fp8.cu` | FP8 e4m3 | FP8 e4m3 | `wgmma` | FP32 | BF16/FP16 | SM100 |
| `scaled_mm_blockwise_sm100_fp8.cu` | FP8 e4m3 | FP8 e4m3 | `wgmma` | FP32 | BF16/FP16 | SM100 |
| `scaled_mm_sm120_fp8.cu` | FP8 e4m3 | FP8 e4m3 | `wgmma` | FP32 | BF16/FP16 | SM120 (Blackwell) |
| `scaled_mm_blockwise_sm120_fp8.cu` | FP8 e4m3 | FP8 e4m3 | `wgmma` | FP32 | BF16/FP16 | SM120 |

CUTLASS 커널 구조:
```cpp
// scaled_mm_sm90_fp8_dispatch.cuh
struct cutlass_3x_gemm_sm90_fp8 {
    using ElementAB = float_e4m3_t;   // A, B 모두 FP8
    using ElementAcc = float;          // 누적 FP32
    using ElementD = bfloat16_t;       // 출력 BF16
};
```

#### Marlin FP8 (mma.sync 기반, SM80+)

| 커널 파일 | A | B | 실제 MMA 인스트럭션 | Min SM |
|-----------|---|---|---------------------|--------|
| `sm80_kernel_float16_fe4m3fn_float16.cu` | FP8 e4m3 | FP8 e4m3 | `mma.sync.m16n8k16.f32.e4m3.e4m3.f32` | SM80 (A100) |
| `sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu` | FP8 e4m3 | FP8 e4m3 | `mma.sync.m16n8k16.f32.e4m3.e4m3.f32` | SM80 |

MMA 인스트럭션 (marlin_mma.h):
```cuda
// FP8 × FP8 → FP32 누적
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
    ...);
```

#### PyTorch Native (SM89+)

| 커널 | A | B | Backend | Min SM |
|------|---|---|---------|--------|
| PerTensor `torch._scaled_mm` | FP8 e4m3 | FP8 e4m3 | cuBLAS | SM89 (L4/RTX 4090) |
| ChannelWise `torch._scaled_mm` | FP8 e4m3 | FP8 e4m3 | cuBLAS + unfused dequant | SM89 |

#### 기타

| 커널 | A | B | Backend | Min SM |
|------|---|---|---------|--------|
| FlashInfer `scaled_fp8_mm` | FP8 e4m3 | FP8 e4m3 | FlashInfer | SM100 |
| DeepGEMM | FP8 e4m3 | FP8 e4m3 | DeepGEMM library | SM90 |
| ROCm wvSplitKQ | FP8 e4m3fnuz | FP8 e4m3fnuz | hipBLASLt | MI3xx (AMD) |

### 2.3 L4 (SM89)에서의 FP8 커널 선택

우선순위 (`kernels/scaled_mm/__init__.py`):

```python
_POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA] = [
    FlashInferFP8ScaledMMLinearKernel,       # SM100+ → X
    CutlassFP8ScaledMMLinearKernel,          # SM90+  → X
    PerTensorTorchFP8ScaledMMLinearKernel,   # SM89+  → O (선택됨)
    ChannelWiseTorchFP8ScaledMMLinearKernel,  # SM89+  → fallback
]
```

**L4에서는 `torch._scaled_mm()` → cuBLAS FP8 텐서코어 사용.**

---

## 3. INT4 커널 전수조사

### 3.1 핵심: INT4 weight는 항상 dequant 후 연산

모든 INT4 커널에서 weight(B)는 MMA 실행 전에 A 타입으로 dequantize됨.
**네이티브 INT4 텐서코어 MMA (`s4 × s4`)를 사용하는 커널은 없음.**

dequant 과정 (`marlin/dequant.h`):
```cuda
// INT4(U4) → FP16 dequantization (레지스터에서)
template <>
__device__ inline void dequant<half2, vllm::kU4.id(), false>(int q, half2* frag_b) {
    const int LO = 0x000f000f;   // 하위 4bit 마스크
    const int HI = 0x00f000f0;   // 상위 4bit 마스크
    const int EX = 0x64006400;   // FP16 변환용 지수

    // lop3 비트 연산으로 4bit 추출 → FP16 구성
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

    // FP16으로 변환 + scale/zero-point 적용
    frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                        *reinterpret_cast<const half2*>(&SUB));
    frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                        *reinterpret_cast<const half2*>(&MUL),
                        *reinterpret_cast<const half2*>(&ADD));
}
```

### 3.2 Marlin INT4 커널 (SM80+)

#### W4A16 (FP16/BF16 activation)

| 커널 파일 | A (저장) | B (저장) | B dequant → | 실제 MMA 인스트럭션 | Min SM |
|-----------|---------|---------|-------------|---------------------|--------|
| `sm80_kernel_float16_u4_float16.cu` | FP16 | U4 | → **FP16** | `f32.f16.f16.f32` | SM80 |
| `sm80_kernel_float16_u4b8_float16.cu` | FP16 | U4B8 | → **FP16** | `f32.f16.f16.f32` | SM80 |
| `sm80_kernel_bfloat16_u4_bfloat16.cu` | BF16 | U4 | → **BF16** | `f32.bf16.bf16.f32` | SM80 |
| `sm80_kernel_bfloat16_u4b8_bfloat16.cu` | BF16 | U4B8 | → **BF16** | `f32.bf16.bf16.f32` | SM80 |

MMA 인스트럭션 (marlin_mma.h):
```cuda
// FP16 × FP16 → FP32 (INT4 weight가 FP16으로 dequant된 후)
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    ...);
```

#### W4A8 (INT8 activation)

| 커널 파일 | A (저장) | B (저장) | B dequant → | 실제 MMA 인스트럭션 | Min SM |
|-----------|---------|---------|-------------|---------------------|--------|
| `sm80_kernel_s8_u4_float16.cu` | INT8 | U4 | → **INT8** | `s32.s8.s8.s32` | SM80 |
| `sm80_kernel_s8_u4_bfloat16.cu` | INT8 | U4 | → **INT8** | `s32.s8.s8.s32` | SM80 |
| `sm80_kernel_s8_u4b8_float16.cu` | INT8 | U4B8 | → **INT8** | `s32.s8.s8.s32` | SM80 |
| `sm80_kernel_s8_u4b8_bfloat16.cu` | INT8 | U4B8 | → **INT8** | `s32.s8.s8.s32` | SM80 |

MMA 인스트럭션:
```cuda
// INT8 × INT8 → INT32 (INT4 weight가 INT8로 dequant된 후)
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
    ...);
```

#### W4A8 (FP8 activation, SM89+)

| 커널 파일 | A (저장) | B (저장) | B dequant → | 실제 MMA 인스트럭션 | Min SM |
|-----------|---------|---------|-------------|---------------------|--------|
| `sm89_kernel_fe4m3fn_u4_float16.cu` | FP8 e4m3 | U4 | → **FP8** | `f32.e4m3.e4m3.f32` | SM89 |
| `sm89_kernel_fe4m3fn_u4b8_float16.cu` | FP8 e4m3 | U4B8 | → **FP8** | `f32.e4m3.e4m3.f32` | SM89 |
| `sm89_kernel_fe4m3fn_u4_bfloat16.cu` | FP8 e4m3 | U4 | → **FP8** | `f32.e4m3.e4m3.f32` | SM89 |
| `sm89_kernel_fe4m3fn_u4b8_bfloat16.cu` | FP8 e4m3 | U4B8 | → **FP8** | `f32.e4m3.e4m3.f32` | SM89 |

### 3.3 AWQ Custom 커널 (SM75+)

| 커널 파일 | A | B (저장) | B dequant → | 실제 연산 | Min SM |
|-----------|---|---------|-------------|-----------|--------|
| `awq/gemm_kernels.cu` | FP16 | INT4 packed | → **FP16** | `fma.rn.f16x2` (MMA 미사용) | SM75 |

### 3.4 CUTLASS W4A8 (SM90+, Hopper 전용)

| 커널 파일 | A | B (저장) | 실제 연산 | Min SM |
|-----------|---|---------|-----------|--------|
| `cutlass_w4a8/w4a8_mm_entry.cu` | FP8 e4m3 | `int4b_t` | `wgmma` (dequant 후) | SM90 |

### 3.5 Machete (SM90+, Hopper 전용)

| 커널 | A | B (저장) | 실제 연산 | Min SM |
|------|---|---------|-----------|--------|
| Machete U4 | FP16/BF16 | U4 | `wgmma` (dequant 후) | SM90 |
| Machete U4B8 | FP16/BF16 | U4B8 | `wgmma` (dequant 후) | SM90 |

---

## 4. L4 (SM89)에서의 실제 커널 경로

### FP8 모델 (W8A8)

```
Activation (BF16)
  → scaled_fp8_quant() → FP8 e4m3
  → torch._scaled_mm(FP8, FP8)
  → cuBLAS FP8 텐서코어 (e4m3 × e4m3)
  → Output (BF16)
```

**FP8 네이티브 텐서코어 사용. L4 FP8 TFLOPS: 242.**

### INT4 모델 (W4A16, compressed-tensors)

```
Activation (FP16)                    Weight (INT4 packed)
  │                                    │
  │                                    ↓
  │                               lop3.b32 비트 추출
  │                                    ↓
  │                               FP16으로 변환 (dequant)
  │                                    │
  ↓                                    ↓
  FP16 ──────── MMA (f16 × f16) ──── FP16
                      ↓
                 Output (FP16)
```

**FP16 텐서코어 사용. L4 FP16 TFLOPS: 120.** INT4 텐서코어 미사용.

---

## 5. INT4 텐서코어를 왜 안 쓰는가

### 하드웨어 지원 현황

NVIDIA 텐서코어 MMA 인스트럭션 중 INT4 관련:
```
mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32    (SM75+ Turing)
mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32   (SM80+ Ampere)
```

L4 (SM89), A100 (SM80), H100 (SM90) 모두 **하드웨어적으로 INT4 텐서코어 지원**.

### 안 쓰는 이유

INT4 텐서코어 MMA는 **A, B 모두 INT4** 필요 → W4**A4** 필요:

```
s4 × s4 → s32   (INT4 × INT4 → INT32)
```

그런데 현재 양자화 스킴은 W4**A16** (weight만 INT4, activation은 FP16):
- Activation을 INT4로 양자화하면 **16개 값**만으로 표현해야 함
- FP8은 **256개 값** + 부동소수점이라 dynamic range 커버 가능
- INT4 activation은 품질 열화가 심해서 실용적이지 않음

### 결과적으로

| 양자화 | 저장 타입 | MMA 전 변환 | 실제 텐서코어 | 속도 |
|--------|----------|------------|--------------|------|
| FP8 (W8A8) | A=FP8, B=FP8 | 변환 불필요 | FP8 네이티브 (242 TFLOPS) | **빠름** |
| INT4 (W4A16) | A=FP16, B=INT4 | B: INT4→FP16 dequant | FP16 (120 TFLOPS) | BF16과 비슷 |
| INT4 (W4A4) | A=INT4, B=INT4 | 변환 불필요 | INT4 네이티브 | **커널 없음** |

**INT4의 이점은 연산 속도가 아닌 메모리 절약(모델 크기 축소).**
연산 속도를 원하면 **FP8 (W8A8)**이 정답.

---

## 6. TensorRT-LLM과의 비교 (NVIDIA 공식 프레임워크)

NVIDIA가 직접 개발한 TensorRT-LLM (`/home/ubuntu/TensorRT-LLM/`)에서도 동일한 조사를 수행.

### TensorRT-LLM의 INT4 커널: `fpA_intB_gemm`

"FP Activation × INT Weight" GEMM. 이름 자체가 mixed-precision dequant 방식을 의미.

**위치**: `cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/`

| 커널 파일 | A (activation) | B (weight) | B dequant → | 실제 MMA | Min SM |
|-----------|---------------|------------|-------------|----------|--------|
| `fp16_int4_gemm_*.cu` | FP16 | `uint4b_t` | → **FP16** | `f32.f16.f16.f32` | SM80 |
| `bf16_int4_gemm_*.cu` | BF16 | `uint4b_t` | → **BF16** | `f32.bf16.bf16.f32` | SM80 |
| `e4m3_int4_gemm_*.cu` | FP8 e4m3 | `uint4b_t` | → **FP8** | `f32.e4m3.e4m3.f32` | SM89 |

### Dequant 구현 (`mma_tensorop_dequantizer.h`)

```cuda
// MmaTensorOpDequantizer: MMA 실행 직전에 INT4 → FP16/BF16 변환
__nv_bfloat162 scalex2 = __bfloat162bfloat162(scale_ptr[mma_n_iter]);
__nv_bfloat162* operand_bf16x2_ptr = reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[...]);
for (int ii = 0; ii < kElements / 2; ++ii) {
    operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii], scalex2);  // dequant!
}
// → 이후 표준 FP16/BF16 텐서코어 MMA 실행
```

### 기타 TensorRT-LLM INT4 커널

| 커널 | 위치 | A | B | 방식 |
|------|------|---|---|------|
| `weightOnlyBatchedGemv` | `kernels/weightOnlyBatchedGemv/` | FP16/BF16 | INT4 | dequant→FP16 후 FMA |
| MOE GEMM (INT4) | `kernels/cutlass_kernels/moe_gemm/` | FP16/BF16/FP8 | `uint4b_t` | dequant→FP16 후 MMA |

### 아키텍처별 처리 (TensorRT-LLM)

| GPU | INT4 weight 처리 | 실제 MMA |
|-----|-----------------|----------|
| Ampere (SM80, A100) | INT4→FP16 in warp (`MmaTensorOpDequantizer`) | `mma.sync.f32.f16.f16.f32` |
| Ada (SM89, L4) | INT4→FP16 in warp | `mma.sync.f32.f16.f16.f32` |
| Hopper (SM90, H100) | INT4→FP16 via TMA + warp | `wgmma` (FP16) |

**H100에서도 INT4 텐서코어(s4×s4) 미사용.**

### vLLM vs TensorRT-LLM 비교

| | vLLM (Marlin) | TensorRT-LLM (fpA_intB) |
|---|---|---|
| INT4 dequant 위치 | 레지스터 (`lop3.b32`) | 워프 레벨 (`MmaTensorOpDequantizer`) |
| Dequant 대상 타입 | FP16/BF16/INT8/FP8 | FP16/BF16/FP8 |
| 실제 MMA | `f16×f16` / `s8×s8` / `e4m3×e4m3` | `f16×f16` / `bf16×bf16` / `e4m3×e4m3` |
| 네이티브 INT4 TC (`s4×s4`) | **없음** | **없음** |

### 결론: 업계 표준이 "dequant-then-compute"

NVIDIA가 직접 만든 TensorRT-LLM조차 INT4 텐서코어를 안 쓴다.
하드웨어에 `mma.sync...s4.s4.s32` 인스트럭션이 존재하지만,
**vLLM, TensorRT-LLM 모두 INT4 weight를 상위 타입으로 dequant한 후 해당 타입의 텐서코어를 사용.**

이유:
1. W4A16 스킴에서 activation이 FP16이므로 INT4×FP16 mixed MMA 불가
2. W4A4로 바꾸면 INT4 텐서코어 사용 가능하나 activation 4bit 양자화는 품질 열화 심각
3. 결과적으로 INT4의 이점은 **메모리 절약**에 한정, 연산 속도 이점 없음

### 최종 검증: MMA 인스트럭션 전수조사

vLLM과 TensorRT-LLM 전체 소스코드에서 실제 사용되는 MMA 인스트럭션을 grep으로 전수조사한 결과:

#### vLLM (`csrc/`) - 실제 사용되는 MMA 전체 목록

```
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16        ← FP16
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32      ← BF16
mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32      ← FP8
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32        ← FP16
mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32          ← INT8
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32      ← FP8
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32          ← INT8
mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32           ← INT8
```

**`.s4` / `.u4` 인스트럭션: 0건**

#### TensorRT-LLM (`cpp/`) - 실제 사용되는 MMA 전체 목록

```
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16        ← FP16
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32      ← BF16
mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32      ← FP8
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32        ← FP16
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16      ← FP8
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32      ← FP8
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32          ← INT8
mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32           ← INT8
```

**`.s4` / `.u4` 인스트럭션: 0건**

#### TensorRT-LLM Hopper wgmma - 전체 목록

```
wgmma.mma_async...f32.f16.f16      ← FP16
wgmma.mma_async...f32.bf16.bf16    ← BF16
wgmma.mma_async...f32.e4m3.e4m3    ← FP8
wgmma.mma_async...f32.e4m3.e5m2    ← FP8 mixed
wgmma.mma_async...f32.e5m2.e4m3    ← FP8 mixed
wgmma.mma_async...f32.e5m2.e5m2    ← FP8
wgmma.mma_async...s32.s8.s8        ← INT8
```

**wgmma에서도 `.s4` / `.u4` 인스트럭션: 0건**

#### CUTLASS 라이브러리에는 정의만 존재 (미사용)

3rdparty CUTLASS 헤더(`cutlass/arch/mma_sm75.h`, `mma_sm80.h`)에 INT4 MMA 템플릿 정의는 있음:

```
mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32    (SM75 Turing)
mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32             (SM80 Ampere)
```

**vLLM, TensorRT-LLM 어디에서도 이 템플릿을 instantiate하지 않음.**

#### TensorRT-LLM `static_assert`로 컴파일 타임 강제

```cpp
// mma_tensorop_compute_B_with_f16.h
static_assert(
    (is_same<ArchMmaOperator::ElementA, half_t>::value
     && is_same<ArchMmaOperator::ElementB, half_t>::value)
    || (is_same<ArchMmaOperator::ElementA, bfloat16_t>::value ...)
    || (is_same<ArchMmaOperator::ElementA, float_e4m3_t>::value ...),
    "MmaTensorOpCvtBToA only supports underlying HMMA/QMMA");
```

MMA 양쪽 operand가 반드시 같은 타입(FP16/BF16/FP8)이어야 한다고 **컴파일 타임에 강제**.
INT4가 MMA operand로 들어가는 것 자체가 불가능한 구조.

#### 최종 결론

| | vLLM | TensorRT-LLM | CUTLASS 라이브러리 |
|---|---|---|---|
| `s4 × s4` MMA 사용 | **없음** | **없음** | 템플릿 정의만 존재 (미사용) |
| INT4 weight 처리 | dequant → FP16/BF16/INT8/FP8 | dequant → FP16/BF16/FP8 | - |
| 실제 텐서코어 | FP16/BF16/FP8/INT8 | FP16/BF16/FP8/INT8 | - |

**하드웨어(Turing~Hopper)는 네이티브 INT4 텐서코어를 지원하지만,
현존하는 주요 LLM 추론 프레임워크 중 이를 사용하는 곳은 없다.**

---

## 7. INT4 Dequant vs Native INT4: 정밀도와 속도 분석

### 7.1 정밀도: Dequant 경로의 정보 손실은?

INT4→BF16 dequant 후 matmul과 native INT4×INT4 matmul의 정밀도 차이를 검증.

**핵심**: INT4는 16개 값(-8~7)만 가능. BF16은 정수 -128~127을 exact 표현.
따라서 **INT4→BF16 변환 자체는 무손실**. 차이는 scale 적용 시 BF16 rounding에서만 발생.

| 경로 | 곱셈 | 누적 | Scale 적용 |
|------|------|------|-----------|
| Native INT4×INT4 | `s4×s4` = exact (INT32) | INT32 exact | matmul 후 1회 |
| Dequant BF16 | `(int4×scale)_bf16 × (int4×scale)_bf16` | FP32 (충분) | dequant 시 원소별 |

BF16 rounding 예시 (`int4_val × scale`):
```
 -6 × 0.1 = -0.6000 → BF16: -0.6016 (err=0.0016)  mantissa 7bit
  7 × 0.1 =  0.7000 → BF16:  0.6992 (err=0.0008)
 -5 × 0.1 = -0.5000 → BF16: -0.5000 (err=0.0000)  ← 2의 거듭제곱은 exact
```

실측 결과 (`verify_int4_matmul_precision.py`, M=16, K=2048, N=2048):

| 비교 | Mean Abs Diff | Relative Error | Exact Match |
|------|--------------|----------------|-------------|
| Native vs BF16 dequant | 0.0023 | ~1.4% | 0% |
| Native vs FP16 dequant | 0.0003 | ~0.17% | 0% |
| Native vs FP32 dequant | 0.0000002 | ~0.00004% | 12.7% |
| **Scale=2^n일 때** (모든 경로) | **0.0000** | **0.0%** | **100%** |

**결론**: Scale이 2의 거듭제곱이면 native와 dequant가 완전 동일.
임의 scale에서도 차이는 BF16 machine epsilon 수준 (~1%)으로, INT4 양자화 자체 오차 대비 무시 가능.
**네이티브 INT4 TC의 이점은 정밀도가 아니라 속도.**

### 7.2 속도: 네이티브 INT4 TC의 이론적 이점

#### GPU별 텐서코어 처리량 (공식 데이터시트)

| GPU | FP16 TC | FP8 TC | INT8 TC | INT4 TC | 메모리 BW |
|-----|---------|--------|---------|---------|----------|
| L4 (SM89, Ada) | 120 TFLOPS | 242 TOPS | 242 TOPS | 미기재* | 300 GB/s |
| A100 (SM80, Ampere) | 312 TFLOPS | - | 624 TOPS | **1,248 TOPS** | 2,039 GB/s |
| H100 (SM90, Hopper) | 989 TFLOPS | 1,979 TOPS | 1,979 TOPS | **~1,600 TOPS** | 3,350 GB/s |

*L4 데이터시트에 INT4 미기재. 패턴상 ~485 TOPS (2×INT8) 추정.

**A100 기준: INT4 TC는 FP16 대비 4x, INT8 대비 2x throughput.**

#### MMA 인스트럭션 단위 처리량 (CUTLASS 헤더)

```cpp
// SM80 (Ampere) - cutlass/arch/mma_sm80.h
FP16: Mma<GemmShape<16, 8, 16>>  →  4,096 ops/instruction   // m16n8k16.f32.f16.f16.f32
INT8: Mma<GemmShape<16, 8, 16>>  →  4,096 ops/instruction   // m16n8k16.s32.s8.s8.s32
INT4: Mma<GemmShape<16, 8, 64>>  → 16,384 ops/instruction   // m16n8k64.s32.s4.s4.s32

// SM89 (Ada)
FP8:  Mma<GemmShape<16, 8, 32>>  →  8,192 ops/instruction   // m16n8k32.f32.e4m3.e4m3.f32
```

INT4 MMA는 K=64로, FP16(K=16) 대비 **같은 인스트럭션 1번에 4배 많은 원소 처리**.

#### 하지만: LLM 추론은 대부분 Memory-Bound

LLM 추론의 두 단계:

```
Prefill (입력 전체 처리)  → Compute-bound (large batch)  → 네이티브 INT4 TC 이점 있음
Decode  (토큰 1개씩 생성) → Memory-bound (weight 로딩)    → 네이티브 INT4 TC 이점 없음
```

Decode 시 소요시간:
```
시간 ≈ weight_size / memory_bandwidth

INT4: weight_size = params × 0.5 byte  → 메모리 절약 (FP16 대비 4x 작음)
FP16: weight_size = params × 2 byte
```

**INT4 weight의 메모리 절약은 dequant 경로에서도 동일하게 확보됨.**
네이티브 TC가 추가로 주는 건 compute 가속뿐인데, decode에서는 compute가 병목이 아님.

#### Batch Size별 실제 이점

| Batch Size | 병목 | 현재 W4A16 (dequant→FP16 TC) | 가상 W4A4 (native INT4 TC) | 차이 |
|-----------|------|------------------------------|---------------------------|------|
| 1~16 | Memory BW | weight 로딩 시간 지배 | 동일 (같은 weight 크기) | **거의 없음** |
| 64~128 | 전환 구간 | 일부 compute 이점 시작 | 일부 가속 | **소폭** |
| 256+ | Compute | FP16 TC 포화 | INT4 TC 여유 | **최대 4x** |

#### 네이티브 INT4 TC를 안 쓰는 종합적 이유

1. **W4A4 품질 문제**: Activation INT4 양자화 시 16개 값으로 표현 → 심각한 품질 열화
2. **Memory-bound 지배**: LLM decode는 compute가 아닌 메모리 대역폭이 병목
3. **이미 메모리 이점 확보**: dequant 경로에서도 INT4 weight 크기 이점은 동일
4. **FP8이 더 실용적**: W8A8로 activation 품질 유지 + 네이티브 TC 사용 가능 + 2x 가속
5. **엔지니어링 비용**: INT4 TC 커널 최적화 대비 얻을 수 있는 실질적 이점 미미

---

## 8. 양자화 포맷과 HuggingFace 업로드

FP8, INT4 모두 동일한 프로세스:

```
llmcompressor oneshot()
  → model.save_pretrained(save_compressed=True)
  → HuggingFace upload (표준 safetensors + config.json)
```

vLLM은 `config.json`의 `quantization_config.quant_method: "compressed-tensors"` 를 보고 자동 감지.

### config.json 비교

| 항목 | FP8 | W4A16 |
|------|-----|-------|
| `format` | `float-quantized` | `pack-quantized` |
| `weights.type` | `float` (8bit) | `int` (4bit) |
| `weights.num_bits` | 8 | 4 |
| `weights.strategy` | `channel` | `group` (group_size=128) |
| `input_activations` | 8bit float (dynamic) | null (없음) |
| `scheme` (recipe) | `FP8_DYNAMIC` | `W4A16` |
| 모델 크기 (2B) | 2.3 GB | 1.5 GB |

---

## 9. 벤치마크 결과 (L4 GPU, Qwen3-VL-Embedding-2B)

### Latency 비교

| Config | VRAM | Batch=1 Avg | Batch=1 Throughput | Batch=16 Avg | Batch=16 Throughput |
|--------|------|-------------|--------------------|--------------|--------------------|
| BF16 | 5,522 MiB | 95.7ms | 10,697 t/s | 456.5ms | 35,898 t/s |
| FP8 | 4,722 MiB | 72.5ms | 14,121 t/s | 338.5ms | 48,399 t/s |
| INT4 (W4A16) | 4,386 MiB | 98.7ms | 10,369 t/s | 461.3ms | 35,522 t/s |

- **FP8**: BF16 대비 **1.3~1.4x 빠름** (FP8 텐서코어 네이티브)
- **INT4**: BF16과 비슷하거나 약간 느림 (FP16 텐서코어 + dequant 오버헤드)
- **INT4 VRAM**: BF16 대비 1.1GB 절약 (메모리 이점만 존재)

### 임베딩 품질 비교 (BF16 기준)

| 비교 | Mean Abs Diff | Cosine Similarity |
|------|--------------|-------------------|
| BF16 vs FP8-online | 0.00196~0.00206 | 0.9929~0.9937 |
| BF16 vs FP8-offline | 0.00204~0.00213 | 0.9921~0.9931 |
| FP8-online vs FP8-offline | 0.00258~0.00259 | 0.9888 |

Cosine similarity 0.99 이상 → 실용적으로 동등한 임베딩 품질.

---

## 10. Native INT4 Tensor Core 구현 및 벤치마크 (W4A4 / W4A16)

> 이전 섹션에서 "네이티브 INT4 TC를 사용하는 커널은 없다"는 것을 확인했다.
> 이 섹션에서는 **직접 CUTLASS INT4×INT4 TC 커널을 구현**하고,
> 실제 L4 GPU에서 BF16 / FP8 / W4A16 / W4A4 GEMM 성능을 비교한다.

### 10.1 구현 아키텍처

```
┌─ Standalone PyTorch C++ Extension ─────────────────────────┐
│  csrc/int4_gemm.cu    → CUTLASS INT4×INT4 device::Gemm    │
│  csrc/int4_quant.cu   → Dynamic INT4 quant / dequant      │
│  csrc/bindings.cpp    → pybind11 bindings                  │
└────────────────────────────────────────────────────────────┘
         ↓ import int4_native_tc
┌─ vLLM Custom Quantization ────────────────────────────────┐
│  w4a4_int4tc.py   → @register("w4a4-int4tc")             │
│  w4a16_int4tc.py  → @register("w4a16-int4tc")            │
└────────────────────────────────────────────────────────────┘
         ↓ --quantization w4a4-int4tc / w4a16-int4tc
┌─ vLLM Server ─────────────────────────────────────────────┐
│  Qwen3-Embedding-4B → BF16 로드 → weight INT4 양자화      │
│  W4A4: act INT4 quant → INT4×INT4 GEMM (native TC)        │
│  W4A16: INT4 weight dequant → BF16 GEMM (cuBLAS)          │
└────────────────────────────────────────────────────────────┘
```

#### 두 가지 접근법 비교

| | W4A4-INT4TC | W4A16-INT4TC |
|---|---|---|
| **Weight** | INT4 per-channel symmetric | INT4 per-channel symmetric |
| **Activation** | INT4 per-token dynamic | BF16 (양자화 없음) |
| **GEMM** | CUTLASS `s4×s4→s32` native TC | cuBLAS `bf16×bf16→f32` |
| **MMA instruction** | `mma.sync.m16n8k64.s32.s4.s4.s32.satfinite` | `mma.sync.m16n8k16.f32.bf16.bf16.f32` |
| **핵심 이점** | Compute throughput 4x (FP16 대비) | 메모리 절약 + BF16 정밀도 유지 |
| **핵심 약점** | Activation INT4 품질 열화 | dequant 오버헤드 |
| **Weight 압축** | 4x (BF16 대비) | 4x (BF16 대비) |

### 10.2 CUTLASS INT4×INT4 GEMM 커널 상세

CUTLASS 2.x `device::Gemm` API 사용. SM80 backward compatible (L4 SM89에서 실행 가능):

```cpp
// 핵심 타입 정의
using ElementA = cutlass::int4b_t;           // Activation (INT4)
using ElementB = cutlass::int4b_t;           // Weight (INT4)
using ElementAccumulator = int32_t;          // 누적 INT32
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;

// SM80 INT4 default (default_gemm_configuration.h:639-659)
using ThreadblockShape = GemmShape<128, 256, 128>;
using WarpShape = GemmShape<64, 64, 128>;
using InstructionShape = GemmShape<16, 8, 64>;
// → mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite
```

Tile shape dispatch (M 크기별):

| M | ThreadblockShape | WarpShape | 이유 |
|---|---|---|---|
| ≤16 | 64×64×128 | 32×32×128 | Decode (소형 batch) |
| ≤256 | 128×128×128 | 64×64×128 | 중간 batch |
| >256 | 128×256×128 | 64×64×128 | 대형 batch (기본) |

#### INT4 Packing Convention

```
byte = (high_nibble << 4) | (low_nibble & 0x0F)
  low_nibble  = element at even index
  high_nibble = element at odd index
  INT4 range: [-8, 7] (2's complement in 4 bits)
```

#### Dynamic INT4 Activation Quantization

vLLM `scaled_quant.cu` INT8 패턴 기반:

```
입력 BF16 [M, K]
  → Phase 1: per-row absmax (CUB BlockReduce)
  → Phase 2: scale = absmax / 7.0
  → Phase 3: q = clamp(round(x / scale), -8, 7)
  → Phase 4: 2개 int4를 1 byte로 pack
출력: packed_int4 [M, K/2] uint8, scale [M] float
```

### 10.3 정확성 검증 결과

CPU reference (numpy)와 비교:

| 테스트 | 행렬 크기 | 결과 |
|--------|----------|------|
| INT4 GEMM (known values) | 4×128×8 | **Exact match** (0 diff) |
| INT4 GEMM (LLM-sized) | 16×2560×2560 | **Exact match** (0 diff) |
| Scaled GEMM | 8×256×16 | **max_rel_err=0.00e+00** |
| W4A16 dequant | 16×256 | max_err=0.0077 (BF16 rounding) |
| W4A16 full path | 4×256×64 | mean_rel=0.032, max_rel=0.119 |
| W4A4 full path | 4×256×64 | mean_rel=0.047, max_rel=0.216 |

INT4×INT4 GEMM 자체는 CPU reference와 **완벽히 일치** (INT32 정수 연산).
W4A4 full path의 오차(~5%)는 활성화 INT4 양자화의 정보 손실에 의한 것.

### 10.4 GEMM Micro-Benchmark 결과 (L4 GPU)

**테스트 환경**: NVIDIA L4 (SM89), CUDA 12.4, CUTLASS v4.2.1

Qwen3-Embedding-4B GEMM dimensions:

| Layer | K | N |
|-------|------|------|
| qkv_proj | 2560 | 3840 |
| o_proj | 2560 | 2560 |
| gate_proj | 2560 | 9728 |
| up_proj | 2560 | 9728 |
| down_proj | 9728 | 2560 |

#### 대표 결과: qkv_proj (K=2560, N=3840)

| M | BF16 (us) | FP8 (us) | W4A16 (us) | W4A4-INT4TC (us) | W4A4 TFLOPS |
|---|-----------|----------|------------|------------------|-------------|
| 1 | 82.9 | 91.1 | 62.2 | **56.3** | 0.48 |
| 4 | 62.5 | 86.2 | 63.6 | **35.6** | 4.04 |
| 16 | 41.0 | 80.9 | 65.5 | **35.0** | 16.17 |
| 64 | 41.0 | 78.7 | 65.5 | **39.7** | 53.43 |
| 128 | 48.1 | 104.4 | 119.7 | **61.6** | 94.52 |
| 256 | 86.0 | 91.1 | 117.8 | **62.5** | 109.23 |
| 512 | 165.9 | 123.8 | 212.0 | **138.2** | 84.02 |
| 1024 | 323.6 | 215.0 | 376.8 | **213.0** | 105.14 |

#### 대표 결과: down_proj (K=9728, N=2560)

| M | BF16 (us) | FP8 (us) | W4A16 (us) | W4A4-INT4TC (us) | W4A4 TFLOPS |
|---|-----------|----------|------------|------------------|-------------|
| 1 | 67.6 | 91.2 | 393.2 | **42.0** | 1.95 |
| 16 | 74.8 | 83.1 | 340.0 | **39.9** | 33.84 |
| 128 | 193.5 | 86.9 | 431.1 | **79.9** | 103.77 |
| 256 | 263.2 | 167.9 | 533.5 | **106.5** | 153.73 |
| 1024 | 1123.3 | 500.7 | 1373.2 | **375.8** | 169.41 |

#### Full Forward Pass 추정 (36 layers × 5 GEMMs)

| M | BF16 (ms) | FP8 (ms) | W4A16 (ms) | W4A4-INT4TC (ms) | W4A4 Speedup vs BF16 |
|---|-----------|----------|------------|------------------|-----------------------|
| 1 | 16.26 | 15.23 | 48.84 | **7.67** | **2.1x** |
| 4 | 10.29 | 14.68 | 39.12 | **6.44** | **1.6x** |
| 16 | 10.07 | 14.79 | 39.85 | **7.19** | **1.4x** |
| 64 | 15.85 | 16.29 | 44.68 | **9.21** | **1.7x** |
| 128 | 23.70 | 16.70 | 53.37 | **11.25** | **2.1x** |
| 256 | 32.48 | 25.07 | 63.19 | **16.85** | **1.9x** |
| 512 | 52.31 | 36.42 | 83.83 | **32.02** | **1.6x** |
| 1024 | 143.88 | 71.48 | 174.96 | **68.57** | **2.1x** |

### 10.5 핵심 발견

#### W4A4-INT4TC: 예상 외로 빠르다

이전 분석(Section 7)에서 "LLM decode는 memory-bound이므로 INT4 TC 이점 없다"고 예측했으나:

1. **소형 batch (M=1~16)에서도 W4A4가 BF16보다 1.4~2.1x 빠름**
   - 이유: INT4 packing으로 weight 크기가 1/4 → 메모리 읽기도 빠름
   - INT4 MMA의 K=64 (FP16의 K=16 대비 4x) → 적은 인스트럭션 수

2. **대형 batch (M=256~1024)에서 W4A4가 FP8과 비슷하거나 더 빠름**
   - M=1024: W4A4 68.57ms vs FP8 71.48ms (W4A4이 FP8보다 빠름!)
   - 이 구간에서 compute-bound → INT4 TC의 높은 throughput 발휘

3. **W4A16은 dequant 오버헤드로 BF16보다 느림**
   - 특히 gate_proj, up_proj, down_proj (K=9728)에서 dequant 시간 ~260us
   - W4A16의 이점은 **메모리 절약**에만 한정

#### 각 방식의 최적 사용 사례

| 방식 | 최적 사례 | GEMM Latency | 메모리 절약 | 품질 |
|------|----------|-------------|-----------|------|
| **BF16** | 품질 최우선 | Baseline | 없음 | 최고 |
| **FP8** | 균형 (속도+품질) | 0.5~2x 빠름 | 2x | 우수 |
| **W4A16-INT4TC** | 메모리 제약 환경 | BF16과 유사~느림 | 4x | 우수 |
| **W4A4-INT4TC** | 최대 속도 | **1.4~2.1x 빠름** | 4x | 열화 있음 |

### 10.6 W4A4 Activation 품질 열화 분석

W4A4의 속도 이점은 크지만, INT4 activation 양자화의 정보 손실이 문제:

```
INT4 값 범위: [-8, -7, ..., 0, ..., 6, 7]  → 16개 레벨
FP8 값 범위: 256개 레벨 + 부동소수점 dynamic range
BF16 값 범위: 65,536개 레벨 + 넓은 dynamic range
```

실측 (4×256×64 GEMM):
- W4A4 full path mean relative error: **~4.7%**
- W4A16 full path mean relative error: **~3.3%**

LLM embedding/generation 품질에 대한 영향은 모델과 태스크에 따라 다르며,
end-to-end 벤치마크에서 검증 필요.

### 10.7 vLLM 통합

`register_quantization_config` 데코레이터를 사용하여 vLLM 코어 수정 없이 외부 등록:

```python
# W4A4 사용
import vllm.model_executor.layers.quantization.w4a4_int4tc
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-4B \
    --quantization w4a4-int4tc

# W4A16 사용
import vllm.model_executor.layers.quantization.w4a16_int4tc
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-4B \
    --quantization w4a16-int4tc
```

### 10.8 파일 목록

```
fp8_inference_toolkit/int4_native_tc/
├── csrc/
│   ├── int4_gemm.cu              # CUTLASS INT4×INT4 GEMM (3 tile sizes)
│   ├── int4_quant.cu             # INT4 quant/dequant (4 kernels)
│   └── bindings.cpp              # pybind11 bindings
├── int4_native_tc/__init__.py    # Python API
├── setup.py                      # Build config
├── pyproject.toml
└── test_correctness.py           # 8 correctness tests

vllm/vllm/model_executor/layers/quantization/
├── w4a4_int4tc.py                # W4A4 INT4 TC quantization
└── w4a16_int4tc.py               # W4A16 INT4 TC quantization

fp8_inference_toolkit/benchmark/
├── benchmark_int4_native_gemm.py # GEMM micro-benchmark
└── run_5way_benchmark.py         # End-to-end 5-way benchmark
```

### 10.9 결론

1. **네이티브 INT4 TC (W4A4)는 업계에서 사용하지 않지만, 실제로 구현하면 상당한 속도 이점이 있다**
   - L4에서 BF16 대비 1.4~2.1x GEMM 가속
   - FP8과도 동등하거나 빠른 수준 (대형 batch에서)

2. **W4A16은 dequant 오버헤드로 속도 이점이 제한적**
   - 메모리 절약(4x)이 유일한 이점
   - 특히 큰 K 차원에서 dequant 시간이 지배적

3. **업계가 W4A4를 안 쓰는 진짜 이유는 "속도"가 아니라 "품질"**
   - INT4 activation 양자화의 16-level 표현 한계
   - 하지만 embedding, 간단한 generation 등에서는 충분할 수 있음

4. **실용적 권장사항**:
   - 속도 + 품질 균형 → **FP8 (W8A8)**
   - 메모리 제약 → **W4A16** (Marlin/GPTQ 등 기존 구현 사용)
   - 최대 속도 + 품질 허용 → **W4A4-INT4TC** (이 구현 사용)

---

## References

- vLLM 소스코드: `csrc/quantization/marlin/marlin_mma.h` (MMA 인스트럭션 정의)
- vLLM 소스코드: `csrc/quantization/marlin/dequant.h` (INT4 dequant 구현)
- vLLM 소스코드: `csrc/quantization/w8a8/cutlass/c3x/` (CUTLASS FP8 커널)
- vLLM 소스코드: `vllm/model_executor/layers/quantization/kernels/scaled_mm/` (커널 선택 로직)
- TensorRT-LLM 소스코드: `cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/` (INT4 GEMM)
- TensorRT-LLM 소스코드: `cpp/tensorrt_llm/cutlass_extensions/.../mma_tensorop_dequantizer.h` (INT4 dequant)
- [NVIDIA PTX ISA - Matrix Multiply-Accumulate Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS 3.x Mixed Input GEMM](https://github.com/NVIDIA/cutlass)
- [NVIDIA L4 Datasheet](https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/l4-gpu-datasheet)
- [NVIDIA A100 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)
- [NVIDIA H100 Datasheet](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)
- [LLM Inference Roofline Analysis](https://arxiv.org/html/2402.16363v4)
