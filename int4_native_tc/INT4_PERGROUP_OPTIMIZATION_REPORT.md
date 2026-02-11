# INT4 Per-Group GEMM Kernel Optimization Report

## Target: NVIDIA L4 GPU (SM89, Ada Lovelace)

**Model**: Qwen3-4B (K=2560, qkv_proj N=6144, gate_up_proj N=19456)
**Quantization**: W4A4 per-group (group_size=128, num_groups=20)
**Batch**: M=500 (prefill)
**Benchmark**: 36 layers x 2 projections = 72 quant+GEMM ops

---

## 1. Problem Statement

Per-group INT4 quantization(W4A4)은 per-channel 대비 정확도가 뛰어나지만, 성능이 크게 저하됨.

**근본 원인: Pipeline Starvation**

```
INT4 x INT4 MMA → INT32 accumulator
Per-group scale은 FP32 → INT32에 직접 곱할 수 없음
→ 그룹마다 INT32→FP32 변환 + scale 곱하기 + accumulator 리셋 필요
```

- `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` (MMA_K=64)
- group_size=128 → 그룹당 MMA **2회**만 수행 후 flush
- K=2560 → 20개 그룹 × 2 MMA = 40 MMA (per-channel은 40회 연속)
- 그룹마다 `__syncthreads()` + INT32→FP32 변환 + scale 적용 + FP32 합산

---

## 2. Kernel Versions Summary

### V1 (Original Baseline)
- **File**: `int4_gemm.cu:1266`
- **Tile**: 32x32, 128 threads, single-buffered
- **Loads**: Standard SMEM loads (no cp.async)
- **용도**: M<=32 fallback으로 사용

### V2 (Constexpr Unrolling)
- **File**: `int4_gemm.cu:1653`
- **Tile**: 64x128, 128 threads, single-buffered
- **개선**: `constexpr GS=128` 템플릿으로 K-loop 완전 unroll
- **Loads**: Standard loads
- **결과**: V1 대비 큰 폭 개선 (타일 확대 효과)

### V3 (First cp.async)
- **File**: `int4_gemm.cu:1906`
- **Tile**: 64x128, 128 threads, 2-stage double buffer
- **개선**: `cp.async 4-byte` GMEM→SMEM 비동기 복사 도입
- **문제**: Runtime `gs` 사용 → loop unroll 실패
- **결과**: V2보다 느림 (unroll 실패 때문)

### V4 (Production Baseline) ★
- **File**: `int4_gemm.cu:2192`
- **Tile**: 64x128, 128 threads, 2-stage double buffer
- **개선**: V2의 constexpr unroll + V3의 cp.async 결합
- **Loads**: `cp.async.ca.shared.global [dst], [src], 4` — 4바이트 × 24회/스레드/그룹
- **Stride**: A_STRIDE=68, B_STRIDE=68 (gs/2+4, bank-conflict-free)
- **SMEM**: 2 × ~14.5KB = ~29KB → SM당 2블록 가능
- **결과**: **41.28ms** (vs PerCh 15.79ms = 2.62x)

### V5 (Multi-Group K-Loop)
- **File**: `int4_gemm.cu:2496`
- **Tile**: 64x128, 128 threads, 2-stage double buffer
- **아이디어**: 여러 그룹을 하나의 SMEM 로드로 처리하여 MMA 횟수 증가
- **Loads**: cp.async 4-byte
- **GROUPS_PER_LOAD**: 2 (4 MMA iterations per load)
- **문제**: 그룹 경계에서 INT32 flush는 여전히 필요 → SMEM 증가만 발생
- **결과**: **53.51ms** (V4 대비 +29%, 더 느림)

### V7 (Large Tile + 3-Stage)
- **File**: `int4_gemm.cu:3430`
- **Tile**: 128x128, 256 threads (8 warps, 4M×2N layout), 3-stage triple buffer
- **아이디어**: 타일 확대 + 3단계 파이프라인 + load-after-compute (sync 1회/그룹)
- **Loads**: cp.async 4-byte
- **SMEM**: 3 × 18432 = 55296 bytes (54KB)
- **문제**: 256 threads/block → SM당 1블록만 실행 (V4는 2블록). 같은 스레드 수지만 warp 스케줄링 다양성 감소
- **결과**: **43.51ms** (V4 대비 +5%, 더 느림)

### V8 (16-Byte Vectorized Loads) ★★ BEST
- **File**: `int4_gemm.cu:2840`
- **Tile**: 64x128, 128 threads, 2-stage double buffer
- **핵심 개선**: `cp.async.ca.shared.global [dst], [src], 16` — 16바이트 × 6회/스레드/그룹
- **Stride**: A_STRIDE=80, B_STRIDE=80 (16바이트 정렬 + bank-conflict-free: 20%32≠0)
- **SMEM**: 2 × 16128 = 32256 bytes → SM당 2블록 유지
- **로드 명령어 4x 감소**: 24회 → 6회 (진짜 병목이 여기였음)
- **결과**: **31.13ms** (V4 대비 -24%, PerCh 대비 1.97x)

### V9 (16B Loads + 3-Stage Pipeline)
- **File**: `int4_gemm.cu:3132`
- **Tile**: 64x128, 128 threads, 3-stage triple buffer
- **아이디어**: V8의 16B 로드 + 3단계 파이프라인 + load-after-compute
- **Loads**: cp.async 16-byte
- **SMEM**: 3 × 16128 = 48384 bytes (47.25KB)
- **문제**: 3-stage SMEM(48KB/블록) → 2블록 시 96KB → L1 캐시 32KB만 남음 (V8은 64KB)
- **SMEM carveout 최적화**: `cudaFuncAttributePreferredSharedMemoryCarveout` 적용했으나 효과 미미
- **결과**: **34.89ms** (V4 대비 -16%, V8보다 12% 느림)

### Dequant+BF16 (W4A16 스타일)
- **File**: `int4_gemm.cu:4395`
- **아이디어**: INT4 activation을 BF16으로 dequant 후, pre-dequanted BF16 weight와 cuBLAS BF16 GEMM
- **문제**: BF16 GEMM 자체가 느림 (L4에서 메모리 대역폭 부족) + dequant 오버헤드
- **결과**: **61.56ms** (V4 대비 +49%)

### Progressive INT8 (QServe-style)
- **File**: `int4_gemm.cu:4479`
- **아이디어**: INT4를 INT8로 unpack 후 INT8 MMA(`mma.m16n8k32.s32.s8.s8.s32`) 사용
- **문제**: INT8 MMA 처리량이 INT4 MMA의 절반 + unpack 오버헤드
- **결과**: **~87ms** (V4 대비 +111%)

---

## 3. Final Benchmark Results

### Full Model (72 ops, M=500, quant+GEMM 포함)

```
#  Config                     Mean(ms)    vs BF16     vs FP8   vs PerCh
───────────────────────────────────────────────────────────────────────
1  INT4 PerCh (quant+GEMM)      15.83      0.27x      0.44x      1.00x
2  INT4 PG V8 (16B async) ★     31.37      0.53x      0.88x      1.98x
3  FP8 (GEMM only)              34.79      0.59x      0.98x      2.20x
4  FP8 (quant+GEMM)             35.66      0.61x      1.00x      2.25x
5  INT4 PG V9 (16B+3stg)        34.89       —          —         2.21x
6  INT4 PG V4 (4B async)        41.53      0.71x      1.16x      2.62x
7  INT4 PG V7 (128x128,3stg)    43.51       —          —         2.75x
8  INT4 PG V5 (multi-group)     53.51       —          —         3.39x
9  BF16 (torch.mm)              58.80      1.00x      1.65x      3.72x
10 Dequant+BF16                 61.56       —          —         3.90x
11 Progressive INT8             ~87         —          —         ~5.5x
```

**주목할 점: V8 (31.37ms) > FP8 (35.66ms)**

INT4 per-group V8이 FP8보다 12% 빠름. 이유:
- INT4: 4x 메모리 압축 → 메모리 대역폭 절약
- FP8: 2x 압축에 불과
- L4 (300GB/s)는 메모리 대역폭 제한 GPU → 압축률이 성능에 직결
- V8의 accumulator flush 오버헤드를 INT4의 2x 추가 압축이 상쇄

```
Weight 메모리 (K=2560, N=6144 기준):
  BF16:  30.0 MB
  FP8:   15.0 MB  (2x 압축)
  INT4:   7.5 MB  (4x 압축)
```

### Single-Layer GEMM-Only (M=500)

```
Kernel      qkv_proj (us)    gate_up_proj (us)    vs V4
─────────────────────────────────────────────────────
PerCh            66               243-249           —
V8              151               484             0.72x
V9              188               570             0.85x
V4              197               669             1.00x
V7              241               686             1.05x
```

---

## 4. Key Insights

### 왜 V8만 성공했는가

| 시도 | 가설 | 실제 | 결과 |
|------|------|------|------|
| V5 | 그룹 합쳐서 MMA 늘리면 파이프라인 충전 | INT32 flush는 여전히 그룹마다 필요 | +29% 느림 |
| V7 | 큰 타일 + 3-stage로 latency hiding | SM당 1블록 → occupancy 감소 | +5% 느림 |
| V9 | V8 + 3-stage 프리페치 | SMEM 증가 → L1 캐시 감소 | V8보다 12% 느림 |
| Dequant | BF16 GEMM으로 flush 회피 | BF16 GEMM 자체 성능 부족 | +49% 느림 |
| Progressive | INT8 MMA로 처리량 확보 | INT8 MMA = INT4의 절반 처리량 | +111% 느림 |
| **V8** | **로드 명령어 수 감소** | **진짜 병목 = 로드 명령어 과다** | **-24% 빠름** |

**핵심 발견**: 병목은 파이프라인 깊이, 타일 크기, MMA 반복 횟수가 아니라 **cp.async 로드 명령어 수**였음.

```
V4: cp.async 4byte × 24회/스레드/그룹 → 명령어 스케줄러 과부하
V8: cp.async 16byte × 6회/스레드/그룹 → 스케줄러에 여유, MMA와 로드 오버랩 개선
```

### 남은 1.97x 갭의 원인 (하드웨어 제약)

Per-channel: K=2560 전체를 INT32로 연속 누적 (40회 MMA 논스톱)
Per-group: 그룹당 2회 MMA 후 INT32→FP32 flush + scale + reset × 20그룹

이 차이는 INT4×INT4→INT32 MMA의 ISA 제약으로, 커널 최적화로는 제거 불가.

---

## 5. Industry Context

### W4A4 INT4는 production에서 사용되지 않음

| 방식 | 실제 연산 | per-group 처리 | production 여부 |
|------|----------|---------------|----------------|
| **W4A16** (GPTQ/AWQ/Marlin) | INT4→FP16 dequant + FP16 MMA | scale을 weight에 미리 곱함 → flush 없음 | **production 표준** |
| **W4A8** (QServe/QQQ) | INT4→INT8 dequant + INT8 MMA | INT8 MMA → flush 횟수 감소 | production 가능 |
| **W8A8** (FP8/SmoothQuant) | FP8 또는 INT8 MMA | per-tensor scale | **production 표준** |
| **W4A4 INT4** (우리 커널) | INT4×INT4 MMA → INT32 flush | 그룹마다 flush 필수 | **연구용** |
| **NVFP4** (Blackwell) | FP4 MMA → FP32 (HW g16 scale) | 하드웨어가 처리 | **차세대 production** |

**주요 논문**: ATOM (MLSys 2024), QuaRot (NeurIPS 2024), SpinQuant (ICLR 2025), COMET (ASPLOS 2025)
모두 W4A4 INT4의 accumulator flush 오버헤드를 보고했으며, production 배포에 성공한 사례 없음.

### NVIDIA Hopper에서 INT4 텐서코어 폐기

- **Ampere (A100)**: INT4 MMA → 전용 IMMA 텐서코어 명령어
- **Hopper (H100)**: INT4 MMA → CUDA core IMAD로 fallback (텐서코어 미사용!)
- **Blackwell**: NVFP4 (FP4) 네이티브 텐서코어 지원 (INT4 아님)

---

## 6. File Structure

```
int4_native_tc/
├── csrc/
│   ├── int4_gemm.cu          # V1~V9 커널 + Dequant + Progressive 전체 구현
│   ├── int4_quant.cu          # quantization/dequantization 유틸리티
│   └── bindings.cpp           # pybind11 바인딩 (V1~V9 등록)
├── int4_native_tc/
│   └── __init__.py            # Python import/export
├── profile_all_kernels.py     # 전체 커널 벤치마크
├── profile_bottleneck.py      # 단일 레이어 roofline 분석
├── test_v5_correctness.py     # 정확도 검증 스크립트
└── setup.py                   # 빌드 설정
```

---

## 7. Conclusion

V4 대비 **24% 성능 개선** (41.28ms → 31.13ms) 달성. 핵심은 `cp.async` 로드 벡터화 (4B→16B).

Per-group 오버헤드를 2.62x → 1.97x로 축소했으나, 나머지 갭은 INT4 MMA의 하드웨어 제약 (그룹마다 INT32 accumulator flush)이므로 커널 최적화로는 해결 불가.

실무 적용 시에는 W4A16 (Marlin) 또는 W4A8 (QServe) 방식이 현 하드웨어에서 더 실용적이며, 차세대 Blackwell의 NVFP4가 W4A4 문제를 하드웨어 수준에서 해결함.
