# Embedding Model Quantization Benchmark Report (BF16 / FP8 / NVFP4)

**Date**: 2026-03-03
**GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (GB202, 96 GB)
**Engine**: vLLM 0.16.0 (CUDA graphs enabled, chunked prefill)
**Quantization**: llmcompressor 0.9.0.2 (compressed-tensors format)

## Target Models

| Model | Parameters | Type | Embedding Dim | Max Context |
|-------|-----------|------|---------------|-------------|
| Qwen3-VL-Embedding-2B | 2B | Vision-Language Embedding | 2048 | 32,768 |
| Qwen3-VL-Embedding-8B | 8B | Vision-Language Embedding | 4096 | 32,768 |
| Qwen3-Embedding-0.6B | 0.6B | Text Embedding | 1024 | 32,768 |
| Qwen3-Embedding-4B | 4B | Text Embedding | 2560 | 32,768 |
| Qwen3-Embedding-8B | 8B | Text Embedding | 4096 | 32,768 |
| BAAI/bge-m3 | 0.6B | Text Embedding (XLM-RoBERTa) | 1024 | 8,192 |

## Quantization Methods

| Method | Scheme | Calibration | Format | vLLM Flag |
|--------|--------|-------------|--------|-----------|
| BF16 | — (baseline) | — | HuggingFace | — |
| FP8 | FP8_DYNAMIC (weight-only, dynamic activation) | Not required | compressed-tensors | `quantization="compressed-tensors"` |
| NVFP4 | W4A4 (weight + activation FP4) | ultrachat_200k, 128 samples, 4096 seq_len | compressed-tensors | `quantization="compressed-tensors"` |

---

## 1. Embedding Accuracy (BF16 vs FP8 vs NVFP4)

BF16을 ground truth로 FP8/NVFP4의 임베딩 품질 오차를 측정.
- **데이터**: War and Peace에서 추출한 100 query-doc 쌍 (128, 256, 512, 1024 토큰 혼합)
- **측정 항목**: Embedding Cosine Similarity, MAE, Query-Doc Similarity 차이

### 1.1 Embedding Cosine Similarity (BF16 emb vs Quantized emb)

값이 1.0에 가까울수록 BF16과 동일한 임베딩을 의미.

| Model | FP8 Mean | FP8 Min | NVFP4 Mean | NVFP4 Min |
|-------|----------|---------|------------|-----------|
| VL-2B | 0.9948 | 0.9910 | 0.9359 | 0.9090 |
| VL-8B | 0.9928 | 0.9869 | 0.9515 | 0.9319 |
| Emb-0.6B | 0.9954 | 0.9893 | 0.9030 | 0.8015 |
| Emb-4B | 0.9959 | 0.9929 | 0.9442 | 0.9222 |
| Emb-8B | **0.9976** | 0.9960 | **0.9671** | 0.9478 |
| bge-m3 | 0.9963 | 0.9934 | **0.5415** | **0.3224** |

### 1.2 Mean Absolute Error (MAE)

| Model | FP8 MAE | FP8 Max AE | NVFP4 MAE | NVFP4 Max AE |
|-------|---------|------------|-----------|--------------|
| VL-2B | 0.00176 | 0.03200 | 0.00618 | 0.11891 |
| VL-8B | 0.00145 | 0.04066 | 0.00381 | 0.06649 |
| Emb-0.6B | 0.00234 | 0.02308 | 0.01076 | 0.09215 |
| Emb-4B | 0.00141 | 0.01348 | 0.00520 | 0.06196 |
| Emb-8B | **0.00083** | 0.02108 | **0.00311** | 0.06239 |
| bge-m3 | 0.00214 | 0.01440 | 0.02236 | 0.47945 |

### 1.3 Query-Document Cosine Similarity Difference

BF16의 query-doc 유사도와 양자화 모델의 query-doc 유사도 차이 (절댓값).
값이 0에 가까울수록 검색 랭킹에 미치는 영향이 작음.

| Model | FP8 Mean | FP8 Max | FP8 P99 | NVFP4 Mean | NVFP4 Max | NVFP4 P99 |
|-------|----------|---------|---------|------------|-----------|-----------|
| VL-2B | 0.0057 | 0.0260 | 0.0195 | 0.0213 | 0.0661 | 0.0614 |
| VL-8B | 0.0079 | 0.0318 | 0.0267 | 0.0217 | 0.0522 | 0.0441 |
| Emb-0.6B | 0.0074 | 0.0341 | 0.0276 | 0.0528 | 0.1492 | 0.1349 |
| Emb-4B | 0.0056 | 0.0226 | 0.0164 | 0.0267 | 0.0750 | 0.0732 |
| Emb-8B | **0.0044** | 0.0219 | 0.0183 | **0.0178** | 0.0551 | 0.0533 |
| bge-m3 | **0.0036** | 0.0135 | 0.0110 | 0.1012 | 0.3079 | 0.2726 |

### 1.4 Accuracy 판정

| Model | FP8 | NVFP4 |
|-------|-----|-------|
| VL-2B | ✅ 프로덕션 사용 가능 | ⚠️ 검토 필요 (CosSim 0.94) |
| VL-8B | ✅ 프로덕션 사용 가능 | ⚠️ 검토 필요 (CosSim 0.95) |
| Emb-0.6B | ✅ 프로덕션 사용 가능 | ⚠️ 주의 필요 (CosSim 0.90, min 0.80) |
| Emb-4B | ✅ 프로덕션 사용 가능 | ⚠️ 검토 필요 (CosSim 0.94) |
| Emb-8B | ✅ 프로덕션 사용 가능 | ✅ 허용 범위 (CosSim 0.97) |
| bge-m3 | ✅ 프로덕션 사용 가능 | ❌ **사용 불가** (CosSim 0.54) |

> **FP8**: 전 모델 CosSim ≥ 0.992, QD Diff ≤ 0.008 → 실질적 무손실
>
> **NVFP4 bge-m3**: XLM-RoBERTa encoder 구조와 W4A4 양자화 비호환으로 임베딩 품질 붕괴

---

## 2. Latency (P50, ms)

모든 값은 BF16 / FP8 / NVFP4 순서로 표기.

### 2.1 Batch = 1

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-2B | 5.0 / 5.8 / 5.1 | 10.7 / 9.0 / 13.1 | 19.3 / 14.5 / 15.1 | 36.6 / 27.3 / 23.5 |
| VL-8B | 16.1 / 11.9 / 10.7 | 37.2 / 21.2 / 18.1 | 57.6 / 38.6 / 27.8 | 113.3 / 74.4 / 58.0 |
| Emb-0.6B | 3.6 / 3.8 / 3.5 | 7.6 / 6.8 / 13.3 | 11.7 / 10.5 / 15.2 | 21.9 / 19.1 / 20.8 |
| Emb-4B | 9.9 / 9.2 / 8.0 | 22.4 / 15.5 / 18.7 | 37.9 / 26.2 / 20.6 | 68.4 / 55.4 / 42.8 |
| Emb-8B | 15.8 / 12.3 / 9.9 | 37.2 / 21.1 / 17.9 | 58.1 / 38.4 / 27.6 | 114.3 / 72.7 / 56.8 |
| bge-m3 | 2.5 / 3.2 / 2.9 | 6.0 / 5.5 / 10.6 | 9.8 / 8.7 / 12.9 | 18.8 / 16.4 / 16.6 |

### 2.2 Batch = 16

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-2B | 30.9 / 21.9 / 27.3 | 99.3 / 67.3 / 55.3 | 190.2 / 128.2 / 104.7 | 403.7 / 286.7 / 236.4 |
| VL-8B | 115.6 / 65.2 / 48.3 | 361.5 / 221.5 / 161.2 | 725.6 / 459.3 / 327.5 | 2,900 / 1,846 / 1,375 |
| Emb-0.6B | 15.8 / 13.3 / 18.1 | 64.6 / 58.1 / 55.2 | 126.3 / 111.1 / 117.9 | 278.2 / 260.9 / 268.0 |
| Emb-4B | 62.1 / 42.7 / 31.1 | 233.1 / 166.4 / 130.5 | 475.5 / 334.4 / 276.6 | 1,814 / 1,282 / 1,059 |
| Emb-8B | 105.3 / 65.5 / 46.3 | 369.7 / 235.5 / 175.9 | 757.4 / 490.9 / 378.6 | 2,981 / 1,917 / 1,451 |
| bge-m3 | 11.3 / 9.9 / 13.9 | 57.8 / 50.0 / 48.6 | 122.9 / 99.0 / 90.8 | 276.3 / 232.3 / 202.0 |

---

## 3. Throughput (tok/s)

### 3.1 Batch = 1

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-2B | 25,256 / 21,840 / 24,757 | 95,476 / 113,427 / 77,729 | 106,101 / 137,625 / 134,943 | 111,831 / 148,133 / 170,635 |
| VL-8B | 7,969 / 10,539 / 11,925 | 27,547 / 48,207 / 55,023 | 35,528 / 52,909 / 73,432 | 36,097 / 55,078 / 70,495 |
| Emb-0.6B | 35,068 / 32,263 / 36,478 | 134,023 / 144,819 / 77,258 | 175,336 / 192,403 / 132,145 | 183,927 / 213,786 / 196,244 |
| Emb-4B | 12,314 / 13,692 / 15,926 | 45,645 / 65,093 / 54,492 | 53,979 / 76,377 / 95,609 | 59,861 / 73,972 / 93,750 |
| Emb-8B | 8,100 / 10,418 / 12,941 | 27,507 / 47,870 / 56,904 | 34,692 / 52,966 / 73,616 | 35,458 / 55,868 / 71,741 |
| bge-m3 | 50,706 / 39,248 / 44,525 | 169,807 / 183,085 / 90,289 | 207,443 / 233,108 / 157,178 | 217,388 / 251,231 / 247,466 |

### 3.2 Batch = 16

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-2B | 81,535 / 104,378 / 90,163 | 210,385 / 279,996 / 335,447 | 222,481 / 315,504 / 367,745 | 159,861 / 231,987 / 273,903 |
| VL-8B | 23,068 / 35,648 / 49,445 | 64,374 / 99,692 / 134,836 | 65,303 / 99,937 / 138,171 | 33,984 / 52,554 / 69,913 |
| Emb-0.6B | 144,139 / 159,276 / 125,609 | 300,631 / 324,472 / 331,385 | 324,950 / 345,921 / 315,809 | 265,873 / 281,649 / 278,468 |
| Emb-4B | 42,498 / 55,996 / 75,254 | 97,949 / 130,494 / 161,453 | 97,630 / 133,383 / 159,307 | 53,680 / 73,753 / 89,817 |
| Emb-8B | 26,002 / 38,070 / 53,950 | 62,175 / 95,432 / 124,218 | 62,626 / 94,630 / 120,413 | 33,059 / 50,716 / 66,244 |
| bge-m3 | 182,269 / 207,100 / 145,681 | 281,525 / 325,604 / 343,292 | 266,593 / 331,929 / 363,253 | 236,638 / 283,925 / 323,823 |

---

## 4. Latency Reduction (%) vs BF16

음수는 레이턴시 증가(성능 악화)를 의미.

### 4.1 Batch = 1

| Model | FP8 128t | FP8 4096t | NVFP4 128t | NVFP4 4096t |
|-------|----------|-----------|------------|-------------|
| VL-2B | -16% | **-25%** | -2% | **-36%** |
| VL-8B | **-26%** | **-34%** | **-34%** | **-49%** |
| Emb-0.6B | +6% | -13% | +3% | -5% |
| Emb-4B | -7% | -19% | -19% | **-37%** |
| Emb-8B | **-22%** | **-36%** | **-37%** | **-50%** |
| bge-m3 | +28% | -13% | +16% | -12% |

### 4.2 Batch = 16

| Model | FP8 128t | FP8 4096t | NVFP4 128t | NVFP4 4096t |
|-------|----------|-----------|------------|-------------|
| VL-2B | **-29%** | **-29%** | -12% | **-41%** |
| VL-8B | **-44%** | **-36%** | **-58%** | **-53%** |
| Emb-0.6B | -16% | -6% | +15% | -4% |
| Emb-4B | **-31%** | **-29%** | **-50%** | **-42%** |
| Emb-8B | **-38%** | **-36%** | **-56%** | **-51%** |
| bge-m3 | -12% | -16% | +23% | **-27%** |

---

## 5. Key Findings

### 5.1 FP8 — 전 모델에서 안전하게 권장

- **정합성**: 전 모델 Embedding CosSim ≥ 0.992 (실질적 무손실)
- **레이턴시**: 8B 모델 기준 batch=16에서 34~44% 감소
- **소형 모델**: 레이턴시 개선폭 작지만 정합성 완벽하므로 기본 적용 권장

### 5.2 NVFP4 — 대형 모델에서 높은 효과, 소형/encoder 모델 주의

- **8B 모델 (VL-8B, Emb-8B)**: batch=16, 4096tok 기준 50~53% 레이턴시 감소. CosSim 0.95~0.97
- **4B 모델**: 37~42% 레이턴시 감소. CosSim 0.94
- **소형 모델 (0.6B, bge-m3)**: 짧은 시퀀스에서 dequantization 오버헤드로 인해 레이턴시 **증가**
- **bge-m3 NVFP4**: CosSim 0.54 → **사용 불가** (XLM-RoBERTa encoder와 W4A4 비호환)

### 5.3 모델 사이즈별 NVFP4 효과

Blackwell FP4 Tensor Core의 이점은 모델이 클수록 (compute-bound일수록) 크게 나타남.
소형 모델은 memory-bound이므로 양자화의 compute 절감 효과가 dequantization 오버헤드에 상쇄됨.

---

## 6. Production Recommendations

| Model | 권장 정밀도 | 근거 |
|-------|------------|------|
| VL-8B | **NVFP4** | 53% 레이턴시↓, CosSim 0.95 (대규모 배치에서 극적 개선) |
| Emb-8B | **NVFP4** | 51% 레이턴시↓, CosSim 0.97 (가장 높은 NVFP4 정합성) |
| Emb-4B | FP8 또는 NVFP4 | 정확도 민감 → FP8, 속도 우선 → NVFP4 (42% 레이턴시↓) |
| VL-2B | **FP8** | NVFP4 정합성 0.94, 속도 이점도 긴 시퀀스에서만 유의미 |
| Emb-0.6B | **FP8** | NVFP4 정합성 0.90 (min 0.80), 속도 이점 미미 |
| bge-m3 | **FP8** | NVFP4 사용 불가 (정합성 붕괴) |

---

## 7. Quantized Model Artifacts

### HuggingFace Repositories

| Model | FP8 | NVFP4 |
|-------|-----|-------|
| VL-2B | [Forturne/Qwen3-VL-Embedding-2B-FP8](https://huggingface.co/Forturne/Qwen3-VL-Embedding-2B-FP8) | [Forturne/Qwen3-VL-Embedding-2B-NVFP4](https://huggingface.co/Forturne/Qwen3-VL-Embedding-2B-NVFP4) |
| VL-8B | [Forturne/Qwen3-VL-Embedding-8B-FP8](https://huggingface.co/Forturne/Qwen3-VL-Embedding-8B-FP8) | [Forturne/Qwen3-VL-Embedding-8B-NVFP4](https://huggingface.co/Forturne/Qwen3-VL-Embedding-8B-NVFP4) |
| Emb-0.6B | [Forturne/Qwen3-Embedding-0.6B-FP8](https://huggingface.co/Forturne/Qwen3-Embedding-0.6B-FP8) | [Forturne/Qwen3-Embedding-0.6B-NVFP4](https://huggingface.co/Forturne/Qwen3-Embedding-0.6B-NVFP4) |
| Emb-4B | [Forturne/Qwen3-Embedding-4B-FP8](https://huggingface.co/Forturne/Qwen3-Embedding-4B-FP8) | [Forturne/Qwen3-Embedding-4B-NVFP4](https://huggingface.co/Forturne/Qwen3-Embedding-4B-NVFP4) |
| Emb-8B | [Forturne/Qwen3-Embedding-8B-FP8](https://huggingface.co/Forturne/Qwen3-Embedding-8B-FP8) | [Forturne/Qwen3-Embedding-8B-NVFP4](https://huggingface.co/Forturne/Qwen3-Embedding-8B-NVFP4) |
| bge-m3 | [Forturne/bge-m3-FP8](https://huggingface.co/Forturne/bge-m3-FP8) | [Forturne/bge-m3-NVFP4](https://huggingface.co/Forturne/bge-m3-NVFP4) |

### Model Sizes (Disk)

| Model | BF16 (HF) | FP8 | NVFP4 |
|-------|-----------|-----|-------|
| VL-2B | ~4.5 GB | 2.3 GB | 2.2 GB |
| VL-8B | ~17 GB | 9.4 GB | 6.3 GB |
| Emb-0.6B | ~1.2 GB | 733 MB | 845 MB |
| Emb-4B | ~8 GB | 4.2 GB | 3.4 GB |
| Emb-8B | ~17 GB | 8.9 GB | 6.0 GB |
| bge-m3 | ~2.3 GB | 1.3 GB | 1.2 GB |

---

## Benchmark Methodology

- **Latency benchmark**: vLLM `LLM(runner="pooling", convert="embed")`, `gpu_memory_utilization=0.90`
- **Exact token count**: tokenizer encode → slice → decode → re-encode verification
- **Prefix cache invalidation**: 배치 내 각 항목마다 다른 텍스트 (staggered offset slicing)
- **CUDA graphs**: enabled (`enforce_eager=False`)
- **Warmup**: 3 runs (discarded), **Timed runs**: 10 per configuration
- **Batch sizes**: 1, 4, 8, 16 / **Input lengths**: 128, 256, 512, 1024, 2048, 4096, 8192
- **Accuracy benchmark**: 100 query-doc pairs from War and Peace, token lengths 128/256/512/1024 mixed
- **Source text**: War and Peace (Project Gutenberg), ~340K tokens

## Raw Data Files

| File | Description |
|------|-------------|
| `embedding_latency_results.json` | BF16 latency raw data |
| `embedding_latency_results_fp8.json` | FP8 latency raw data |
| `embedding_latency_results_nvfp4.json` | NVFP4 latency raw data |
| `embedding_accuracy_results.json` | BF16 vs FP8 vs NVFP4 accuracy metrics |
