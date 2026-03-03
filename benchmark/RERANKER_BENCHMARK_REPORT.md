# Reranker Model Quantization Benchmark Report (BF16 / FP8 / NVFP4)

**Date**: 2026-03-03
**GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (GB202, 96 GB)
**Engine**: vLLM 0.16.0 (CUDA graphs enabled, chunked prefill)
**Quantization**: llmcompressor 0.9.0.2 (compressed-tensors format)

## Target Models

| Model | Parameters | Type | Native Arch | vLLM Override | Max Context |
|-------|-----------|------|-------------|---------------|-------------|
| Qwen3-VL-Reranker-2B | 2B | Vision-Language Reranker | Qwen3VLForConditionalGeneration | Qwen3VLForSequenceClassification | 32,768 |
| Qwen3-VL-Reranker-8B | 8B | Vision-Language Reranker | 동일 | 동일 | 32,768 |
| Qwen3-Reranker-0.6B | 0.6B | Text Reranker | Qwen3ForCausalLM | Qwen3ForSequenceClassification | 32,768 |
| Qwen3-Reranker-4B | 4B | Text Reranker | 동일 | 동일 | 32,768 |
| Qwen3-Reranker-8B | 8B | Text Reranker | 동일 | 동일 | 32,768 |
| bge-reranker-v2-m3 | 0.6B | Text Reranker (XLM-RoBERTa) | XLMRobertaForSequenceClassification | — (native) | 8,192 |

## Quantization Methods

| Method | Scheme | Calibration | Format | vLLM Flag |
|--------|--------|-------------|--------|-----------|
| BF16 | — (baseline) | — | HuggingFace | — |
| FP8 | FP8_DYNAMIC (weight-only, dynamic activation) | Not required | compressed-tensors | `quantization="compressed-tensors"` |
| NVFP4 | W4A4 (weight + activation FP4) | ultrachat_200k, 128 samples, 4096 seq_len | compressed-tensors | `quantization="compressed-tensors"` |

---

## 0. Critical: vLLM 0.16.0 Qwen3 Reranker Quantization 비호환

> **Qwen3 계열 리랭커 5개 모델은 vLLM 0.16.0에서 FP8/NVFP4 양자화 추론이 불가능합니다.**

### 근본 원인

Qwen3 리랭커는 CausalLM 아키텍처를 SequenceClassification으로 변환하는 `from_2_way_softmax` 방식을 사용합니다:

```
score.weight = lm_head.weight[true_token_id] - lm_head.weight[false_token_id]
```

이 과정에서 두 가지 vLLM 버그가 발생합니다:

| 문제 | 양자화 | 원인 | 증상 |
|------|--------|------|------|
| Bug 1 | FP8 | `tie_word_embeddings=True` → checkpoint에 별도 `lm_head.weight` 미존재 → `from_2_way_softmax`가 새 `ParallelLMHead` 생성하나 weight 미로드 | score = 0.0 (전부) |
| Bug 2 | NVFP4 | `from_2_way_softmax`가 `score_layer.weight` 접근 → NVFP4 양자화된 `ReplicatedLinear`는 `.weight` 속성 없음 | `AttributeError` crash |

### 영향 범위

- **영향 모델**: Qwen3-VL-Reranker-2B/8B, Qwen3-Reranker-0.6B/4B/8B (5개)
- **영향 없는 모델**: bge-reranker-v2-m3 (native SequenceClassification, `from_2_way_softmax` 미사용)
- **BF16 추론은 모든 모델에서 정상 동작**

### 레이턴시 벤치마크 유효성

- FP8 레이턴시 데이터는 모델 로딩/연산 자체는 정상이므로 **latency 측정값은 유효** (score 출력만 0)
- NVFP4 레이턴시는 Qwen3 모델에서 **측정 불가** (crash)

---

## 1. Reranker Accuracy (BF16 vs FP8 vs NVFP4)

BF16을 ground truth로 FP8/NVFP4의 리랭커 score 오차를 측정.
- **데이터**: War and Peace에서 추출한 100 query-doc 쌍 (128, 256, 512, 1024 토큰 혼합)
- **측정 항목**: Score MAE, Max Diff, Spearman Rank Correlation, Top-K Overlap
- **Qwen3 모델**: FP8/NVFP4 비호환으로 측정 불가 (위 Section 0 참조)

### 1.1 Score MAE / Max Diff

| Model | FP8 MAE | FP8 Max Diff | NVFP4 MAE | NVFP4 Max Diff |
|-------|---------|--------------|-----------|----------------|
| VL-Reranker-2B | N/A (Bug 1) | N/A | N/A (Bug 2) | N/A |
| VL-Reranker-8B | N/A (Bug 1) | N/A | N/A (Bug 2) | N/A |
| Reranker-0.6B | N/A (Bug 1) | N/A | N/A (Bug 2) | N/A |
| Reranker-4B | N/A (Bug 1) | N/A | N/A (Bug 2) | N/A |
| Reranker-8B | N/A (Bug 1) | N/A | N/A (Bug 2) | N/A |
| bge-reranker-v2-m3 | 0.003613 | 0.067467 | 0.062492 | 0.947552 |

### 1.2 Spearman Rank Correlation

값이 1.0에 가까울수록 BF16과 동일한 랭킹을 의미.

| Model | FP8 Spearman | NVFP4 Spearman |
|-------|-------------|----------------|
| VL-Reranker-2B | N/A | N/A |
| VL-Reranker-8B | N/A | N/A |
| Reranker-0.6B | N/A | N/A |
| Reranker-4B | N/A | N/A |
| Reranker-8B | N/A | N/A |
| bge-reranker-v2-m3 | **0.998** | 0.569 |

### 1.3 Top-K Rank Overlap

BF16 기준 상위 K개 문서와 양자화 모델의 상위 K개 문서의 일치율.

| Model | FP8 Top-10 | FP8 Top-20 | NVFP4 Top-10 | NVFP4 Top-20 |
|-------|-----------|-----------|-------------|-------------|
| VL-Reranker-2B | N/A | N/A | N/A | N/A |
| VL-Reranker-8B | N/A | N/A | N/A | N/A |
| Reranker-0.6B | N/A | N/A | N/A | N/A |
| Reranker-4B | N/A | N/A | N/A | N/A |
| Reranker-8B | N/A | N/A | N/A | N/A |
| bge-reranker-v2-m3 | **1.00** | **0.95** | 0.50 | 0.40 |

### 1.4 Accuracy 판정

| Model | FP8 | NVFP4 |
|-------|-----|-------|
| VL-Reranker-2B | N/A (vLLM bug) | N/A (vLLM bug) |
| VL-Reranker-8B | N/A (vLLM bug) | N/A (vLLM bug) |
| Reranker-0.6B | N/A (vLLM bug) | N/A (vLLM bug) |
| Reranker-4B | N/A (vLLM bug) | N/A (vLLM bug) |
| Reranker-8B | N/A (vLLM bug) | N/A (vLLM bug) |
| bge-reranker-v2-m3 | **PASS** (Spearman 0.998) | **FAIL** (Spearman 0.569, Top-10 50%) |

---

## 2. Latency (P50, ms)

모든 값은 BF16 / FP8 / NVFP4 순서로 표기.
- Qwen3 모델의 FP8 레이턴시는 **연산 성능 측정으로 유효** (score 출력이 0이지만 모델 연산 자체는 정상)
- Qwen3 NVFP4는 crash로 측정 불가 (— 표기)

### 2.1 Batch = 1

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-Reranker-2B | 5.3 / 6.1 / — | 11.1 / 9.0 / — | 19.2 / 14.1 / — | 37.5 / 26.8 / — |
| VL-Reranker-8B | 16.3 / 12.7 / — | 37.2 / 21.4 / — | 58.3 / 37.9 / — | 114.8 / 72.1 / — |
| Reranker-0.6B | 3.8 / 4.0 / — | 7.6 / 6.8 / — | 11.7 / 10.7 / — | 22.6 / 19.5 / — |
| Reranker-4B | 10.2 / 9.2 / — | 22.5 / 15.8 / — | 38.6 / 25.4 / — | 69.7 / 51.0 / — |
| Reranker-8B | 15.9 / 11.8 / — | 36.6 / 20.8 / — | 57.2 / 37.0 / — | 112.4 / 70.1 / — |
| bge-reranker-v2-m3 | 2.6 / 5.4 / 4.8 | 6.2 / 7.3 / 11.3 | 10.6 / 10.3 / 12.3 | 20.3 / 18.0 / 15.6 |

### 2.2 Batch = 16

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-Reranker-2B | 34.5 / 25.2 / — | 113.3 / 85.2 / — | 226.2 / 171.5 / — | 494.0 / 371.5 / — |
| VL-Reranker-8B | 117.9 / 68.2 / — | 377.9 / 236.4 / — | 768.6 / 494.3 / — | 2996.9 / 1936.0 / — |
| Reranker-0.6B | 17.1 / 14.8 / — | 65.0 / 54.3 / — | 131.6 / 117.3 / — | 298.4 / 258.7 / — |
| Reranker-4B | 66.4 / 43.4 / — | 229.5 / 153.3 / — | 470.0 / 325.0 / — | 1825.8 / 1273.2 / — |
| Reranker-8B | 114.8 / 69.4 / — | 377.4 / 233.6 / — | 757.1 / 484.6 / — | 2979.9 / 1902.2 / — |
| bge-reranker-v2-m3 | 13.0 / 11.8 / 15.4 | 63.5 / 49.2 / 42.2 | 132.6 / 96.3 / 82.7 | 294.9 / 225.9 / 195.3 |

---

## 3. Throughput (tok/s)

### 3.1 Batch = 1

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-Reranker-2B | 23,879 / 21,067 / — | 90,209 / 113,032 / — | 106,777 / 145,121 / — | 109,140 / 152,655 / — |
| VL-Reranker-8B | 7,854 / 10,069 / — | 27,514 / 47,867 / — | 35,111 / 53,857 / — | 35,695 / 56,530 / — |
| Reranker-0.6B | 33,793 / 31,843 / — | 134,862 / 149,782 / — | 175,043 / 192,164 / — | 180,733 / 209,846 / — |
| Reranker-4B | 12,532 / 13,794 / — | 45,405 / 63,341 / — | 52,745 / 79,454 / — | 58,681 / 79,984 / — |
| Reranker-8B | 8,039 / 10,685 / — | 27,944 / 49,208 / — | 35,789 / 55,282 / — | 36,419 / 58,438 / — |
| bge-reranker-v2-m3 | 49,095 / 23,893 / 26,487 | 163,852 / 140,094 / 90,990 | 191,234 / 199,932 / 166,328 | 200,839 / 227,303 / 254,954 |

### 3.2 Batch = 16

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-Reranker-2B | 73,266 / 94,007 / — | 187,223 / 237,179 / — | 190,043 / 242,151 / — | 136,298 / 190,026 / — |
| VL-Reranker-8B | 23,159 / 36,213 / — | 61,059 / 93,837 / — | 61,915 / 93,579 / — | 32,803 / 49,997 / — |
| Reranker-0.6B | 137,526 / 146,869 / — | 304,334 / 343,406 / — | 307,561 / 337,992 / — | 246,593 / 286,702 / — |
| Reranker-4B | 39,494 / 55,967 / — | 97,547 / 140,618 / — | 99,225 / 137,853 / — | 53,183 / 75,243 / — |
| Reranker-8B | 23,516 / 36,670 / — | 61,753 / 96,613 / — | 62,960 / 95,793 / — | 33,034 / 51,233 / — |
| bge-reranker-v2-m3 | 156,838 / 173,970 / 132,312 | 257,485 / 332,755 / 389,228 | 247,108 / 339,827 / 395,846 | 221,946 / 288,893 / 335,586 |

---

## 4. Latency Reduction (%) vs BF16

음수는 레이턴시 증가(성능 악화)를 의미.

### 4.1 Batch = 1

| Model | FP8 128t | FP8 4096t | NVFP4 128t | NVFP4 4096t |
|-------|----------|-----------|------------|-------------|
| VL-Reranker-2B | -13.5% | +28.5% | — | — |
| VL-Reranker-8B | +22.0% | +37.2% | — | — |
| Reranker-0.6B | -6.1% | +13.8% | — | — |
| Reranker-4B | +9.4% | +26.9% | — | — |
| Reranker-8B | +25.6% | +37.6% | — | — |
| bge-reranker-v2-m3 | -106.2% | +11.3% | -85.8% | +23.2% |

### 4.2 Batch = 16

| Model | FP8 128t | FP8 4096t | NVFP4 128t | NVFP4 4096t |
|-------|----------|-----------|------------|-------------|
| VL-Reranker-2B | +27.1% | +24.8% | — | — |
| VL-Reranker-8B | +42.1% | +35.4% | — | — |
| Reranker-0.6B | +13.3% | +13.3% | — | — |
| Reranker-4B | +34.7% | +30.3% | — | — |
| Reranker-8B | +39.6% | +36.2% | — | — |
| bge-reranker-v2-m3 | +9.2% | +23.4% | -19.1% | +33.8% |

---

## 5. Key Findings

### 5.1 vLLM 0.16.0 Qwen3 Reranker 양자화 비호환

- Qwen3 리랭커 5개 모델은 `from_2_way_softmax` weight loading 방식과 양자화 간 비호환으로 FP8/NVFP4 추론이 **불가능**
- FP8: `tie_word_embeddings` + 양자화 → `lm_head` weight 미로드 → score = 0
- NVFP4: `ReplicatedLinear.weight` 속성 미존재 → crash
- vLLM upstream 수정이 필요하며, 현재 버전에서는 **BF16만 사용 가능**

### 5.2 FP8 Latency (Qwen3 모델, 연산 성능 참고용)

- 8B 모델 (Reranker-8B, VL-Reranker-8B): **25~42% 레이턴시 감소** (batch=16 기준)
- 4B 모델 (Reranker-4B): **30~35% 감소**
- 0.6B 모델 (Reranker-0.6B): **13% 감소** (소형 모델은 커널 오버헤드가 지배적)
- 2B 모델 (VL-Reranker-2B): **25~28% 감소**
- **128 토큰 단일 배치**에서는 FP8이 오히려 느린 경우 있음 (커널 오버헤드 > 연산 절감)

### 5.3 bge-reranker-v2-m3 (유일한 정상 양자화 모델)

- **FP8**: Spearman 0.998, Top-10 100%, Top-20 95% → **프로덕션 사용 가능**
- **NVFP4**: Spearman 0.569, Top-10 50% → **프로덕션 사용 불가** (임베딩 벤치마크의 bge-m3 NVFP4 CosSim 0.54 붕괴와 동일 패턴)
- FP8 레이턴시: 대입력(4096tok)에서 11~23% 감소, 소입력(128tok)에서는 오버헤드로 인한 성능 악화

### 5.4 XLM-RoBERTa NVFP4 품질 붕괴

bge-reranker-v2-m3 (XLM-RoBERTa 기반)의 NVFP4 품질 열화는 임베딩 벤치마크의 bge-m3와 동일한 패턴:
- 임베딩 bge-m3 NVFP4: CosSim 0.54 → **FAIL**
- 리랭커 bge-reranker-v2-m3 NVFP4: Spearman 0.569, Top-10 50% → **FAIL**
- 원인: XLM-RoBERTa 아키텍처가 W4A4 양자화에 민감 (RoBERTa self-attention의 precision 요구)

---

## 6. Production Recommendations

### 6.1 Qwen3 리랭커 (5개 모델)

| 권장사항 | 세부 |
|----------|------|
| **현재**: BF16만 사용 | FP8/NVFP4 비호환 (vLLM 0.16.0 버그) |
| **향후**: vLLM 업데이트 대기 | `from_2_way_softmax` 양자화 호환 패치 후 재측정 |
| **대안**: GGUF/GPTQ 등 다른 양자화 형식 검토 | vLLM 외 엔진도 고려 |

### 6.2 bge-reranker-v2-m3

| Precision | 권장 여부 | 근거 |
|-----------|----------|------|
| BF16 | **기본 권장** | 최고 정확도, 96GB GPU에서 충분 |
| FP8 | **조건부 권장** | Spearman 0.998 → 랭킹 품질 유지, 대입력 레이턴시 11~23% 개선 |
| NVFP4 | **비권장** | Spearman 0.569 → 랭킹 품질 심각 붕괴 |

### 6.3 종합

- 리랭커 양자화는 임베딩보다 제약이 많음 (Qwen3 5개 모델 전부 비호환)
- bge-reranker-v2-m3 FP8만이 유일하게 프로덕션 가능한 양자화 조합
- 96GB GPU 환경에서는 BF16으로 8B 리랭커도 충분히 구동 가능 → 양자화 필요성 낮음

---

## 7. Quantized Model Artifacts

### HuggingFace Repositories

| Model | FP8 | NVFP4 |
|-------|-----|-------|
| VL-Reranker-2B | Forturne/Qwen3-VL-Reranker-2B-FP8 | Forturne/Qwen3-VL-Reranker-2B-NVFP4 |
| VL-Reranker-8B | Forturne/Qwen3-VL-Reranker-8B-FP8 | Forturne/Qwen3-VL-Reranker-8B-NVFP4 |
| Reranker-0.6B | Forturne/Qwen3-Reranker-0.6B-FP8 | Forturne/Qwen3-Reranker-0.6B-NVFP4 |
| Reranker-4B | Forturne/Qwen3-Reranker-4B-FP8 | Forturne/Qwen3-Reranker-4B-NVFP4 |
| Reranker-8B | Forturne/Qwen3-Reranker-8B-FP8 | Forturne/Qwen3-Reranker-8B-NVFP4 |
| bge-reranker-v2-m3 | Forturne/bge-reranker-v2-m3-FP8 | Forturne/bge-reranker-v2-m3-NVFP4 |

### Model Sizes (Disk)

| Model | FP8 | NVFP4 |
|-------|-----|-------|
| VL-Reranker-2B | 2.9 GB | 3.3 GB |
| VL-Reranker-8B | 11 GB | 7.5 GB |
| Reranker-0.6B | 733 MB | 845 MB |
| Reranker-4B | 4.2 GB | 3.4 GB |
| Reranker-8B | 8.9 GB | 6.0 GB |
| bge-reranker-v2-m3 | 1.3 GB | 1.2 GB |

---

## Benchmark Methodology

- **Latency benchmark**: vLLM `LLM(runner="pooling")` + `llm.score()`, `gpu_memory_utilization=0.90`
- **hf_overrides**: Qwen3 models require architecture override + `classifier_from_token: ["no", "yes"]` + `is_original_qwen3_reranker: true`
- **Input**: 고정 query (~20 tok) + 가변 document (128~8192 토큰)
- **Prefix cache invalidation**: 배치 내 각 항목마다 다른 텍스트 (staggered offset slicing)
- **CUDA graphs**: enabled (`enforce_eager=False`)
- **Warmup**: 3 runs (discarded), **Timed runs**: 10 per configuration
- **Batch sizes**: 1, 4, 8, 16 / **Input lengths**: 128, 256, 512, 1024, 2048, 4096, 8192
- **Accuracy benchmark**: 100 query-doc pairs from War and Peace, doc lengths 128/256/512/1024 mixed
- **Source text**: War and Peace (Project Gutenberg), ~340K tokens
- **FP8 quantization**: classifier head excluded from quantization (kept in BF16) for encoder-only models
- **NVFP4 quantization**: classifier head excluded, calibration with ultrachat_200k

## Raw Data Files

| File | Description |
|------|-------------|
| `reranker_latency_results_bf16.json` | BF16 latency raw data (6 models) |
| `reranker_latency_results_fp8.json` | FP8 latency raw data (6 models) |
| `reranker_latency_results_nvfp4.json` | NVFP4 latency raw data (bge-reranker-v2-m3 only) |
| `reranker_accuracy_results.json` | BF16 vs FP8 vs NVFP4 accuracy metrics |
