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

## 0. Note: vLLM 0.16.0 Qwen3 Reranker Quantization 버그 패치

> **vLLM 0.16.0에서 Qwen3 리랭커 FP8/NVFP4 양자화 추론 버그가 존재했으나, `adapters.py` 패치 적용 후 전 모델 정상 동작 확인.**

### 버그 요약

`as_seq_cls_model()`에서 score layer 생성 시 `quant_config`를 전달하여 발생:
- FP8: score layer output_dim=1 → Marlin tile alignment(64) 위반 → score=0
- NVFP4: score layer에 `weight_packed`만 등록, `.weight` 접근 시 crash

### 패치 내용

```diff
# vllm/model_executor/models/adapters.py:300
- quant_config=quant_config,
+ quant_config=None,
```

상세 분석: [`docs/VLLM_QWEN3_RERANKER_QUANTIZATION_BUG.md`](../docs/VLLM_QWEN3_RERANKER_QUANTIZATION_BUG.md) · 관련 이슈: [vllm#33970](https://github.com/vllm-project/vllm/issues/33970)

---

## 1. Reranker Accuracy (BF16 vs FP8 vs NVFP4)

BF16을 ground truth로 FP8/NVFP4의 리랭커 score 오차를 측정.
- **데이터**: War and Peace에서 추출한 100 query-doc 쌍 (128, 256, 512, 1024 토큰 혼합)
- **측정 항목**: Score MAE, Max Diff, Spearman Rank Correlation, Top-K Overlap
- **vLLM**: 패치 적용 버전 (score layer `quant_config=None`)

### 1.1 Score MAE / Max Diff

| Model | FP8 MAE | FP8 Max Diff | NVFP4 MAE | NVFP4 Max Diff |
|-------|---------|--------------|-----------|----------------|
| VL-Reranker-2B | 0.019800 | 0.132811 | 0.055900 | 0.331587 |
| VL-Reranker-8B | 0.021694 | 0.155255 | 0.047264 | 0.374410 |
| Reranker-0.6B | 0.025739 | 0.154471 | 0.135583 | 0.550437 |
| Reranker-4B | 0.022516 | 0.178986 | 0.069312 | 0.469278 |
| Reranker-8B | 0.018897 | 0.108588 | 0.044691 | 0.227421 |
| bge-reranker-v2-m3 | 0.003616 | 0.067467 | 0.062495 | 0.947552 |

### 1.2 Spearman Rank Correlation

값이 1.0에 가까울수록 BF16과 동일한 랭킹을 의미.

| Model | FP8 Spearman | NVFP4 Spearman |
|-------|-------------|----------------|
| VL-Reranker-2B | **0.994** | 0.950 |
| VL-Reranker-8B | **0.993** | 0.962 |
| Reranker-0.6B | **0.986** | 0.839 |
| Reranker-4B | **0.993** | 0.925 |
| Reranker-8B | **0.995** | 0.966 |
| bge-reranker-v2-m3 | **0.998** | 0.569 |

### 1.3 Top-K Rank Overlap

BF16 기준 상위 K개 문서와 양자화 모델의 상위 K개 문서의 일치율.

| Model | FP8 Top-10 | FP8 Top-20 | NVFP4 Top-10 | NVFP4 Top-20 |
|-------|-----------|-----------|-------------|-------------|
| VL-Reranker-2B | **1.00** | **0.95** | 0.90 | 0.90 |
| VL-Reranker-8B | 0.90 | 0.90 | 0.90 | 0.85 |
| Reranker-0.6B | 0.90 | 0.90 | 0.80 | 0.60 |
| Reranker-4B | 0.90 | 0.90 | 0.70 | 0.90 |
| Reranker-8B | **1.00** | **0.95** | 0.80 | **0.95** |
| bge-reranker-v2-m3 | **1.00** | **0.95** | 0.50 | 0.40 |

### 1.4 Accuracy 판정

| Model | FP8 | NVFP4 |
|-------|-----|-------|
| VL-Reranker-2B | **PASS** (Spearman 0.994) | **PASS** (Spearman 0.950) |
| VL-Reranker-8B | **PASS** (Spearman 0.993) | **PASS** (Spearman 0.962) |
| Reranker-0.6B | **PASS** (Spearman 0.986) | **CAUTION** (Spearman 0.839, Top-20 60%) |
| Reranker-4B | **PASS** (Spearman 0.993) | **CAUTION** (Spearman 0.925, Top-10 70%) |
| Reranker-8B | **PASS** (Spearman 0.995) | **PASS** (Spearman 0.966) |
| bge-reranker-v2-m3 | **PASS** (Spearman 0.998) | **FAIL** (Spearman 0.569, Top-10 50%) |

---

## 2. Latency (P50, ms)

모든 값은 BF16 / FP8 / NVFP4 순서로 표기. vLLM 패치 적용 후 전 모델 측정 완료.

### 2.1 Batch = 1

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-Reranker-2B | 5.3 / 5.9 / 5.1 | 11.1 / 9.2 / 13.3 | 19.2 / 14.6 / 14.9 | 37.5 / 27.7 / 23.8 |
| VL-Reranker-8B | 16.3 / 12.7 / 10.4 | 37.2 / 21.7 / 18.0 | 58.3 / 37.7 / 28.2 | 114.8 / 71.7 / 55.4 |
| Reranker-0.6B | 3.8 / 4.1 / 3.7 | 7.6 / 6.6 / 12.8 | 11.7 / 10.3 / 14.5 | 22.6 / 19.7 / 18.0 |
| Reranker-4B | 10.2 / 9.1 / 7.6 | 22.5 / 15.3 / 18.2 | 38.6 / 24.5 / 19.9 | 69.7 / 50.4 / 41.7 |
| Reranker-8B | 15.9 / 11.9 / 10.0 | 36.6 / 20.6 / 17.8 | 57.2 / 36.8 / 27.1 | 112.4 / 70.1 / 54.2 |
| bge-reranker-v2-m3 | 2.6 / 3.3 / 3.2 | 6.2 / 5.6 / 10.3 | 10.6 / 8.8 / 11.5 | 20.3 / 17.4 / 15.3 |

### 2.2 Batch = 16

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-Reranker-2B | 34.5 / 25.3 / 20.6 | 113.3 / 87.6 / 77.1 | 226.2 / 168.3 / 146.7 | 494.0 / 377.2 / 330.5 |
| VL-Reranker-8B | 117.9 / 70.8 / 49.4 | 377.9 / 235.4 / 182.8 | 768.6 / 496.1 / 376.8 | 2996.9 / 1938.6 / 1457.3 |
| Reranker-0.6B | 17.1 / 14.6 / 18.7 | 65.0 / 53.9 / 57.1 | 131.6 / 112.3 / 103.9 | 298.4 / 263.0 / 239.6 |
| Reranker-4B | 66.4 / 43.2 / 33.1 | 229.5 / 152.5 / 124.2 | 470.0 / 323.8 / 269.8 | 1825.8 / 1263.3 / 1054.6 |
| Reranker-8B | 114.8 / 66.8 / 47.7 | 377.4 / 232.5 / 171.2 | 757.1 / 485.1 / 367.5 | 2979.9 / 1902.9 / 1435.9 |
| bge-reranker-v2-m3 | 13.0 / 11.6 / 15.4 | 63.5 / 49.8 / 48.6 | 132.6 / 104.8 / 95.0 | 294.9 / 242.2 / 207.2 |

---

## 3. Throughput (tok/s)

### 3.1 Batch = 1

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-Reranker-2B | 23,879 / 21,556 / 24,965 | 90,209 / 110,562 / 75,768 | 106,777 / 140,052 / 136,506 | 109,140 / 146,504 / 173,014 |
| VL-Reranker-8B | 7,854 / 10,091 / 12,318 | 27,514 / 47,098 / 56,800 | 35,111 / 54,344 / 72,628 | 35,695 / 57,129 / 73,908 |
| Reranker-0.6B | 33,793 / 29,911 / 34,627 | 134,862 / 156,224 / 79,833 | 175,043 / 197,709 / 141,283 | 180,733 / 207,838 / 227,647 |
| Reranker-4B | 12,532 / 14,001 / 16,795 | 45,405 / 66,811 / 55,230 | 52,745 / 83,616 / 102,882 | 58,681 / 81,244 / 98,237 |
| Reranker-8B | 8,039 / 10,755 / 12,752 | 27,944 / 49,738 / 57,637 | 35,789 / 55,696 / 75,680 | 36,419 / 58,253 / 75,623 |
| bge-reranker-v2-m3 | 49,095 / 38,571 / 39,706 | 163,852 / 185,081 / 95,820 | 191,234 / 232,679 / 177,807 | 200,839 / 235,850 / 267,417 |

### 3.2 Batch = 16

| Model | 128 tok | 1024 tok | 2048 tok | 4096 tok |
|-------|---------|----------|----------|----------|
| VL-Reranker-2B | 73,266 / 92,945 / 105,646 | 187,223 / 228,821 / 254,840 | 190,043 / 245,535 / 273,936 | 136,298 / 188,152 / 211,652 |
| VL-Reranker-8B | 23,159 / 35,878 / 49,954 | 61,059 / 94,832 / 120,669 | 61,915 / 93,264 / 120,862 | 32,803 / 50,010 / 65,853 |
| Reranker-0.6B | 137,526 / 151,542 / 120,794 | 304,334 / 339,859 / 337,122 | 307,561 / 349,159 / 367,946 | 246,593 / 281,902 / 304,233 |
| Reranker-4B | 39,494 / 56,561 / 72,046 | 97,547 / 141,769 / 167,785 | 99,225 / 140,481 / 163,876 | 53,183 / 75,918 / 89,872 |
| Reranker-8B | 23,516 / 37,474 / 51,745 | 61,753 / 97,017 / 127,500 | 62,960 / 95,707 / 124,163 | 33,034 / 51,139 / 67,037 |
| bge-reranker-v2-m3 | 156,838 / 176,519 / 132,114 | 257,485 / 329,234 / 332,179 | 247,108 / 311,698 / 345,367 | 221,946 / 270,396 / 314,733 |

---

## 4. Latency Reduction (%) vs BF16

음수는 레이턴시 증가(성능 악화)를 의미.

### 4.1 Batch = 1

| Model | FP8 128t | FP8 4096t | NVFP4 128t | NVFP4 4096t |
|-------|----------|-----------|------------|-------------|
| VL-Reranker-2B | -11.0% | +26.0% | +4.3% | +36.6% |
| VL-Reranker-8B | +22.1% | +37.5% | +36.4% | +51.7% |
| Reranker-0.6B | -7.4% | +12.7% | +2.9% | +20.5% |
| Reranker-4B | +10.7% | +27.7% | +25.6% | +40.2% |
| Reranker-8B | +25.3% | +37.6% | +37.1% | +51.8% |
| bge-reranker-v2-m3 | -28.1% | +14.5% | -24.6% | +24.7% |

### 4.2 Batch = 16

| Model | FP8 128t | FP8 4096t | NVFP4 128t | NVFP4 4096t |
|-------|----------|-----------|------------|-------------|
| VL-Reranker-2B | +26.8% | +23.6% | +40.5% | +33.1% |
| VL-Reranker-8B | +39.9% | +35.3% | +58.1% | +51.4% |
| Reranker-0.6B | +14.9% | +11.9% | -9.5% | +19.7% |
| Reranker-4B | +34.9% | +30.8% | +50.2% | +42.2% |
| Reranker-8B | +41.8% | +36.1% | +58.5% | +51.8% |
| bge-reranker-v2-m3 | +10.4% | +17.9% | -19.0% | +29.8% |

---

## 5. Key Findings

### 5.1 FP8: 전 모델 프로덕션 사용 가능

- **Spearman ≥ 0.986** (전 6모델), Top-10 Overlap ≥ 90%
- 8B급 모델이 FP8에서 가장 높은 정합성: Reranker-8B (0.995), bge (0.998)
- 0.6B 모델이 상대적으로 가장 낮지만 여전히 0.986으로 프로덕션 충분

### 5.2 FP8 Latency

- 8B 모델 (Reranker-8B, VL-Reranker-8B): **36~42% 레이턴시 감소** (batch=16 기준)
- 4B 모델 (Reranker-4B): **31~35% 감소**
- 0.6B 모델 (Reranker-0.6B): **12~15% 감소** (소형 모델은 커널 오버헤드가 지배적)
- 2B 모델 (VL-Reranker-2B): **24~27% 감소**
- **128 토큰 단일 배치**에서는 FP8이 오히려 느린 경우 있음 (커널 오버헤드 > 연산 절감)

### 5.3 NVFP4: 레이턴시는 우수, 정합성은 모델별 차이

**레이턴시** (Blackwell 네이티브 W4A4):
- 8B 모델: **batch=16에서 51~58% 감소** — FP8보다 더 빠름
- 4B 모델: **42~50% 감소**
- 2B 모델: **33~41% 감소**
- 0.6B 모델: **12~20% 감소** (소형 모델은 개선폭 제한)

**정합성**:
- **고품질 (Spearman ≥ 0.95)**: Reranker-8B (0.966), VL-Reranker-8B (0.962), VL-Reranker-2B (0.950)
- **주의 필요 (0.85~0.95)**: Reranker-4B (0.925), Reranker-0.6B (0.839)
- **사용 불가 (< 0.6)**: bge-reranker-v2-m3 (0.569)
- **경향**: 모델이 클수록 NVFP4 정합성이 높음. 0.6B 소형 모델은 NVFP4에 민감

### 5.4 XLM-RoBERTa NVFP4 품질 붕괴

bge-reranker-v2-m3 (XLM-RoBERTa 기반)의 NVFP4 품질 열화는 임베딩 벤치마크의 bge-m3와 동일한 패턴:
- 임베딩 bge-m3 NVFP4: CosSim 0.54 → **FAIL**
- 리랭커 bge-reranker-v2-m3 NVFP4: Spearman 0.569, Top-10 50% → **FAIL**
- 원인: XLM-RoBERTa 아키텍처가 W4A4 양자화에 민감 (RoBERTa self-attention의 precision 요구)

### 5.5 vLLM 버그 패치 필요

- vLLM 0.16.0 vanilla에서는 Qwen3 리랭커 FP8/NVFP4가 동작하지 않음
- `adapters.py` score layer `quant_config=None` 패치 필요 (상세: Section 0)

---

## 6. Production Recommendations

### 6.1 FP8 (전 모델 권장)

| Model | Spearman | 판정 | 비고 |
|-------|----------|------|------|
| VL-Reranker-2B | 0.994 | **PASS** | Top-10 100% |
| VL-Reranker-8B | 0.993 | **PASS** | batch=16 레이턴시 42% 감소 |
| Reranker-0.6B | 0.986 | **PASS** | 소형 모델, 레이턴시 개선 13% |
| Reranker-4B | 0.993 | **PASS** | 레이턴시 30~35% 감소 |
| Reranker-8B | 0.995 | **PASS** | Top-10 100%, 레이턴시 36~40% 감소 |
| bge-reranker-v2-m3 | 0.998 | **PASS** | 최고 정합성 |

### 6.2 NVFP4 (모델별 판단 필요)

| Model | Spearman | 판정 | 비고 |
|-------|----------|------|------|
| VL-Reranker-2B | 0.950 | **PASS** | Top-10 90% |
| VL-Reranker-8B | 0.962 | **PASS** | Top-10 90% |
| Reranker-0.6B | 0.839 | **CAUTION** | Top-20 60%, 소형 모델 NVFP4에 민감 |
| Reranker-4B | 0.925 | **CAUTION** | Top-10 70% |
| Reranker-8B | 0.966 | **PASS** | Top-20 95% |
| bge-reranker-v2-m3 | 0.569 | **FAIL** | XLM-RoBERTa NVFP4 품질 붕괴 |

### 6.3 종합

- **FP8**: 전 6모델 프로덕션 사용 가능. Spearman ≥ 0.986, 8B급 모델에서 레이턴시 35~42% 감소
- **NVFP4**: 8B급 Qwen3 모델에서 Spearman ≥ 0.95로 사용 가능. 소형 모델(0.6B, 4B)은 주의 필요. bge는 사용 불가
- **BF16**: 96GB GPU 환경에서는 양자화 없이도 8B 모델 구동 가능. 정확도 우선 시 권장
- **주의**: vLLM 0.16.0에는 `adapters.py` 패치 필요 (Section 0 참조)

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
| `reranker_latency_results_nvfp4.json` | NVFP4 latency raw data (6 models) |
| `reranker_accuracy_results.json` | BF16 vs FP8 vs NVFP4 accuracy metrics |
