# Retrieval Evaluation Report — BF16 vs FP8 vs NVFP4

**Date**: 2026-03-03
**GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB)
**Engine**: vLLM 0.16.0
**Models**: Qwen3-VL-Embedding-8B, Qwen3-VL-Embedding-2B

## Overview

15개 실제 데이터셋(도메인 4 + 클라이언트 11)에서 BF16/FP8/NVFP4 양자화 모델의 검색 품질을 비교.
합성 데이터(War and Peace)가 아닌 실제 고객사 데이터 기반 평가.

## Datasets

| Category | Dataset | Queries | Corpus | Domain |
|----------|---------|---------|--------|--------|
| Domain | finance | 645 | 3,281 | 금융 |
| Domain | hotpot | 688 | 3,218 | Multi-hop QA |
| Domain | legal | 605 | 4,090 | 법률 |
| Domain | patent | 480 | 3,339 | 특허 |
| Client | gugak | 43 | 1,023 | 국악 |
| Client | hanhwa_insurance | 183 | 1,615 | 보험 |
| Client | isu_system | 130 | 219 | IT 시스템 |
| Client | jacs | 32 | 5,599 | 학술 |
| Client | mirae_asset | 629 | 374 | 자산운용 |
| Client | ok_finance | 22 | 781 | 금융(저축은행) |
| Client | sejong | 23 | 5,753 | 세종학당 |
| Client | skens | 190 | 626 | 에너지 |
| Client | sumitomo | 95 | 2,427 | 화학 |
| Client | trans_cosmos | 87 | 267 | CS운영 |
| Client | yuhan_kimberly | 203 | 2,015 | 소비재 |
| **Total** | | **4,055** | **34,627** | |

## Metrics

- **Coverage@10**: gold_chunk_groups 중 top-10 검색 결과에 1개 이상 포함된 그룹 비율
- **Top@10 (Perfect Match)**: 모든 gold_chunk_groups가 top-10에 포함된 query 비율
- **NDCG@10**: group-level DCG with log discount
- **MRR**: group-level Mean Reciprocal Rank

---

## 1. Qwen3-VL-Embedding-8B Results

### 1.1 Overall Summary

| Metric | BF16 | FP8 | Δ FP8 | NVFP4 | Δ NVFP4 |
|--------|------|-----|-------|-------|---------|
| Coverage@10 | **82.18%** | 82.33% | +0.15 | 81.35% | -0.83 |
| Top@10 | **75.33%** | 75.55% | +0.22 | 74.54% | -0.79 |
| NDCG@10 | **70.06%** | 69.97% | -0.09 | 68.36% | -1.70 |
| MRR | **60.56%** | 60.37% | -0.19 | 58.60% | -1.96 |

### 1.2 Per-Dataset — Coverage@10 (%)

| Dataset | BF16 | FP8 | Δ | NVFP4 | Δ |
|---------|------|-----|---|-------|---|
| finance | 71.30 | 71.72 | +0.42 | 72.00 | +0.70 |
| hotpot | 90.55 | 90.13 | -0.42 | 88.78 | -1.77 |
| legal | 64.31 | 64.64 | +0.33 | 63.62 | -0.69 |
| patent | 81.93 | 81.89 | -0.04 | 83.04 | +1.11 |
| gugak | 93.02 | 93.02 | 0.00 | 84.88 | **-8.14** |
| hanhwa_insurance | 77.87 | 76.78 | -1.09 | 79.51 | +1.64 |
| isu_system | 95.38 | 95.38 | 0.00 | 93.08 | -2.30 |
| jacs | 93.75 | 93.75 | 0.00 | 96.88 | +3.13 |
| mirae_asset | 95.31 | 94.99 | -0.32 | 94.36 | -0.95 |
| ok_finance | 59.09 | 59.09 | 0.00 | 59.09 | 0.00 |
| sejong | 76.09 | 76.09 | 0.00 | 73.91 | -2.18 |
| skens | 89.47 | 90.00 | +0.53 | 91.32 | +1.85 |
| sumitomo | 75.79 | 75.79 | 0.00 | 72.11 | -3.68 |
| trans_cosmos | 71.84 | 74.14 | +2.30 | 72.41 | +0.57 |
| yuhan_kimberly | 97.04 | 97.54 | +0.50 | 95.32 | -1.72 |

### 1.3 Per-Dataset — NDCG@10 (%)

| Dataset | BF16 | FP8 | Δ | NVFP4 | Δ |
|---------|------|-----|---|-------|---|
| finance | 63.74 | 64.56 | +0.82 | 64.49 | +0.75 |
| hotpot | 89.40 | 89.28 | -0.12 | 87.58 | -1.82 |
| legal | 54.74 | 54.75 | +0.01 | 53.26 | -1.48 |
| patent | 76.29 | 76.34 | +0.05 | 76.25 | -0.04 |
| gugak | 78.40 | 79.55 | +1.15 | 70.96 | **-7.44** |
| hanhwa_insurance | 58.55 | 58.48 | -0.07 | 57.01 | -1.54 |
| isu_system | 81.21 | 80.74 | -0.47 | 78.28 | -2.93 |
| jacs | 85.57 | 83.73 | -1.84 | 85.66 | +0.09 |
| mirae_asset | 82.24 | 82.58 | +0.34 | 80.70 | -1.54 |
| ok_finance | 45.70 | 46.66 | +0.96 | 40.58 | -5.12 |
| sejong | 66.36 | 66.01 | -0.35 | 68.69 | +2.33 |
| skens | 73.81 | 73.20 | -0.61 | 71.46 | -2.35 |
| sumitomo | 54.18 | 54.67 | +0.49 | 52.92 | -1.26 |
| trans_cosmos | 52.81 | 50.95 | -1.86 | 52.51 | -0.30 |
| yuhan_kimberly | 87.84 | 88.11 | +0.27 | 85.10 | -2.74 |

### 1.4 Embedding Time (seconds)

| Dataset | Texts | BF16 | FP8 | Speedup | NVFP4 | Speedup |
|---------|-------|------|-----|---------|-------|---------|
| finance | 3,926 | 44.9 | 27.6 | 1.63x | 19.8 | 2.27x |
| hotpot | 3,906 | 15.0 | 9.4 | 1.60x | 6.8 | 2.21x |
| legal | 4,695 | 69.4 | 42.1 | 1.65x | 30.0 | 2.31x |
| patent | 3,819 | 52.9 | 32.2 | 1.64x | 22.9 | 2.31x |
| gugak | 1,066 | 118.9 | 74.9 | 1.59x | 55.2 | 2.15x |
| hanhwa_insurance | 1,798 | 77.3 | 48.0 | 1.61x | 34.9 | 2.21x |
| isu_system | 349 | 6.5 | 4.0 | 1.63x | 2.9 | 2.24x |
| jacs | 5,631 | 188.7 | 115.8 | 1.63x | 83.5 | 2.26x |
| mirae_asset | 1,003 | 10.9 | 6.7 | 1.63x | 4.9 | 2.22x |
| ok_finance | 803 | 63.2 | 40.5 | 1.56x | 30.4 | 2.08x |
| sejong | 5,776 | 138.6 | 84.9 | 1.63x | 61.2 | 2.26x |
| skens | 816 | 11.7 | 7.1 | 1.65x | 5.1 | 2.29x |
| sumitomo | 2,522 | 56.2 | 34.9 | 1.61x | 25.5 | 2.20x |
| trans_cosmos | 354 | 14.7 | 9.3 | 1.58x | 6.9 | 2.13x |
| yuhan_kimberly | 2,218 | 125.6 | 78.0 | 1.61x | 56.6 | 2.22x |
| **Total** | **38,682** | **994.5** | **615.4** | **1.62x** | **446.6** | **2.23x** |

---

## 2. Qwen3-VL-Embedding-2B Results

### 2.1 Overall Summary

| Metric | BF16 | FP8 | Δ FP8 | NVFP4 | Δ NVFP4 |
|--------|------|-----|-------|-------|---------|
| Coverage@10 | **77.54%** | 76.98% | -0.56 | 74.61% | -2.93 |
| Top@10 | **69.98%** | 69.68% | -0.30 | 66.72% | -3.26 |
| NDCG@10 | **62.53%** | 62.03% | -0.50 | 59.63% | -2.90 |
| MRR | **52.87%** | 52.38% | -0.49 | 50.26% | -2.61 |

### 2.2 Per-Dataset — Coverage@10 (%)

| Dataset | BF16 | FP8 | Δ | NVFP4 | Δ |
|---------|------|-----|---|-------|---|
| finance | 68.13 | 67.65 | -0.48 | 66.65 | -1.48 |
| hotpot | 83.14 | 83.38 | +0.24 | 81.88 | -1.26 |
| legal | 59.83 | 59.82 | -0.01 | 58.71 | -1.12 |
| patent | 80.54 | 81.35 | +0.81 | 78.84 | -1.70 |
| gugak | 75.58 | 74.42 | -1.16 | 77.91 | +2.33 |
| hanhwa_insurance | 67.21 | 66.94 | -0.27 | 60.38 | **-6.83** |
| isu_system | 90.00 | 90.00 | 0.00 | 86.54 | -3.46 |
| jacs | 90.62 | 90.62 | 0.00 | 90.62 | 0.00 |
| mirae_asset | 90.86 | 90.14 | -0.72 | 87.52 | -3.34 |
| ok_finance | 77.27 | 72.73 | -4.54 | 68.18 | **-9.09** |
| sejong | 69.57 | 69.57 | 0.00 | 73.91 | +4.34 |
| skens | 81.58 | 82.37 | +0.79 | 76.58 | -5.00 |
| sumitomo | 67.37 | 66.32 | -1.05 | 63.68 | -3.69 |
| trans_cosmos | 67.24 | 64.37 | -2.87 | 58.62 | **-8.62** |
| yuhan_kimberly | 94.09 | 95.07 | +0.98 | 89.16 | -4.93 |

### 2.3 Per-Dataset — NDCG@10 (%)

| Dataset | BF16 | FP8 | Δ | NVFP4 | Δ |
|---------|------|-----|---|-------|---|
| finance | 60.76 | 60.53 | -0.23 | 59.46 | -1.30 |
| hotpot | 81.37 | 81.61 | +0.24 | 79.43 | -1.94 |
| legal | 49.54 | 49.17 | -0.37 | 48.14 | -1.40 |
| patent | 73.92 | 73.62 | -0.30 | 72.15 | -1.77 |
| gugak | 61.73 | 61.97 | +0.24 | 67.35 | +5.62 |
| hanhwa_insurance | 47.85 | 46.27 | -1.58 | 42.33 | **-5.52** |
| isu_system | 73.46 | 71.77 | -1.69 | 67.64 | -5.82 |
| jacs | 70.44 | 72.23 | +1.79 | 67.79 | -2.65 |
| mirae_asset | 75.52 | 75.08 | -0.44 | 71.10 | -4.42 |
| ok_finance | 50.46 | 48.44 | -2.02 | 47.34 | -3.12 |
| sejong | 55.51 | 56.17 | +0.66 | 55.88 | +0.37 |
| skens | 62.53 | 60.92 | -1.61 | 57.66 | -4.87 |
| sumitomo | 48.99 | 47.97 | -1.02 | 46.19 | -2.80 |
| trans_cosmos | 43.16 | 41.93 | -1.23 | 35.98 | **-7.18** |
| yuhan_kimberly | 82.71 | 82.81 | +0.10 | 76.00 | -6.71 |

### 2.4 Embedding Time (seconds)

| Dataset | Texts | BF16 | FP8 | Speedup | NVFP4 | Speedup |
|---------|-------|------|-----|---------|-------|---------|
| finance | 3,926 | 10.8 | 7.2 | 1.50x | 6.0 | 1.80x |
| hotpot | 3,906 | 3.7 | 2.5 | 1.48x | 2.2 | 1.68x |
| legal | 4,695 | 16.4 | 10.7 | 1.53x | 8.9 | 1.84x |
| patent | 3,819 | 12.5 | 8.2 | 1.52x | 6.8 | 1.84x |
| gugak | 1,066 | 29.2 | 19.9 | 1.47x | 16.8 | 1.74x |
| hanhwa_insurance | 1,798 | 18.8 | 12.5 | 1.50x | 10.5 | 1.79x |
| isu_system | 349 | 1.6 | 1.0 | 1.60x | 0.9 | 1.78x |
| jacs | 5,631 | 45.5 | 29.8 | 1.53x | 24.8 | 1.83x |
| mirae_asset | 1,003 | 2.7 | 1.8 | 1.50x | 1.5 | 1.80x |
| ok_finance | 803 | 15.9 | 11.0 | 1.45x | 9.5 | 1.67x |
| sejong | 5,776 | 33.3 | 21.9 | 1.52x | 18.3 | 1.82x |
| skens | 816 | 2.8 | 1.9 | 1.47x | 1.5 | 1.87x |
| sumitomo | 2,522 | 13.8 | 9.2 | 1.50x | 7.8 | 1.77x |
| trans_cosmos | 354 | 3.7 | 2.5 | 1.48x | 2.1 | 1.76x |
| yuhan_kimberly | 2,218 | 30.6 | 20.3 | 1.51x | 17.0 | 1.80x |
| **Total** | **38,682** | **241.3** | **160.4** | **1.50x** | **134.6** | **1.79x** |

---

## 3. 8B vs 2B Comparison

### 3.1 Overall

| Model | Precision | Cov@10 | Top@10 | NDCG@10 | MRR |
|-------|-----------|--------|--------|---------|-----|
| **VL-8B** | BF16 | 82.18 | 75.33 | 70.06 | 60.56 |
| **VL-8B** | FP8 | 82.33 | 75.55 | 69.97 | 60.37 |
| **VL-8B** | NVFP4 | 81.35 | 74.54 | 68.36 | 58.60 |
| **VL-2B** | BF16 | 77.54 | 69.98 | 62.53 | 52.87 |
| **VL-2B** | FP8 | 76.98 | 69.68 | 62.03 | 52.38 |
| **VL-2B** | NVFP4 | 74.61 | 66.72 | 59.63 | 50.26 |

### 3.2 Quantization Degradation

| Model | FP8 Cov Δ | FP8 NDCG Δ | NVFP4 Cov Δ | NVFP4 NDCG Δ |
|-------|-----------|------------|-------------|--------------|
| VL-8B | +0.15 | -0.09 | -0.83 | -1.70 |
| VL-2B | -0.56 | -0.50 | -2.93 | -2.90 |

### 3.3 Speedup

| Model | FP8 Speedup | NVFP4 Speedup | BF16 Total Time | NVFP4 Total Time |
|-------|-------------|---------------|-----------------|------------------|
| VL-8B | 1.62x | 2.23x | 994.5s | 446.6s |
| VL-2B | 1.50x | 1.79x | 241.3s | 134.6s |

### 3.4 Worst-Case Degradation (NVFP4)

**VL-8B NVFP4** worst: gugak -8.14pp (Coverage), ok_finance -5.12pp (NDCG)
**VL-2B NVFP4** worst: ok_finance -9.09pp (Coverage), trans_cosmos -7.18pp (NDCG)

---

## 4. Key Findings

### FP8 — 사실상 무손실

- **VL-8B FP8**: 모든 메트릭에서 BF16 대비 ±0.2pp 이내. Coverage는 오히려 +0.15pp 상승.
- **VL-2B FP8**: BF16 대비 -0.5pp 수준의 미미한 하락. 실용적으로 무시 가능.
- **속도**: VL-8B 1.62x, VL-2B 1.50x 가속.

### NVFP4 — 모델 크기에 비례하는 열화

- **VL-8B NVFP4**: 평균 -0.83pp (Coverage), -1.70pp (NDCG). 대부분 데이터셋에서 2pp 이내.
  - 예외: gugak (-8.14pp Coverage, -7.44pp NDCG) — 소규모 특수 도메인.
- **VL-2B NVFP4**: 평균 -2.93pp (Coverage), -2.90pp (NDCG). 8B보다 3배 이상 큰 열화.
  - ok_finance -9.09pp, trans_cosmos -8.62pp 등 일부 데이터셋에서 현저한 하락.
- **속도**: VL-8B 2.23x, VL-2B 1.79x 가속.

### 모델 크기 효과

- 8B → 2B 축소 시 BF16 기준 -4.64pp (Coverage), -7.53pp (NDCG) 하락.
- 양자화 민감도: 2B 모델이 NVFP4에서 3배 이상 큰 성능 저하 (0.83pp vs 2.93pp).
- **VL-8B NVFP4 > VL-2B BF16**: 8B의 NVFP4(81.35%)가 2B의 BF16(77.54%)보다 높아, 큰 모델의 양자화가 작은 모델 원본보다 우수.

### 권장 사항

| 시나리오 | 추천 | 근거 |
|----------|------|------|
| 품질 최우선 | VL-8B BF16 | 최고 정확도 |
| 품질-속도 균형 | VL-8B FP8 | 무손실 + 1.6x 가속 |
| 최대 처리량 | VL-8B NVFP4 | -0.8pp로 2.2x 가속, 2B BF16보다 우수 |
| 저사양 환경 | VL-2B FP8 | 2B 원본 대비 -0.5pp로 1.5x 가속 |
| 비추천 | VL-2B NVFP4 | -2.9pp 열화, 일부 데이터셋 -9pp |

---

## 5. Methodology

- **임베딩 엔진**: vLLM 0.16.0 (pooling runner, `convert="embed"`)
- **max_model_len**: 8,192 tokens (초과 시 tokenizer 기반 truncation, headroom 200 tokens)
- **검색**: Cosine similarity (L2-normalized embeddings → matrix multiply → argsort)
- **메트릭**: Group-based (gold_chunk_groups 기반) — 단일 청크가 아닌 정보 단위(group) 기준 평가
- **양자화**: FP8 = `compressed-tensors` (FP8_DYNAMIC), NVFP4 = `compressed-tensors` (NVFP4, calibrated with ultrachat_200k)

## 6. Files

| File | Description |
|------|-------------|
| `client_eval/run_retrieval_eval.py` | 평가 스크립트 |
| `client_eval/results/retrieval_eval_results.json` | 전체 결과 (2모델 × 3정밀도 × 15데이터셋) |
| `client_eval/RETRIEVAL_EVAL_REPORT.md` | 이 보고서 |
| `dataset/DATASET_GUIDE.md` | 데이터셋 가이드 |
| `dataset/client/*.json` | 클라이언트 데이터셋 (11개) |
| `dataset/domain/*.json` | 도메인 데이터셋 (4개) |
