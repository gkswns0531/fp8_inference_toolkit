# Embedding & Reranker Model Inference Benchmark — Project Status

**Epic**: [AI-2251](https://linear.app/allganize/issue/AI-2251/embedding-model-inference-benchmark-bf16fp8fp4) Embedding Model Inference Benchmark (BF16/FP8/FP4)
**GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB)
**Engine**: vLLM 0.16.0

## Embedding Target Models

| Model | Parameters | Type | Max Context |
|-------|-----------|------|-------------|
| Qwen3-Embedding-0.6B | 0.6B | Text Embedding | 32,768 |
| Qwen3-Embedding-4B | 4B | Text Embedding | 32,768 |
| Qwen3-Embedding-8B | 8B | Text Embedding | 32,768 |
| Qwen3-VL-Embedding-2B | 2B | Vision-Language Embedding | 32,768 |
| Qwen3-VL-Embedding-8B | 8B | Vision-Language Embedding | 32,768 |
| BAAI/bge-m3 | 0.6B | Text Embedding | 8,192 |

## Reranker Target Models

| Model | Parameters | Type | Native Arch | vLLM Override | Max Context |
|-------|-----------|------|-------------|---------------|-------------|
| Qwen3-VL-Reranker-2B | 2B | VL Reranker | Qwen3VLForConditionalGeneration | Qwen3VLForSequenceClassification | 32,768 |
| Qwen3-VL-Reranker-8B | 8B | VL Reranker | 동일 | 동일 | 32,768 |
| Qwen3-Reranker-0.6B | 0.6B | Text Reranker | Qwen3ForCausalLM | Qwen3ForSequenceClassification | 32,768 |
| Qwen3-Reranker-4B | 4B | Text Reranker | 동일 | 동일 | 32,768 |
| Qwen3-Reranker-8B | 8B | Text Reranker | 동일 | 동일 | 32,768 |
| bge-reranker-v2-m3 | 0.6B | Text Reranker | XLMRobertaForSequenceClassification | — (native) | 8,192 |

## Embedding Task Tracker

### Done

| Ticket | Task | Output |
|--------|------|--------|
| [AI-2252](https://linear.app/allganize/issue/AI-2252/bf16-임베딩-6모델-레이턴시-측정-및-보고서) | BF16 임베딩 6모델 레이턴시 측정 및 보고서 | [`EMBEDDING_LATENCY_REPORT.md`](./EMBEDDING_LATENCY_REPORT.md), [`embedding_latency_results.json`](./embedding_latency_results.json) |
| [AI-2253](https://linear.app/allganize/issue/AI-2253/fp4nvfp4-임베딩-추론-가능성-조사) | FP4/NVFP4 임베딩 추론 가능성 조사 | [`FP4_INFERENCE_RESEARCH.md`](./FP4_INFERENCE_RESEARCH.md) |
| [AI-2254](https://linear.app/allganize/issue/AI-2254/fp8-임베딩-6모델-레이턴시-측정) | FP8 임베딩 6모델 레이턴시 측정 | [`embedding_latency_results_fp8.json`](./embedding_latency_results_fp8.json) |
| [AI-2255](https://linear.app/allganize/issue/AI-2255/fp4-임베딩-모델-nvfp4-양자화) | FP4 임베딩 모델 NVFP4 양자화 (6모델 전체) | HuggingFace `Forturne/*-NVFP4` repos |
| [AI-2256](https://linear.app/allganize/issue/AI-2256/fp4-임베딩-모델-레이턴시-측정) | FP4 임베딩 모델 레이턴시 측정 | [`embedding_latency_results_nvfp4.json`](./embedding_latency_results_nvfp4.json) |
| [AI-2257](https://linear.app/allganize/issue/AI-2257/bf16-vs-fp8-vs-fp4-임베딩-정확도-오차-검증) | BF16 vs FP8 vs FP4 정확도 오차 검증 | [`embedding_accuracy_results.json`](./embedding_accuracy_results.json) |
| [AI-2258](https://linear.app/allganize/issue/AI-2258/bf16fp8fp4-임베딩-통합-비교-보고서) | BF16/FP8/FP4 임베딩 통합 비교 보고서 | [`EMBEDDING_BENCHMARK_REPORT.md`](./EMBEDDING_BENCHMARK_REPORT.md) |

## Reranker Task Tracker

### Done

| Step | Task | Output | Note |
|------|------|--------|------|
| Step 0 | Smoke Test — vLLM `score()` + FP8 호환 검증 | PASS | BF16 전 모델 정상, FP8 로딩 정상 (score 출력은 Qwen3 모델에서 0) |
| Step 1 | FP8 리랭커 6모델 양자화 | `/home/ubuntu/models/*-fp8` | 완료, HF 업로드 대기 |
| Step 2 | NVFP4 리랭커 6모델 양자화 | `/home/ubuntu/models/*-nvfp4` | 완료, HF 업로드 대기 |
| Step 3 | BF16/FP8/NVFP4 리랭커 레이턴시 측정 | `reranker_latency_results_*.json` | BF16 6/6, FP8 6/6, NVFP4 1/6 (bge only) |
| Step 4 | BF16 vs FP8 vs NVFP4 리랭커 정합성 검증 | `reranker_accuracy_results.json` | Qwen3 FP8/NVFP4 비호환, bge FP8 PASS / NVFP4 FAIL |
| Step 5 | 통합 보고서 | [`RERANKER_BENCHMARK_REPORT.md`](./RERANKER_BENCHMARK_REPORT.md) | 완료 |

## Methodology

모든 레이턴시 벤치마크는 동일한 공정 비교 방법론을 사용한다:

- **Exact token count**: tokenizer encode → slice → decode → re-encode 검증
- **Prefix cache 무효화**: 배치 내 각 항목마다 다른 텍스트 (staggered offset slicing)
- **CUDA graphs 활성화**: `enforce_eager=False`
- **Source text**: War and Peace (Project Gutenberg), ~340K tokens
- **Input lengths**: 128, 256, 512, 1024, 2048, 4096, 8192 tokens
- **Batch sizes**: 1, 4, 8, 16
- **Warmup**: 3회 (discarded), **Timed runs**: 10회
- **Metrics**: P50, P99, avg, std, throughput (tok/s)

## Accuracy Verification Methodology (AI-2257)

BF16을 ground truth로 FP8/FP4의 임베딩 품질 오차를 측정한다:

1. **데이터**: 100개 query-document 쌍 (다양한 도메인, 다양한 길이)
2. **측정 항목**:
   - Query-document cosine similarity 차이 (BF16 sim vs FP8/FP4 sim)
   - Output embedding tensor MAE (mean absolute error)
   - Embedding-level cosine similarity (BF16 embedding vs FP8/FP4 embedding)
3. **산출물**: 모델별(6개) × precision별(3개) 오차 비교표

## File Structure

```
benchmark/
├── PROJECT_STATUS.md                              ← 이 문서
├── EMBEDDING_BENCHMARK_REPORT.md                  ← 임베딩 BF16/FP8/NVFP4 통합 최종 보고서
├── RERANKER_BENCHMARK_REPORT.md                   ← 리랭커 BF16/FP8/NVFP4 통합 보고서
├── EMBEDDING_LATENCY_REPORT.md                    ← BF16 레이턴시 보고서
├── FP4_INFERENCE_RESEARCH.md                      ← FP4 추론 가능성 조사
├── embedding_latency_results.json                 ← 임베딩 BF16 레이턴시 raw 데이터
├── embedding_latency_results_fp8.json             ← 임베딩 FP8 레이턴시 raw 데이터
├── embedding_latency_results_nvfp4.json           ← 임베딩 NVFP4 레이턴시 raw 데이터
├── embedding_accuracy_results.json                ← 임베딩 BF16 vs FP8 vs NVFP4 정합성 데이터
├── reranker_latency_results_bf16.json             ← 리랭커 BF16 레이턴시 raw 데이터
├── reranker_latency_results_fp8.json              ← 리랭커 FP8 레이턴시 raw 데이터
├── reranker_latency_results_nvfp4.json            ← 리랭커 NVFP4 레이턴시 raw 데이터
├── reranker_accuracy_results.json                 ← 리랭커 BF16 vs FP8 vs NVFP4 정합성 데이터
├── benchmark_embedding_latency.py                 ← 임베딩 레이턴시 벤치마크
├── benchmark_accuracy_bf16_fp8_fp4.py             ← 임베딩 정합성 검증
├── benchmark_reranker_latency.py                  ← 리랭커 레이턴시 벤치마크
├── benchmark_reranker_accuracy_bf16_fp8_fp4.py    ← 리랭커 정합성 검증
├── prepare_test_data.py                           ← 테스트 데이터 준비 스크립트
└── test_data/                                     ← War and Peace 원본 + 토큰화 데이터
```
