# Embedding Model Inference Benchmark — Project Status

**Epic**: [AI-2251](https://linear.app/allganize/issue/AI-2251/embedding-model-inference-benchmark-bf16fp8fp4) Embedding Model Inference Benchmark (BF16/FP8/FP4)
**GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB)
**Engine**: vLLM 0.16.0

## Target Models

| Model | Parameters | Type | Max Context |
|-------|-----------|------|-------------|
| Qwen3-Embedding-0.6B | 0.6B | Text Embedding | 32,768 |
| Qwen3-Embedding-4B | 4B | Text Embedding | 32,768 |
| Qwen3-Embedding-8B | 8B | Text Embedding | 32,768 |
| Qwen3-VL-Embedding-2B | 2B | Vision-Language Embedding | 32,768 |
| Qwen3-VL-Embedding-8B | 8B | Vision-Language Embedding | 32,768 |
| BAAI/bge-m3 | 0.6B | Text Embedding | 8,192 |

## Task Tracker

### Done

| Ticket | Task | Output |
|--------|------|--------|
| [AI-2252](https://linear.app/allganize/issue/AI-2252/bf16-임베딩-6모델-레이턴시-측정-및-보고서) | BF16 임베딩 6모델 레이턴시 측정 및 보고서 | [`EMBEDDING_LATENCY_REPORT.md`](./EMBEDDING_LATENCY_REPORT.md), [`embedding_latency_results.json`](./embedding_latency_results.json) |
| [AI-2253](https://linear.app/allganize/issue/AI-2253/fp4nvfp4-임베딩-추론-가능성-조사) | FP4/NVFP4 임베딩 추론 가능성 조사 | [`FP4_INFERENCE_RESEARCH.md`](./FP4_INFERENCE_RESEARCH.md) |

### Todo

| Ticket | Task | Blocked By | Details |
|--------|------|------------|---------|
| [AI-2254](https://linear.app/allganize/issue/AI-2254/fp8-임베딩-6모델-레이턴시-측정) | FP8 임베딩 6모델 레이턴시 측정 | — | Forturne/FP8 모델 또는 on-the-fly `quantization="fp8"` 사용. BF16과 동일 방법론. |
| [AI-2255](https://linear.app/allganize/issue/AI-2255/fp4-임베딩-모델-nvfp4-양자화) | FP4 임베딩 모델 NVFP4 양자화 | — | 0.6B, 4B, VL-2B, VL-8B 직접 양자화. 8B는 `alexliap/Qwen3-Embedding-8B-NVFP4` 활용 가능. `quantization/nvfp4_quantize.py` 사용. |
| [AI-2256](https://linear.app/allganize/issue/AI-2256/fp4-임베딩-모델-레이턴시-측정) | FP4 임베딩 모델 레이턴시 측정 | AI-2255 | NVFP4 양자화 완료 후 진행. `quantization="compressed-tensors"`로 로드. |
| [AI-2257](https://linear.app/allganize/issue/AI-2257/bf16-vs-fp8-vs-fp4-임베딩-정확도-오차-검증) | BF16 vs FP8 vs FP4 정확도 오차 검증 | AI-2254, AI-2256 | 100개 query-doc 쌍 cosine similarity 비교 + embedding MAE 측정. |
| [AI-2258](https://linear.app/allganize/issue/AI-2258/bf16fp8fp4-임베딩-통합-비교-보고서) | BF16/FP8/FP4 임베딩 통합 비교 보고서 | AI-2254, AI-2256, AI-2257 | 레이턴시 + 정확도 통합 분석. 프로덕션 권장 설정 제안. |

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
├── PROJECT_STATUS.md                  ← 이 문서
├── EMBEDDING_LATENCY_REPORT.md        ← BF16 레이턴시 보고서 (Done)
├── FP4_INFERENCE_RESEARCH.md          ← FP4 추론 가능성 조사 (Done)
├── embedding_latency_results.json     ← BF16 레이턴시 raw 데이터
├── benchmark_embedding_latency.py     ← 레이턴시 벤치마크 스크립트
├── prepare_test_data.py               ← 테스트 데이터 준비 스크립트
└── test_data/                         ← War and Peace 원본 + 토큰화 데이터
```
