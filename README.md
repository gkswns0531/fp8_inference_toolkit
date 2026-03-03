# FP8 Inference Toolkit

LLM/VLM 모델의 양자화(FP8, INT4 GPTQ/AWQ, NVFP4), 벤치마크, Hugging Face 업로드, vLLM 추론을 위한 통합 툴킷입니다.

**GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (GB202, 96 GB)
**Engine**: vLLM 0.16.0 · **Quantization**: llmcompressor 0.9.0.2

---

## Benchmark Reports

| Report | Description | Link |
|--------|-------------|------|
| **Embedding Benchmark** | 6모델 BF16/FP8/NVFP4 레이턴시 + 정합성 | [`EMBEDDING_BENCHMARK_REPORT.md`](benchmark/EMBEDDING_BENCHMARK_REPORT.md) |
| **Reranker Benchmark** | 6모델 BF16/FP8/NVFP4 레이턴시 + 정합성 | [`RERANKER_BENCHMARK_REPORT.md`](benchmark/RERANKER_BENCHMARK_REPORT.md) |
| **Project Status** | 전체 태스크 트래커 (Linear 티켓 연동) | [`PROJECT_STATUS.md`](benchmark/PROJECT_STATUS.md) |
| **FP4 Research** | FP4/NVFP4 추론 가능성 조사 | [`FP4_INFERENCE_RESEARCH.md`](benchmark/FP4_INFERENCE_RESEARCH.md) |

### Key Results Summary

**Embedding Models (6개)**:
- FP8: 전 모델 CosSim ≥ 0.999, 레이턴시 최대 37% 감소 → **프로덕션 권장**
- NVFP4: Qwen3 계열 CosSim ≥ 0.998, bge-m3 CosSim 0.54 → **Qwen3만 사용 가능**

**Reranker Models (6개)**:
- Qwen3 리랭커 5개: vLLM 0.16.0 FP8/NVFP4 비호환 (BF16만 사용 가능)
- bge-reranker-v2-m3 FP8: Spearman 0.998, Top-10 100% → **프로덕션 사용 가능**
- bge-reranker-v2-m3 NVFP4: Spearman 0.569 → **사용 불가**

---

## Supported Quantization Methods

| 방식 | 스크립트 | 비트 | Calibration | GPU 요구 | 특징 |
|------|---------|------|-------------|---------|------|
| **FP8** (W8A8) | `fp8_quantize_all.py` | 8 | 불필요 | H100 네이티브 / L4 폴백 | 빠른 양자화, 최소 정확도 손실 |
| **INT4 GPTQ** (W4A16) | `int4_quantize_gptq.py` | 4 | 필요 (128+) | 대부분 GPU | Hessian 기반, 넓은 호환성 |
| **INT4 AWQ** (W4A16) | `int4_quantize_awq.py` | 4 | 필요 (128+) | 대부분 GPU | Activation-aware, 높은 품질 |
| **NVFP4** (W4A4) | `nvfp4_quantize.py` | 4.5 | 필요 (20+) | **Blackwell 네이티브** | INT4 대비 2.35x 빠름 (B200) |
| **NVFP4A16** (W4A16) | `nvfp4_quantize.py --scheme NVFP4A16` | 4.5 | 불필요 | Blackwell / 기타 Marlin 폴백 | Weight-only, 간편 |

---

## Directory Structure

```
fp8_inference_toolkit/
├── README.md
├── quantization/                                    # 양자화 스크립트
│   ├── fp8_quantize_all.py                         # FP8 임베딩 모델 양자화
│   ├── fp8_quantize_rerankers.py                   # FP8 리랭커 모델 양자화
│   ├── fp8_quantize_hf_standard.py                 # FP8 auto_fp8 표준 포맷
│   ├── nvfp4_quantize.py                           # NVFP4 (W4A4 / W4A16) 범용
│   ├── int4_quantize_gptq.py                       # INT4 GPTQ (W4A16)
│   ├── int4_quantize_awq.py                        # INT4 AWQ (W4A16)
│   └── verify_int4_vllm.py                         # INT4 모델 vLLM 검증
├── benchmark/                                       # 벤치마크
│   ├── PROJECT_STATUS.md                           # 전체 태스크 트래커
│   ├── EMBEDDING_BENCHMARK_REPORT.md               # 임베딩 BF16/FP8/NVFP4 통합 보고서
│   ├── RERANKER_BENCHMARK_REPORT.md                # 리랭커 BF16/FP8/NVFP4 통합 보고서
│   ├── EMBEDDING_LATENCY_REPORT.md                 # 임베딩 BF16 레이턴시 보고서
│   ├── FP4_INFERENCE_RESEARCH.md                   # FP4 추론 가능성 조사
│   ├── benchmark_embedding_latency.py              # 임베딩 레이턴시 벤치마크
│   ├── benchmark_accuracy_bf16_fp8_fp4.py          # 임베딩 정합성 검증
│   ├── benchmark_reranker_latency.py               # 리랭커 레이턴시 벤치마크
│   ├── benchmark_reranker_accuracy_bf16_fp8_fp4.py # 리랭커 정합성 검증
│   ├── smoke_test_reranker.py                      # vLLM score() 호환 검증
│   ├── prepare_test_data.py                        # 테스트 데이터 준비
│   ├── embedding_latency_results.json              # 임베딩 BF16 레이턴시 데이터
│   ├── embedding_latency_results_fp8.json          # 임베딩 FP8 레이턴시 데이터
│   ├── embedding_latency_results_nvfp4.json        # 임베딩 NVFP4 레이턴시 데이터
│   ├── embedding_accuracy_results.json             # 임베딩 정합성 데이터
│   ├── reranker_latency_results_bf16.json          # 리랭커 BF16 레이턴시 데이터
│   ├── reranker_latency_results_fp8.json           # 리랭커 FP8 레이턴시 데이터
│   ├── reranker_latency_results_nvfp4.json         # 리랭커 NVFP4 레이턴시 데이터
│   ├── reranker_accuracy_results.json              # 리랭커 정합성 데이터
│   └── test_data/                                  # War and Peace 원본 + 토큰화 데이터
├── serving/                                         # 추론 서빙
│   ├── triton/                                     # Triton Inference Server
│   └── vllm/                                       # vLLM Direct 서빙
├── int4_native_tc/                                  # INT4 Native Tensor Core (실험)
│   ├── csrc/                                       # CUDA kernel 소스
│   └── calibration/                                # Calibration 도구
└── docs/                                            # 추가 문서
```

---

## Benchmark Target Models

### Embedding Models (6)

| Model | Parameters | Type | HF ID |
|-------|-----------|------|-------|
| Qwen3-Embedding-0.6B | 0.6B | Text | `Qwen/Qwen3-Embedding-0.6B` |
| Qwen3-Embedding-4B | 4B | Text | `Qwen/Qwen3-Embedding-4B` |
| Qwen3-Embedding-8B | 8B | Text | `Qwen/Qwen3-Embedding-8B` |
| Qwen3-VL-Embedding-2B | 2B | Vision-Language | `Qwen/Qwen3-VL-Embedding-2B` |
| Qwen3-VL-Embedding-8B | 8B | Vision-Language | `Qwen/Qwen3-VL-Embedding-8B` |
| bge-m3 | 0.6B | Text | `BAAI/bge-m3` |

### Reranker Models (6)

| Model | Parameters | Type | HF ID | vLLM Override |
|-------|-----------|------|-------|---------------|
| Qwen3-VL-Reranker-2B | 2B | VL Reranker | `Qwen/Qwen3-VL-Reranker-2B` | `Qwen3VLForSequenceClassification` |
| Qwen3-VL-Reranker-8B | 8B | VL Reranker | `Qwen/Qwen3-VL-Reranker-8B` | 동일 |
| Qwen3-Reranker-0.6B | 0.6B | Text Reranker | `Qwen/Qwen3-Reranker-0.6B` | `Qwen3ForSequenceClassification` |
| Qwen3-Reranker-4B | 4B | Text Reranker | `Qwen/Qwen3-Reranker-4B` | 동일 |
| Qwen3-Reranker-8B | 8B | Text Reranker | `Qwen/Qwen3-Reranker-8B` | 동일 |
| bge-reranker-v2-m3 | 0.6B | Text Reranker | `BAAI/bge-reranker-v2-m3` | — (native) |

---

## Benchmark Methodology

모든 벤치마크는 동일한 공정 비교 방법론을 사용합니다:

- **Exact token count**: tokenizer encode → slice → decode → re-encode 검증
- **Prefix cache 무효화**: 배치 내 각 항목마다 다른 텍스트 (staggered offset slicing)
- **CUDA graphs 활성화**: `enforce_eager=False`
- **Source text**: War and Peace (Project Gutenberg), ~340K tokens
- **Input lengths**: 128, 256, 512, 1024, 2048, 4096, 8192 tokens
- **Batch sizes**: 1, 4, 8, 16
- **Warmup**: 3회 (discarded), **Timed runs**: 10회
- **Accuracy**: 100개 query-document 쌍 (다양한 도메인, 다양한 길이)

---

## Quick Start

### 1. 의존성 설치

```bash
pip install llmcompressor transformers huggingface_hub torch datasets vllm scipy
```

### 2. HF 토큰 설정

```bash
export HF_TOKEN="your_token_here"
```

### 3. 양자화 실행

```bash
# FP8 임베딩 모델 (6개, calibration 불필요)
python quantization/fp8_quantize_all.py

# FP8 리랭커 모델 (6개, calibration 불필요)
python quantization/fp8_quantize_rerankers.py

# NVFP4 (모델별 개별 실행, calibration 필요)
python quantization/nvfp4_quantize.py --model Qwen/Qwen3-Embedding-8B

# INT4 GPTQ
python quantization/int4_quantize_gptq.py --model Qwen/Qwen3-Next-80B-A3B-Instruct

# INT4 AWQ
python quantization/int4_quantize_awq.py --model Qwen/Qwen3-Next-80B-A3B-Instruct
```

### 4. 벤치마크 실행

```bash
# 임베딩 레이턴시 (BF16)
python benchmark/benchmark_embedding_latency.py

# 임베딩 정합성 (BF16 vs FP8 vs NVFP4)
python benchmark/benchmark_accuracy_bf16_fp8_fp4.py

# 리랭커 레이턴시 (BF16/FP8/NVFP4)
python benchmark/benchmark_reranker_latency.py --quantization bf16
python benchmark/benchmark_reranker_latency.py --quantization compressed-tensors --model-suffix fp8

# 리랭커 정합성
python benchmark/benchmark_reranker_accuracy_bf16_fp8_fp4.py
```

### 5. 서빙

```bash
# 임베딩 모델
vllm serve Forturne/Qwen3-Embedding-8B-FP8 \
  --runner pooling --dtype auto --trust-remote-code

# 리랭커 모델 (Qwen3 — hf_overrides 필요)
vllm serve Qwen/Qwen3-Reranker-8B \
  --runner pooling --dtype auto --trust-remote-code \
  --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'

# 리랭커 모델 (bge — native)
vllm serve BAAI/bge-reranker-v2-m3 \
  --runner pooling --dtype auto --trust-remote-code
```

---

## Quantized Model Artifacts (HuggingFace)

### Embedding Models

| Model | FP8 | NVFP4 |
|-------|-----|-------|
| Qwen3-Embedding-0.6B | `Forturne/Qwen3-Embedding-0.6B-FP8` | `Forturne/Qwen3-Embedding-0.6B-NVFP4` |
| Qwen3-Embedding-4B | `Forturne/Qwen3-Embedding-4B-FP8` | `Forturne/Qwen3-Embedding-4B-NVFP4` |
| Qwen3-Embedding-8B | `Forturne/Qwen3-Embedding-8B-FP8` | `Forturne/Qwen3-Embedding-8B-NVFP4` |
| Qwen3-VL-Embedding-2B | `Forturne/Qwen3-VL-Embedding-2B-FP8` | `Forturne/Qwen3-VL-Embedding-2B-NVFP4` |
| Qwen3-VL-Embedding-8B | `Forturne/Qwen3-VL-Embedding-8B-FP8` | `Forturne/Qwen3-VL-Embedding-8B-NVFP4` |
| bge-m3 | `Forturne/bge-m3-FP8` | `Forturne/bge-m3-NVFP4` |

### Reranker Models

| Model | FP8 | NVFP4 |
|-------|-----|-------|
| Qwen3-VL-Reranker-2B | `Forturne/Qwen3-VL-Reranker-2B-FP8` | `Forturne/Qwen3-VL-Reranker-2B-NVFP4` |
| Qwen3-VL-Reranker-8B | `Forturne/Qwen3-VL-Reranker-8B-FP8` | `Forturne/Qwen3-VL-Reranker-8B-NVFP4` |
| Qwen3-Reranker-0.6B | `Forturne/Qwen3-Reranker-0.6B-FP8` | `Forturne/Qwen3-Reranker-0.6B-NVFP4` |
| Qwen3-Reranker-4B | `Forturne/Qwen3-Reranker-4B-FP8` | `Forturne/Qwen3-Reranker-4B-NVFP4` |
| Qwen3-Reranker-8B | `Forturne/Qwen3-Reranker-8B-FP8` | `Forturne/Qwen3-Reranker-8B-NVFP4` |
| bge-reranker-v2-m3 | `Forturne/bge-reranker-v2-m3-FP8` | `Forturne/bge-reranker-v2-m3-NVFP4` |

---

## Known Issues

### vLLM 0.16.0: Qwen3 Reranker FP8/NVFP4 비호환

Qwen3 리랭커는 `from_2_way_softmax` weight loading + `tie_word_embeddings=True` 조합으로 인해 FP8/NVFP4 양자화 추론이 불가합니다:
- FP8: `lm_head.weight` 미로드 → score = 0
- NVFP4: `ReplicatedLinear.weight` 속성 없음 → crash

**해결**: vLLM upstream 패치 대기. 현재는 BF16만 사용.

### XLM-RoBERTa NVFP4 품질 붕괴

bge-m3 / bge-reranker-v2-m3 (XLM-RoBERTa 기반)는 NVFP4 양자화 시 심각한 품질 붕괴:
- 임베딩 bge-m3: CosSim 0.54
- 리랭커 bge-reranker-v2-m3: Spearman 0.569

**해결**: XLM-RoBERTa 모델은 FP8만 사용 권장.

### Encoder 모델 양자화 시 classifier head 보존

리랭커 등 SequenceClassification 모델 양자화 시 `AutoModelForSequenceClassification`으로 로드해야 classifier head가 보존됩니다. `AutoModel`을 사용하면 classifier가 누락되어 score가 비정상 출력됩니다.

### MoE 모델 gate weight 경고

`shared_expert_gate.weight_packed not found` → `get_ignore_patterns()`로 자동 제외됨. 구버전 모델은 재양자화 필요.

---

## GPU Compatibility

| GPU | FP8 | INT4 (GPTQ/AWQ) | NVFP4 |
|-----|-----|-----------------|-------|
| **B200/GB200** (Blackwell) | 네이티브 W8A8 | Marlin 커널 | **네이티브 W4A4** |
| **H100/H200** (Hopper) | 네이티브 W8A8 | Marlin 커널 | W4A16 폴백 |
| **L40S/RTX 4090** (Ada) | 네이티브 W8A8 | Marlin 커널 | W4A16 폴백 |
| **L4** (Ada) | W8A16 폴백 | Marlin 커널 | W4A16 폴백 |
| **A100** (Ampere) | W8A16 폴백 | Marlin 커널 | W4A16 폴백 |

---

## References

- [llmcompressor](https://github.com/vllm-project/llm-compressor)
- [vLLM Documentation](https://docs.vllm.ai/)
- [NVFP4 Technical Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [GPTQ Paper (ICLR 2023)](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
