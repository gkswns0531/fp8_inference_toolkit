# FP8 Inference Toolkit

LLM/VLM 모델의 양자화(FP8, INT4 GPTQ/AWQ, NVFP4), Hugging Face 업로드, vLLM 추론을 위한 통합 툴킷입니다.

## Supported Quantization Methods

| 방식 | 스크립트 | 비트 | Calibration | GPU 요구 | 특징 |
|------|---------|------|-------------|---------|------|
| **FP8** (W8A8) | `fp8_quantize_all.py` | 8 | 불필요 | H100 네이티브 / L4 폴백 | 빠른 양자화, 최소 정확도 손실 |
| **INT4 GPTQ** (W4A16) | `int4_quantize_gptq.py` | 4 | 필요 (128+) | 대부분 GPU | Hessian 기반, 넓은 호환성 |
| **INT4 AWQ** (W4A16) | `int4_quantize_awq.py` | 4 | 필요 (128+) | 대부분 GPU | Activation-aware, 높은 품질 |
| **NVFP4** (W4A4) | `nvfp4_quantize.py` | 4.5 | 필요 (20+) | **Blackwell 네이티브** | INT4 대비 2.35x 빠름 (B200) |
| **NVFP4A16** (W4A16) | `nvfp4_quantize.py --scheme NVFP4A16` | 4.5 | 불필요 | Blackwell / 기타 Marlin 폴백 | Weight-only, 간편 |

## Directory Structure

```
fp8_inference_toolkit/
├── README.md
├── quantization/                        # 양자화 스크립트
│   ├── fp8_quantize_all.py             # FP8 (W8A8) 전체 레이어
│   ├── fp8_quantize_hf_standard.py     # FP8 auto_fp8 표준 포맷
│   ├── int4_quantize_gptq.py           # INT4 GPTQ (W4A16)
│   ├── int4_quantize_awq.py            # INT4 AWQ (W4A16)
│   ├── nvfp4_quantize.py              # NVFP4 (W4A4 / W4A16)
│   └── verify_int4_vllm.py            # INT4 모델 vLLM 검증
├── serving/                             # 추론 서빙
│   ├── triton/                          # Triton Inference Server
│   │   ├── models/
│   │   ├── patch_vllm.py
│   │   └── start_triton.sh
│   └── vllm/                            # vLLM Direct 서빙
│       └── start_vllm_server.py
└── benchmark/                           # 벤치마크
    ├── benchmark_client.py
    ├── benchmark_bf16_vs_fp8.py
    └── benchmark_embedding_comparison.py  # BF16/FP8/GPTQ/AWQ 비교
```

---

## Quick Start

### 1. 의존성 설치

```bash
pip install llmcompressor transformers huggingface_hub torch datasets vllm
```

### 2. HF 토큰 설정

```bash
export HF_TOKEN="your_token_here"
```

### 3. 양자화 실행

```bash
# FP8 (calibration 불필요, 가장 간편)
python quantization/fp8_quantize_all.py

# INT4 GPTQ (calibration 필요)
python quantization/int4_quantize_gptq.py --model Qwen/Qwen3-Next-80B-A3B-Instruct

# INT4 AWQ (calibration 필요)
python quantization/int4_quantize_awq.py --model Qwen/Qwen3-Next-80B-A3B-Instruct

# NVFP4 W4A4 (Blackwell GPU, calibration 필요)
python quantization/nvfp4_quantize.py --model Qwen/Qwen3-Next-80B-A3B-Instruct

# NVFP4 W4A16 (weight-only, calibration 불필요)
python quantization/nvfp4_quantize.py --model Qwen/Qwen3-Next-80B-A3B-Instruct --scheme NVFP4A16
```

각 스크립트가 양자화 → 로컬 저장 → HF 업로드 → 시간 측정까지 자동으로 수행합니다.

---

## Quantization Details

### FP8 (W8A8)

```bash
python quantization/fp8_quantize_all.py
```

- `FP8_DYNAMIC` scheme — calibration 불필요
- H100/H200에서 네이티브 FP8 Tensor Core 가속
- L4/A100에서는 W8A16 Marlin 커널로 폴백
- 메모리 BF16 대비 ~2x 절감, 정확도 손실 최소

### INT4 GPTQ (W4A16)

```bash
python quantization/int4_quantize_gptq.py \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --num-samples 128 \
  --max-seq-length 8192
```

- Hessian 기반 weight 재구성 — layer-by-layer 처리로 메모리 효율적
- `compressed-tensors` 포맷 → vLLM 자동 감지 (Marlin 커널 가속)
- 옵션: `--calibration-dataset`, `--group-size`, `--num-samples`

### INT4 AWQ (W4A16)

```bash
python quantization/int4_quantize_awq.py \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --num-samples 128
```

- Activation-aware weight quantization — 중요 채널 보존
- GPTQ 대비 높은 품질 보존 (~95% vs ~90%)
- 메모리 사용량 높음 (calibration 시 forward 입력 캐시)

### NVFP4 (W4A4 / W4A16) — Blackwell GPU

```bash
# W4A4: weight + activation 모두 FP4 (최대 성능)
python quantization/nvfp4_quantize.py

# W4A16: weight만 FP4 (calibration 불필요)
python quantization/nvfp4_quantize.py --scheme NVFP4A16
```

- E2M1 포맷 + 2단계 스케일링 (블록 FP8 + 텐서 FP32)
- B200에서 INT4 대비 **2.35x 빠른 추론** (dequantization 불필요)
- H100 이하에서는 W4A16 Marlin 커널로 자동 폴백

### MoE / Hybrid 모델 자동 처리

모든 양자화 스크립트가 MoE/Hybrid 모델을 자동 감지하여 민감한 레이어를 제외합니다:

| 자동 감지 대상 | 이유 | 예시 |
|---------------|------|------|
| MoE gate/router | 양자화 시 expert 라우팅 결정 붕괴 | `mlp.gate`, `block_sparse_moe.gate` |
| shared_expert_gate | shared/routed expert 혼합 비율 민감 | Qwen3-Next |
| DeltaNet (linear_attn) | recurrent state 오차 누적, conv1d 비호환 | Qwen3-Next |

---

## Inference Serving

### vLLM Direct (권장)

```bash
# FP8 모델
vllm serve Forturne/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --dtype auto --trust-remote-code --tensor-parallel-size 2

# GPTQ 모델 (자동 감지)
vllm serve Forturne/Qwen3-Next-80B-A3B-Instruct-INT4-GPTQ \
  --dtype auto --trust-remote-code

# NVFP4 모델 (Blackwell)
vllm serve Forturne/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --dtype auto --trust-remote-code --tensor-parallel-size 2

# 또는 이 프로젝트의 서빙 스크립트 사용
python serving/vllm/start_vllm_server.py \
  --model-path Forturne/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --mode openai --port 8000
```

### Triton + vLLM Backend

```bash
cd serving/triton
./start_triton.sh
```

> Triton vLLM Backend은 decoupled model로 동작 — HTTP는 501 에러, **gRPC streaming** 사용 필수

### Embedding 모델

Embedding 모델은 `--runner pooling` 추가:

```bash
vllm serve Forturne/Qwen3-Embedding-8B-FP8 \
  --runner pooling --dtype auto --trust-remote-code
```

---

## Benchmarking

### Embedding 모델 종합 비교 (BF16/FP8/GPTQ/AWQ)

```bash
python benchmark/benchmark_embedding_comparison.py \
  --base-model Qwen/Qwen3-Embedding-8B \
  --hf-username Forturne
```

- 배치 사이즈 (1,2,4,8,16) × 1024 토큰 레이턴시 측정
- BF16 대비 품질 비교 (cosine similarity, mean absolute diff, MSE)
- 매 run마다 랜덤 토큰 시퀀스 생성으로 캐시 방지
- 결과 JSON 자동 저장

### BF16 vs FP8 비교

```bash
python benchmark/benchmark_bf16_vs_fp8.py
```

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

## Known Issues & Solutions

### MoE 모델 gate weight 경고

**증상:** `shared_expert_gate.weight_packed not found in params_dict, skip loading`

**원인:** gate 레이어가 양자화된 상태로 저장됨

**해결:** 최신 스크립트는 `get_ignore_patterns()`로 자동 제외. 구버전 양자화 모델은 재양자화 필요.

### vLLM guided_json 무시

**증상:** `The following fields were present in the request but ignored: {'guided_json'}`

**원인:** vLLM API 변경. 양자화와 무관.

**해결:** `response_format={"type": "json_schema", "json_schema": {...}}` 사용

### lm_head.weight ValueError

**증상:** `ValueError: Following weights were not initialized from checkpoint: ['lm_head.weight']`

**해결:** `python serving/triton/patch_vllm.py`

### Triton HTTP 501

**원인:** vLLM Backend은 decoupled model — HTTP 미지원

**해결:** gRPC streaming 사용

---

## References

- [llmcompressor](https://github.com/vllm-project/llm-compressor)
- [vLLM Documentation](https://docs.vllm.ai/)
- [NVFP4 Technical Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [GPTQ Paper (ICLR 2023)](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [Qwen3-Next Architecture](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
