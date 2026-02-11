# FP8 Inference Toolkit

LLM/VLM 모델의 FP8 양자화, Hugging Face 업로드, Triton + vLLM 추론을 위한 통합 툴킷입니다.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FP8 Inference Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────────────┐  │
│  │  Original Model │ ──▶ │ FP8 Quantized  │ ──▶ │   Hugging Face Hub    │  │
│  │   (BF16/FP16)   │     │   (llmcompressor)│     │   (Upload & Share)   │  │
│  └────────────────┘     └────────────────┘     └────────────────────────┘  │
│                                                           │                  │
│                                                           ▼                  │
│                              ┌────────────────────────────────────────────┐ │
│                              │          Inference Serving                 │ │
│                              │  ┌──────────────┐  ┌──────────────────┐   │ │
│                              │  │ Triton Server │  │   vLLM Direct    │   │ │
│                              │  │  + vLLM      │  │  (OpenAI API)    │   │ │
│                              │  │  Backend     │  │                  │   │ │
│                              │  └──────────────┘  └──────────────────┘   │ │
│                              └────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
fp8_inference_toolkit/
├── README.md                           # This file
├── quantization/                       # FP8 양자화 스크립트
│   ├── fp8_quantize_all.py            # 전체 레이어 FP8 (Vision 포함)
│   ├── fp8_quantize_llm_only.py       # LLM만 FP8, Vision은 BF16
│   └── fp8_quantize_hf_standard.py    # auto_fp8 표준 HF 포맷
├── serving/                            # 추론 서빙
│   ├── triton/                         # Triton Inference Server
│   │   ├── models/                     # Triton 모델 저장소
│   │   │   └── qwen3_vl_embedding/
│   │   │       ├── config.pbtxt
│   │   │       └── 1/model.json
│   │   ├── patch_vllm.py              # vLLM weight validation 패치
│   │   └── start_triton.sh            # Triton 시작 스크립트
│   └── vllm/                           # vLLM Direct 서빙
│       └── start_vllm_server.py
└── benchmark/                          # 벤치마크
    ├── benchmark_client.py            # Triton/vLLM 벤치마크
    └── benchmark_bf16_vs_fp8.py       # BF16 vs FP8 비교
```

---

## 1. FP8 Quantization (양자화)

### 1.1 llmcompressor로 전체 레이어 양자화

Vision Encoder를 포함한 모든 Linear 레이어를 FP8로 양자화합니다.

```bash
cd quantization
python fp8_quantize_all.py
```

**특징:**
- `compressed-tensors` 포맷으로 저장
- vLLM에서 자동 감지하여 FP8 모드로 로드
- HuggingFace Hub에 자동 업로드

**주요 코드:**
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["re:.*lm_head"],  # lm_head 제외
)
oneshot(model=model, recipe=recipe)
model.save_pretrained(output_dir, save_compressed=True)
```

### 1.2 LLM만 FP8 (Vision Encoder BF16 유지)

VLM에서 Vision 품질을 유지하면서 LLM 부분만 FP8로 양자화합니다.

```bash
python fp8_quantize_llm_only.py
```

**핵심 설정:**
```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",
        "re:visual.*",  # Vision Encoder 제외
    ]
)
```

### 1.3 Hugging Face 업로드

양자화된 모델을 HuggingFace Hub에 업로드:

```python
from huggingface_hub import HfApi, create_repo

api = HfApi(token=HF_TOKEN)
create_repo(repo_id, token=HF_TOKEN, exist_ok=True)
api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    commit_message=f"Upload FP8 quantized model"
)
```

---

## 2. Triton + vLLM Serving

### 2.1 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Triton Inference Server                   │
│  ┌────────────────────────────────────────────────────┐  │
│  │              vLLM Backend (official)               │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │          vLLM Pooling Engine                 │  │  │
│  │  │      (AsyncLLMEngine, runner=pooling)        │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Ports:  8000 (HTTP)  8001 (gRPC)  8002 (Metrics)       │
└──────────────────────────────────────────────────────────┘
                        ▲
                        │ gRPC Streaming (REQUIRED!)
                        │ (embedding_request)
              ┌─────────┴─────────┐
              │   Client (gRPC)   │
              └───────────────────┘
```

> **IMPORTANT**: vLLM Backend은 **decoupled model**로 동작합니다.
> HTTP API는 501 에러를 반환하므로 반드시 **gRPC streaming**을 사용해야 합니다.

### 2.2 Triton 모델 설정

**config.pbtxt** (변경 불필요):
```protobuf
backend: "vllm"

instance_group [
    {
        kind: KIND_MODEL
    }
]
```

**model.json** (모델별 수정):
```json
{
    "model": "Qwen/Qwen3-VL-Embedding-8B",
    "runner": "pooling",
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "trust_remote_code": true,
    "quantization": "fp8"
}
```

| 파라미터 | 필수 | 설명 |
|---|---|---|
| `model` | O | HuggingFace model ID 또는 로컬 경로 |
| `runner` | O | `"pooling"` — embedding 모델 필수 |
| `dtype` | - | `"bfloat16"`, `"float16"`, `"auto"` |
| `quantization` | - | `"fp8"` (동적 양자화), `null` (BF16) |
| `max_model_len` | - | 최대 시퀀스 길이 |
| `gpu_memory_utilization` | - | GPU 메모리 사용률 (0.0~1.0) |
| `trust_remote_code` | - | HF 커스텀 코드 허용 |

### 2.3 Triton Server 실행

```bash
cd serving/triton
chmod +x start_triton.sh
./start_triton.sh
```

또는 직접 Docker 실행:

```bash
docker run -d --gpus all \
  --name triton_embedding \
  --shm-size=16g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/patch_vllm.py:/tmp/patch_vllm.py:ro \
  tritonserver:25.01-vllm015 \
  bash -c 'tritonserver --model-repository=/models'
```

### 2.4 gRPC Client Example

```python
import numpy as np
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:8001")

text = "Hello, this is a test sentence for embedding."

# 입력 텐서 생성
input_text = grpcclient.InferInput("embedding_request", [1], "BYTES")
input_text.set_data_from_numpy(np.array([text], dtype=object))

# Streaming 요청
results = []
client.start_stream(callback=lambda result, error: results.append((result, error)))
client.async_stream_infer(
    model_name="qwen3_vl_embedding",
    inputs=[input_text],
    request_id="1"
)
client.stop_stream()

# 결과 파싱
result, error = results[0]
output = result.as_numpy("embedding_output")
print(f"Embedding shape: {output.shape}")
```

---

## 3. vLLM Direct Serving

Triton 없이 vLLM을 직접 OpenAI-compatible 서버로 실행:

```bash
cd serving/vllm

# BF16
python start_vllm_server.py \
  --model-path Qwen/Qwen3-VL-Embedding-8B \
  --mode openai \
  --port 8000

# FP8
python start_vllm_server.py \
  --model-path Qwen/Qwen3-VL-Embedding-8B \
  --mode openai \
  --quantization fp8 \
  --port 8000
```

**Client Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "model": "Qwen/Qwen3-VL-Embedding-8B",
        "input": "Hello, world!"
    }
)
embedding = response.json()["data"][0]["embedding"]
```

---

## 4. Known Issues & Solutions

### Issue 1: lm_head.weight ValueError

**증상:**
```
ValueError: Following weights were not initialized from checkpoint: ['lm_head.weight']
```

**원인:** `tie_word_embeddings: false`인 모델에서 발생

**영향받는 모델:**
- Qwen3-Embedding-8B
- Qwen3-VL-Embedding-8B

**영향받지 않는 모델:**
- Qwen3-Embedding-0.6B, 4B
- Qwen3-VL-Embedding-2B

**해결:**
```bash
python serving/triton/patch_vllm.py
```

### Issue 2: HTTP 501 Error

**원인:** Triton vLLM Backend은 decoupled model로 HTTP 미지원

**해결:** gRPC streaming 사용 (위 클라이언트 예시 참조)

### Issue 3: VL 모델 opencv 의존성

**증상:** `ImportError: libxcb.so.1`

**해결:**
```bash
apt-get install -y libxcb1 libx11-6
pip install opencv-python-headless
```

### Issue 4: 8B FP8 Triton OOM (L4 22GB)

**원인:** Triton 오버헤드 (~248MiB) + FP8 weight loading으로 OOM

**해결:**
- L4에서는 vLLM Direct 사용
- A10G (24GB), A100 이상 GPU 사용

### Issue 5: max_model_len 초과로 hang

**원인:** EOS 토큰 자동 추가로 길이 초과

**해결:**
```python
OVERHEAD = 2  # EOS + safety margin
max_input_tokens = max_model_len - OVERHEAD
```

---

## 5. Benchmarking

### 단일 서버 벤치마크

```bash
cd benchmark

# Triton 벤치마크
python benchmark_client.py \
  --grpc-port 8001 \
  --model-name qwen3_vl_embedding \
  --output results_triton.json

# vLLM Direct 벤치마크
python benchmark_client.py \
  --use-vllm-direct \
  --vllm-port 8000 \
  --vllm-model Qwen/Qwen3-VL-Embedding-8B \
  --output results_vllm.json
```

### BF16 vs FP8 비교

```bash
python benchmark_bf16_vs_fp8.py
```

자동으로 BF16과 FP8 서버를 순차적으로 시작하고 벤치마크를 실행합니다.

### Expected Performance (vLLM, L4 GPU)

| Model | Dtype | Speedup | VRAM Saving |
|-------|-------|---------|-------------|
| VL-2B | FP8 vs BF16 | 1.3~1.4x | ~2GB |
| VL-8B | FP8 vs BF16 | 1.5~1.7x | ~4GB |

---

## 6. GPU Memory Requirements (L4 22GB)

| 모델 | BF16 Triton | BF16 vLLM | FP8 Triton | FP8 vLLM |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B | ~21GB | ~21GB | ~21GB | ~21GB |
| Qwen3-VL-Embedding-2B | ~20GB | ~21GB | ~20GB | ~20GB |
| Qwen3-VL-Embedding-8B | ~20GB | ~21GB | **OOM** | ~20GB |

> gpu_memory_utilization=0.9 기준 (KV cache 포함)

---

## 7. Quick Start

### 1. FP8 양자화 & 업로드
```bash
cd quantization
# HF_TOKEN 환경변수 설정 후
python fp8_quantize_all.py
```

### 2. vLLM Direct로 빠른 테스트
```bash
cd serving/vllm
python start_vllm_server.py \
  --model-path Forturne/Qwen3-VL-Embedding-2B-FP8 \
  --port 8000
```

### 3. 벤치마크
```bash
cd benchmark
python benchmark_client.py --use-vllm-direct --vllm-port 8000
```

---

## References

- [Qwen3-Embedding Models](https://huggingface.co/collections/Qwen/qwen3-embedding-67de357648730e6f573a8277)
- [llmcompressor](https://github.com/vllm-project/llm-compressor)
- [Triton vLLM Backend](https://github.com/triton-inference-server/vllm_backend)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM FP8 Guide](https://docs.vllm.ai/en/latest/quantization/fp8.html)
