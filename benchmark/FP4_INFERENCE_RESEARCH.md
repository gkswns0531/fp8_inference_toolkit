# FP4/NVFP4 Inference Feasibility Research

**Date**: 2026-03-02
**Target GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (GB202, sm_120, 96 GB)
**Engine**: vLLM 0.16.0

## Summary

FP4 임베딩 모델 추론은 RTX PRO 6000 Blackwell에서 **기술적으로 가능**하나, FP8과 달리 **사전 양자화(pre-quantization)가 필수**이다.

## 1. vLLM 0.16.0 FP4 지원 현황

vLLM 0.16.0은 다음 FP4 관련 양자화 방식을 지원한다:

| Method | Format | 설명 |
|--------|--------|------|
| `compressed-tensors` | llm-compressor NVFP4 | 표준 경로. W4A4 또는 W4A16 |
| `modelopt_fp4` | NVIDIA ModelOpt | TensorRT-LLM 배포 대상 |
| `mxfp4` | Microscaling FP4 | v0.14.0+, weight-only, 실험적 |
| `petit_nvfp4` | Petit backend | AMD GPU 대상, CUDA 미지원 |

**권장 경로**: llm-compressor로 NVFP4 사전 양자화 → `quantization="compressed-tensors"`로 로드

## 2. FP8 vs FP4 비교

| 항목 | FP8 | FP4/NVFP4 |
|------|-----|-----------|
| On-the-fly 양자화 | **가능** (`--quantization fp8`) | **불가** |
| 사전 양자화 | 선택적 (정확도 개선) | **필수** |
| 캘리브레이션 데이터 | 불필요 (dynamic per-tensor scaling) | W4A4: 필요 / W4A16: 불필요 |
| 메모리 절감 | ~50% (vs BF16) | ~75% (vs BF16) |
| 양자화 도구 | vLLM 내장 | llm-compressor offline `oneshot()` |
| Tensor Core 가속 | FP8 TC (Hopper+) | FP4 TC (Blackwell only) |

### NVFP4 양자화 방식

- **W4A4 (NVFP4)**: Weight + Activation 모두 FP4. 캘리브레이션 필요. 최대 성능.
- **W4A16 (NVFP4A16)**: Weight만 FP4, Activation은 FP16. 캘리브레이션 불필요. 구현 간단.

## 3. HuggingFace NVFP4 임베딩 모델 존재 여부

| Model | NVFP4 체크포인트 | 비고 |
|-------|:---:|------|
| Qwen3-Embedding-0.6B | 없음 | GGUF만 존재 |
| Qwen3-Embedding-4B | 없음 | GGUF만 존재 |
| Qwen3-Embedding-8B | **있음** | `alexliap/Qwen3-Embedding-8B-NVFP4` |
| Qwen3-VL-Embedding-2B | 없음 | 양자화 버전 전무 |
| Qwen3-VL-Embedding-8B | 없음 | 양자화 버전 전무 |
| bge-m3 | 없음 | FP8 (`Forturne/bge-m3-FP8`) 존재 |

### alexliap/Qwen3-Embedding-8B-NVFP4 상세

- **양자화 도구**: llm-compressor
- **방식**: W4A4 NVFP4, `scheme="NVFP4"`, `targets="Linear"`
- **포맷**: compressed-tensors
- **캘리브레이션**: 100 samples (CohereLabs/aya_dataset Greek subset), max_seq_length=4096
- **vLLM 로드**: `LLM(model="...", task="embed", quantization="compressed-tensors")`
- **정확도 벤치마크**: 미공개

## 4. RTX PRO 6000 Blackwell 호환성

RTX PRO 6000 (GB202, sm_120)은 데이터센터 Blackwell(B200, sm_100)과 동일한 FP4 Tensor Core를 탑재한다.

### vLLM sm_120 지원 이력

| 버전 | 상태 |
|------|------|
| v0.10.2 | sm_120 미인식 — NVFP4 실패 |
| v0.14.0+ | PR #21309: CUTLASS NVFP4 sm_120 지원 추가 |
| v0.15.0 | PR #24968: NVFP4 MoE sm_120 지원 추가 |
| v0.16.0 | Dense 모델 정상 동작. MoE는 issue #33416 일부 엣지케이스 |

**Dense 임베딩 모델(Qwen3-Embedding, bge-m3 등)은 vLLM 0.16.0에서 sm_120 NVFP4 정상 지원.**

## 5. FP4 벤치마크 실행 방안

### Option A: 기존 NVFP4 체크포인트 사용 (8B만)

```bash
# alexliap/Qwen3-Embedding-8B-NVFP4 바로 벤치마크
python benchmark/benchmark_embedding_latency.py \
    --model alexliap/Qwen3-Embedding-8B-NVFP4
```

단점: 8B 모델 1개만 비교 가능

### Option B: 직접 NVFP4 양자화 후 벤치마크 (전체 모델)

```bash
# 1. 각 모델을 NVFP4로 양자화 (기존 nvfp4_quantize.py 활용)
python quantization/nvfp4_quantize.py \
    --model_id Qwen/Qwen3-Embedding-0.6B \
    --scheme NVFP4 \
    --num_calibration_samples 128

# 2. 양자화된 모델로 벤치마크
python benchmark/benchmark_embedding_latency.py \
    --model ./output/Qwen3-Embedding-0.6B-NVFP4
```

필요 작업:
- `nvfp4_quantize.py`가 임베딩 모델 지원하는지 검증
- 임베딩 워크로드에 적합한 캘리브레이션 데이터셋 선정
- `benchmark_embedding_latency.py`에 `quantization="compressed-tensors"` 옵션 추가

### Option C: W4A16 weight-only (캘리브레이션 불필요)

```python
from llmcompressor.modifiers.quantization import QuantizationModifier
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16")
```

장점: 캘리브레이션 불필요, 정확도 손실 최소
단점: W4A4 대비 성능 이점 제한적

## 6. 알려진 제약사항

1. **Pooling + 양자화 호환성**: vLLM issue #33970에서 reranker(score) 모델의 FP8 양자화가 출력 차원 정렬 문제로 실패. 단, 임베딩 모델은 고차원 출력(1024-4096d)이므로 해당 없음.

2. **품질 영향 미검증**: NVFP4 양자화 후 임베딩 품질(cosine similarity, MTEB 점수)에 대한 공개 벤치마크 없음. BF16 대비 cosine similarity 측정 필수.

3. **캘리브레이션 데이터 민감성**: W4A4 양자화 시 캘리브레이션 데이터가 임베딩 워크로드를 대표해야 정확도 유지. 범용 텍스트(wikitext, c4) vs 도메인 특화 텍스트 비교 필요.

## References

- [vLLM Quantization Documentation](https://docs.vllm.ai/en/latest/features/quantization/)
- [LLM Compressor FP4 Guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)
- [NVIDIA NVFP4 Technical Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [Red Hat NVFP4 Article](https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization)
- [alexliap/Qwen3-Embedding-8B-NVFP4](https://huggingface.co/alexliap/Qwen3-Embedding-8B-NVFP4)
- [vLLM SM120 NVFP4 Support PR #21309](https://github.com/vllm-project/vllm/pull/21309)
- [vLLM Pooling + Quantization Issue #33970](https://github.com/vllm-project/vllm/issues/33970)
