# vLLM Qwen3 Reranker Quantization Bug Analysis

**Date**: 2025-06-25
**vLLM Version**: v0.16.1rc0 (commit 8fa68a8ce)
**Affected Models**: All Qwen3 Reranker / VL-Reranker models (FP8 & NVFP4)
**Status**: Fixed — PR [vllm-project/vllm#35849](https://github.com/vllm-project/vllm/pull/35849)

## Executive Summary

Qwen3 Reranker 모델 (text + VL)은 vLLM에서 FP8/NVFP4 양자화 추론 시 두 가지 별개의 버그로 실패한다:

| Precision | 증상 | Root Cause |
|-----------|------|------------|
| **FP8** | 모든 score가 0.0 반환 | `score` layer (output_dim=1)에 FP8 quantization 적용 → Marlin tile alignment 위반 |
| **NVFP4** | 모델 로드 시 crash | `score` layer의 NVFP4 scheme이 `weight_packed`만 등록 → `from_2_way_softmax`가 `.weight` 접근 시 `AttributeError` |

**근본 원인**: vLLM이 SequenceClassification score layer를 동적 생성할 때 `quant_config`를 전달하지만, 이 layer는 checkpoint에 존재하지 않는 1×hidden_size 크기의 작은 분류 헤드이므로 양자화가 불가능하고 불필요하다.

**제안 수정**: `score` layer 생성 시 `quant_config=None`으로 변경 (1줄 수정).

---

## 1. Background: Qwen3 Reranker Architecture in vLLM

### 1.1 Native vs vLLM Architecture

Qwen3 Reranker 모델은 원래 CausalLM (또는 VL) 아키텍처이다. vLLM은 `hf_overrides`를 통해 SequenceClassification으로 변환한다:

```python
# vLLM 로드 시 hf_overrides 설정
hf_overrides = {
    "architectures": ["Qwen3ForSequenceClassification"],  # or Qwen3VLForSequenceClassification
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}
llm = LLM(model="Qwen/Qwen3-Reranker-0.6B", hf_overrides=hf_overrides, runner="pooling")
```

### 1.2 `from_2_way_softmax` Weight Loading

Qwen3 Reranker의 checkpoint에는 `lm_head.weight`만 존재하고 `score.weight`는 없다. vLLM은 `from_2_way_softmax` 방식으로 score weight를 합성한다:

```
score.weight = lm_head[true_token_id] - lm_head[false_token_id]
```

여기서 `true_token_id`/`false_token_id`는 `classifier_from_token: ["no", "yes"]` 설정에서 "yes"/"no" 토큰의 vocabulary ID이다.

### 1.3 Score Layer 생성 경로

```
as_seq_cls_model()                      # adapters.py:260
  └─ _init_pooler()                     # adapters.py:286
       └─ self.score = ReplicatedLinear(  # adapters.py:295-303
              hidden_size, num_labels=1,
              quant_config=quant_config,   # ← BUG: should be None
          )

load_weights()                           # adapters.py:310
  └─ from_2_way_softmax()               # adapters.py:447-526
       ├─ Create temp ParallelLMHead     # line 471-484
       ├─ Tie to embed_tokens if needed  # line 474-484
       ├─ Load base model weights        # line 492
       ├─ Compute score = lm[true]-lm[false]  # line 505-508
       └─ score_layer.weight = score_weight    # line 510-513 ← FAILS
```

---

## 2. Bug #1: FP8 — Score가 항상 0.0

### 2.1 증상

```python
llm = LLM(model="path/to/fp8-model", quantization="compressed-tensors",
          hf_overrides=QWEN3_RERANKER_OVERRIDES, runner="pooling")
outputs = llm.score(["query"], ["document"])
# outputs[0].outputs.score == 0.0  (모든 입력에 대해)
```

### 2.2 Root Cause

`score` layer가 `ReplicatedLinear(hidden_size, num_labels=1, quant_config=quant_config)`로 생성된다.

FP8 scheme (`CompressedTensorsW8A8Fp8`)은 `create_weights()` 시 `.weight` 파라미터를 정상 등록하므로 (`compressed_tensors_w8a8_fp8.py:132`), `from_2_way_softmax`의 `score_layer.weight` 접근은 성공한다. 그러나:

1. **Marlin tile alignment 위반**: FP8 quantized Linear의 output_dim이 Marlin 커널의 tile size(64)로 나누어떨어져야 함. score layer의 output_dim=1은 이를 위반.
   - 에러: `size_n = 1 is not divisible by tile_n_size = 64`
   - 이로 인해 Marlin repack이 실패하고, fallback으로 잘못된 zero 텐서가 사용됨

2. **`tie_word_embeddings=True` 상호작용**: Qwen3 모델은 `tie_word_embeddings=True`이므로 checkpoint에 별도 `lm_head.weight`가 없음. `from_2_way_softmax`는 `embed_tokens.weight`를 tie하여 lm_head를 복원하지만, 이 과정에서 float32→FP8 변환이 추가 오차를 발생시킬 수 있음.

### 2.3 관련 GitHub Issue

- **[vllm-project/vllm#33970](https://github.com/vllm-project/vllm/issues/33970)** (OPEN)
  - Title: "Qwen3-Reranker-8B fp8 fails vllm score"
  - 동일 증상: `size_n = 1 is not divisible by tile_n_size = 64`
  - `from_2_way_softmax` → score layer의 Marlin repack 실패
  - 해결 PR 없음 (2025년 6월 기준)

### 2.4 Code References

| File | Line | Description |
|------|------|-------------|
| `vllm/model_executor/models/adapters.py` | 295-303 | `score = ReplicatedLinear(..., quant_config=quant_config)` — quant_config 전달 |
| `vllm/model_executor/models/adapters.py` | 505-513 | `from_2_way_softmax`에서 score weight 합성 및 할당 |
| `vllm/.../compressed_tensors_w8a8_fp8.py` | 132 | FP8: `layer.register_parameter("weight", weight)` |

---

## 3. Bug #2: NVFP4 — 모델 로드 시 Crash

### 3.1 증상

```python
llm = LLM(model="path/to/nvfp4-model", quantization="compressed-tensors",
          hf_overrides=QWEN3_RERANKER_OVERRIDES, runner="pooling")
# AttributeError: 'ReplicatedLinear' object has no attribute 'weight'
```

### 3.2 Root Cause

NVFP4 scheme (`CompressedTensorsW4A4Fp4`, `CompressedTensorsW4A16Fp4`)은 `create_weights()` 시 `weight_packed`를 등록하고, `weight`는 `process_weights_after_loading()`에서만 생성한다:

```python
# create_weights() — during model init (BEFORE load_weights)
layer.register_parameter("weight_packed", weight)   # ← "weight_packed", NOT "weight"

# process_weights_after_loading() — AFTER load_weights completes
layer.weight = layer.weight_packed                   # ← too late!
del layer.weight_packed
```

실행 순서:
```
1. __init__()  → score = ReplicatedLinear(quant_config=nvfp4)
                  → create_weights() → registers "weight_packed"
2. load_weights() → from_2_way_softmax()
                     → score_layer.weight  ← AttributeError! (only "weight_packed" exists)
3. process_weights_after_loading()  ← never reached
```

### 3.3 Code References

| File | Line | Description |
|------|------|-------------|
| `vllm/.../compressed_tensors_w4a4_nvfp4.py` | 59 | `layer.register_parameter("weight_packed", weight)` |
| `vllm/.../compressed_tensors_w4a4_nvfp4.py` | 90 | `layer.weight = layer.weight_packed` (too late) |
| `vllm/.../compressed_tensors_w4a16_nvfp4.py` | 58 | Same: `weight_packed` |
| `vllm/.../compressed_tensors_w4a16_nvfp4.py` | 85 | Same: `layer.weight = layer.weight_packed` |
| `vllm/model_executor/models/adapters.py` | 511 | `param = score_layer.weight` — crash point |

---

## 4. Proposed Fix

### 4.1 핵심 수정 (1줄)

`score` layer 생성 시 `quant_config=None`으로 변경. 이 layer는:
- Checkpoint에 존재하지 않는 동적 생성 layer
- `from_2_way_softmax`가 float32로 합성하는 weight
- output_dim=1인 작은 분류 헤드 — 양자화 불필요

```diff
--- a/vllm/model_executor/models/adapters.py
+++ b/vllm/model_executor/models/adapters.py
@@ -295,7 +295,7 @@
             self.score = ReplicatedLinear(
                 model_config.get_hidden_size(),
                 text_config.num_labels,
                 bias=False,
                 params_dtype=vllm_config.model_config.head_dtype,
-                quant_config=quant_config,
+                quant_config=None,
                 return_bias=False,
                 prefix=maybe_prefix(prefix, "score"),
             )
```

### 4.2 추가 개선 (선택)

`from_2_way_softmax` 내 임시 `ParallelLMHead`도 양자화할 필요 없음 (삭제 예정이므로):

```diff
@@ -471,7 +471,7 @@
-    language_model.lm_head = ParallelLMHead(
-        text_config.vocab_size, text_config.hidden_size, quant_config=quant_config
+    language_model.lm_head = ParallelLMHead(
+        text_config.vocab_size, text_config.hidden_size, quant_config=None
     )
```

단, `ParallelLMHead`는 `VocabParallelEmbedding` 기반으로 quantization method 선택 시 `LinearBase`가 아니어서 자동으로 unquantized로 처리되므로, 이 변경은 안전장치 성격이다.

### 4.3 Fix 효과

| Precision | Before | After |
|-----------|--------|-------|
| FP8 | score=0.0 (Marlin tile error) | score layer unquantized → 정상 출력 |
| NVFP4 | AttributeError crash | score layer unquantized → 정상 출력 |
| BF16 | 정상 | 영향 없음 (quant_config=None이면 기존과 동일) |

---

## 5. Related GitHub Issues & PRs

### 5.1 직접 관련

| # | Title | Status | Relevance |
|---|-------|--------|-----------|
| [#33970](https://github.com/vllm-project/vllm/issues/33970) | Qwen3-Reranker-8B fp8 fails vllm score | **OPEN** | **Primary** — FP8 score layer Marlin tile alignment |
| [#19260](https://github.com/vllm-project/vllm/pull/19260) | Support Qwen3 Embedding & Reranker models | Merged | Original PR adding `from_2_way_softmax` support |

### 5.2 간접 관련 (이전 수정)

| # | Title | Status | Relevance |
|---|-------|--------|-----------|
| [#20670](https://github.com/vllm-project/vllm/pull/20670) | Fix `from_2_way_softmax` with TP | Merged | Tensor parallelism 환경 수정 |
| [#20682](https://github.com/vllm-project/vllm/pull/20682) | Fix `from_2_way_softmax` weight access | Merged | TP에서 weight 접근 수정 |
| [#32086](https://github.com/vllm-project/vllm/pull/32086) | Fix VL reranker loading | Merged | VL 모델 로딩 경로 수정 |
| [#32089](https://github.com/vllm-project/vllm/pull/32089) | Fix VL reranker tokenizer | Merged | VL 모델 tokenizer 수정 |
| [#31563](https://github.com/vllm-project/vllm/pull/31563) | SentenceTransformers V6 reranker config | Open (draft) | 향후 reranker config 표준화 |

### 5.3 미해결 상태

- #33970 외에 FP8/NVFP4 reranker 양자화 문제를 다루는 PR은 없음
- `quant_config=None` 수정을 제안하는 PR 없음

---

## 6. Workaround

vLLM 수정 전까지 사용할 수 있는 우회 방법:

### 6.1 BF16 추론 사용 (권장)

```python
llm = LLM(model="Qwen/Qwen3-Reranker-0.6B",
          hf_overrides=QWEN3_RERANKER_OVERRIDES, runner="pooling")
```

### 6.2 Encoder-only Reranker (bge-reranker-v2-m3)

bge-reranker-v2-m3는 XLMRobertaForSequenceClassification으로, `from_2_way_softmax`를 사용하지 않아 FP8 정상 동작 (단, classifier head는 양자화에서 제외 필요):

```python
llm = LLM(model="path/to/bge-reranker-v2-m3-fp8",
          quantization="compressed-tensors", runner="pooling")
```

---

## 7. Affected Models

| Model | FP8 | NVFP4 | BF16 |
|-------|-----|-------|------|
| Qwen3-Reranker-0.6B | score=0 | crash | OK |
| Qwen3-Reranker-4B | score=0 | crash | OK |
| Qwen3-Reranker-8B | score=0 | crash | OK |
| Qwen3-VL-Reranker-2B | score=0 | crash | OK |
| Qwen3-VL-Reranker-8B | score=0 | crash | OK |
| bge-reranker-v2-m3 | **OK** | **FAIL** (품질) | OK |

Note: bge-reranker-v2-m3는 `from_2_way_softmax`를 사용하지 않으므로 FP8는 정상. NVFP4는 로드는 되나 정합성 품질이 현저히 낮음 (Spearman 0.57).
