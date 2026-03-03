# vLLM Qwen3 Reranker Quantization Patch

**Date**: 2026-03-03
**PR**: [vllm-project/vllm#35849](https://github.com/vllm-project/vllm/pull/35849)
**Issue**: [vllm-project/vllm#33970](https://github.com/vllm-project/vllm/issues/33970)
**Bug Analysis**: [VLLM_QWEN3_RERANKER_QUANTIZATION_BUG.md](./VLLM_QWEN3_RERANKER_QUANTIZATION_BUG.md)

---

## 1. Problem

vLLM 0.16.0에서 Qwen3 Reranker 모델 5종을 FP8/NVFP4 양자화로 추론 시 실패:

| Precision | 증상 | Root Cause |
|-----------|------|------------|
| FP8 | 모든 score가 0.0 | score layer output_dim=1 → Marlin tile alignment(64) 위반 |
| NVFP4 | 모델 로드 시 crash | `weight_packed`만 등록, `.weight` 접근 시 `AttributeError` |

근본 원인: `as_seq_cls_model()`에서 score layer 생성 시 `quant_config`를 전달하지만, 이 layer는 checkpoint에 없는 동적 생성 분류 헤드(output_dim=1)로 양자화가 불가능하고 불필요하다.

## 2. Fix

파일: `vllm/model_executor/models/adapters.py` (1 file, +4/-7 lines)

### 2.1 Score layer — `quant_config=None` (핵심)

```diff
-            quant_config = vllm_config.quant_config
-
+            # Don't quantize: dynamic classification head, not in checkpoint
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

`quant_config=None`이면 `LinearBase`가 `UnquantizedLinearMethod()`를 사용하여 표준 `.weight` 파라미터를 등록한다. 이는 codebase에서 MoE gate/router layer 등 46곳에서 이미 사용하는 확립된 패턴이다.

### 2.2 임시 lm_head — `quant_config` 제거 (방어적)

`load_weights_using_from_2_way_softmax()`와 `load_weights_no_post_processing()` 2곳에서 임시 `ParallelLMHead` 생성 시 `quant_config`를 제거:

```diff
-    quant_config = model.vllm_config.quant_config
     ...
     language_model.lm_head = ParallelLMHead(
-        text_config.vocab_size, text_config.hidden_size, quant_config=quant_config
+        text_config.vocab_size, text_config.hidden_size,
     )
```

이 `lm_head`는 토큰 임베딩 추출 후 즉시 삭제(`del language_model.lm_head`)되는 임시 객체로, `.weight.data[token_ids]`에 직접 접근해야 하므로 양자화하면 안 된다.

## 3. Scope 판단

### 포함한 것
- `adapters.py` 내 3곳: score layer 1곳 + 임시 lm_head 2곳

### 제외한 것
- `nemotron_vl.py`: 동일 패턴이지만 이슈 #33970 범위 밖이고 검증되지 않아 제외

### 온라인/오프라인 양자화
- **온라인 FP8/NVFP4**: 직접 테스트 완료, 정상 동작
- **오프라인 (GPTQ/AWQ/FP8 static)**: score layer는 항상 동적 생성이므로 동일하게 동작. 임시 lm_head는 `ParallelLMHead` 기본값이 `quant_config=None`이라 실질적 차이 없음

## 4. Verification

### 4.1 FP8 Smoke Test

```
GPU: NVIDIA RTX PRO 6000 Blackwell (96 GB)
vLLM: 0.16.0, CUDA 13.0

Before (vanilla):
  doc[0] score = 0.000000
  doc[1] score = 0.000000
  doc[2] score = 0.000000

After (patched):
  doc[0] score = 0.972774  # Paris — capital of France
  doc[1] score = 0.896133  # Eiffel Tower — related
  doc[2] score = 0.705444  # Berlin — unrelated
```

### 4.2 Accuracy (BF16 ground truth, 100 query-doc pairs)

| Model | FP8 Spearman | FP8 Top-10 | NVFP4 Spearman | NVFP4 Top-10 |
|-------|-------------|-----------|---------------|-------------|
| Qwen3-VL-Reranker-2B | 0.994 | 100% | 0.950 | 90% |
| Qwen3-VL-Reranker-8B | 0.993 | 90% | 0.962 | 90% |
| Qwen3-Reranker-0.6B | 0.986 | 90% | 0.839 | 80% |
| Qwen3-Reranker-4B | 0.993 | 90% | 0.925 | 70% |
| Qwen3-Reranker-8B | 0.995 | 100% | 0.966 | 80% |

### 4.3 Latency (P50 ms, batch=16, seq_len=2048)

| Model | BF16 | FP8 | NVFP4 |
|-------|------|-----|-------|
| Qwen3-VL-Reranker-2B | 226.2 | 168.3 | 146.7 |
| Qwen3-VL-Reranker-8B | 768.6 | 496.1 | 376.8 |
| Qwen3-Reranker-0.6B | 131.6 | 112.3 | 103.9 |
| Qwen3-Reranker-4B | 470.0 | 323.8 | 269.8 |
| Qwen3-Reranker-8B | 757.1 | 485.1 | 367.5 |

## 5. How to Apply (Before PR Merge)

패치된 `adapters.py`를 설치된 vllm 패키지에 복사:

```bash
cp /root/vllm/vllm/model_executor/models/adapters.py \
   $(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")/model_executor/models/adapters.py
```

주의: `pip install vllm` 또는 `pip install --upgrade vllm` 시 덮어씌워진다.

## 6. PR Convention

- Title: `[Bugfix] Fix score layer quantization for sequence classification models`
- DCO: `Signed-off-by` 필수 (`git commit -s`)
- PR template: `## Purpose` / `## Test Plan` / `## Test Result`
