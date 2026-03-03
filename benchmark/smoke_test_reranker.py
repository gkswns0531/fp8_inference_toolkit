#!/usr/bin/env python3
"""Smoke test: vLLM reranker score() + FP8 호환 검증."""

import gc
import time
import torch
from vllm import LLM

QUERY = "What is the capital of France?"
DOCUMENTS = [
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    "Berlin is the capital of Germany.",
]


def test_model(label: str, model_id: str, hf_overrides: dict | None = None, quantization: str | None = None) -> bool:
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {label}")
    print(f"  model={model_id}")
    if hf_overrides:
        print(f"  hf_overrides={hf_overrides}")
    if quantization:
        print(f"  quantization={quantization}")
    print(f"{'='*60}")

    try:
        kwargs: dict = dict(
            model=model_id,
            runner="pooling",
            max_model_len=1024,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            enforce_eager=False,
        )
        if hf_overrides:
            kwargs["hf_overrides"] = hf_overrides
        if quantization:
            kwargs["quantization"] = quantization

        llm = LLM(**kwargs)
        outputs = llm.score(QUERY, DOCUMENTS)

        print(f"  Results:")
        for i, o in enumerate(outputs):
            print(f"    doc[{i}] score = {o.outputs.score:.6f}")

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

        print(f"  PASSED")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return False


def main() -> None:
    results = {}

    # Test 1: BF16 bge-reranker-v2-m3 (native SequenceClassification)
    results["bge-reranker-v2-m3 BF16"] = test_model(
        "bge-reranker-v2-m3 (BF16)",
        "BAAI/bge-reranker-v2-m3",
    )

    # Test 2: BF16 Qwen3-Reranker-0.6B with hf_overrides
    results["Qwen3-Reranker-0.6B BF16"] = test_model(
        "Qwen3-Reranker-0.6B (BF16 + hf_overrides)",
        "Qwen/Qwen3-Reranker-0.6B",
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )

    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
