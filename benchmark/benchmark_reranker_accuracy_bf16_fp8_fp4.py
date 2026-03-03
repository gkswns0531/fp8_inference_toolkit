#!/usr/bin/env python3
"""
BF16 vs FP8 vs NVFP4 Reranker Accuracy Verification

BF16을 ground truth로 FP8/NVFP4 리랭커 score 오차를 측정합니다.

측정 항목:
  1. Score MAE (Mean Absolute Error) / Max Diff
  2. Spearman Rank Correlation
  3. Top-10 / Top-20 Rank Overlap

Usage:
    python benchmark_reranker_accuracy_bf16_fp8_fp4.py
"""

import gc
import json
import os
import time

import numpy as np
from scipy import stats


QWEN3_RERANKER_OVERRIDES: dict = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}
QWEN3_VL_RERANKER_OVERRIDES: dict = {
    "architectures": ["Qwen3VLForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}

MODELS = [
    {
        "model_id": "Qwen/Qwen3-VL-Reranker-2B",
        "short": "VL-Reranker-2B",
        "fp8_path": "/home/ubuntu/models/qwen3-vl-reranker-2b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-vl-reranker-2b-nvfp4",
        "hf_overrides": QWEN3_VL_RERANKER_OVERRIDES,
    },
    {
        "model_id": "Qwen/Qwen3-VL-Reranker-8B",
        "short": "VL-Reranker-8B",
        "fp8_path": "/home/ubuntu/models/qwen3-vl-reranker-8b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-vl-reranker-8b-nvfp4",
        "hf_overrides": QWEN3_VL_RERANKER_OVERRIDES,
    },
    {
        "model_id": "Qwen/Qwen3-Reranker-0.6B",
        "short": "Reranker-0.6B",
        "fp8_path": "/home/ubuntu/models/qwen3-reranker-0.6b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-reranker-0.6b-nvfp4",
        "hf_overrides": QWEN3_RERANKER_OVERRIDES,
    },
    {
        "model_id": "Qwen/Qwen3-Reranker-4B",
        "short": "Reranker-4B",
        "fp8_path": "/home/ubuntu/models/qwen3-reranker-4b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-reranker-4b-nvfp4",
        "hf_overrides": QWEN3_RERANKER_OVERRIDES,
    },
    {
        "model_id": "Qwen/Qwen3-Reranker-8B",
        "short": "Reranker-8B",
        "fp8_path": "/home/ubuntu/models/qwen3-reranker-8b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-reranker-8b-nvfp4",
        "hf_overrides": QWEN3_RERANKER_OVERRIDES,
    },
    {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "short": "bge-reranker-v2-m3",
        "fp8_path": "/home/ubuntu/models/bge-reranker-v2-m3-fp8",
        "nvfp4_path": "/home/ubuntu/models/bge-reranker-v2-m3-nvfp4",
        "hf_overrides": None,
    },
]

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
NUM_PAIRS = 100
DOC_LENGTHS = [128, 256, 512, 1024]
QUERY_TOKEN_TARGET = 20


def load_source_text() -> str:
    path = os.path.join(TEST_DATA_DIR, "war_and_peace.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prepare_query_doc_pairs(
    tokenizer, source_tokens: list[int], num_pairs: int, doc_lengths: list[int], stride: int = 200,
) -> tuple[list[str], list[str]]:
    """Create num_pairs of (query, document) text pairs at varied document lengths."""
    queries: list[str] = []
    documents: list[str] = []
    offset = 0
    for i in range(num_pairs):
        doc_length = doc_lengths[i % len(doc_lengths)]
        # Query (~20 tokens)
        if offset + QUERY_TOKEN_TARGET > len(source_tokens):
            offset = 0
        q_tokens = source_tokens[offset:offset + QUERY_TOKEN_TARGET]
        q_text = tokenizer.decode(q_tokens, skip_special_tokens=True)
        queries.append(q_text)
        offset += stride
        # Document (variable length)
        if offset + doc_length > len(source_tokens):
            offset = 0
        d_tokens = source_tokens[offset:offset + doc_length]
        d_text = tokenizer.decode(d_tokens, skip_special_tokens=True)
        documents.append(d_text)
        offset += stride
    return queries, documents


def score_pairs(
    model_path: str,
    queries: list[str],
    documents: list[str],
    hf_overrides: dict | None = None,
    quantization: str | None = None,
) -> np.ndarray:
    """Load model, score (query, document) pairs, return scores as numpy array."""
    from vllm import LLM
    import torch

    kwargs: dict = dict(
        model=model_path,
        runner="pooling",
        max_model_len=1536,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=False,
    )
    if quantization:
        kwargs["quantization"] = quantization
    if hf_overrides:
        kwargs["hf_overrides"] = hf_overrides

    llm = LLM(**kwargs)
    outputs = llm.score(queries, documents)
    scores = np.array([o.outputs.score for o in outputs])

    del llm
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(2)

    return scores


def compute_metrics(bf16_scores: np.ndarray, quant_scores: np.ndarray) -> dict:
    """Compute accuracy metrics: MAE, Max Diff, Spearman, Top-K overlap."""
    n = len(bf16_scores)

    # 1. Score MAE / Max Diff
    diffs = np.abs(bf16_scores - quant_scores)
    mae = float(np.mean(diffs))
    max_diff = float(np.max(diffs))

    # 2. Spearman Rank Correlation
    spearman_corr, spearman_p = stats.spearmanr(bf16_scores, quant_scores)

    # 3. Top-K Rank Overlap
    bf16_ranking = np.argsort(-bf16_scores)  # descending
    quant_ranking = np.argsort(-quant_scores)

    top10_bf16 = set(bf16_ranking[:10])
    top10_quant = set(quant_ranking[:10])
    top10_overlap = len(top10_bf16 & top10_quant) / 10

    top20_bf16 = set(bf16_ranking[:20])
    top20_quant = set(quant_ranking[:20])
    top20_overlap = len(top20_bf16 & top20_quant) / 20

    return {
        "score_mae": round(mae, 6),
        "score_max_diff": round(max_diff, 6),
        "spearman_corr": round(float(spearman_corr), 6),
        "spearman_p": round(float(spearman_p), 8),
        "top10_overlap": round(top10_overlap, 2),
        "top20_overlap": round(top20_overlap, 2),
    }


def main() -> None:
    source_text = load_source_text()
    all_results: list[dict] = []

    for model_cfg in MODELS:
        model_id = model_cfg["model_id"]
        short = model_cfg["short"]
        hf_overrides = model_cfg.get("hf_overrides")
        print(f"\n{'=' * 70}")
        print(f" {short} ({model_id})")
        print(f"{'=' * 70}")

        # Tokenize source for this model
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        source_tokens = tokenizer.encode(source_text, add_special_tokens=False)

        queries, documents = prepare_query_doc_pairs(tokenizer, source_tokens, NUM_PAIRS, DOC_LENGTHS)
        print(f"  Prepared {NUM_PAIRS} query-doc pairs")

        # BF16 scores
        print(f"  [BF16] Scoring...")
        t0 = time.time()
        bf16_scores = score_pairs(model_id, queries, documents, hf_overrides=hf_overrides)
        print(f"  [BF16] Done ({time.time() - t0:.1f}s), scores shape={bf16_scores.shape}")

        # FP8 scores
        print(f"  [FP8]  Scoring...")
        t0 = time.time()
        fp8_scores = score_pairs(
            model_cfg["fp8_path"], queries, documents,
            hf_overrides=hf_overrides, quantization="compressed-tensors",
        )
        print(f"  [FP8]  Done ({time.time() - t0:.1f}s)")

        # NVFP4 scores
        fp4_metrics = None
        print(f"  [FP4]  Scoring...")
        t0 = time.time()
        try:
            fp4_scores = score_pairs(
                model_cfg["nvfp4_path"], queries, documents,
                hf_overrides=hf_overrides, quantization="compressed-tensors",
            )
            print(f"  [FP4]  Done ({time.time() - t0:.1f}s)")
            fp4_metrics = compute_metrics(bf16_scores, fp4_scores)
            del fp4_scores
        except Exception as e:
            print(f"  [FP4]  FAILED: {e}")

        # Compute metrics
        fp8_metrics = compute_metrics(bf16_scores, fp8_scores)

        result: dict = {
            "model": model_id,
            "short": short,
            "num_pairs": NUM_PAIRS,
            "fp8": fp8_metrics,
        }
        if fp4_metrics is not None:
            result["nvfp4"] = fp4_metrics
        else:
            result["nvfp4"] = "incompatible"
        all_results.append(result)

        # Print summary for this model
        if fp4_metrics:
            print(f"\n  {'Metric':<25} {'FP8':>12} {'NVFP4':>12}")
            print(f"  {'─' * 50}")
            print(f"  {'Score MAE':<25} {fp8_metrics['score_mae']:>12.6f} {fp4_metrics['score_mae']:>12.6f}")
            print(f"  {'Score Max Diff':<25} {fp8_metrics['score_max_diff']:>12.6f} {fp4_metrics['score_max_diff']:>12.6f}")
            print(f"  {'Spearman Corr':<25} {fp8_metrics['spearman_corr']:>12.6f} {fp4_metrics['spearman_corr']:>12.6f}")
            print(f"  {'Top-10 Overlap':<25} {fp8_metrics['top10_overlap']:>12.2f} {fp4_metrics['top10_overlap']:>12.2f}")
            print(f"  {'Top-20 Overlap':<25} {fp8_metrics['top20_overlap']:>12.2f} {fp4_metrics['top20_overlap']:>12.2f}")
        else:
            print(f"\n  {'Metric':<25} {'FP8':>12}")
            print(f"  {'─' * 38}")
            print(f"  {'Score MAE':<25} {fp8_metrics['score_mae']:>12.6f}")
            print(f"  {'Score Max Diff':<25} {fp8_metrics['score_max_diff']:>12.6f}")
            print(f"  {'Spearman Corr':<25} {fp8_metrics['spearman_corr']:>12.6f}")
            print(f"  {'Top-10 Overlap':<25} {fp8_metrics['top10_overlap']:>12.2f}")
            print(f"  {'Top-20 Overlap':<25} {fp8_metrics['top20_overlap']:>12.2f}")

        del bf16_scores, fp8_scores, tokenizer
        gc.collect()

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "reranker_accuracy_results.json",
    )
    with open(output_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": all_results}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
