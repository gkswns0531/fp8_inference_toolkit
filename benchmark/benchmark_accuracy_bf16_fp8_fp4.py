#!/usr/bin/env python3
"""
BF16 vs FP8 vs NVFP4 Embedding Accuracy Verification

BF16을 ground truth로 FP8/NVFP4 임베딩 품질 오차를 측정합니다.

측정 항목:
  1. Embedding cosine similarity (BF16 emb vs quantized emb)
  2. Embedding MAE (mean absolute error)
  3. Query-document cosine similarity 차이

Usage:
    python benchmark_accuracy_bf16_fp8_fp4.py
"""

import gc
import json
import os
import time

import numpy as np


MODELS = [
    {
        "model_id": "Qwen/Qwen3-VL-Embedding-2B",
        "short": "VL-2B",
        "fp8_path": "/home/ubuntu/models/qwen3-vl-embedding-2b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-vl-embedding-2b-nvfp4",
    },
    {
        "model_id": "Qwen/Qwen3-VL-Embedding-8B",
        "short": "VL-8B",
        "fp8_path": "/home/ubuntu/models/qwen3-vl-embedding-8b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-vl-embedding-8b-nvfp4",
    },
    {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "short": "Emb-0.6B",
        "fp8_path": "/home/ubuntu/models/qwen3-embedding-0.6b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-embedding-0.6b-nvfp4",
    },
    {
        "model_id": "Qwen/Qwen3-Embedding-4B",
        "short": "Emb-4B",
        "fp8_path": "/home/ubuntu/models/qwen3-embedding-4b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-embedding-4b-nvfp4",
    },
    {
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "short": "Emb-8B",
        "fp8_path": "/home/ubuntu/models/qwen3-embedding-8b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-embedding-8b-nvfp4",
    },
    {
        "model_id": "BAAI/bge-m3",
        "short": "bge-m3",
        "fp8_path": "/home/ubuntu/models/bge-m3-fp8",
        "nvfp4_path": "/home/ubuntu/models/bge-m3-nvfp4",
    },
]

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
NUM_PAIRS = 100
LENGTHS = [128, 256, 512, 1024]


def load_source_text() -> str:
    path = os.path.join(TEST_DATA_DIR, "war_and_peace.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prepare_text_pairs(
    tokenizer, source_tokens: list[int], num_pairs: int, lengths: list[int], stride: int = 200,
) -> tuple[list[str], list[str]]:
    """Create num_pairs of (query, document) text pairs at varied lengths."""
    queries: list[str] = []
    documents: list[str] = []
    offset = 0
    for i in range(num_pairs):
        length = lengths[i % len(lengths)]
        # Query
        if offset + length > len(source_tokens):
            offset = 0
        q_tokens = source_tokens[offset:offset + length]
        q_text = tokenizer.decode(q_tokens, skip_special_tokens=True)
        queries.append(q_text)
        offset += stride
        # Document (different offset)
        if offset + length > len(source_tokens):
            offset = 0
        d_tokens = source_tokens[offset:offset + length]
        d_text = tokenizer.decode(d_tokens, skip_special_tokens=True)
        documents.append(d_text)
        offset += stride
    return queries, documents


def embed_texts(model_path: str, texts: list[str], quantization: str | None = None) -> np.ndarray:
    """Load model, embed texts, return embeddings as numpy array."""
    from vllm import LLM
    import torch

    kwargs = dict(
        model=model_path,
        runner="pooling",
        convert="embed",
        max_model_len=1536,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=False,
    )
    if quantization:
        kwargs["quantization"] = quantization

    llm = LLM(**kwargs)
    outputs = llm.embed(texts)
    embeddings = np.array([np.array(o.outputs.embedding) for o in outputs])

    del llm
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(2)

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_metrics(
    bf16_embs: np.ndarray, quant_embs: np.ndarray,
    bf16_q_embs: np.ndarray, bf16_d_embs: np.ndarray,
    quant_q_embs: np.ndarray, quant_d_embs: np.ndarray,
) -> dict:
    n = bf16_embs.shape[0]

    # 1. Embedding-level cosine similarity (BF16 emb vs quantized emb)
    emb_cos_sims = [cosine_similarity(bf16_embs[i], quant_embs[i]) for i in range(n)]

    # 2. MAE (element-wise)
    mae = float(np.mean(np.abs(bf16_embs - quant_embs)))
    max_ae = float(np.max(np.abs(bf16_embs - quant_embs)))

    # 3. Query-document cosine similarity difference
    n_pairs = bf16_q_embs.shape[0]
    bf16_qd_sims = [cosine_similarity(bf16_q_embs[i], bf16_d_embs[i]) for i in range(n_pairs)]
    quant_qd_sims = [cosine_similarity(quant_q_embs[i], quant_d_embs[i]) for i in range(n_pairs)]
    qd_sim_diffs = [abs(bf16_qd_sims[i] - quant_qd_sims[i]) for i in range(n_pairs)]

    return {
        "emb_cos_sim_mean": round(float(np.mean(emb_cos_sims)), 6),
        "emb_cos_sim_min": round(float(np.min(emb_cos_sims)), 6),
        "emb_cos_sim_p99": round(float(np.percentile(emb_cos_sims, 1)), 6),  # worst 1%
        "mae": round(mae, 8),
        "max_ae": round(max_ae, 6),
        "qd_sim_diff_mean": round(float(np.mean(qd_sim_diffs)), 6),
        "qd_sim_diff_max": round(float(np.max(qd_sim_diffs)), 6),
        "qd_sim_diff_p99": round(float(np.percentile(qd_sim_diffs, 99)), 6),
    }


def main() -> None:
    source_text = load_source_text()
    all_results: list[dict] = []

    for model_cfg in MODELS:
        model_id = model_cfg["model_id"]
        short = model_cfg["short"]
        print(f"\n{'=' * 70}")
        print(f" {short} ({model_id})")
        print(f"{'=' * 70}")

        # Tokenize source for this model
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        source_tokens = tokenizer.encode(source_text, add_special_tokens=False)

        queries, documents = prepare_text_pairs(tokenizer, source_tokens, NUM_PAIRS, LENGTHS)
        all_texts = queries + documents  # embed all at once

        print(f"  Prepared {NUM_PAIRS} query-doc pairs ({len(all_texts)} texts total)")

        # BF16 embeddings
        print(f"  [BF16] Embedding...")
        t0 = time.time()
        bf16_embs = embed_texts(model_id, all_texts)
        print(f"  [BF16] Done ({time.time() - t0:.1f}s), shape={bf16_embs.shape}")
        bf16_q = bf16_embs[:NUM_PAIRS]
        bf16_d = bf16_embs[NUM_PAIRS:]

        # FP8 embeddings
        print(f"  [FP8]  Embedding...")
        t0 = time.time()
        fp8_embs = embed_texts(model_cfg["fp8_path"], all_texts, quantization="compressed-tensors")
        print(f"  [FP8]  Done ({time.time() - t0:.1f}s)")
        fp8_q = fp8_embs[:NUM_PAIRS]
        fp8_d = fp8_embs[NUM_PAIRS:]

        # NVFP4 embeddings
        print(f"  [FP4]  Embedding...")
        t0 = time.time()
        fp4_embs = embed_texts(model_cfg["nvfp4_path"], all_texts, quantization="compressed-tensors")
        print(f"  [FP4]  Done ({time.time() - t0:.1f}s)")
        fp4_q = fp4_embs[:NUM_PAIRS]
        fp4_d = fp4_embs[NUM_PAIRS:]

        # Compute metrics
        fp8_metrics = compute_metrics(bf16_embs, fp8_embs, bf16_q, bf16_d, fp8_q, fp8_d)
        fp4_metrics = compute_metrics(bf16_embs, fp4_embs, bf16_q, bf16_d, fp4_q, fp4_d)

        result = {
            "model": model_id,
            "short": short,
            "num_pairs": NUM_PAIRS,
            "embedding_dim": int(bf16_embs.shape[1]),
            "fp8": fp8_metrics,
            "nvfp4": fp4_metrics,
        }
        all_results.append(result)

        # Print summary for this model
        print(f"\n  {'Metric':<25} {'FP8':>12} {'NVFP4':>12}")
        print(f"  {'─' * 50}")
        print(f"  {'Emb CosSim (mean)':<25} {fp8_metrics['emb_cos_sim_mean']:>12.6f} {fp4_metrics['emb_cos_sim_mean']:>12.6f}")
        print(f"  {'Emb CosSim (min)':<25} {fp8_metrics['emb_cos_sim_min']:>12.6f} {fp4_metrics['emb_cos_sim_min']:>12.6f}")
        print(f"  {'MAE':<25} {fp8_metrics['mae']:>12.8f} {fp4_metrics['mae']:>12.8f}")
        print(f"  {'QD Sim Diff (mean)':<25} {fp8_metrics['qd_sim_diff_mean']:>12.6f} {fp4_metrics['qd_sim_diff_mean']:>12.6f}")
        print(f"  {'QD Sim Diff (max)':<25} {fp8_metrics['qd_sim_diff_max']:>12.6f} {fp4_metrics['qd_sim_diff_max']:>12.6f}")

        del bf16_embs, fp8_embs, fp4_embs, tokenizer
        gc.collect()

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "embedding_accuracy_results.json",
    )
    with open(output_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": all_results}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
