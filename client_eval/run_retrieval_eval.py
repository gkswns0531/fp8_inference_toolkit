#!/usr/bin/env python3
"""
Retrieval Evaluation — BF16 vs FP8 vs NVFP4

도메인(4) + 클라이언트(11) = 15개 데이터셋으로 임베딩 모델의 실제 검색 성능을 비교합니다.
vLLM으로 query/corpus 임베딩 → cosine similarity 검색 → group-based 메트릭 계산.

Usage:
    # VL-8B, 전체 15개 데이터셋
    python client_eval/run_retrieval_eval.py --models vl-8b

    # 특정 데이터셋만
    python client_eval/run_retrieval_eval.py --models vl-8b --datasets gugak finance

    # 특정 정밀도만
    python client_eval/run_retrieval_eval.py --models vl-8b --precisions bf16 fp8
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "vl-2b": {
        "model_id": "Qwen/Qwen3-VL-Embedding-2B",
        "bf16_path": "Qwen/Qwen3-VL-Embedding-2B",
        "fp8_path": "/home/ubuntu/models/qwen3-vl-embedding-2b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-vl-embedding-2b-nvfp4",
    },
    "vl-8b": {
        "model_id": "Qwen/Qwen3-VL-Embedding-8B",
        "bf16_path": "Qwen/Qwen3-VL-Embedding-8B",
        "fp8_path": "/home/ubuntu/models/qwen3-vl-embedding-8b-fp8",
        "nvfp4_path": "/home/ubuntu/models/qwen3-vl-embedding-8b-nvfp4",
    },
}

PRECISION_CONFIGS: dict[str, dict[str, Any]] = {
    "bf16": {"path_key": "bf16_path", "quantization": None},
    "fp8": {"path_key": "fp8_path", "quantization": "compressed-tensors"},
    "nvfp4": {"path_key": "nvfp4_path", "quantization": "compressed-tensors"},
}

TOP_K = 10
MAX_MODEL_LEN = 8192
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_CLIENT_DIR = ROOT_DIR / "dataset" / "client"
DATASET_DOMAIN_DIR = ROOT_DIR / "dataset" / "domain"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# Domain dataset name -> filename mapping
DOMAIN_DATASETS: dict[str, str] = {
    "finance": "finance_eval_dataset.json",
    "hotpot": "hotpot_eval_dataset.json",
    "legal": "legal_eval_dataset.json",
    "patent": "patent_eval_dataset.json",
}


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------
_tokenizer_cache: dict[str, Any] = {}


def get_tokenizer(model_path: str):
    """Get or create a cached tokenizer for truncation."""
    if model_path not in _tokenizer_cache:
        from transformers import AutoTokenizer
        _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
    return _tokenizer_cache[model_path]


def truncate_texts(texts: list[str], model_path: str, max_tokens: int) -> list[str]:
    """Truncate texts that exceed max_tokens. Returns list with same length."""
    tokenizer = get_tokenizer(model_path)
    truncated = 0
    result: list[str] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            text = tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
            truncated += 1
        result.append(text)
    if truncated:
        print(f"    Truncated {truncated}/{len(texts)} texts to {max_tokens} tokens")
    return result


# ---------------------------------------------------------------------------
# vLLM embedding
# ---------------------------------------------------------------------------
def create_llm(model_path: str, quantization: str | None = None):
    """Create a vLLM LLM instance for embedding."""
    from vllm import LLM

    kwargs: dict[str, Any] = dict(
        model=model_path,
        runner="pooling",
        convert="embed",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=False,
    )
    if quantization:
        kwargs["quantization"] = quantization

    return LLM(**kwargs)


def embed_with_llm(llm, texts: list[str]) -> np.ndarray:
    """Embed texts using an already-loaded LLM. Returns (N, dim) numpy array."""
    outputs = llm.embed(texts)
    return np.array([np.array(o.outputs.embedding) for o in outputs])


def destroy_llm(llm) -> None:
    """Clean up a vLLM instance and free GPU memory."""
    import torch

    del llm
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(2)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_full_ranking(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    corpus_ids: list[str],
) -> list[list[str]]:
    """Return full ranking of corpus IDs per query via cosine similarity."""
    q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    c_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-10)
    sim_matrix = q_norm @ c_norm.T

    results: list[list[str]] = []
    for i in range(sim_matrix.shape[0]):
        ranked_indices = np.argsort(sim_matrix[i])[::-1]
        results.append([corpus_ids[idx] for idx in ranked_indices])
    return results


# ---------------------------------------------------------------------------
# Group-based metrics (from DATASET_GUIDE.md)
# ---------------------------------------------------------------------------
def evaluate_query(
    retrieved: list[str],
    gold_groups: list[list[str]],
    top_k: int = TOP_K,
) -> dict[str, float]:
    """Compute group-based metrics for a single query."""
    retrieved_top_k = retrieved[:top_k]
    retrieved_set = set(retrieved_top_k)
    num_groups = len(gold_groups)

    if num_groups == 0:
        return {
            f"coverage@{top_k}": 0.0,
            f"perfect_match@{top_k}": 0.0,
            f"ndcg@{top_k}": 0.0,
            "mrr": 0.0,
        }

    # Coverage@K
    covered = sum(1 for group in gold_groups if any(cid in retrieved_set for cid in group))
    coverage = covered / num_groups

    # Perfect Match@K
    perfect_match = 1.0 if coverage == 1.0 else 0.0

    # NDCG@K (group-level)
    dcg = 0.0
    for group in gold_groups:
        for rank_idx, cid in enumerate(retrieved_top_k):
            if cid in group:
                dcg += 1.0 / math.log2(rank_idx + 2)
                break

    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_groups))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # MRR (group-level, full ranking)
    rr_sum = 0.0
    for group in gold_groups:
        for rank_idx, cid in enumerate(retrieved):
            if cid in group:
                rr_sum += 1.0 / (rank_idx + 1)
                break
    mrr = rr_sum / num_groups

    return {
        f"coverage@{top_k}": coverage,
        f"perfect_match@{top_k}": perfect_match,
        f"ndcg@{top_k}": ndcg,
        "mrr": mrr,
    }


def evaluate_dataset(
    retrieved_per_query: list[list[str]],
    queries: list[dict],
    top_k: int = TOP_K,
) -> dict[str, float]:
    """Compute average metrics over all queries in a dataset."""
    all_metrics: list[dict[str, float]] = []
    for i, query in enumerate(queries):
        gold_groups = query["gold_chunk_groups"]
        metrics = evaluate_query(retrieved_per_query[i], gold_groups, top_k)
        all_metrics.append(metrics)

    if not all_metrics:
        return {}

    avg: dict[str, float] = {}
    for key in all_metrics[0]:
        avg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    return avg


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(name: str) -> dict[str, Any]:
    """Load a dataset by name (client or domain)."""
    # Try client first
    client_path = DATASET_CLIENT_DIR / f"{name}.json"
    if client_path.exists():
        with open(client_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Try domain
    if name in DOMAIN_DATASETS:
        domain_path = DATASET_DOMAIN_DIR / DOMAIN_DATASETS[name]
        with open(domain_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError(f"Dataset '{name}' not found in client or domain directories")


def list_all_datasets() -> list[str]:
    """List all available dataset names (domain first, then client)."""
    domain_names = sorted(DOMAIN_DATASETS.keys())
    client_names = sorted(p.stem for p in DATASET_CLIENT_DIR.glob("*.json"))
    return domain_names + client_names


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_single_eval(
    model_key: str,
    precision: str,
    dataset_names: list[str],
) -> dict[str, Any]:
    """Run evaluation for one model+precision across all datasets."""
    mcfg = MODEL_CONFIGS[model_key]
    pcfg = PRECISION_CONFIGS[precision]
    model_path = mcfg[pcfg["path_key"]]
    quantization = pcfg["quantization"]

    # For truncation, use the original HF model id (tokenizer is always from bf16)
    tokenizer_path = mcfg["model_id"]

    print(f"\n{'=' * 70}")
    print(f"  Model: {mcfg['model_id']}  |  Precision: {precision.upper()}")
    print(f"  Path: {model_path}")
    print(f"{'=' * 70}")

    per_dataset_results: dict[str, dict] = {}

    # Load model once for all datasets
    print("  Loading model...")
    t_load = time.time()
    llm = create_llm(model_path, quantization)
    print(f"  Model loaded in {time.time() - t_load:.1f}s")

    for di, ds_name in enumerate(dataset_names, 1):
        print(f"\n  [{di}/{len(dataset_names)}] Dataset: {ds_name}")
        dataset = load_dataset(ds_name)
        queries = dataset["queries"]
        corpus = dataset["corpus"]
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["content"] for cid in corpus_ids]
        query_texts = [q["question"] for q in queries]

        print(f"    queries={len(query_texts)}, corpus={len(corpus_texts)}")

        # Truncate long texts
        # Leave headroom for special tokens (~100 tokens)
        max_text_tokens = MAX_MODEL_LEN - 200
        all_texts = query_texts + corpus_texts
        all_texts = truncate_texts(all_texts, tokenizer_path, max_text_tokens)

        print(f"    Embedding {len(all_texts)} texts...")
        t0 = time.time()
        all_embs = embed_with_llm(llm, all_texts)
        elapsed = time.time() - t0
        print(f"    Embedded in {elapsed:.1f}s")

        query_embs = all_embs[: len(query_texts)]
        corpus_embs = all_embs[len(query_texts) :]

        # Retrieve full ranking
        retrieved = retrieve_full_ranking(query_embs, corpus_embs, corpus_ids)

        # Evaluate
        metrics = evaluate_dataset(retrieved, queries, TOP_K)
        per_dataset_results[ds_name] = {
            "num_queries": len(query_texts),
            "num_corpus": len(corpus_texts),
            "metrics": {k: round(v, 6) for k, v in metrics.items()},
        }

        print(f"    Coverage@{TOP_K}: {metrics[f'coverage@{TOP_K}'] * 100:.2f}%"
              f"  Top@{TOP_K}: {metrics[f'perfect_match@{TOP_K}'] * 100:.2f}%"
              f"  NDCG@{TOP_K}: {metrics[f'ndcg@{TOP_K}'] * 100:.2f}%"
              f"  MRR: {metrics['mrr'] * 100:.2f}%")

        del all_embs, query_embs, corpus_embs
        gc.collect()

    # Release model
    destroy_llm(llm)

    # Compute overall average
    metric_keys = list(next(iter(per_dataset_results.values()))["metrics"].keys())
    avg_metrics: dict[str, float] = {}
    for key in metric_keys:
        vals = [per_dataset_results[d]["metrics"][key] for d in per_dataset_results]
        avg_metrics[key] = round(sum(vals) / len(vals), 6)

    return {
        "model": mcfg["model_id"],
        "model_key": model_key,
        "precision": precision,
        "per_dataset": per_dataset_results,
        "overall": avg_metrics,
    }


def print_summary(all_results: list[dict]) -> None:
    """Print a comparison table across all model+precision combos."""
    print(f"\n{'=' * 100}")
    print("RETRIEVAL EVALUATION SUMMARY")
    print(f"{'=' * 100}")

    models_seen: list[str] = []
    for r in all_results:
        if r["model_key"] not in models_seen:
            models_seen.append(r["model_key"])

    for model_key in models_seen:
        model_results = [r for r in all_results if r["model_key"] == model_key]
        model_id = model_results[0]["model"]
        print(f"\nModel: {model_id}")
        print("-" * 95)
        print(f"  {'Dataset':<25}  {'Prec':<6}  {'Coverage@10':>12}  {'Top@10':>10}  {'NDCG@10':>10}  {'MRR':>10}")
        print("-" * 95)

        dataset_names = list(model_results[0]["per_dataset"].keys())
        for ds in dataset_names:
            for ri, r in enumerate(model_results):
                prec = r["precision"].upper()
                m = r["per_dataset"][ds]["metrics"]
                label = ds if ri == 0 else ""
                print(f"  {label:<25}  {prec:<6}"
                      f"  {m[f'coverage@{TOP_K}'] * 100:>11.2f}%"
                      f"  {m[f'perfect_match@{TOP_K}'] * 100:>9.2f}%"
                      f"  {m[f'ndcg@{TOP_K}'] * 100:>9.2f}%"
                      f"  {m['mrr'] * 100:>9.2f}%")
            print()

        # Overall row
        print("-" * 95)
        for r in model_results:
            prec = r["precision"].upper()
            m = r["overall"]
            print(f"  {'OVERALL':<25}  {prec:<6}"
                  f"  {m[f'coverage@{TOP_K}'] * 100:>11.2f}%"
                  f"  {m[f'perfect_match@{TOP_K}'] * 100:>9.2f}%"
                  f"  {m[f'ndcg@{TOP_K}'] * 100:>9.2f}%"
                  f"  {m['mrr'] * 100:>9.2f}%")

    print(f"\n{'=' * 100}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Evaluation (BF16/FP8/NVFP4)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()), help="Models to evaluate")
    parser.add_argument("--precisions", nargs="+", default=["bf16", "fp8", "nvfp4"],
                        choices=["bf16", "fp8", "nvfp4"], help="Precisions to evaluate")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names to evaluate (default: all 15)")
    args = parser.parse_args()

    dataset_names = args.datasets or list_all_datasets()
    print(f"Models: {args.models}")
    print(f"Precisions: {args.precisions}")
    print(f"Datasets ({len(dataset_names)}): {', '.join(dataset_names)}")

    all_results: list[dict] = []

    for model_key in args.models:
        for precision in args.precisions:
            result = run_single_eval(model_key, precision, dataset_names)
            all_results.append(result)

    print_summary(all_results)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "retrieval_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "top_k": TOP_K, "results": all_results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
