#!/usr/bin/env python3
"""
Reranker Retrieval Evaluation

VL-8B embedding → top-64 candidates → VL-8B Reranker (BF16/FP8/NVFP4) → @10 metrics

Pipeline:
1. VL-8B 임베딩으로 15개 데이터셋 top-64 후보 추출 (1회, 캐싱)
2. 각 precision별 reranker로 64 candidates 리랭킹 (3회: bf16/fp8/nvfp4, 캐싱)
3. candidate pool size 16/32/64에 대해 top-10 결과로 Coverage, PerfectMatch, NDCG, MRR 계산

Usage:
    python client_eval/run_reranker_eval.py
    python client_eval/run_reranker_eval.py --embedder-precision nvfp4
    python client_eval/run_reranker_eval.py --precisions bf16 fp8
    python client_eval/run_reranker_eval.py --no-cache
"""

import argparse
import gc
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CANDIDATE_KS = [16, 32, 64]          # candidate pool sizes to feed reranker
EVAL_TOP_K = 10                       # final evaluation cutoff (always top-10)
MAX_CANDIDATE_K = max(CANDIDATE_KS)   # retrieve this many from embedder (64)

MAX_MODEL_LEN_EMBED = 8192
MAX_MODEL_LEN_RERANK = 8192

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_CLIENT_DIR = ROOT_DIR / "dataset" / "client"
DATASET_DOMAIN_DIR = ROOT_DIR / "dataset" / "domain"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = OUTPUT_DIR / "cache"

DOMAIN_DATASETS: dict[str, str] = {
    "finance": "finance_eval_dataset.json",
    "hotpot": "hotpot_eval_dataset.json",
    "legal": "legal_eval_dataset.json",
    "patent": "patent_eval_dataset.json",
}

EMBEDDER_MODEL_ID = "Qwen/Qwen3-VL-Embedding-8B"
EMBEDDER_PRECISIONS: dict[str, dict[str, Any]] = {
    "bf16": {
        "path": "Qwen/Qwen3-VL-Embedding-8B",
        "quantization": None,
    },
    "nvfp4": {
        "path": "/home/ubuntu/models/qwen3-vl-embedding-8b-nvfp4",
        "quantization": "compressed-tensors",
    },
}

RERANKER_MODEL_ID = "Qwen/Qwen3-VL-Reranker-8B"
RERANKER_PRECISIONS: dict[str, dict[str, Any]] = {
    "bf16": {
        "path": "Qwen/Qwen3-VL-Reranker-8B",
        "quantization": None,
    },
    "fp8": {
        "path": "/home/ubuntu/models/qwen3-vl-reranker-8b-fp8",
        "quantization": "compressed-tensors",
    },
    "nvfp4": {
        "path": "/home/ubuntu/models/qwen3-vl-reranker-8b-nvfp4",
        "quantization": "compressed-tensors",
    },
}

QWEN3_VL_RERANKER_OVERRIDES: dict[str, Any] = {
    "architectures": ["Qwen3VLForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
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
    """Truncate texts that exceed max_tokens."""
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
def create_embedder(model_path: str, quantization: str | None = None):
    """Create vLLM LLM for embedding."""
    from vllm import LLM
    kwargs: dict[str, Any] = dict(
        model=model_path,
        runner="pooling",
        convert="embed",
        max_model_len=MAX_MODEL_LEN_EMBED,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=False,
    )
    if quantization:
        kwargs["quantization"] = quantization
    return LLM(**kwargs)


def embed_with_llm(llm, texts: list[str]) -> np.ndarray:
    """Embed texts. Returns (N, dim) numpy array."""
    outputs = llm.embed(texts)
    return np.array([np.array(o.outputs.embedding) for o in outputs])


def destroy_llm(llm) -> None:
    """Clean up vLLM instance and free GPU memory."""
    import torch
    del llm
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(2)


# ---------------------------------------------------------------------------
# vLLM reranker
# ---------------------------------------------------------------------------
def create_reranker(model_path: str, quantization: str | None = None):
    """Create vLLM LLM for reranking (score API)."""
    from vllm import LLM
    kwargs: dict[str, Any] = dict(
        model=model_path,
        runner="pooling",
        max_model_len=MAX_MODEL_LEN_RERANK,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=False,
        hf_overrides=QWEN3_VL_RERANKER_OVERRIDES,
    )
    if quantization:
        kwargs["quantization"] = quantization
    return LLM(**kwargs)


def rerank_candidates(
    reranker,
    query_texts: list[str],
    candidate_texts_per_query: list[list[str]],
    candidate_ids_per_query: list[list[str]],
) -> list[list[str]]:
    """Rerank candidates for all queries. Returns reranked chunk ID lists."""
    queries_flat: list[str] = []
    docs_flat: list[str] = []
    boundaries: list[int] = [0]

    for query, cand_texts in zip(query_texts, candidate_texts_per_query):
        for doc in cand_texts:
            queries_flat.append(query)
            docs_flat.append(doc)
        boundaries.append(len(queries_flat))

    outputs = reranker.score(queries_flat, docs_flat)
    scores = [o.outputs.score for o in outputs]

    reranked: list[list[str]] = []
    for qi in range(len(query_texts)):
        start, end = boundaries[qi], boundaries[qi + 1]
        q_scores = scores[start:end]
        q_ids = candidate_ids_per_query[qi]
        sorted_pairs = sorted(zip(q_scores, q_ids), key=lambda x: x[0], reverse=True)
        reranked.append([cid for _, cid in sorted_pairs])

    return reranked


# ---------------------------------------------------------------------------
# Retrieval: top-K candidates via cosine similarity
# ---------------------------------------------------------------------------
def retrieve_top_k(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    corpus_ids: list[str],
    top_k: int,
) -> list[list[str]]:
    """Return top-K corpus IDs per query via cosine similarity."""
    q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    c_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-10)
    sim_matrix = q_norm @ c_norm.T

    results: list[list[str]] = []
    for i in range(sim_matrix.shape[0]):
        ranked_indices = np.argsort(sim_matrix[i])[::-1][:top_k]
        results.append([corpus_ids[idx] for idx in ranked_indices])
    return results


# ---------------------------------------------------------------------------
# Group-based metrics
# ---------------------------------------------------------------------------
def evaluate_query(
    retrieved: list[str],
    gold_groups: list[list[str]],
    top_k: int = EVAL_TOP_K,
) -> dict[str, float]:
    """Compute group-based metrics for a single query at a given top_k."""
    retrieved_top_k = retrieved[:top_k]
    retrieved_set = set(retrieved_top_k)
    num_groups = len(gold_groups)

    if num_groups == 0:
        return {"coverage": 0.0, "perfect_match": 0.0, "ndcg": 0.0, "mrr": 0.0}

    covered = sum(1 for group in gold_groups if any(cid in retrieved_set for cid in group))
    coverage = covered / num_groups
    perfect_match = 1.0 if coverage == 1.0 else 0.0

    dcg = 0.0
    for group in gold_groups:
        for rank_idx, cid in enumerate(retrieved_top_k):
            if cid in group:
                dcg += 1.0 / math.log2(rank_idx + 2)
                break
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_groups))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    rr_sum = 0.0
    for group in gold_groups:
        for rank_idx, cid in enumerate(retrieved_top_k):
            if cid in group:
                rr_sum += 1.0 / (rank_idx + 1)
                break
    mrr = rr_sum / num_groups

    return {"coverage": coverage, "perfect_match": perfect_match, "ndcg": ndcg, "mrr": mrr}


def evaluate_dataset(
    retrieved_per_query: list[list[str]],
    queries: list[dict],
    top_k: int = EVAL_TOP_K,
) -> dict[str, float]:
    """Compute average metrics over all queries at a single top_k."""
    all_metrics: list[dict[str, float]] = []
    for i, query in enumerate(queries):
        gold_groups = query["gold_chunk_groups"]
        all_metrics.append(evaluate_query(retrieved_per_query[i], gold_groups, top_k))

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
    client_path = DATASET_CLIENT_DIR / f"{name}.json"
    if client_path.exists():
        with open(client_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
# Phase 1: Embed and retrieve candidates (with caching)
# ---------------------------------------------------------------------------
def embed_and_retrieve(
    dataset_names: list[str],
    embedder_precision: str = "bf16",
    use_cache: bool = True,
) -> dict[str, dict[str, Any]]:
    """Embed all datasets and retrieve top-64 candidates."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"candidates_{embedder_precision}.pkl"

    # Try loading from cache
    if use_cache and cache_path.exists():
        print(f"\n  Phase 1: Loading candidates from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        # Verify all requested datasets are in cache
        if all(ds in cached for ds in dataset_names):
            print(f"  Cache hit: {len(dataset_names)} datasets loaded")
            return {ds: cached[ds] for ds in dataset_names}
        print("  Cache miss (incomplete), re-computing...")

    ecfg = EMBEDDER_PRECISIONS[embedder_precision]

    print(f"\n{'=' * 70}")
    print(f"  Phase 1: Embedding with VL-Embedding-8B ({embedder_precision.upper()})")
    print(f"  Path: {ecfg['path']}")
    print(f"  Retrieve top-{MAX_CANDIDATE_K} candidates per query")
    print(f"{'=' * 70}")

    print("  Loading embedder...")
    t_load = time.time()
    llm = create_embedder(ecfg["path"], ecfg["quantization"])
    print(f"  Embedder loaded in {time.time() - t_load:.1f}s")

    candidates: dict[str, dict[str, Any]] = {}

    for di, ds_name in enumerate(dataset_names, 1):
        print(f"\n  [{di}/{len(dataset_names)}] Dataset: {ds_name}")
        dataset = load_dataset(ds_name)
        queries = dataset["queries"]
        corpus = dataset["corpus"]
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["content"] for cid in corpus_ids]
        query_texts = [q["question"] for q in queries]

        print(f"    queries={len(query_texts)}, corpus={len(corpus_texts)}")

        # Truncate for embedding
        max_text_tokens = MAX_MODEL_LEN_EMBED - 200
        all_texts = query_texts + corpus_texts
        all_texts = truncate_texts(all_texts, EMBEDDER_MODEL_ID, max_text_tokens)

        print(f"    Embedding {len(all_texts)} texts...")
        t0 = time.time()
        all_embs = embed_with_llm(llm, all_texts)
        embed_time = time.time() - t0
        print(f"    Embedded in {embed_time:.1f}s")

        query_embs = all_embs[:len(query_texts)]
        corpus_embs = all_embs[len(query_texts):]

        # Retrieve top-64 candidates
        top_k_ids = retrieve_top_k(query_embs, corpus_embs, corpus_ids, MAX_CANDIDATE_K)

        # Build candidate texts for reranking (truncated to fit reranker context)
        max_doc_tokens = MAX_MODEL_LEN_RERANK - 500  # headroom for query + template
        raw_cand_texts: list[str] = []
        for q_top_ids in top_k_ids:
            raw_cand_texts.extend(corpus[cid]["content"] for cid in q_top_ids)
        truncated_cand_texts = truncate_texts(raw_cand_texts, EMBEDDER_MODEL_ID, max_doc_tokens)

        candidate_texts: list[list[str]] = []
        offset = 0
        for q_top_ids in top_k_ids:
            n = len(q_top_ids)
            candidate_texts.append(truncated_cand_texts[offset:offset + n])
            offset += n

        # Embed-only baseline: evaluate top-10 from embedder
        baseline_metrics = evaluate_dataset(top_k_ids, queries, EVAL_TOP_K)

        candidates[ds_name] = {
            "queries": queries,
            "query_texts": query_texts,
            "candidate_ids": top_k_ids,
            "candidate_texts": candidate_texts,
            "embed_time_sec": round(embed_time, 1),
            "num_queries": len(query_texts),
            "num_corpus": len(corpus_texts),
            "embed_only_metrics": {k: round(v, 6) for k, v in baseline_metrics.items()},
        }

        cov = baseline_metrics.get("coverage", 0) * 100
        pm = baseline_metrics.get("perfect_match", 0) * 100
        ndcg = baseline_metrics.get("ndcg", 0) * 100
        print(f"    Embed-only @{EVAL_TOP_K}: Cov={cov:.2f}% PM={pm:.2f}% NDCG={ndcg:.2f}%")

        del all_embs, query_embs, corpus_embs
        gc.collect()

    destroy_llm(llm)

    # Save cache
    with open(cache_path, "wb") as f:
        pickle.dump(candidates, f)
    print(f"\n  Candidates cached: {cache_path}")

    return candidates


# ---------------------------------------------------------------------------
# Phase 2: Rerank with reranker (with caching)
# ---------------------------------------------------------------------------
def rerank_and_evaluate(
    candidates: dict[str, dict[str, Any]],
    precision: str,
    embedder_precision: str = "bf16",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Rerank all datasets with a specific precision reranker.

    Reranks all 64 candidates once per precision, then evaluates @10
    for each candidate pool size (16, 32, 64).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"rerank_{embedder_precision}_{precision}.pkl"

    # Try loading from cache
    if use_cache and cache_path.exists():
        print(f"\n  Phase 2 [{precision.upper()}]: Loading from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cached_result = pickle.load(f)
        # Verify all datasets present
        if all(ds in cached_result["per_dataset"] for ds in candidates):
            print(f"  Cache hit: {precision.upper()} results loaded")
            return cached_result
        print("  Cache miss (incomplete), re-computing...")

    rcfg = RERANKER_PRECISIONS[precision]

    print(f"\n{'=' * 70}")
    print(f"  Phase 2: Reranking with VL-Reranker-8B ({precision.upper()})")
    print(f"  Path: {rcfg['path']}")
    print(f"  Candidate pools: {CANDIDATE_KS}, Eval: @{EVAL_TOP_K}")
    print(f"{'=' * 70}")

    print("  Loading reranker...")
    t_load = time.time()
    reranker = create_reranker(rcfg["path"], rcfg["quantization"])
    print(f"  Reranker loaded in {time.time() - t_load:.1f}s")

    per_dataset_results: dict[str, dict] = {}

    for di, (ds_name, cand_data) in enumerate(candidates.items(), 1):
        print(f"\n  [{di}/{len(candidates)}] Dataset: {ds_name}")
        n_queries = cand_data["num_queries"]
        print(f"    queries={n_queries}, candidates_per_query={MAX_CANDIDATE_K}, "
              f"total_pairs={n_queries * MAX_CANDIDATE_K}")

        t0 = time.time()
        reranked_ids = rerank_candidates(
            reranker,
            cand_data["query_texts"],
            cand_data["candidate_texts"],
            cand_data["candidate_ids"],
        )
        rerank_time = time.time() - t0
        print(f"    Reranked in {rerank_time:.1f}s")

        # Evaluate @10 for each candidate pool size
        cand_k_metrics: dict[int, dict[str, float]] = {}
        for cand_k in CANDIDATE_KS:
            # Filter reranked results: keep only IDs that were in the embedder's top-cand_k
            filtered_per_query: list[list[str]] = []
            for qi in range(n_queries):
                embed_top_k_set = set(cand_data["candidate_ids"][qi][:cand_k])
                filtered = [cid for cid in reranked_ids[qi] if cid in embed_top_k_set]
                filtered_per_query.append(filtered)

            metrics = evaluate_dataset(filtered_per_query, cand_data["queries"], EVAL_TOP_K)
            cand_k_metrics[cand_k] = {k: round(v, 6) for k, v in metrics.items()}

            cov = metrics.get("coverage", 0) * 100
            pm = metrics.get("perfect_match", 0) * 100
            ndcg = metrics.get("ndcg", 0) * 100
            print(f"    cand={cand_k:>2} @{EVAL_TOP_K}: Cov={cov:.2f}% PM={pm:.2f}% NDCG={ndcg:.2f}%")

        per_dataset_results[ds_name] = {
            "num_queries": cand_data["num_queries"],
            "num_corpus": cand_data["num_corpus"],
            "metrics_by_cand_k": cand_k_metrics,
            "rerank_time_sec": round(rerank_time, 1),
        }

    destroy_llm(reranker)

    # Compute overall average per cand_k
    overall_by_cand_k: dict[int, dict[str, float]] = {}
    for cand_k in CANDIDATE_KS:
        metric_keys = list(next(iter(per_dataset_results.values()))["metrics_by_cand_k"][cand_k].keys())
        avg: dict[str, float] = {}
        for key in metric_keys:
            vals = [per_dataset_results[d]["metrics_by_cand_k"][cand_k][key] for d in per_dataset_results]
            avg[key] = round(sum(vals) / len(vals), 6)
        overall_by_cand_k[cand_k] = avg

    result = {
        "reranker": RERANKER_MODEL_ID,
        "precision": precision,
        "per_dataset": per_dataset_results,
        "overall_by_cand_k": overall_by_cand_k,
    }

    # Save cache
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Results cached: {cache_path}")

    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(
    all_results: list[dict],
    candidates: dict[str, dict[str, Any]],
) -> None:
    """Print comparison table: embed-only vs reranker at each candidate pool size."""
    print(f"\n{'=' * 120}")
    print(f"RERANKER EVALUATION SUMMARY (All metrics @{EVAL_TOP_K})")
    print(f"Embedder: {EMBEDDER_MODEL_ID} (BF16) -> Reranker: {RERANKER_MODEL_ID}")
    print(f"Candidate pool sizes: {CANDIDATE_KS}")
    print(f"{'=' * 120}")

    dataset_names = list(candidates.keys())

    header = (f"  {'Dataset':<25}  {'Method':<18}  {'Coverage':>10}  {'PerfMatch':>10}"
              f"  {'NDCG':>10}  {'MRR':>10}")
    print(f"\n{header}")
    print(f"  {'-' * 95}")

    for ds in dataset_names:
        # Embed-only baseline
        em = candidates[ds]["embed_only_metrics"]
        print(f"  {ds:<25}  {'Embed @10':<18}"
              f"  {em.get('coverage', 0) * 100:>9.2f}%"
              f"  {em.get('perfect_match', 0) * 100:>9.2f}%"
              f"  {em.get('ndcg', 0) * 100:>9.2f}%"
              f"  {em.get('mrr', 0) * 100:>9.2f}%")

        # Reranker results per (precision, cand_k)
        for r in all_results:
            prec = r["precision"].upper()
            for cand_k in CANDIDATE_KS:
                m = r["per_dataset"][ds]["metrics_by_cand_k"][cand_k]
                label = f"{prec} cand={cand_k}"
                print(f"  {'':<25}  {label:<18}"
                      f"  {m.get('coverage', 0) * 100:>9.2f}%"
                      f"  {m.get('perfect_match', 0) * 100:>9.2f}%"
                      f"  {m.get('ndcg', 0) * 100:>9.2f}%"
                      f"  {m.get('mrr', 0) * 100:>9.2f}%")
        print()

    # Overall averages
    print(f"  {'-' * 95}")
    em_avg: dict[str, float] = {}
    for key in candidates[dataset_names[0]]["embed_only_metrics"]:
        vals = [candidates[d]["embed_only_metrics"][key] for d in dataset_names]
        em_avg[key] = sum(vals) / len(vals)

    print(f"  {'OVERALL':<25}  {'Embed @10':<18}"
          f"  {em_avg.get('coverage', 0) * 100:>9.2f}%"
          f"  {em_avg.get('perfect_match', 0) * 100:>9.2f}%"
          f"  {em_avg.get('ndcg', 0) * 100:>9.2f}%"
          f"  {em_avg.get('mrr', 0) * 100:>9.2f}%")

    for r in all_results:
        prec = r["precision"].upper()
        for cand_k in CANDIDATE_KS:
            m = r["overall_by_cand_k"][cand_k]
            label = f"{prec} cand={cand_k}"
            print(f"  {'':<25}  {label:<18}"
                  f"  {m.get('coverage', 0) * 100:>9.2f}%"
                  f"  {m.get('perfect_match', 0) * 100:>9.2f}%"
                  f"  {m.get('ndcg', 0) * 100:>9.2f}%"
                  f"  {m.get('mrr', 0) * 100:>9.2f}%")

    print(f"\n{'=' * 120}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Reranker Retrieval Evaluation")
    parser.add_argument("--embedder-precision", default="bf16",
                        choices=["bf16", "nvfp4"], help="Embedder precision")
    parser.add_argument("--precisions", nargs="+", default=["bf16", "fp8", "nvfp4"],
                        choices=["bf16", "fp8", "nvfp4"], help="Reranker precisions to evaluate")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names to evaluate (default: all 15)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cache and re-compute everything")
    args = parser.parse_args()

    emb_prec = args.embedder_precision
    use_cache = not args.no_cache
    dataset_names = args.datasets or list_all_datasets()
    print(f"Embedder: VL-Embedding-8B ({emb_prec.upper()})")
    print(f"Reranker precisions: {args.precisions}")
    print(f"Datasets ({len(dataset_names)}): {', '.join(dataset_names)}")
    print(f"Candidate pools: {CANDIDATE_KS}, Eval: @{EVAL_TOP_K}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")

    # Phase 1: Embed and retrieve candidates (cached per embedder precision)
    candidates = embed_and_retrieve(dataset_names, embedder_precision=emb_prec, use_cache=use_cache)

    # Phase 2: Rerank with each precision (cached per embedder+reranker precision)
    all_results: list[dict] = []
    for precision in args.precisions:
        result = rerank_and_evaluate(candidates, precision, embedder_precision=emb_prec, use_cache=use_cache)
        all_results.append(result)

    # Print summary
    print_summary(all_results, candidates)

    # Build embed-only data for JSON output
    embed_only_per_dataset: dict[str, dict] = {}
    for ds_name in dataset_names:
        embed_only_per_dataset[ds_name] = {
            "num_queries": candidates[ds_name]["num_queries"],
            "num_corpus": candidates[ds_name]["num_corpus"],
            "metrics": candidates[ds_name]["embed_only_metrics"],
            "embed_time_sec": candidates[ds_name]["embed_time_sec"],
        }

    embed_only_overall: dict[str, float] = {}
    for key in candidates[dataset_names[0]]["embed_only_metrics"]:
        vals = [candidates[d]["embed_only_metrics"][key] for d in dataset_names]
        embed_only_overall[key] = round(sum(vals) / len(vals), 6)

    # Convert cand_k int keys to string for JSON serialization
    serializable_results: list[dict] = []
    for r in all_results:
        sr = {
            "reranker": r["reranker"],
            "precision": r["precision"],
            "per_dataset": {},
            "overall_by_cand_k": {str(k): v for k, v in r["overall_by_cand_k"].items()},
        }
        for ds, ds_data in r["per_dataset"].items():
            sr["per_dataset"][ds] = {
                "num_queries": ds_data["num_queries"],
                "num_corpus": ds_data["num_corpus"],
                "metrics_by_cand_k": {str(k): v for k, v in ds_data["metrics_by_cand_k"].items()},
                "rerank_time_sec": ds_data["rerank_time_sec"],
            }
        serializable_results.append(sr)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"reranker_eval_results_{emb_prec}.json"

    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedder": EMBEDDER_MODEL_ID,
        "embedder_precision": emb_prec,
        "reranker": RERANKER_MODEL_ID,
        "candidate_ks": CANDIDATE_KS,
        "eval_top_k": EVAL_TOP_K,
        "embed_only": {
            "per_dataset": embed_only_per_dataset,
            "overall": embed_only_overall,
        },
        "results": serializable_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
