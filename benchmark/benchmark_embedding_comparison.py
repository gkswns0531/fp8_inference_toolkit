#!/usr/bin/env python3
"""
Embedding Model Quantization Benchmark

BF16, FP8, INT4-GPTQ, INT4-AWQ 4가지 양자화 방식의 임베딩 모델을 비교합니다.

측정 항목:
1. Latency: batch_size × seq_len 조합별 추론 시간
2. Quality: BF16 대비 cosine similarity, mean absolute difference

Usage:
    python benchmark_embedding_comparison.py \
        --base-model Qwen/Qwen3-Embedding-8B \
        --hf-username Forturne

    # FP8 v2 사용
    python benchmark_embedding_comparison.py \
        --base-model Qwen/Qwen3-Embedding-8B \
        --fp8-suffix FP8-v2
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Test Data Generation
# ─────────────────────────────────────────────────────────────────────────────

# 16개의 다양한 주제 텍스트 (batch=16 quality 비교용)
BASE_TEXTS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future.",
    "The theory of general relativity, published by Albert Einstein in 1915, describes gravity not as a force, but as a consequence of the curvature of spacetime caused by the uneven distribution of mass and energy. This theory has been confirmed by numerous experiments and observations, including the bending of light around massive objects and the detection of gravitational waves.",
    "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures.",
    "The human genome contains approximately 3 billion base pairs of DNA, organized into 23 pairs of chromosomes. The Human Genome Project, completed in 2003, identified all the genes in human DNA, determined the sequences of the chemical base pairs that make up human DNA, and stored this information in databases for further research and analysis.",
    "Quantum computing harnesses the phenomena of quantum mechanics to deliver a huge leap forward in computation to solve certain problems. Quantum computers use quantum bits, or qubits, which can exist in multiple states simultaneously through a property called superposition. This allows quantum computers to process a vast number of possibilities simultaneously.",
    "The Renaissance was a period of cultural, artistic, political and economic rebirth following the Middle Ages. Generally described as taking place from the 14th century to the 17th century, the Renaissance promoted the rediscovery of classical philosophy, literature and art. Some of the greatest thinkers, authors, statesmen, scientists and artists in human history thrived during this era.",
    "Blockchain technology is a decentralized, distributed ledger that records transactions across many computers. This technology ensures that records cannot be altered retroactively without the alteration of all subsequent blocks and the consensus of the network. Originally devised for Bitcoin, the technology has since been adapted for use in many different applications.",
    "Natural language processing is a field of computer science and artificial intelligence concerned with the interactions between computers and human language. NLP combines computational linguistics, machine learning, and deep learning models to process human language in the form of text or voice data and to understand its full meaning, complete with the speaker's intent and sentiment.",
    "The Amazon rainforest is the world's largest tropical rainforest, covering over 5.5 million square kilometers. It is home to an estimated 390 billion individual trees divided into 16,000 species. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world.",
    "Cybersecurity involves protecting computer systems and networks from information disclosure, theft, or damage to their hardware, software, or electronic data, as well as from the disruption or misdirection of the services they provide. The field is becoming increasingly significant due to the expanding reliance on computer systems, the Internet, and wireless network standards.",
    "The theory of evolution by natural selection, first formulated by Charles Darwin, is the process by which organisms change over time as a result of changes in heritable physical or behavioral traits. Changes that allow an organism to better adapt to its environment will help it survive and have more offspring, passing those beneficial traits to the next generation.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision and natural language processing.",
    "The International Space Station is a modular space station in low Earth orbit. It is a multinational collaborative project involving five participating space agencies: NASA, Roscosmos, JAXA, ESA, and CSA. The ISS serves as a microgravity and space environment research laboratory in which scientific research is conducted in astrobiology, astronomy, meteorology, physics, and other fields.",
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. Some of this chemical energy is stored in carbohydrate molecules, such as sugars and starches, which are synthesized from carbon dioxide and water.",
    "The global economy is increasingly driven by digital transformation, with artificial intelligence, cloud computing, and data analytics reshaping industries worldwide. Companies that embrace digital technologies are better positioned to adapt to changing market conditions, improve operational efficiency, and deliver enhanced customer experiences in an increasingly competitive landscape.",
    "한국어 자연어 처리는 한국어의 특수한 언어적 특성을 고려하여 설계되어야 합니다. 한국어는 교착어로서 어미 활용이 복잡하고, 조사에 의해 문법적 관계가 결정되며, 띄어쓰기가 영어와 다른 규칙을 따릅니다. 최근에는 BERT, GPT 계열의 대규모 언어 모델이 한국어 NLP 작업에서도 뛰어난 성능을 보여주고 있습니다.",
]


def generate_test_texts(tokenizer, target_seq_len: int, num_texts: int) -> list[str]:
    """주어진 토큰 길이에 맞는 테스트 텍스트를 생성합니다."""
    texts = []
    for i in range(num_texts):
        base = BASE_TEXTS[i % len(BASE_TEXTS)]
        # 텍스트를 반복하여 target_seq_len에 가까운 길이로 만듦
        repeated = (base + " ") * (target_seq_len // 50 + 1)
        tokens = tokenizer.encode(repeated, add_special_tokens=False)
        # target_seq_len - 2 (special tokens 여유)
        truncated_tokens = tokens[:target_seq_len - 2]
        text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        texts.append(text)
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_latency_benchmark(
    model_path: str,
    variant_name: str,
    test_texts: list[str],
    batch_sizes: list[int],
    max_model_len: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """단일 모델 변형에 대한 레이턴시 벤치마크를 실행합니다."""
    from vllm import LLM

    print(f"\n{'─'*60}")
    print(f"Loading {variant_name}: {model_path}")
    print(f"{'─'*60}")

    load_start = time.time()
    llm = LLM(
        model=model_path,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        runner="pooling",
    )
    load_time = time.time() - load_start

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"  Loaded in {load_time:.1f}s, VRAM: {mem_gb:.2f} GB")

    results = {"variant": variant_name, "model": model_path, "load_time": load_time}
    latencies = {}
    embeddings_16 = None

    for bs in batch_sizes:
        batch_texts = test_texts[:bs]
        label = f"{bs}x1024"

        # Warmup
        for _ in range(num_warmup):
            llm.embed(batch_texts)

        # Timed runs
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.time()
            outputs = llm.embed(batch_texts)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        p50_ms = np.percentile(times, 50) * 1000
        p99_ms = np.percentile(times, 99) * 1000

        latencies[label] = {
            "avg_ms": round(avg_ms, 2),
            "std_ms": round(std_ms, 2),
            "p50_ms": round(p50_ms, 2),
            "p99_ms": round(p99_ms, 2),
        }
        print(f"  {label}: avg={avg_ms:.2f}ms, std={std_ms:.2f}ms, "
              f"p50={p50_ms:.2f}ms, p99={p99_ms:.2f}ms")

        # batch=16일 때 임베딩 저장 (품질 비교용)
        if bs == max(batch_sizes):
            embeddings_16 = np.array([
                np.array(o.outputs.embedding) for o in outputs
            ])

    results["latencies"] = latencies
    results["embeddings"] = embeddings_16

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)  # GPU 메모리 해제 대기

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quality Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_quality(bf16_embs: np.ndarray, variant_embs: np.ndarray, name: str) -> dict:
    """BF16 대비 임베딩 품질을 비교합니다."""
    n = bf16_embs.shape[0]

    # Per-sample cosine similarity
    cos_sims = []
    for i in range(n):
        a, b = bf16_embs[i], variant_embs[i]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        cos_sims.append(cos_sim)

    # Mean absolute difference (element-wise)
    abs_diffs = np.abs(bf16_embs - variant_embs)
    mean_abs_diff = float(np.mean(abs_diffs))
    max_abs_diff = float(np.max(abs_diffs))

    # Mean squared error
    mse = float(np.mean((bf16_embs - variant_embs) ** 2))

    result = {
        "variant": name,
        "cos_sim_mean": round(float(np.mean(cos_sims)), 6),
        "cos_sim_min": round(float(np.min(cos_sims)), 6),
        "cos_sim_std": round(float(np.std(cos_sims)), 6),
        "mean_abs_diff": round(mean_abs_diff, 6),
        "max_abs_diff": round(max_abs_diff, 6),
        "mse": round(mse, 8),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Embedding Model Quantization Benchmark"
    )
    parser.add_argument(
        "--base-model", type=str, default="Qwen/Qwen3-Embedding-8B",
        help="Base model (BF16 original)",
    )
    parser.add_argument(
        "--hf-username", type=str, default="Forturne",
        help="HuggingFace username for quantized model repos",
    )
    parser.add_argument(
        "--fp8-suffix", type=str, default="FP8",
        help="FP8 model suffix (FP8 or FP8-v2)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=1536,
        help="Maximum model length for vLLM (default: 1536, enough for 1024 token inputs)",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num-runs", type=int, default=10,
        help="Number of timed runs per batch size",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--local-dir", type=str, default="/home/ubuntu/models",
        help="Local directory for quantized models",
    )

    args = parser.parse_args()

    model_name = args.base_model.split("/")[-1]
    username = args.hf_username
    batch_sizes = [1, 2, 4, 8, 16]

    # 모델 변형 정의
    # 로컬 경로가 있으면 로컬 사용, 없으면 HF repo 사용
    local_dir = args.local_dir
    variants = {}

    # BF16
    variants["BF16"] = args.base_model

    # FP8
    fp8_local = os.path.join(local_dir, f"{model_name.lower()}-fp8")
    fp8_hf = f"{username}/{model_name}-{args.fp8_suffix}"
    variants["FP8"] = fp8_local if os.path.isdir(fp8_local) else fp8_hf

    # INT4-GPTQ
    gptq_local = os.path.join(local_dir, f"{model_name.lower()}-int4-gptq")
    gptq_hf = f"{username}/{model_name}-INT4-GPTQ"
    variants["INT4-GPTQ"] = gptq_local if os.path.isdir(gptq_local) else gptq_hf

    # INT4-AWQ
    awq_local = os.path.join(local_dir, f"{model_name.lower()}-int4-awq")
    awq_hf = f"{username}/{model_name}-INT4-AWQ"
    variants["INT4-AWQ"] = awq_local if os.path.isdir(awq_local) else awq_hf

    print("=" * 70)
    print(f"Embedding Model Quantization Benchmark")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"Batch sizes: {batch_sizes} x seq_len=1024")
    print(f"Warmup: {args.num_warmup}, Runs: {args.num_runs}")
    print()
    for name, path in variants.items():
        exists = "LOCAL" if os.path.isdir(path) else "HF"
        print(f"  {name:12s}: {path} [{exists}]")
    print("=" * 70)

    # 테스트 텍스트 생성
    from transformers import AutoTokenizer
    print("\nGenerating test texts (target: 1024 tokens each)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    test_texts = generate_test_texts(tokenizer, target_seq_len=1024, num_texts=16)
    actual_lens = [len(tokenizer.encode(t)) for t in test_texts]
    print(f"  Generated {len(test_texts)} texts, token lengths: "
          f"min={min(actual_lens)}, max={max(actual_lens)}, avg={np.mean(actual_lens):.0f}")
    del tokenizer

    # 벤치마크 실행
    all_results = {}
    for name, path in variants.items():
        try:
            result = run_latency_benchmark(
                model_path=path,
                variant_name=name,
                test_texts=test_texts,
                batch_sizes=batch_sizes,
                max_model_len=args.max_model_len,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
            all_results[name] = result
        except Exception as e:
            print(f"\n  FAILED: {name} ({path})")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

    # ─────────────────────────────────────────────────────────────────────
    # 결과 출력
    # ─────────────────────────────────────────────────────────────────────

    print("\n")
    print("=" * 70)
    print("LATENCY RESULTS (ms)")
    print("=" * 70)

    # 테이블 헤더
    header = f"{'Variant':12s}"
    for bs in batch_sizes:
        header += f" | {bs}x1024:>10s"
    print(header)
    print("-" * 70)

    for name in ["BF16", "FP8", "INT4-GPTQ", "INT4-AWQ"]:
        if name not in all_results:
            continue
        row = f"{name:12s}"
        for bs in batch_sizes:
            label = f"{bs}x1024"
            lat = all_results[name]["latencies"].get(label, {})
            avg = lat.get("avg_ms", 0)
            row += f" | {avg:>10.2f}"
        print(row)

    # ─────────────────────────────────────────────────────────────────────
    # 품질 비교 (batch=16)
    # ─────────────────────────────────────────────────────────────────────

    print("\n")
    print("=" * 70)
    print("QUALITY COMPARISON vs BF16 (batch=16, seq_len=1024)")
    print("=" * 70)

    bf16_embs = all_results.get("BF16", {}).get("embeddings")
    quality_results = []

    if bf16_embs is not None:
        for name in ["FP8", "INT4-GPTQ", "INT4-AWQ"]:
            if name not in all_results:
                continue
            var_embs = all_results[name].get("embeddings")
            if var_embs is not None:
                qr = compare_quality(bf16_embs, var_embs, name)
                quality_results.append(qr)

        # 테이블 출력
        print(f"{'Variant':12s} | {'CosSim(mean)':>12s} | {'CosSim(min)':>12s} | "
              f"{'MeanAbsDiff':>12s} | {'MaxAbsDiff':>12s} | {'MSE':>12s}")
        print("-" * 85)
        for qr in quality_results:
            print(f"{qr['variant']:12s} | {qr['cos_sim_mean']:>12.6f} | "
                  f"{qr['cos_sim_min']:>12.6f} | {qr['mean_abs_diff']:>12.6f} | "
                  f"{qr['max_abs_diff']:>12.6f} | {qr['mse']:>12.8f}")
    else:
        print("  BF16 embeddings not available for comparison")

    # ─────────────────────────────────────────────────────────────────────
    # JSON 출력
    # ─────────────────────────────────────────────────────────────────────

    # embeddings는 JSON에서 제외 (너무 큼)
    json_results = {
        "base_model": args.base_model,
        "batch_sizes": [f"{bs}x1024" for bs in batch_sizes],
        "num_warmup": args.num_warmup,
        "num_runs": args.num_runs,
        "latencies": {},
        "quality": quality_results,
    }
    for name, res in all_results.items():
        json_results["latencies"][name] = {
            "model_path": res["model"],
            "load_time_s": round(res["load_time"], 1),
            "batch_latencies": res["latencies"],
        }

    output_path = args.output or os.path.join(
        os.path.dirname(__file__),
        f"embedding_benchmark_{model_name.lower()}.json"
    )
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
