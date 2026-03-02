# Embedding Model Latency Benchmark Report

**Date**: 2026-03-02
**GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB)
**Engine**: vLLM 0.16.0 (CUDA graphs enabled, chunked prefill)
**Methodology**: Exact token-count texts from War and Peace, different text per batch item (no prefix cache)

## Models Tested

| Model | Parameters | Type | Max Context |
|-------|-----------|------|-------------|
| Qwen3-Embedding-0.6B | 0.6B | Text Embedding | 32,768 |
| Qwen3-Embedding-4B | 4B | Text Embedding | 32,768 |
| Qwen3-Embedding-8B | 8B | Text Embedding | 32,768 |
| Qwen3-VL-Embedding-2B | 2B | Vision-Language Embedding | 32,768 |
| Qwen3-VL-Embedding-8B | 8B | Vision-Language Embedding | 32,768 |
| BAAI/bge-m3 | 0.6B | Text Embedding | 8,192 |

## Key Results (Batch=1, P50 Latency)

| Model | 128 tok | 512 tok | 1024 tok | 4096 tok | 8192 tok |
|-------|---------|---------|----------|----------|----------|
| Qwen3-Embedding-0.6B | 3.6 ms | 4.5 ms | 7.6 ms | 21.9 ms | 69.9 ms |
| Qwen3-Embedding-4B | 9.9 ms | 15.2 ms | 22.4 ms | 68.4 ms | 267.9 ms |
| Qwen3-Embedding-8B | 15.8 ms | 24.1 ms | 37.2 ms | 114.3 ms | 416.1 ms |
| Qwen3-VL-Embedding-2B | 5.0 ms | 7.4 ms | 10.7 ms | 36.6 ms | 118.6 ms |
| Qwen3-VL-Embedding-8B | 16.1 ms | 24.3 ms | 37.2 ms | 113.3 ms | 415.3 ms |
| bge-m3 | 2.5 ms | 4.5 ms | 6.0 ms | 18.8 ms | N/A |

## Key Results (Batch=16, P50 Latency)

| Model | 128 tok | 512 tok | 1024 tok | 4096 tok | 8192 tok |
|-------|---------|---------|----------|----------|----------|
| Qwen3-Embedding-0.6B | 15.8 ms | 33.9 ms | 64.6 ms | 278.2 ms | 1,059 ms |
| Qwen3-Embedding-4B | 62.1 ms | 116.9 ms | 233.1 ms | 1,814 ms | 4,210 ms |
| Qwen3-Embedding-8B | 105.3 ms | 194.8 ms | 369.7 ms | 2,981 ms | 6,526 ms |
| Qwen3-VL-Embedding-2B | 30.9 ms | 52.3 ms | 99.3 ms | 403.7 ms | 1,670 ms |
| Qwen3-VL-Embedding-8B | 115.6 ms | 199.9 ms | 361.5 ms | 2,900 ms | 6,362 ms |
| bge-m3 | 11.3 ms | 31.1 ms | 57.8 ms | 276.3 ms | N/A |

## Peak Throughput (tok/s, best batch size)

| Model | GPU Memory | Peak Throughput | Best Config |
|-------|-----------|----------------|-------------|
| Qwen3-Embedding-0.6B | 88,175 MiB | 324,950 tok/s | batch=16, 2048 tok |
| Qwen3-Embedding-4B | 87,549 MiB | 97,949 tok/s | batch=16, 1024 tok |
| Qwen3-Embedding-8B | 87,321 MiB | 62,626 tok/s | batch=16, 2048 tok |
| Qwen3-VL-Embedding-2B | 87,189 MiB | 222,481 tok/s | batch=16, 2048 tok |
| Qwen3-VL-Embedding-8B | 86,783 MiB | 65,303 tok/s | batch=16, 2048 tok |
| bge-m3 | 2,129 MiB | 281,525 tok/s | batch=16, 1024 tok |

## Observations

1. **Embedding-8B vs VL-Embedding-8B**: Nearly identical latency (same backbone size, ~415ms at batch=1/8192tok). VL variant has minimal overhead for text-only workloads.

2. **VL-Embedding-2B is efficient**: 2x faster than Embedding-4B despite being a VL model, making it the best quality/speed tradeoff in the VL lineup.

3. **bge-m3 baseline**: Fastest model with lowest memory footprint (2.1 GB), but limited to 8K context. Comparable latency to Qwen3-Embedding-0.6B.

4. **GPU memory**: All Qwen models show ~87-88 GB usage due to 90% GPU memory utilization setting — most is allocated for KV cache, not model weights.

5. **Batch efficiency**: All models show good batching efficiency. Throughput peaks at batch=16 with mid-range input lengths (1024-2048 tokens).

6. **High std on large batches**: Due to vLLM chunked prefill splitting batches that exceed `max_num_batched_tokens=16384`. P50 is more representative than avg for these cases.

## Benchmark Configuration

- Warmup runs: 3
- Timed runs: 10
- Batch sizes: 1, 4, 8, 16
- Input lengths: 128, 256, 512, 1024, 2048, 4096, 8192 tokens
- GPU memory utilization: 90%
- CUDA graphs: enabled (enforce_eager=False)
- Prefix caching: defeated (different text per batch item)
- Token count: exact (tokenizer-verified slicing from source text)
