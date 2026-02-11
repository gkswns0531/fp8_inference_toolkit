#!/bin/bash
# 4-Model FP8 Benchmark via vLLM Direct (Sequential)
set -o pipefail

PORT=8000
RESULT_FILE="/home/ubuntu/fp8_inference_toolkit/benchmark/4model_fp8_benchmark_results.json"
BENCHMARK_SCRIPT="/home/ubuntu/fp8_inference_toolkit/benchmark/benchmark_client.py"

# Model configs: name|model_path|gpu_mem
MODELS=(
    "Qwen3-Embedding-0.6B-FP8|Forturne/Qwen3-Embedding-0.6B-FP8|0.80"
    "bge-m3-FP8|Forturne/bge-m3-FP8|0.80"
    "Qwen3-VL-Embedding-2B-FP8|Forturne/Qwen3-VL-Embedding-2B-FP8|0.80"
    "Qwen3-VL-Embedding-8B-FP8|Forturne/Qwen3-VL-Embedding-8B-FP8|0.85"
)

cleanup_port() {
    local pids
    pids=$(lsof -ti :${PORT} 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null
        sleep 5
    fi
}

wait_for_server() {
    local timeout=300
    local start=$SECONDS
    while (( SECONDS - start < timeout )); do
        if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

echo "======================================================================"
echo "4-Model FP8 Benchmark (vLLM Direct - Sequential)"
echo "======================================================================"
echo ""

ALL_RESULTS="[]"
IDX=0
TOTAL=${#MODELS[@]}

for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME MODEL_PATH GPU_MEM <<< "$entry"
    IDX=$((IDX + 1))

    echo "----------------------------------------------------------------------"
    echo "[${IDX}/${TOTAL}] ${NAME}"
    echo "  Model: ${MODEL_PATH}"
    echo "----------------------------------------------------------------------"

    # 1. Cleanup
    cleanup_port

    # 2. Start vLLM server
    echo -n "  Starting vLLM server... "
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --dtype auto \
        --max-model-len 4096 \
        --gpu-memory-utilization "${GPU_MEM}" \
        --trust-remote-code \
        --runner pooling \
        --host 0.0.0.0 \
        --port ${PORT} \
        --no-enable-prefix-caching \
        > "/tmp/vllm_bench_${NAME}.log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "TIMEOUT"
        echo "  Last log lines:"
        tail -5 "/tmp/vllm_bench_${NAME}.log" | sed 's/^/    /'
        kill -9 $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        cleanup_port
        sleep 5
        continue
    fi
    echo "READY (PID=${SERVER_PID})"

    # 3. Benchmark
    OUTPUT_JSON="/tmp/bench_${NAME}.json"
    python3 "${BENCHMARK_SCRIPT}" \
        --use-vllm-direct \
        --vllm-port ${PORT} \
        --vllm-model "${MODEL_PATH}" \
        --output "${OUTPUT_JSON}" 2>&1 | sed 's/^/  /'

    # 4. Stop server
    echo -n "  Stopping server... "
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    cleanup_port
    sleep 5
    echo "OK"
    echo ""
done

echo "======================================================================"
echo "ALL BENCHMARKS COMPLETE"
echo "======================================================================"
echo ""

# Merge results
echo "Individual results:"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME MODEL_PATH GPU_MEM <<< "$entry"
    OUTPUT_JSON="/tmp/bench_${NAME}.json"
    if [ -f "${OUTPUT_JSON}" ]; then
        echo ""
        echo "--- ${NAME} ---"
        python3 -c "
import json
with open('${OUTPUT_JSON}') as f:
    data = json.load(f)
for r in data.get('results', []):
    if r['success']:
        print(f\"  batch={r['batch_size']:>2}, tokens={r['target_tokens']:>4} -> \"
              f\"avg={r['avg_latency_ms']:.1f}ms  p50={r['p50_latency_ms']:.1f}ms  \"
              f\"p99={r['p99_latency_ms']:.1f}ms  throughput={r['throughput_tokens_per_sec']:.0f} tok/s\")
    else:
        print(f\"  batch={r['batch_size']:>2}, tokens={r['target_tokens']:>4} -> FAILED\")
"
    else
        echo ""
        echo "--- ${NAME} --- SKIPPED (no results)"
    fi
done

echo ""
echo "======================================================================"
