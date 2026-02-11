#!/bin/bash
# FP8 vs BF16 Benchmark - 4 Models × 2 Precisions × 7 Token Lengths
# Batch size: 1 (fixed)
# Token lengths: 64, 128, 256, 512, 1024, 2048, 4096
set -o pipefail

PORT=8000
BENCHMARK_SCRIPT="/home/ubuntu/fp8_inference_toolkit/benchmark/benchmark_client.py"
RESULT_DIR="/home/ubuntu/fp8_inference_toolkit/benchmark/results"
mkdir -p "${RESULT_DIR}"

BATCH_SIZES="1"
TOKEN_LENGTHS="64,128,256,512,1024,2048,4096"

# Model configs: label|model_path|gpu_mem|max_model_len
MODELS=(
    # FP8
    "Qwen3-Embedding-0.6B-FP8|Forturne/Qwen3-Embedding-0.6B-FP8|0.80|4096"
    "bge-m3-FP8|Forturne/bge-m3-FP8|0.80|4096"
    "Qwen3-VL-Embedding-2B-FP8|Forturne/Qwen3-VL-Embedding-2B-FP8|0.80|4096"
    "Qwen3-VL-Embedding-8B-FP8|Forturne/Qwen3-VL-Embedding-8B-FP8|0.85|4096"
    # BF16
    "Qwen3-Embedding-0.6B-BF16|Qwen/Qwen3-Embedding-0.6B|0.80|4096"
    "bge-m3-BF16|BAAI/bge-m3|0.80|4096"
    "Qwen3-VL-Embedding-2B-BF16|Qwen/Qwen3-VL-Embedding-2B|0.80|4096"
    "Qwen3-VL-Embedding-8B-BF16|Qwen/Qwen3-VL-Embedding-8B|0.90|4096"
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
echo "FP8 vs BF16 Benchmark (Batch=1, Token Lengths: ${TOKEN_LENGTHS})"
echo "======================================================================"
echo ""

IDX=0
TOTAL=${#MODELS[@]}

for entry in "${MODELS[@]}"; do
    IFS='|' read -r LABEL MODEL_PATH GPU_MEM MAX_LEN <<< "$entry"
    IDX=$((IDX + 1))

    echo "----------------------------------------------------------------------"
    echo "[${IDX}/${TOTAL}] ${LABEL}"
    echo "  Model: ${MODEL_PATH}"
    echo "----------------------------------------------------------------------"

    # 1. Cleanup
    cleanup_port

    # 2. Start vLLM server
    echo -n "  Starting vLLM server... "
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --dtype auto \
        --max-model-len "${MAX_LEN}" \
        --gpu-memory-utilization "${GPU_MEM}" \
        --trust-remote-code \
        --runner pooling \
        --host 0.0.0.0 \
        --port ${PORT} \
        --no-enable-prefix-caching \
        > "/tmp/vllm_bench_${LABEL}.log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "TIMEOUT"
        echo "  Last log lines:"
        tail -5 "/tmp/vllm_bench_${LABEL}.log" 2>/dev/null | sed 's/^/    /'
        kill -9 $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        cleanup_port
        sleep 5
        continue
    fi
    echo "READY (PID=${SERVER_PID})"

    # 3. Benchmark
    OUTPUT_JSON="${RESULT_DIR}/${LABEL}.json"
    python3 "${BENCHMARK_SCRIPT}" \
        --use-vllm-direct \
        --vllm-port ${PORT} \
        --vllm-model "${MODEL_PATH}" \
        --tokenizer "${MODEL_PATH}" \
        --batch-sizes "${BATCH_SIZES}" \
        --token-lengths "${TOKEN_LENGTHS}" \
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

# ─── Summary Table ───────────────────────────────────────────────────
echo "======================================================================"
echo "SUMMARY TABLE"
echo "======================================================================"

python3 << 'PYEOF'
import json, glob, os

result_dir = "/home/ubuntu/fp8_inference_toolkit/benchmark/results"
files = sorted(glob.glob(os.path.join(result_dir, "*.json")))

all_results = {}
for f in files:
    label = os.path.splitext(os.path.basename(f))[0]
    with open(f) as fh:
        data = json.load(fh)
    for r in data.get("results", []):
        if r["success"]:
            key = (label, r["target_tokens"])
            all_results[key] = r["avg_latency_ms"]

# Group by base model
base_models = ["Qwen3-Embedding-0.6B", "bge-m3", "Qwen3-VL-Embedding-2B", "Qwen3-VL-Embedding-8B"]
token_lengths = [64, 128, 256, 512, 1024, 2048, 4096]

header = f"{'Model':<32} {'Prec':>5}"
for t in token_lengths:
    header += f" {t:>8}"
print(header)
print("─" * len(header))

for base in base_models:
    for prec in ["FP8", "BF16"]:
        label = f"{base}-{prec}"
        row = f"{label:<32} {prec:>5}"
        for t in token_lengths:
            val = all_results.get((label, t))
            if val is not None:
                row += f" {val:>7.1f}ms"  # changed from 8 to 7+ms
            else:
                row += f" {'FAIL':>8}"
        print(row)
    print()

# Speedup
print()
print("FP8 Speedup (BF16_ms / FP8_ms):")
print(f"{'Model':<32}", end="")
for t in token_lengths:
    print(f" {t:>8}", end="")
print()
print("─" * (32 + 9 * len(token_lengths)))

for base in base_models:
    row = f"{base:<32}"
    for t in token_lengths:
        fp8 = all_results.get((f"{base}-FP8", t))
        bf16 = all_results.get((f"{base}-BF16", t))
        if fp8 and bf16:
            row += f" {bf16/fp8:>7.2f}x"
        else:
            row += f" {'N/A':>8}"
    print(row)

PYEOF

echo ""
echo "======================================================================"
