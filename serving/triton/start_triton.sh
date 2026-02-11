#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Triton vLLM Backend Startup Script
#
# Qwen3-VL-Embedding 모델을 Triton + vLLM으로 서빙합니다.
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Configuration ──
CONTAINER_NAME="${CONTAINER_NAME:-triton_embedding}"
MODEL_REPO="${MODEL_REPO:-$(dirname "$0")/models}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VLLM_BACKEND_PATH="${VLLM_BACKEND_PATH:-/home/ubuntu/vllm_backend}"
TRITON_IMAGE="${TRITON_IMAGE:-tritonserver:25.01-vllm015}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Ports ──
HTTP_PORT="${HTTP_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"

# ── Options ──
UPGRADE_VLLM="${UPGRADE_VLLM:-false}"
APPLY_PATCH="${APPLY_PATCH:-false}"

echo "============================================================"
echo "Triton vLLM Embedding Server"
echo "============================================================"
echo "Container:  $CONTAINER_NAME"
echo "Model Repo: $MODEL_REPO"
echo "HF Cache:   $HF_CACHE"
echo "Image:      $TRITON_IMAGE"
echo "Ports:      HTTP=$HTTP_PORT, gRPC=$GRPC_PORT, Metrics=$METRICS_PORT"
echo "============================================================"

# ── Stop existing container ──
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# ── Build startup command ──
STARTUP_CMD=""

if [ "$UPGRADE_VLLM" = "true" ]; then
    STARTUP_CMD+="apt-get update -qq && apt-get install -y -qq libxcb1 libx11-6 > /dev/null 2>&1 && "
    STARTUP_CMD+="pip install --upgrade vllm transformers opencv-python-headless 2>&1 | tail -3 && "
fi

if [ "$APPLY_PATCH" = "true" ]; then
    STARTUP_CMD+="python3 /tmp/patch_vllm.py && "
fi

STARTUP_CMD+="tritonserver --model-repository=/models"

# ── Volume mounts ──
MOUNTS="-v ${MODEL_REPO}:/models"
MOUNTS+=" -v ${HF_CACHE}:/root/.cache/huggingface"

# Mount vLLM backend source if available
if [ -d "$VLLM_BACKEND_PATH" ]; then
    MOUNTS+=" -v ${VLLM_BACKEND_PATH}/src/model.py:/opt/tritonserver/backends/vllm/model.py:ro"
    MOUNTS+=" -v ${VLLM_BACKEND_PATH}/src/utils:/opt/tritonserver/backends/vllm/utils:ro"
fi

# Mount patch script
MOUNTS+=" -v ${SCRIPT_DIR}/patch_vllm.py:/tmp/patch_vllm.py:ro"

# ── Run container ──
echo "Starting Triton server..."

docker run -d --gpus all \
    --name "$CONTAINER_NAME" \
    --shm-size=16g \
    -p "${HTTP_PORT}:8000" \
    -p "${GRPC_PORT}:8001" \
    -p "${METRICS_PORT}:8002" \
    $MOUNTS \
    "$TRITON_IMAGE" \
    bash -c "$STARTUP_CMD"

echo ""
echo "Container started. Waiting for server..."
echo "View logs: docker logs -f $CONTAINER_NAME"
echo ""

# ── Wait for server ──
echo "Checking server status..."
for i in {1..60}; do
    if docker exec "$CONTAINER_NAME" curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 5
    echo "  Waiting... ($i/60)"
done

echo ""
echo "============================================================"
echo "Server Information"
echo "============================================================"
echo "gRPC endpoint: localhost:$GRPC_PORT"
echo "HTTP endpoint: localhost:$HTTP_PORT (Note: 501 for embedding - use gRPC)"
echo "Metrics:       localhost:$METRICS_PORT"
echo ""
echo "Test with:"
echo "  python3 -c \""
echo "  import tritonclient.grpc as grpcclient"
echo "  c = grpcclient.InferenceServerClient('localhost:$GRPC_PORT')"
echo "  print('Server ready:', c.is_server_ready())"
echo "  print('Model ready:', c.is_model_ready('qwen3_vl_embedding'))"
echo "  \""
echo "============================================================"
