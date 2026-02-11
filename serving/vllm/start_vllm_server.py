#!/usr/bin/env python3
"""
vLLM Embedding Server for Qwen3-VL-Embedding Models

Triton 없이 vLLM을 직접 OpenAI-compatible 서버로 실행합니다.
두 가지 모드 지원:
    1. openai: vLLM의 내장 OpenAI API 서버
    2. pooling: 커스텀 FastAPI 서버 (더 많은 제어 가능)

Requirements:
    pip install vllm uvicorn fastapi

Usage:
    # OpenAI mode (권장)
    python start_vllm_server.py --mode openai --model-path Qwen/Qwen3-VL-Embedding-8B

    # FP8 with OpenAI mode
    python start_vllm_server.py --mode openai --model-path Qwen/Qwen3-VL-Embedding-8B --quantization fp8

    # Pooling mode (커스텀)
    python start_vllm_server.py --mode pooling --model-path Qwen/Qwen3-VL-Embedding-2B

주의사항:
    1. VL 모델은 opencv-python-headless 설치 필요
    2. 8B 모델은 lm_head 패치 필요 (tie_word_embeddings=false)
    3. FP8 사용 시 GPU 메모리 ~40% 절약
"""

import argparse
import os
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_DTYPE = "bfloat16"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85


def format_input_to_conversation(text: str, instruction: str = "Represent the user's input.") -> List[Dict]:
    """Format input using Qwen3-VL chat template."""
    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]


def start_vllm_openai_server(args):
    """Start vLLM OpenAI-compatible server."""

    # Build command line arguments for vLLM server
    server_args = [
        "--model", args.model_path,
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--runner", "pooling",
        "--host", args.host,
        "--port", str(args.port),
    ]

    if args.tensor_parallel_size > 1:
        server_args.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])

    if args.quantization:
        server_args.extend(["--quantization", args.quantization])

    if args.kv_cache_dtype and args.kv_cache_dtype != "auto":
        server_args.extend(["--kv-cache-dtype", args.kv_cache_dtype])

    if args.no_enable_prefix_caching:
        server_args.append("--no-enable-prefix-caching")

    print("="*60)
    print("vLLM Embedding Server (OpenAI Mode)")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Dtype: {args.dtype}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Quantization: {args.quantization or 'None'}")
    print(f"Endpoint: http://{args.host}:{args.port}/v1/embeddings")
    print("="*60)

    # Execute vLLM server
    os.execvp("python", ["python", "-m", "vllm.entrypoints.openai.api_server"] + server_args)


def start_vllm_pooling_server(args):
    """Start vLLM pooling server with custom FastAPI endpoint."""
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional, Union
    import numpy as np
    from vllm import LLM, EngineArgs

    app = FastAPI(title="Qwen3-VL Embedding Server")

    # Initialize LLM engine
    print("Initializing vLLM pooling engine...")

    engine_args = EngineArgs(
        model=args.model_path,
        runner="pooling",
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization if args.quantization else None,
        kv_cache_dtype=args.kv_cache_dtype,
    )

    llm = LLM(**vars(engine_args))
    print("Engine initialized successfully!")

    class EmbeddingRequest(BaseModel):
        input: Union[str, List[str]]
        model: Optional[str] = None
        instruction: Optional[str] = "Represent the user's input."
        encoding_format: Optional[str] = "float"

    class EmbeddingResponse(BaseModel):
        object: str = "list"
        data: List[dict]
        model: str
        usage: dict

    def prepare_vllm_input(text: str, instruction: str) -> Dict:
        """Prepare input for vLLM pooling engine."""
        conversation = format_input_to_conversation(text, instruction)

        prompt = llm.llm_engine.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        return {"prompt": prompt, "multi_modal_data": None}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": args.model_path, "object": "model"}]
        }

    @app.post("/v1/embeddings")
    async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            # Handle single or batch input
            if isinstance(request.input, str):
                texts = [request.input]
            else:
                texts = request.input

            instruction = request.instruction or "Represent the user's input."

            # Prepare inputs
            vllm_inputs = [prepare_vllm_input(t, instruction) for t in texts]

            # Get embeddings
            outputs = llm.embed(vllm_inputs)

            # Format response
            data = []
            total_tokens = 0
            for i, output in enumerate(outputs):
                embedding = output.outputs.embedding
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                data.append({
                    "object": "embedding",
                    "embedding": embedding,
                    "index": i
                })
                # Approximate token count
                total_tokens += len(texts[i].split()) * 2

            return EmbeddingResponse(
                data=data,
                model=args.model_path,
                usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/embed")
    async def embed_direct(request: EmbeddingRequest):
        """Direct embedding endpoint (non-OpenAI format)."""
        return await create_embeddings(request)

    print(f"\nStarting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="vLLM Embedding Server")

    # Model configuration
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL,
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE,
                        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
                        help="Data type for model weights")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
                        help="Maximum sequence length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION,
                        help="GPU memory utilization (0.0 to 1.0)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size for multi-GPU")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=[None, "fp8", "awq", "gptq"],
                        help="Quantization method")
    parser.add_argument("--kv-cache-dtype", type=str, default="auto",
                        help="KV cache data type")
    parser.add_argument("--no-enable-prefix-caching", action="store_true",
                        help="Disable prefix caching for accurate benchmarking")

    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--mode", type=str, default="openai",
                        choices=["pooling", "openai"],
                        help="Server mode: openai (recommended) or pooling (custom)")

    args = parser.parse_args()

    print("="*60)
    print("Qwen3-VL Embedding Server")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Mode: {args.mode}")
    print(f"Dtype: {args.dtype}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Quantization: {args.quantization or 'None'}")
    print(f"Host: {args.host}:{args.port}")
    print("="*60)

    if args.mode == "openai":
        start_vllm_openai_server(args)
    else:
        start_vllm_pooling_server(args)


if __name__ == "__main__":
    main()
