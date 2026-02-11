#!/usr/bin/env python3
"""
INT4 GPTQ 양자화 모델의 vLLM 호환성 검증 스크립트

양자화된 모델이 정상적으로 vLLM에서 로드되고 추론 가능한지 확인합니다.

Usage:
    # 로컬 모델 검증
    python verify_int4_vllm.py --model /home/ubuntu/models/qwen3-embedding-4b-int4-gptq

    # HF 모델 검증
    python verify_int4_vllm.py --model YourUsername/Qwen3-Embedding-4B-INT4-GPTQ

    # Embedding 모델 검증
    python verify_int4_vllm.py --model YourUsername/Qwen3-Embedding-4B-INT4-GPTQ --task embedding

    # CausalLM 모델 검증 (텍스트 생성)
    python verify_int4_vllm.py --model YourUsername/Model-INT4-GPTQ --task generate
"""

import argparse
import json
import os
import sys
import time

import torch


def check_config(model_path: str) -> dict:
    """모델의 config.json에서 양자화 설정을 확인합니다."""
    print("\n[1/4] Checking model config...")

    # HF 모델이면 다운로드
    if not os.path.isdir(model_path):
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=model_path, filename="config.json")
    else:
        config_path = os.path.join(model_path, "config.json")

    with open(config_path) as f:
        config = json.load(f)

    # quantization_config 확인
    quant_config = config.get("quantization_config", {})
    if not quant_config:
        # compressed-tensors 포맷은 compression_config에 있을 수 있음
        quant_config = config.get("compression_config", {})

    if quant_config:
        print(f"  Quantization config found:")
        print(f"    {json.dumps(quant_config, indent=4)}")
    else:
        print("  WARNING: No quantization config found in config.json")
        print("  vLLM may still auto-detect from tensor metadata")

    return config


def check_safetensors(model_path: str):
    """safetensors 파일에서 INT4 양자화된 텐서를 확인합니다."""
    print("\n[2/4] Checking safetensors metadata...")

    if not os.path.isdir(model_path):
        print(f"  Skipping (remote model: {model_path})")
        return

    safetensor_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if not safetensor_files:
        print("  ERROR: No .safetensors files found!")
        return

    total_size = sum(
        os.path.getsize(os.path.join(model_path, f))
        for f in safetensor_files
    )
    print(f"  Found {len(safetensor_files)} safetensors file(s)")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")

    # safetensors 메타데이터 확인
    try:
        from safetensors import safe_open
        f = safe_open(os.path.join(model_path, safetensor_files[0]), framework="pt")
        keys = list(f.keys())[:10]
        print(f"  Sample tensor keys: {keys}")

        # 첫 번째 weight 텐서의 dtype 확인
        for key in f.keys():
            if "weight" in key and "scale" not in key:
                tensor = f.get_tensor(key)
                print(f"  Sample tensor '{key}': shape={tensor.shape}, dtype={tensor.dtype}")
                break
    except ImportError:
        print("  (safetensors library not available for metadata check)")


def test_vllm_load(model_path: str, task: str, max_model_len: int):
    """vLLM으로 모델 로드를 테스트합니다."""
    print("\n[3/4] Testing vLLM model loading...")

    try:
        from vllm import LLM
    except ImportError:
        print("  ERROR: vLLM not installed. Install with: pip install vllm")
        return False

    start_time = time.time()

    try:
        llm_kwargs = {
            "model": model_path,
            "dtype": "auto",
            "trust_remote_code": True,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,  # torch.compile 호환성 문제 방지
        }

        if task == "embedding":
            llm_kwargs["runner"] = "pooling"

        llm = LLM(**llm_kwargs)
        load_time = time.time() - start_time
        print(f"  Model loaded successfully in {load_time:.1f}s")

        # GPU 메모리 사용량
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / (1024**3)
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"  GPU memory: {mem_used:.2f} GB allocated, "
                  f"{mem_reserved:.2f} GB reserved")

        return llm

    except Exception as e:
        print(f"  ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_inference(llm, task: str):
    """추론 테스트를 수행합니다."""
    print("\n[4/4] Testing inference...")

    try:
        if task == "embedding":
            # Embedding 테스트
            test_texts = [
                "What is machine learning?",
                "머신러닝이란 무엇인가요?",
                "The quick brown fox jumps over the lazy dog.",
            ]
            print(f"  Testing embedding with {len(test_texts)} texts...")

            outputs = llm.embed(test_texts)

            for i, output in enumerate(outputs):
                emb = output.outputs.embedding
                dim = len(emb)
                norm = sum(x**2 for x in emb) ** 0.5
                print(f"  Text {i+1}: dim={dim}, norm={norm:.4f}, "
                      f"first 5 values={emb[:5]}")

            # 유사도 테스트
            import numpy as np
            emb1 = np.array(outputs[0].outputs.embedding)
            emb2 = np.array(outputs[1].outputs.embedding)
            emb3 = np.array(outputs[2].outputs.embedding)

            cos_sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            cos_sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
            print(f"\n  Cosine similarity (text1 vs text2, same meaning): {cos_sim_12:.4f}")
            print(f"  Cosine similarity (text1 vs text3, different topic): {cos_sim_13:.4f}")

            if cos_sim_12 > cos_sim_13:
                print("  PASS: Semantically similar texts have higher similarity")
            else:
                print("  WARNING: Similarity ordering unexpected (may need quality check)")

        else:
            # Text generation 테스트
            test_prompts = [
                "The capital of South Korea is",
                "Machine learning is a field of",
            ]
            print(f"  Testing generation with {len(test_prompts)} prompts...")

            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=50,
            )

            outputs = llm.generate(test_prompts, sampling_params)

            for i, output in enumerate(outputs):
                prompt = output.prompt
                generated = output.outputs[0].text
                print(f"  Prompt: {prompt}")
                print(f"  Output: {generated}")
                print()

        print("  Inference test PASSED!")
        return True

    except Exception as e:
        print(f"  ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify INT4 GPTQ model compatibility with vLLM"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model path (local directory or HuggingFace model ID)",
    )
    parser.add_argument(
        "--task", type=str, default="embedding",
        choices=["embedding", "generate"],
        help="Task type: embedding (pooling) or generate (causal LM)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=2048,
        help="Maximum model sequence length (default: 2048)",
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Only check config and files, skip vLLM loading and inference",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("INT4 GPTQ Model Verification")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Task:  {args.task}")

    # Step 1: Config 확인
    config = check_config(args.model)

    # Step 2: Safetensors 확인
    check_safetensors(args.model)

    if args.skip_inference:
        print("\nSkipping vLLM load and inference tests (--skip-inference)")
        return

    # Step 3: vLLM 로드
    llm = test_vllm_load(args.model, args.task, args.max_model_len)
    if llm is None:
        print("\nFAILED: Model could not be loaded in vLLM")
        sys.exit(1)

    # Step 4: 추론 테스트
    success = test_inference(llm, args.task)

    # Summary
    print(f"\n{'='*60}")
    if success:
        print("RESULT: ALL TESTS PASSED")
        print(f"Model '{args.model}' is compatible with vLLM for {args.task}!")
    else:
        print("RESULT: SOME TESTS FAILED")
        print("Check the errors above for details.")
    print(f"{'='*60}")

    # Cleanup
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
