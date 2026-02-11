#!/usr/bin/env python3
"""
FP8 Quantization Script - Standard HuggingFace Format

auto_fp8 라이브러리를 사용하여 표준 HuggingFace FP8 체크포인트를 생성합니다.
config.json에 quantization_config를 명시적으로 추가하여 호환성을 높입니다.

llmcompressor와의 차이점:
    - llmcompressor: compressed-tensors 포맷 (vLLM 자동 감지)
    - auto_fp8: 표준 HF 포맷 + quantization_config 명시

Requirements:
    pip install auto_fp8 transformers huggingface_hub torch

Usage:
    python fp8_quantize_hf_standard.py

주의사항:
    1. auto_fp8은 weight만 양자화 (activation은 dynamic)
    2. config.json에 quantization_config 추가됨
    3. preprocessor_config.json 복사 필요 (VL 모델)
"""

import os
import gc
import json
import shutil
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
from auto_fp8.quantize import quantize_weights
from auto_fp8.config import BaseQuantizeConfig
from huggingface_hub import hf_hub_download, HfApi

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODELS = [
    # (model_id, output_dir, hf_repo)
    ("Qwen/Qwen3-VL-Embedding-2B", "/home/ubuntu/qwen3_vl_2b_fp8_std", "YourUsername/Qwen3-VL-Embedding-2B-FP8-Standard"),
    ("Qwen/Qwen3-VL-Embedding-8B", "/home/ubuntu/qwen3_vl_8b_fp8_std", "YourUsername/Qwen3-VL-Embedding-8B-FP8-Standard"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def quantize_vl_model(model_id: str, output_dir: str, hf_repo: str):
    """VL 모델을 FP8로 양자화하고 표준 HF 포맷으로 저장."""

    print(f"\n{'='*60}")
    print(f"Quantizing: {model_id}")
    print(f"Output: {output_dir}")
    print(f"HF Repo: {hf_repo}")
    print(f"{'='*60}")

    # 1. 모델 로드
    print("\nLoading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. FP8 양자화
    print("\nQuantizing weights to FP8...")
    quantize_config = BaseQuantizeConfig(
        quant_method="fp8",
        activation_scheme="dynamic",  # 동적 activation 양자화
    )
    quantize_weights(model, quantize_config)

    # 3. 저장
    print(f"\nSaving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # 4. preprocessor_config.json 복사 (VL 모델 필수)
    try:
        path = hf_hub_download(repo_id=model_id, filename="preprocessor_config.json")
        shutil.copy(path, output_dir)
        print("Copied preprocessor_config.json")
    except Exception:
        print("No preprocessor_config.json found (text-only model)")

    # 5. config.json에 quantization_config 추가
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        saved_config = json.load(f)

    # 양자화 설정 메타데이터 추가
    saved_config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic"
    }

    with open(config_path, "w") as f:
        json.dump(saved_config, f, indent=2)
    print("Updated config.json with quantization_config")

    # 6. HuggingFace 업로드
    print(f"\nUploading to {hf_repo}...")
    api = HfApi()
    api.create_repo(repo_id=hf_repo, exist_ok=True)
    api.upload_folder(folder_path=output_dir, repo_id=hf_repo)
    print(f"Uploaded to https://huggingface.co/{hf_repo}")

    # 메모리 정리
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return True


def main():
    for model_id, output_dir, hf_repo in MODELS:
        try:
            quantize_vl_model(model_id, output_dir, hf_repo)
        except Exception as e:
            print(f"Error quantizing {model_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
