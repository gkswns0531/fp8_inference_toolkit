#!/usr/bin/env python3
"""
FP8 Quantization Script - All Layers

llmcompressor를 사용하여 모델 전체를 FP8로 양자화하고 Hugging Face에 업로드합니다.
지원 모델 타입: Vision-Language, CausalLM, Encoder-only (BERT/RoBERTa 계열)

Requirements:
    pip install llmcompressor transformers huggingface_hub torch

Usage:
    python fp8_quantize_all.py

주의사항:
    1. HF_TOKEN 환경변수 또는 코드 내 토큰 설정 필요
    2. 충분한 GPU 메모리 필요 (8B 모델: ~32GB 권장)
    3. save_compressed=True로 저장하면 llmcompressor의 compressed-tensors 포맷으로 저장됨
    4. vLLM에서 로드 시 자동으로 FP8 포맷 감지됨
"""

import os
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoConfig, AutoModel
from transformers import Qwen3VLForConditionalGeneration, AutoModelForCausalLM
from huggingface_hub import HfApi, create_repo
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "your_hf_token_here")

# 양자화할 모델 목록
MODELS = [
    "Qwen/Qwen3-VL-Embedding-2B",
    "Qwen/Qwen3-VL-Embedding-8B",
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-m3",
    # "Qwen/Qwen3-Embedding-4B",
    # "Qwen/Qwen3-Embedding-8B",
]

OUTPUT_BASE_DIR = "/home/ubuntu/models"

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ENCODER_ONLY_TYPES = {"xlm-roberta", "roberta", "bert", "albert", "electra", "camembert", "deberta", "deberta-v2"}


def detect_model_type(model_id: str) -> str:
    """모델 타입 판별: 'vl', 'encoder', 'causal'"""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    architectures = getattr(config, "architectures", [])
    model_type = getattr(config, "model_type", "")

    if any("VL" in arch or "Vision" in arch for arch in architectures):
        return "vl"
    if model_type in ENCODER_ONLY_TYPES:
        return "encoder"
    return "causal"


def quantize_and_upload(model_id: str, output_dir: str, repo_id: str, hf_token: str):
    """단일 모델 FP8 양자화 및 업로드."""

    print(f"\n{'='*60}")
    print(f"Processing: {model_id}")
    print(f"Output: {output_dir}")
    print(f"HF Repo: {repo_id}")
    print(f"{'='*60}")

    mtype = detect_model_type(model_id)
    type_labels = {"vl": "VL (Vision-Language)", "encoder": "Encoder-only", "causal": "CausalLM"}
    print(f"Model type: {type_labels[mtype]}")

    # 1. 프로세서/토크나이저 로드
    if mtype == "vl":
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    else:
        print("Loading tokenizer...")
        processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 2. 모델 로드
    print("Loading model...")
    if mtype == "vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    elif mtype == "encoder":
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    print(f"Model class: {type(model).__name__}")

    # 3. FP8 양자화 설정
    # Vision encoder 포함 모든 Linear 레이어 양자화
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=[
            "re:.*lm_head",  # lm_head는 제외 (embedding 모델에서 불필요)
            # Vision 패턴 제외하지 않음 -> Vision Encoder도 FP8 양자화됨
        ],
    )

    # 4. 양자화 실행
    print("Applying FP8 quantization...")
    oneshot(model=model, recipe=recipe)

    # 5. 저장
    print(f"Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # save_compressed=True: llmcompressor의 compressed-tensors 포맷으로 저장
    # vLLM에서 자동으로 FP8 포맷 감지
    model.save_pretrained(output_dir, save_compressed=True)
    processor.save_pretrained(output_dir)

    # 6. HF 업로드
    print(f"Uploading to {repo_id}...")
    api = HfApi(token=hf_token)
    create_repo(repo_id, token=hf_token, exist_ok=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message=f"Upload {model_id.split('/')[-1]} FP8 quantized (all layers)"
    )

    print(f"SUCCESS: https://huggingface.co/{repo_id}")

    # 메모리 정리
    del model
    torch.cuda.empty_cache()


def main():
    api = HfApi(token=HF_TOKEN)
    username = api.whoami()["name"]

    for model_id in MODELS:
        model_name = model_id.split("/")[-1]
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"{model_name.lower()}-fp8")
        repo_id = f"{username}/{model_name}-FP8"

        try:
            quantize_and_upload(model_id, output_dir, repo_id, HF_TOKEN)
        except Exception as e:
            print(f"FAILED: {model_id}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue

    print("\n" + "="*60)
    print("All done!")


if __name__ == "__main__":
    main()
