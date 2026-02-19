#!/usr/bin/env python3
"""
NVFP4 (W4A4) Quantization Script for Blackwell GPUs (B200, GB200)

llmcompressor의 QuantizationModifier를 사용하여 모델을 NVFP4로 양자화합니다.
NVFP4는 weight와 activation 모두 FP4(E2M1)로 양자화하며,
16값 블록 단위 FP8 스케일 + 텐서 단위 FP32 스케일의 2단계 스케일링을 사용합니다.

INT4 대비 장점 (Blackwell GPU):
    - dequantization 불필요 → INT4 대비 2.35x 빠른 추론
    - 부동소수점 특성으로 아웃라이어 처리에 유리
    - FP8 대비 ~1.8x 메모리 절감

Scheme 선택:
    - NVFP4 (W4A4): weight + activation 모두 FP4. Calibration 필요. Blackwell 네이티브.
    - NVFP4A16 (W4A16): weight만 FP4. Calibration 불필요. H100 이하에서도 사용 가능.

Requirements:
    pip install llmcompressor transformers huggingface_hub torch datasets

Usage:
    # W4A4 (Blackwell 네이티브, calibration 필요)
    python nvfp4_quantize.py --model Qwen/Qwen3-Next-80B-A3B-Instruct

    # W4A16 (weight-only, calibration 불필요)
    python nvfp4_quantize.py --model Qwen/Qwen3-Next-80B-A3B-Instruct --scheme NVFP4A16

주의사항:
    1. NVFP4 W4A4 네이티브 가속은 Blackwell GPU (B200, GB200 등) 필요
    2. H100 이하에서는 W4A16 Marlin 커널로 폴백 (weight-only 압축)
    3. HF_TOKEN 환경변수 필요
"""

import argparse
import gc
import os
import time
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoConfig, AutoModel
from transformers import AutoModelForCausalLM
from huggingface_hub import HfApi, create_repo
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")

DEFAULT_MODELS = [
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
]

OUTPUT_BASE_DIR = "/home/ubuntu/models"

# Calibration 설정 (W4A4에서만 사용)
DEFAULT_CALIBRATION_DATASET = "ultrachat_200k"
DEFAULT_NUM_SAMPLES = 128
DEFAULT_MAX_SEQ_LENGTH = 8192

# Encoder-only 모델 타입
ENCODER_ONLY_TYPES = {
    "xlm-roberta", "roberta", "bert", "albert",
    "electra", "camembert", "deberta", "deberta-v2",
}

# ─────────────────────────────────────────────────────────────────────────────
# Calibration Dataset Helpers
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "ultrachat_200k": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "text_field": None,
    },
    "wikitext": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "train",
        "text_field": "text",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "split": "train",
        "text_field": "text",
        "streaming": True,
    },
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "streaming": True,
    },
}


def prepare_calibration_dataset(dataset_name: str, num_samples: int):
    """Calibration용 Dataset 서브셋을 준비합니다."""
    from datasets import Dataset, load_dataset

    config = DATASET_CONFIGS.get(dataset_name)
    if config is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    print(f"Loading calibration dataset: {config['path']} "
          f"(subset: {num_samples} samples)...")

    load_kwargs = {"path": config["path"], "streaming": True}
    if "name" in config:
        load_kwargs["name"] = config["name"]
    load_kwargs["split"] = config["split"]

    ds = load_dataset(**load_kwargs)

    text_field = config["text_field"]
    texts = []
    for sample in ds:
        if len(texts) >= num_samples:
            break

        if text_field:
            text = sample[text_field]
        else:
            messages = sample.get("messages", [])
            text = " ".join(m.get("content", "") for m in messages)

        if text and len(text.strip()) > 100:
            texts.append(text.strip())

    print(f"Collected {len(texts)} calibration samples")
    return Dataset.from_dict({"text": texts})


# ─────────────────────────────────────────────────────────────────────────────
# Model Type Detection
# ─────────────────────────────────────────────────────────────────────────────

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


def get_ignore_patterns(model) -> list[str]:
    """모델 구조를 분석하여 양자화에서 제외할 레이어 패턴을 반환합니다.

    제외 대상:
    - MoE gate/router: 양자화 시 expert 라우팅 결정 붕괴
    - shared_expert_gate: shared/routed expert 혼합 비율 결정
    - DeltaNet (linear_attn): recurrent state 누적 오차, conv1d 비호환
    """
    ignore = ["re:.*lm_head"]

    detected = set()
    has_moe = False
    has_linear_attn = False

    for name, _ in model.named_modules():
        module_name = name.split(".")[-1]

        if module_name == "gate" and any(k in name for k in ("expert", "moe", "mlp.gate")):
            parent = ".".join(name.split(".")[-2:])
            detected.add(f"re:.*{parent}$")
            has_moe = True

        if module_name == "shared_expert_gate":
            detected.add("re:.*shared_expert_gate")
            has_moe = True

        if module_name == "linear_attn" or (
            "linear_attn" in name and module_name in ("in_proj_qkvz", "in_proj_ba", "out_proj", "conv1d")
        ):
            if not has_linear_attn:
                detected.add("re:.*linear_attn.*")
                has_linear_attn = True

    if detected:
        ignore.extend(sorted(detected))

    if has_moe:
        print(f"MoE model detected — gate/router layers excluded from quantization")
    if has_linear_attn:
        print(f"DeltaNet (linear_attn) detected — excluded from quantization")
    print(f"Ignore patterns: {ignore}")

    return ignore


# ─────────────────────────────────────────────────────────────────────────────
# Quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_and_upload(
    model_id: str,
    output_dir: str,
    repo_id: str,
    hf_token: str,
    scheme: str,
    calibration_dataset: str,
    num_samples: int,
    max_seq_length: int,
) -> dict:
    """단일 모델 NVFP4 양자화 및 업로드. Returns timing info."""

    is_w4a4 = scheme == "NVFP4"

    print(f"\n{'='*60}")
    print(f"NVFP4 Quantization ({scheme})")
    print(f"{'='*60}")
    print(f"Model:       {model_id}")
    print(f"Scheme:      {scheme} ({'W4A4 — weights+activations FP4' if is_w4a4 else 'W4A16 — weights FP4, activations FP16'})")
    print(f"Output:      {output_dir}")
    print(f"HF Repo:     {repo_id}")
    if is_w4a4:
        print(f"Calibration: {calibration_dataset} ({num_samples} samples)")
        print(f"Max Seq Len: {max_seq_length}")
    else:
        print(f"Calibration: Not required (weight-only)")
    print(f"{'='*60}")

    timing = {}
    total_start = time.time()

    mtype = detect_model_type(model_id)
    type_labels = {
        "vl": "VL (Vision-Language)",
        "encoder": "Encoder-only",
        "causal": "CausalLM",
    }
    print(f"Model type: {type_labels[mtype]}")

    # 1. 토크나이저 로드
    if mtype == "vl":
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    else:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        processor = tokenizer

    # 2. 모델 로드
    t0 = time.time()
    print("Loading model...")
    if mtype == "vl":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True,
        )
    elif mtype == "encoder":
        model = AutoModel.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True,
        )

    timing["model_load"] = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model class: {type(model).__name__}")
    print(f"Parameters: {param_count:,}")
    bf16_size_gb = param_count * 2 / (1024**3)
    fp4_size_gb = param_count * 0.5625 / (1024**3)  # ~4.5 bits = 0.5625 bytes (4bit + FP8 scale overhead)
    print(f"Estimated size: BF16={bf16_size_gb:.2f}GB → NVFP4≈{fp4_size_gb:.2f}GB "
          f"({(1 - fp4_size_gb/bf16_size_gb)*100:.0f}% reduction)")

    # 3. NVFP4 양자화 설정
    ignore = get_ignore_patterns(model)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=ignore,
    )

    # 4. Calibration + 양자화 실행
    t0 = time.time()
    if is_w4a4:
        # W4A4: calibration 필요 (global activation scale 계산)
        cal_dataset = prepare_calibration_dataset(calibration_dataset, num_samples)
        timing["data_load"] = time.time() - t0

        print(f"\nRunning NVFP4 W4A4 calibration + quantization...")
        t0 = time.time()
        oneshot(
            model=model,
            dataset=cal_dataset,
            recipe=recipe,
            max_seq_length=max_seq_length,
            num_calibration_samples=num_samples,
            tokenizer=tokenizer,
        )
    else:
        # W4A16: calibration 불필요
        timing["data_load"] = 0
        print(f"\nRunning NVFP4 W4A16 quantization (no calibration)...")
        t0 = time.time()
        oneshot(model=model, recipe=recipe)

    timing["quantization"] = time.time() - t0
    print(f"Quantization complete! ({timing['quantization']:.1f}s)")

    # 5. 저장
    t0 = time.time()
    print(f"\nSaving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir, save_compressed=True)
    processor.save_pretrained(output_dir)

    total_size = 0
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            if f.endswith(".safetensors"):
                print(f"  {f}: {size / (1024**3):.2f} GB")
    print(f"Total saved size: {total_size / (1024**3):.2f} GB")
    timing["save"] = time.time() - t0

    # 6. HF 업로드
    if hf_token:
        t0 = time.time()
        print(f"\nUploading to {repo_id}...")
        api = HfApi(token=hf_token)
        create_repo(repo_id, token=hf_token, exist_ok=True)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            commit_message=(
                f"Upload {model_id.split('/')[-1]} {scheme} "
                f"({num_samples} cal samples)" if is_w4a4
                else f"Upload {model_id.split('/')[-1]} {scheme} (weight-only)"
            ),
        )
        timing["upload"] = time.time() - t0
        print(f"SUCCESS: https://huggingface.co/{repo_id}")
    else:
        print("\nSkipping HF upload (no HF_TOKEN set)")
        print(f"To upload manually:")
        print(f"  huggingface-cli upload {repo_id} {output_dir}")

    timing["total"] = time.time() - total_start

    # 7. 타이밍 요약
    print(f"\n{'='*60}")
    print("Timing Summary")
    print(f"{'='*60}")
    print(f"  Model load:    {timing['model_load']:.1f}s")
    if is_w4a4:
        print(f"  Data load:     {timing['data_load']:.1f}s")
    print(f"  Quantization:  {timing['quantization']:.1f}s")
    print(f"  Save:          {timing['save']:.1f}s")
    if 'upload' in timing:
        print(f"  Upload:        {timing['upload']:.1f}s")
    print(f"  TOTAL:         {timing['total']:.1f}s")
    print(f"{'='*60}")

    # 8. vLLM 추론 가이드
    print(f"\n{'='*60}")
    print("vLLM Inference Guide")
    print(f"{'='*60}")
    print(f"""
# Blackwell GPU (B200, GB200) — 네이티브 NVFP4 가속
vllm serve {repo_id} \\
    --dtype auto \\
    --trust-remote-code \\
    --tensor-parallel-size 2

# H100 이하 — W4A16 Marlin 커널 폴백 (weight-only 압축)
# 동일 명령어로 서빙 가능하나, activation은 FP16으로 폴백됨
""")

    # 메모리 정리
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return timing


def main():
    parser = argparse.ArgumentParser(
        description="NVFP4 Quantization for Blackwell GPUs"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model to quantize. If not set, DEFAULT_MODELS are processed.",
    )
    parser.add_argument(
        "--scheme", type=str, default="NVFP4",
        choices=["NVFP4", "NVFP4A16"],
        help="NVFP4 (W4A4, calibration required) or NVFP4A16 (W4A16, no calibration). Default: NVFP4",
    )
    parser.add_argument(
        "--calibration-dataset", type=str, default=DEFAULT_CALIBRATION_DATASET,
        help=f"Calibration dataset (default: {DEFAULT_CALIBRATION_DATASET}). "
             f"Only used with NVFP4 scheme.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help=f"Number of calibration samples (default: {DEFAULT_NUM_SAMPLES}). "
             "Only used with NVFP4 scheme.",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Max sequence length for calibration (default: {DEFAULT_MAX_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_BASE_DIR,
        help=f"Base output directory (default: {OUTPUT_BASE_DIR})",
    )
    parser.add_argument(
        "--hf-username", type=str, default=None,
        help="HuggingFace username for repo. Auto-detected if not set.",
    )

    args = parser.parse_args()

    hf_token = HF_TOKEN
    if not hf_token:
        print("WARNING: HF_TOKEN not set. Model will be saved locally only.")

    username = args.hf_username
    if not username and hf_token:
        api = HfApi(token=hf_token)
        username = api.whoami()["name"]

    models = [args.model] if args.model else DEFAULT_MODELS

    for model_id in models:
        model_name = model_id.split("/")[-1]
        suffix = "nvfp4" if args.scheme == "NVFP4" else "nvfp4a16"
        output_dir = os.path.join(args.output_dir, f"{model_name.lower()}-{suffix}")

        if username:
            repo_id = f"{username}/{model_name}-{suffix.upper()}"
        else:
            repo_id = f"local/{model_name}-{suffix.upper()}"

        try:
            quantize_and_upload(
                model_id=model_id,
                output_dir=output_dir,
                repo_id=repo_id,
                hf_token=hf_token,
                scheme=args.scheme,
                calibration_dataset=args.calibration_dataset,
                num_samples=args.num_samples,
                max_seq_length=args.max_seq_length,
            )
        except Exception as e:
            print(f"\nFAILED: {model_id}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
