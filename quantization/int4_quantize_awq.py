#!/usr/bin/env python3
"""
INT4 (W4A16) AWQ Quantization Script with Calibration

llmcompressor의 AWQModifier를 사용하여 모델을 INT4(W4A16)로 양자화합니다.
Activation-Weighted Quantization: activation 패턴을 분석하여
중요 weight 채널을 보존하는 방식으로 양자화합니다.

GPTQ와의 차이점:
    - AWQ: activation 통계 기반 scaling → 더 빠른 양자화, 더 나은 품질 보존
    - GPTQ: Hessian 기반 weight 재구성 → 더 느리지만 잘 알려진 방식
    - 둘 다 vLLM에서 Marlin 커널로 동일하게 가속

Requirements:
    pip install llmcompressor transformers huggingface_hub torch datasets

Usage:
    python int4_quantize_awq.py --model Qwen/Qwen3-Embedding-8B
    python int4_quantize_awq.py --model Qwen/Qwen3-Embedding-4B --num-samples 256
"""

import argparse
import os
import gc
import time
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoConfig, AutoModel
from transformers import AutoModelForCausalLM
from huggingface_hub import HfApi, create_repo
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")

DEFAULT_MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
]

OUTPUT_BASE_DIR = "/home/ubuntu/models"

DEFAULT_CALIBRATION_DATASET = "ultrachat_200k"
DEFAULT_NUM_SAMPLES = 512
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_GROUP_SIZE = 128

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

    MoE 모델의 gate/router는 양자화하면 expert 라우팅이 깨지므로 반드시 제외.
    """
    ignore = ["re:.*lm_head"]

    gate_patterns = set()
    for name, module in model.named_modules():
        module_name = name.split(".")[-1]

        # MoE routing gate (expert 선택용)
        if module_name in ("gate",) and "moe" in name.lower():
            parent = ".".join(name.split(".")[-2:])
            gate_patterns.add(f"re:.*{parent}$")

        # shared_expert_gate (shared/routed expert 혼합 비율)
        if module_name == "shared_expert_gate":
            gate_patterns.add("re:.*shared_expert_gate")

    if gate_patterns:
        ignore.extend(sorted(gate_patterns))
        print(f"MoE gate patterns detected (auto-ignored): {sorted(gate_patterns)}")

    return ignore


# ─────────────────────────────────────────────────────────────────────────────
# Quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_and_upload(
    model_id: str,
    output_dir: str,
    repo_id: str,
    hf_token: str,
    calibration_dataset: str,
    num_samples: int,
    max_seq_length: int,
    group_size: int,
) -> dict:
    """단일 모델 INT4 AWQ 양자화 및 업로드. Returns timing info."""

    print(f"\n{'='*60}")
    print(f"INT4 AWQ Quantization")
    print(f"{'='*60}")
    print(f"Model:       {model_id}")
    print(f"Output:      {output_dir}")
    print(f"HF Repo:     {repo_id}")
    print(f"Calibration: {calibration_dataset} ({num_samples} samples)")
    print(f"Group Size:  {group_size}")
    print(f"Max Seq Len: {max_seq_length}")
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

    # 1. 토크나이저/프로세서 로드
    if mtype == "vl":
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    else:
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
    int4_size_gb = param_count * 0.5 / (1024**3)
    print(f"Estimated: BF16={bf16_size_gb:.2f}GB -> INT4={int4_size_gb:.2f}GB "
          f"({(1 - int4_size_gb/bf16_size_gb)*100:.0f}% reduction)")

    # 3. INT4 AWQ 양자화 설정
    # AWQ: activation 통계 기반 scaling으로 중요 weight 채널 보존
    # 메모리 참고: AWQ는 calibration 샘플의 forward 입력을 캐시하므로
    #   충분한 GPU/시스템 메모리 필요 (8B: ~40GB+ 권장)
    ignore = get_ignore_patterns(model)
    print(f"Ignore patterns: {ignore}")

    if group_size == 128:
        recipe = AWQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=ignore,
        )
    else:
        recipe = AWQModifier(
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=4,
                        type=QuantizationType.INT,
                        symmetric=True,
                        strategy=QuantizationStrategy.GROUP,
                        group_size=group_size,
                    ),
                ),
            },
            ignore=ignore,
        )

    # 4. Calibration 데이터 준비
    t0 = time.time()
    cal_dataset = prepare_calibration_dataset(calibration_dataset, num_samples)
    timing["data_load"] = time.time() - t0

    # 5. AWQ Calibration + 양자화 실행
    print(f"\nRunning AWQ calibration + quantization...")
    t0 = time.time()

    oneshot(
        model=model,
        dataset=cal_dataset,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_samples,
    )
    timing["quantization"] = time.time() - t0
    print(f"Quantization complete! ({timing['quantization']:.1f}s)")

    # 6. 저장
    t0 = time.time()
    print(f"\nSaving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    processor.save_pretrained(output_dir)
    timing["save"] = time.time() - t0

    total_size = 0
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            if f.endswith(".safetensors"):
                print(f"  {f}: {size / (1024**3):.2f} GB")
    print(f"Total saved size: {total_size / (1024**3):.2f} GB")

    # 7. HF 업로드
    if hf_token:
        t0 = time.time()
        print(f"\nUploading to {repo_id}...")
        api = HfApi(token=hf_token)
        create_repo(repo_id, token=hf_token, exist_ok=True)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            commit_message=(
                f"Upload {model_id.split('/')[-1]} INT4-AWQ "
                f"(W4A16, g{group_size}, {num_samples} cal samples)"
            ),
        )
        timing["upload"] = time.time() - t0
        print(f"SUCCESS: https://huggingface.co/{repo_id}")

    timing["total"] = time.time() - total_start

    # 8. 타이밍 요약
    print(f"\n{'='*60}")
    print("Timing Summary")
    print(f"{'='*60}")
    print(f"  Model load:    {timing['model_load']:.1f}s")
    print(f"  Data load:     {timing['data_load']:.1f}s")
    print(f"  Quantization:  {timing['quantization']:.1f}s")
    print(f"  Save:          {timing['save']:.1f}s")
    if 'upload' in timing:
        print(f"  Upload:        {timing['upload']:.1f}s")
    print(f"  TOTAL:         {timing['total']:.1f}s")
    print(f"{'='*60}")

    # 메모리 정리
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return timing


def main():
    parser = argparse.ArgumentParser(
        description="INT4 AWQ Quantization with Calibration (llmcompressor)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single model to quantize. If not set, all DEFAULT_MODELS are processed.",
    )
    parser.add_argument(
        "--calibration-dataset", type=str, default=DEFAULT_CALIBRATION_DATASET,
        help=f"Calibration dataset (default: {DEFAULT_CALIBRATION_DATASET})",
    )
    parser.add_argument(
        "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help=f"Number of calibration samples (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--group-size", type=int, default=DEFAULT_GROUP_SIZE,
        help=f"AWQ group size (default: {DEFAULT_GROUP_SIZE})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_BASE_DIR,
        help=f"Base output directory (default: {OUTPUT_BASE_DIR})",
    )
    parser.add_argument(
        "--hf-username", type=str, default=None,
        help="HuggingFace username. Auto-detected if not set.",
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
    all_timings = {}

    for model_id in models:
        model_name = model_id.split("/")[-1]
        output_dir = os.path.join(
            args.output_dir, f"{model_name.lower()}-int4-awq"
        )

        if username:
            repo_id = f"{username}/{model_name}-INT4-AWQ"
        else:
            repo_id = f"local/{model_name}-INT4-AWQ"

        try:
            timing = quantize_and_upload(
                model_id=model_id,
                output_dir=output_dir,
                repo_id=repo_id,
                hf_token=hf_token,
                calibration_dataset=args.calibration_dataset,
                num_samples=args.num_samples,
                max_seq_length=args.max_seq_length,
                group_size=args.group_size,
            )
            all_timings[model_id] = timing
        except Exception as e:
            print(f"\nFAILED: {model_id}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

    # 전체 타이밍 비교
    if all_timings:
        print(f"\n{'='*60}")
        print("All AWQ Quantization Timings")
        print(f"{'='*60}")
        for model_id, t in all_timings.items():
            name = model_id.split("/")[-1]
            print(f"  {name}: quantization={t['quantization']:.1f}s, total={t['total']:.1f}s")

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
