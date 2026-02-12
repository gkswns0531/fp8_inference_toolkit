#!/usr/bin/env python3
"""
INT4 (W4A16) GPTQ Quantization Script with Calibration

llmcompressor의 GPTQModifier를 사용하여 모델을 INT4(W4A16)로 양자화합니다.
calibration 데이터를 사용하여 최적의 양자화 파라미터를 찾고,
HuggingFace 호환 포맷으로 저장 후 업로드합니다.

FP8과의 차이점:
    - FP8: calibration 불필요 (FP8_DYNAMIC), 빠른 양자화
    - INT4 GPTQ: calibration 필수, 더 높은 압축률 (4bit vs 8bit)
    - INT4: VRAM ~50-60% 절약 (FP8 대비 추가 절약)

vLLM 지원:
    - compressed-tensors 포맷으로 저장 → vLLM 자동 감지
    - 또는 --quantization gptq_marlin 플래그로 명시적 로드
    - Marlin 커널로 INT4 추론 가속

Requirements:
    pip install llmcompressor transformers huggingface_hub torch datasets

Usage:
    # 기본 실행 (ultrachat_200k calibration)
    python int4_quantize_gptq.py

    # 커스텀 calibration 데이터셋
    python int4_quantize_gptq.py --calibration-dataset wikitext --num-samples 512

    # 특정 모델만
    python int4_quantize_gptq.py --model Qwen/Qwen3-Embedding-4B

    # group size 변경 (기본 128)
    python int4_quantize_gptq.py --group-size 64

주의사항:
    1. HF_TOKEN 환경변수 필요
    2. calibration에 GPU 메모리 필요 (8B 모델: ~32GB 권장)
    3. calibration 시간: 모델 크기에 따라 30분~2시간+
    4. group_size가 작을수록 품질 좋지만 저장 크기 증가
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
from llmcompressor.modifiers.quantization import GPTQModifier
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

# 양자화할 모델 목록 (기본)
DEFAULT_MODELS = [
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
]

OUTPUT_BASE_DIR = "/home/ubuntu/models"

# Calibration 설정
DEFAULT_CALIBRATION_DATASET = "ultrachat_200k"
DEFAULT_NUM_SAMPLES = 128
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_GROUP_SIZE = 128

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
        "text_field": None,  # chat format - needs special handling
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


def prepare_calibration_dataset(
    dataset_name: str,
    num_samples: int,
):
    """Calibration용 Dataset 서브셋을 준비합니다.

    전체 데이터셋을 토큰화하지 않고, 필요한 샘플 수만큼만 로드합니다.
    oneshot()에 Dataset 객체를 직접 전달하면 전체 토큰화를 피할 수 있습니다.
    """
    from datasets import Dataset, load_dataset

    config = DATASET_CONFIGS.get(dataset_name)
    if config is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    print(f"Loading calibration dataset: {config['path']} "
          f"(subset: {num_samples} samples)...")

    # 필요한 만큼만 로드 (streaming으로 전체 다운로드 방지)
    load_kwargs = {"path": config["path"], "streaming": True}
    if "name" in config:
        load_kwargs["name"] = config["name"]
    load_kwargs["split"] = config["split"]

    ds = load_dataset(**load_kwargs)

    # 텍스트 필드가 있으면 그대로, chat 형식이면 결합
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

    # Dataset 객체로 변환 (oneshot()에 직접 전달 가능)
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

        # MoE routing gate (expert 선택 결정 — 양자화 시 라우팅 붕괴)
        # Qwen3-Next: mlp.gate, Qwen3-MoE: mlp_moe.gate, Mixtral: block_sparse_moe.gate
        if module_name == "gate" and any(k in name for k in ("expert", "moe", "mlp.gate")):
            # mlp.gate$ 패턴으로 gate_proj와 구분
            parent = ".".join(name.split(".")[-2:])
            detected.add(f"re:.*{parent}$")
            has_moe = True

        # shared_expert_gate (shared/routed expert 혼합 비율)
        if module_name == "shared_expert_gate":
            detected.add("re:.*shared_expert_gate")
            has_moe = True

        # DeltaNet (Gated DeltaNet / linear attention)
        # beta/alpha 게이트가 recurrence state에 직접 영향 — 양자화 오차 누적
        # conv1d는 Linear 타겟과 비호환
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
    calibration_dataset: str,
    num_samples: int,
    max_seq_length: int,
    group_size: int,
) -> dict:
    """단일 모델 INT4 GPTQ 양자화 및 업로드. Returns timing info."""

    print(f"\n{'='*60}")
    print(f"INT4 GPTQ Quantization")
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

    if mtype == "vl":
        print("WARNING: VL 모델의 INT4 GPTQ calibration은 "
              "multimodal 데이터가 필요할 수 있습니다.")
        print("텍스트 전용 calibration으로 진행합니다.")

    # 1. 토크나이저/프로세서 로드
    if mtype == "vl":
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    else:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        processor = tokenizer

    # 2. 모델 로드
    t0 = time.time()
    print("Loading model...")
    if mtype == "vl":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    elif mtype == "encoder":
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    timing["model_load"] = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model class: {type(model).__name__}")
    print(f"Parameters: {param_count:,}")
    bf16_size_gb = param_count * 2 / (1024**3)
    int4_size_gb = param_count * 0.5 / (1024**3)  # 4bit = 0.5 bytes
    print(f"Estimated size: BF16={bf16_size_gb:.2f}GB → INT4={int4_size_gb:.2f}GB "
          f"({(1 - int4_size_gb/bf16_size_gb)*100:.0f}% reduction)")

    # 3. INT4 GPTQ 양자화 설정
    # W4A16: weight 4bit, activation 16bit (BF16/FP16)
    # W4A16 프리셋: group_size=128, symmetric=True, strategy=group
    ignore = get_ignore_patterns(model)
    print(f"Ignore patterns: {ignore}")

    if group_size == 128:
        # W4A16 프리셋 사용 (기본 group_size=128)
        recipe = GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=ignore,
        )
    else:
        # 커스텀 group_size는 config_groups로 지정
        recipe = GPTQModifier(
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

    # 4. Calibration 데이터 준비 (서브셋만 로드)
    t0 = time.time()
    cal_dataset = prepare_calibration_dataset(calibration_dataset, num_samples)
    timing["data_load"] = time.time() - t0

    # 5. Calibration + 양자화 실행
    print(f"\nRunning GPTQ calibration...")
    print("This may take a while depending on model size...")
    t0 = time.time()

    oneshot(
        model=model,
        dataset=cal_dataset,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_samples,
        tokenizer=tokenizer,
    )
    timing["quantization"] = time.time() - t0
    print(f"Quantization complete! ({timing['quantization']:.1f}s)")

    # 6. 저장
    t0 = time.time()
    print(f"\nSaving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # compressed-tensors 포맷으로 저장 (vLLM 자동 감지)
    model.save_pretrained(output_dir, save_compressed=True)
    processor.save_pretrained(output_dir)

    # 저장된 파일 크기 확인
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
                f"Upload {model_id.split('/')[-1]} INT4-GPTQ "
                f"(W4A16, g{group_size}, {num_samples} cal samples)"
            ),
        )
        timing["upload"] = time.time() - t0
        print(f"SUCCESS: https://huggingface.co/{repo_id}")
    else:
        print("\nSkipping HF upload (no HF_TOKEN set)")
        print(f"To upload manually:")
        print(f"  huggingface-cli upload {repo_id} {output_dir}")

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

    # 9. vLLM 추론 가이드 출력
    print(f"\n{'='*60}")
    print("vLLM Inference Guide")
    print(f"{'='*60}")
    print(f"""
# 방법 1: vLLM 자동 감지 (compressed-tensors 포맷)
python -m vllm.entrypoints.openai.api_server \\
    --model {repo_id} \\
    --dtype auto \\
    --max-model-len {max_seq_length} \\
    --trust-remote-code \\
    --gpu-memory-utilization 0.85

# 방법 2: 명시적 quantization 지정
python -m vllm.entrypoints.openai.api_server \\
    --model {repo_id} \\
    --quantization gptq_marlin \\
    --dtype auto \\
    --max-model-len {max_seq_length} \\
    --trust-remote-code

# 방법 3: 이 프로젝트의 start_vllm_server.py 사용
python serving/vllm/start_vllm_server.py \\
    --mode openai \\
    --model-path {repo_id} \\
    --quantization gptq \\
    --max-model-len {max_seq_length}

# Embedding 모델의 경우 --runner pooling 추가
python -m vllm.entrypoints.openai.api_server \\
    --model {repo_id} \\
    --runner pooling \\
    --dtype auto \\
    --max-model-len {max_seq_length} \\
    --trust-remote-code
""")

    # 메모리 정리
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return timing


def main():
    parser = argparse.ArgumentParser(
        description="INT4 GPTQ Quantization with Calibration"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single model to quantize (e.g., Qwen/Qwen3-Embedding-4B). "
             "If not set, all DEFAULT_MODELS are processed.",
    )
    parser.add_argument(
        "--calibration-dataset", type=str, default=DEFAULT_CALIBRATION_DATASET,
        help=f"Calibration dataset (default: {DEFAULT_CALIBRATION_DATASET}). "
             f"Options: {list(DATASET_CONFIGS.keys())} or any HF dataset name",
    )
    parser.add_argument(
        "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help=f"Number of calibration samples (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Max sequence length for calibration (default: {DEFAULT_MAX_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--group-size", type=int, default=DEFAULT_GROUP_SIZE,
        help=f"GPTQ group size (default: {DEFAULT_GROUP_SIZE}). "
             "Smaller = better quality but larger model file.",
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
        print("Set HF_TOKEN environment variable to enable auto-upload.")

    # HF username
    username = args.hf_username
    if not username and hf_token:
        api = HfApi(token=hf_token)
        username = api.whoami()["name"]

    # 모델 목록
    models = [args.model] if args.model else DEFAULT_MODELS

    for model_id in models:
        model_name = model_id.split("/")[-1]
        output_dir = os.path.join(
            args.output_dir, f"{model_name.lower()}-int4-gptq"
        )

        if username:
            repo_id = f"{username}/{model_name}-INT4-GPTQ"
        else:
            repo_id = f"local/{model_name}-INT4-GPTQ"

        try:
            quantize_and_upload(
                model_id=model_id,
                output_dir=output_dir,
                repo_id=repo_id,
                hf_token=hf_token,
                calibration_dataset=args.calibration_dataset,
                num_samples=args.num_samples,
                max_seq_length=args.max_seq_length,
                group_size=args.group_size,
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
