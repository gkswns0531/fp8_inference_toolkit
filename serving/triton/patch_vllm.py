#!/usr/bin/env python3
"""
vLLM Weight Validation Patch

Qwen3-Embedding-8B, Qwen3-VL-Embedding-8B 등 tie_word_embeddings=false인 모델에서
lm_head.weight가 체크포인트에 없어서 발생하는 ValueError를 warning으로 완화합니다.

영향 받는 모델:
    - Qwen3-Embedding-8B (tie_word_embeddings: false)
    - Qwen3-VL-Embedding-8B (tie_word_embeddings: false)

영향 받지 않는 모델:
    - Qwen3-Embedding-0.6B, 4B (tie_word_embeddings: true)
    - Qwen3-VL-Embedding-2B (tie_word_embeddings: true)

Usage:
    # Docker 컨테이너 시작 시:
    python3 /path/to/patch_vllm.py

    # 또는 호스트에서:
    python3 patch_vllm.py

Note:
    vLLM 0.15+에서는 quantization != None일 때 weight 검증을 건너뛰므로
    FP8 모델에서는 이 패치가 불필요합니다.
    BF16 8B 모델(tie_word_embeddings=false)에서만 필요합니다.
"""

import sys
import os

# 가능한 경로들 (Python 버전에 따라 다름)
POSSIBLE_PATHS = [
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/default_loader.py",
    "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader/default_loader.py",
    "/usr/local/lib/python3.11/dist-packages/vllm/model_executor/model_loader/default_loader.py",
    os.path.expanduser("~/.local/lib/python3.10/site-packages/vllm/model_executor/model_loader/default_loader.py"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages/vllm/model_executor/model_loader/default_loader.py"),
]


def find_loader_path():
    """vLLM default_loader.py 경로 찾기."""
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            return path

    # site-packages에서 직접 검색
    try:
        import vllm
        vllm_path = os.path.dirname(vllm.__file__)
        loader_path = os.path.join(vllm_path, "model_executor/model_loader/default_loader.py")
        if os.path.exists(loader_path):
            return loader_path
    except ImportError:
        pass

    return None


def patch_vllm():
    """Weight validation을 warning으로 변경."""

    loader_path = find_loader_path()
    if not loader_path:
        print("ERROR: Could not find vLLM default_loader.py", file=sys.stderr)
        sys.exit(1)

    print(f"Found vLLM loader at: {loader_path}")

    with open(loader_path) as f:
        content = f.read()

    # 패치할 원본 코드 (vLLM 0.15+: quantization is None 조건 하에 존재)
    old_code = '''            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )'''

    # 패치된 코드 (warning으로 변경)
    new_code = '''            if weights_not_loaded:
                import logging
                logging.getLogger(__name__).warning(
                    "Skipping strict weight check: %s", weights_not_loaded)'''

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(loader_path, "w") as f:
            f.write(content)
        print("PATCHED: weight validation relaxed (ValueError -> warning)")
    elif "Skipping strict weight check" in content:
        print("ALREADY PATCHED: no changes needed")
    else:
        print("WARNING: Could not find target code to patch", file=sys.stderr)
        print("This might be a different vLLM version", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    patch_vllm()
