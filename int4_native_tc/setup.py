"""
Build configuration for INT4 Native Tensor Core extension.

Usage:
    pip install -e .
    # or
    python setup.py develop
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUTLASS source path
CUTLASS_DIR = os.environ.get(
    "CUTLASS_DIR",
    "/home/ubuntu/vllm/.deps/cutlass-src"
)

# vLLM csrc path (for cutlass_extensions headers like broadcast_load_epilogue)
VLLM_CSRC_DIR = os.environ.get(
    "VLLM_CSRC_DIR",
    "/home/ubuntu/vllm/csrc"
)

setup(
    name="int4_native_tc",
    version="0.2.0",
    description="Native INT4 Tensor Core GEMM via CUTLASS for W4A4 quantization",
    packages=["int4_native_tc"],
    ext_modules=[
        CUDAExtension(
            name="int4_native_tc_ops",
            sources=[
                "csrc/int4_gemm.cu",
                "csrc/int4_quant.cu",
                "csrc/bindings.cpp",
            ],
            include_dirs=[
                os.path.join(CUTLASS_DIR, "include"),
                os.path.join(CUTLASS_DIR, "tools", "util", "include"),
                VLLM_CSRC_DIR,  # for cutlass_extensions/epilogue/ headers
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-DCUTLASS_ARCH_MMA_SM80_ENABLED=1",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
