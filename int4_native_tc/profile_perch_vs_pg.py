#!/usr/bin/env python3
"""
Profile PerCh vs PG: per-kernel CUDA timing via torch.profiler.

Uses vLLM offline inference (no HTTP) to eliminate server overhead,
then profiles the encode() call to see where the 28ms gap comes from.
"""

import gc
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

MODEL = "Qwen/Qwen3-Embedding-4B"
MAX_MODEL_LEN = 512
NUM_SENTENCES = 10
WARMUP = 3

WORDS = [
    "mountain", "river", "ocean", "forest", "desert", "island", "valley", "bridge",
    "guitar", "piano", "violin", "trumpet", "flute", "cello", "drum", "harp",
    "python", "rust", "java", "swift", "ruby", "perl", "scala", "kotlin",
    "matrix", "vector", "tensor", "graph", "queue", "stack", "tree", "node",
]


def clear_compile_cache():
    cache_dir = Path.home() / ".cache" / "vllm" / "torch_compile_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("  Cleared torch.compile cache")


def run_profile(label, env_vars, sentences, output_dir):
    """Run profiling for a single config in a subprocess to avoid GPU memory issues."""
    import subprocess
    import json

    env = os.environ.copy()
    env.update(env_vars)

    # Write sentences to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sentences, f)
        sentences_file = f.name

    script = f'''
import json
import sys
import time
import gc
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

# Clear compile cache
cache_dir = Path.home() / ".cache" / "vllm" / "torch_compile_cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir)

# Load sentences
with open("{sentences_file}") as f:
    sentences = json.load(f)

# Import vLLM
from vllm import LLM

quantization = "w4a8-enhanced-int4tc"
print(f"Loading model with quantization={{quantization}}, env={{dict((k,v) for k,v in os.environ.items() if k.startswith('INT4TC'))}}", flush=True)

llm = LLM(
    model="{MODEL}",
    runner="pooling",
    dtype="auto",
    max_model_len={MAX_MODEL_LEN},
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    enable_prefix_caching=False,
    quantization=quantization,
    enforce_eager=True,
)

# Warmup
print("Warmup...", flush=True)
for _ in range({WARMUP}):
    llm.embed(sentences)

torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()

# Profile run
print("Profiling...", flush=True)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    profile_memory=False,
) as prof:
    llm.embed(sentences)
    torch.cuda.synchronize()

# Export chrome trace
trace_path = "{output_dir}/{label}_trace.json"
prof.export_chrome_trace(trace_path)
print(f"Trace saved: {{trace_path}}", flush=True)

# Print summary - top CUDA kernels by total time
print("\\n" + "="*80)
print(f"  CUDA Kernel Summary: {label}")
print("="*80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

# Also print CPU op summary
print("\\n" + "="*80)
print(f"  CPU Op Summary: {label}")
print("="*80)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

# Print self CUDA time for int4 related ops
print("\\n" + "="*80)
print(f"  INT4-related ops: {label}")
print("="*80)
for evt in sorted(prof.key_averages(), key=lambda e: -e.cuda_time_total):
    name = evt.key
    if any(kw in name.lower() for kw in ["int4", "quant", "cutlass", "gemm", "scaled_mm", "torch::mm"]):
        print(f"  {{name:<60}} count={{evt.count:>4}}  cuda_total={{evt.cuda_time_total/1000:>8.2f}}ms  cuda_avg={{evt.cuda_time/1000:>8.3f}}ms")

print("\\nDone.", flush=True)
'''

    clear_compile_cache()
    print(f"\n{'='*70}")
    print(f"  Profiling: {label}")
    print(f"  Env: {env_vars}")
    print(f"{'='*70}", flush=True)

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=False,
        text=True,
        timeout=600,
    )
    os.unlink(sentences_file)
    return result.returncode


def main():
    import random
    random.seed(42)
    sentences = [" ".join(random.choices(WORDS, k=50)) for _ in range(NUM_SENTENCES)]

    output_dir = "/tmp/int4_profiles"
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        ("PerCh", {"INT4TC_PER_CHANNEL": "1"}),
        ("PG_sym", {}),
    ]

    for label, env_vars in configs:
        rc = run_profile(label, env_vars, sentences, output_dir)
        if rc != 0:
            print(f"  FAILED with return code {rc}")

    print(f"\nTraces saved in {output_dir}/")
    print("View with: chrome://tracing or https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
