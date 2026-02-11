#!/usr/bin/env python3
"""
Direct kernel profiling: PerCh vs PG quantization + GEMM.

Loads the Qwen3-Embedding-4B model weights directly, runs quantize+GEMM
through our INT4 kernels with torch.profiler, and compares per-kernel timing.

No vLLM overhead — pure kernel-level comparison across all 36 layers.
"""

import gc
import os
import sys
import time

import torch
import numpy as np
from torch.profiler import ProfilerActivity, profile, record_function
from safetensors.torch import load_file

MODEL_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots")
HIDDEN_SIZE = 2560
NUM_LAYERS = 36
GROUP_SIZE = 128
BATCH_TOKENS = 500  # ~10 sentences × 50 tokens

# W4A4 layers (INT4 quantized activations)
W4A4_LAYERS = ["qkv_proj", "gate_up_proj"]
# W4A16 layers (BF16 activations, just torch.mm)
W4A16_LAYERS = ["o_proj", "down_proj"]

# Layer dimensions for Qwen3-4B
LAYER_DIMS = {
    "qkv_proj":     (HIDDEN_SIZE, HIDDEN_SIZE * 3 // 2 * 3),  # K=2560, N=6144  (actually 3*head*dim but merged)
    "o_proj":       (HIDDEN_SIZE * 3 // 2, HIDDEN_SIZE),       # K=4096 (num_heads*head_dim), N=2560
    "gate_up_proj": (HIDDEN_SIZE, HIDDEN_SIZE * 76 // 10),     # K=2560, N=19456 (2*intermediate)
    "down_proj":    (HIDDEN_SIZE * 76 // 20, HIDDEN_SIZE),     # K=9728, N=2560
}


def find_model_dir():
    """Find the actual snapshot directory."""
    if os.path.isdir(MODEL_DIR):
        snapshots = os.listdir(MODEL_DIR)
        if snapshots:
            return os.path.join(MODEL_DIR, snapshots[0])
    # Try direct path
    alt = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B")
    if os.path.isdir(alt):
        for root, dirs, files in os.walk(alt):
            if any(f.endswith('.safetensors') for f in files):
                return root
    raise FileNotFoundError(f"Model not found at {MODEL_DIR}")


def pack_int4_perchannel(weight_bf16):
    """Simulate per-channel INT4 weight packing (N, K) -> packed (N, K//2), scale (N,)"""
    import int4_native_tc as ops
    N, K = weight_bf16.shape
    # Quantize weight to INT4 per-channel
    absmax = weight_bf16.abs().amax(dim=1)
    scale = absmax / 7.0
    scale = scale.clamp(min=1e-10)
    w_int4 = torch.clamp(torch.round(weight_bf16 / scale.unsqueeze(1)), -8, 7).to(torch.int8)
    # Pack pairs into uint8
    w_even = w_int4[:, 0::2] & 0x0F
    w_odd = (w_int4[:, 1::2] & 0x0F) << 4
    packed = (w_even | w_odd).to(torch.uint8)
    return packed.contiguous().cuda(), scale.to(torch.float32).contiguous().cuda()


def pack_int4_pergroup(weight_bf16, group_size=128):
    """Simulate per-group INT4 weight packing (N, K) -> packed (ng, N, gs//2), scale (ng, N)"""
    N, K = weight_bf16.shape
    ng = K // group_size
    w_grouped = weight_bf16.reshape(N, ng, group_size).permute(1, 0, 2)  # (ng, N, gs)
    absmax = w_grouped.abs().amax(dim=2)  # (ng, N)
    scale = absmax / 7.0
    scale = scale.clamp(min=1e-10)
    w_int4 = torch.clamp(torch.round(w_grouped / scale.unsqueeze(2)), -8, 7).to(torch.int8)
    w_even = w_int4[:, :, 0::2] & 0x0F
    w_odd = (w_int4[:, :, 1::2] & 0x0F) << 4
    packed = (w_even | w_odd).to(torch.uint8)
    return packed.contiguous().cuda(), scale.to(torch.float32).contiguous().cuda()


def run_perchannel_profile(model_dir, M):
    """Profile per-channel quantization across all layers."""
    import int4_native_tc as ops

    print(f"\n{'='*70}")
    print(f"  Loading weights and profiling PerCh (M={M})")
    print(f"{'='*70}")

    # Prepare all layer weights
    layers_data = []
    safetensors_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.safetensors')])
    all_weights = {}
    for sf in safetensors_files:
        all_weights.update(load_file(os.path.join(model_dir, sf), device='cpu'))

    for layer_idx in range(NUM_LAYERS):
        for proj_name in W4A4_LAYERS:
            key = f"model.layers.{layer_idx}.{'self_attn' if 'proj' in proj_name and proj_name != 'gate_up_proj' else 'mlp'}.{proj_name}.weight"
            if proj_name in ["qkv_proj"]:
                key = f"model.layers.{layer_idx}.self_attn.{proj_name}.weight"
            elif proj_name in ["gate_up_proj"]:
                key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"

            if key not in all_weights:
                # Try without merged qkv
                print(f"  Warning: {key} not found, using random")
                K, N = LAYER_DIMS[proj_name]
                w = torch.randn(N, K, dtype=torch.bfloat16)
            else:
                w = all_weights[key].to(torch.bfloat16)

            N, K = w.shape
            w_packed, w_scale = pack_int4_perchannel(w)
            layers_data.append((f"L{layer_idx}.{proj_name}", K, N, w_packed, w_scale))

    del all_weights
    gc.collect()

    # Create activation
    x = torch.randn(M, HIDDEN_SIZE, dtype=torch.bfloat16, device='cuda')

    # Warmup
    print("  Warmup...", flush=True)
    for _ in range(3):
        for name, K, N, w_packed, w_scale in layers_data:
            x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            x_packed, x_scale = ops.dynamic_int4_quant(x_input)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            ops.cutlass_int4_scaled_mm(x_packed, w_packed, x_scale, w_scale, out, M, N, K)
    torch.cuda.synchronize()

    # Profile
    print("  Profiling...", flush=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for name, K, N, w_packed, w_scale in layers_data:
            with record_function(f"perch_{name}"):
                x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
                x_packed, x_scale = ops.dynamic_int4_quant(x_input)
                out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
                ops.cutlass_int4_scaled_mm(x_packed, w_packed, x_scale, w_scale, out, M, N, K)
        torch.cuda.synchronize()

    return prof


def run_pergroup_profile(model_dir, M):
    """Profile per-group quantization across all layers."""
    import int4_native_tc as ops

    print(f"\n{'='*70}")
    print(f"  Loading weights and profiling PG (M={M})")
    print(f"{'='*70}")

    layers_data = []
    safetensors_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.safetensors')])
    all_weights = {}
    for sf in safetensors_files:
        all_weights.update(load_file(os.path.join(model_dir, sf), device='cpu'))

    for layer_idx in range(NUM_LAYERS):
        for proj_name in W4A4_LAYERS:
            if proj_name in ["qkv_proj"]:
                key = f"model.layers.{layer_idx}.self_attn.{proj_name}.weight"
            elif proj_name in ["gate_up_proj"]:
                key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"

            if key not in all_weights:
                K, N = LAYER_DIMS[proj_name]
                w = torch.randn(N, K, dtype=torch.bfloat16)
            else:
                w = all_weights[key].to(torch.bfloat16)

            N, K = w.shape
            gs = GROUP_SIZE
            ng = K // gs
            w_packed, w_scale = pack_int4_pergroup(w, gs)
            layers_data.append((f"L{layer_idx}.{proj_name}", K, N, gs, ng, w_packed, w_scale))

    del all_weights
    gc.collect()

    x = torch.randn(M, HIDDEN_SIZE, dtype=torch.bfloat16, device='cuda')

    # Warmup
    print("  Warmup...", flush=True)
    for _ in range(3):
        for name, K, N, gs, ng, w_packed, w_scale in layers_data:
            x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            x_packed, x_scale = ops.dynamic_int4_quant_grouped(x_input, gs)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            ops.cutlass_int4_fused_grouped_gemm(x_packed, w_packed, x_scale, w_scale, out, M, N, gs, ng)
    torch.cuda.synchronize()

    # Profile
    print("  Profiling...", flush=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for name, K, N, gs, ng, w_packed, w_scale in layers_data:
            with record_function(f"pg_{name}"):
                x_input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
                x_packed, x_scale = ops.dynamic_int4_quant_grouped(x_input, gs)
                out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
                ops.cutlass_int4_fused_grouped_gemm(x_packed, w_packed, x_scale, w_scale, out, M, N, gs, ng)
        torch.cuda.synchronize()

    return prof


def print_profile_summary(label, prof):
    """Print profiling summary."""
    print(f"\n{'='*80}")
    print(f"  {label} — CUDA Kernel Summary (sorted by CUDA time)")
    print(f"{'='*80}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Compute totals for our kernels
    total_quant_cuda = 0
    total_gemm_cuda = 0
    total_other_cuda = 0
    quant_count = 0
    gemm_count = 0

    for evt in prof.key_averages():
        name = evt.key.lower()
        cuda_time = getattr(evt, 'self_cuda_time_total', 0)
        if "quant" in name or "absmax" in name or "pack" in name:
            total_quant_cuda += cuda_time
            quant_count += evt.count
        elif "gemm" in name or "scaled_mm" in name or "cutlass" in name:
            total_gemm_cuda += cuda_time
            gemm_count += evt.count
        elif cuda_time > 0:
            total_other_cuda += cuda_time

    total = total_quant_cuda + total_gemm_cuda + total_other_cuda
    print(f"\n  Breakdown:")
    print(f"    Quant kernels:  {total_quant_cuda/1000:>8.2f}ms  ({quant_count} calls)")
    print(f"    GEMM kernels:   {total_gemm_cuda/1000:>8.2f}ms  ({gemm_count} calls)")
    print(f"    Other CUDA:     {total_other_cuda/1000:>8.2f}ms")
    print(f"    Total CUDA:     {total/1000:>8.2f}ms")
    return total_quant_cuda, total_gemm_cuda, total_other_cuda


def main():
    model_dir = find_model_dir()
    print(f"Model dir: {model_dir}")

    M = BATCH_TOKENS  # batch tokens

    # Run PerCh profile
    prof_perch = run_perchannel_profile(model_dir, M)
    perch_quant, perch_gemm, perch_other = print_profile_summary("PerCh", prof_perch)

    # Free GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # Run PG profile
    prof_pg = run_pergroup_profile(model_dir, M)
    pg_quant, pg_gemm, pg_other = print_profile_summary("PG-sym", prof_pg)

    # Comparison
    print(f"\n\n{'='*80}")
    print(f"  COMPARISON: PerCh vs PG (M={M}, {NUM_LAYERS} layers × {len(W4A4_LAYERS)} W4A4 projections)")
    print(f"{'='*80}")
    print(f"                    PerCh(ms)    PG(ms)     Diff(ms)    Ratio")
    print(f"  Quant kernels:  {perch_quant/1000:>9.2f}   {pg_quant/1000:>9.2f}   {(pg_quant-perch_quant)/1000:>+9.2f}   {pg_quant/max(perch_quant,1):>6.2f}x")
    print(f"  GEMM kernels:   {perch_gemm/1000:>9.2f}   {pg_gemm/1000:>9.2f}   {(pg_gemm-perch_gemm)/1000:>+9.2f}   {pg_gemm/max(perch_gemm,1):>6.2f}x")
    print(f"  Other CUDA:     {perch_other/1000:>9.2f}   {pg_other/1000:>9.2f}   {(pg_other-perch_other)/1000:>+9.2f}")
    total_perch = perch_quant + perch_gemm + perch_other
    total_pg = pg_quant + pg_gemm + pg_other
    print(f"  {'─'*60}")
    print(f"  Total CUDA:     {total_perch/1000:>9.2f}   {total_pg/1000:>9.2f}   {(total_pg-total_perch)/1000:>+9.2f}   {total_pg/max(total_perch,1):>6.2f}x")
    print()


if __name__ == "__main__":
    main()
