#!/usr/bin/env python3
"""Run calibration to collect activation statistics for SmoothQuant.

Usage:
    python calibration/run_calibration.py \
        --model Qwen/Qwen3-Embedding-4B \
        --num-samples 128 --seq-len 512 \
        --output calibration_stats.pt

    # With custom dataset
    python calibration/run_calibration.py \
        --model Qwen/Qwen3-Embedding-4B \
        --dataset wikitext --dataset-split test \
        --num-samples 256 --output calibration_stats.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add parent dir to path for calibration package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibration.collector import ActivationStatsCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect activation statistics for SmoothQuant calibration")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for calibration (default: 512)")
    parser.add_argument("--output", type=str, default="calibration_stats.pt",
                        help="Output path for stats file (default: calibration_stats.pt)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (e.g., wikitext). "
                             "If not specified, random tokens are used.")
    parser.add_argument("--dataset-split", type=str, default="test",
                        help="Dataset split to use (default: test)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run calibration on (default: cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (default: bfloat16)")
    return parser.parse_args()


def get_calibration_tokens(
    args: argparse.Namespace,
    tokenizer,
) -> torch.Tensor:
    """Get calibration input_ids tensor [num_samples, seq_len]."""
    if args.dataset is not None:
        from datasets import load_dataset
        # Support "name,config" format (e.g. "wikitext,wikitext-2-raw-v1")
        ds_parts = args.dataset.split(",")
        if len(ds_parts) == 2:
            dataset = load_dataset(ds_parts[0], ds_parts[1],
                                   split=args.dataset_split)
        else:
            dataset = load_dataset(args.dataset, split=args.dataset_split)
        # Concatenate text samples
        text_key = "text" if "text" in dataset.column_names else dataset.column_names[0]
        texts = [s[text_key] for s in dataset if len(s[text_key].strip()) > 0]

        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            if len(all_ids) >= args.num_samples * args.seq_len:
                break

        all_ids = all_ids[:args.num_samples * args.seq_len]
        if len(all_ids) < args.num_samples * args.seq_len:
            print(f"Warning: only got {len(all_ids)} tokens, "
                  f"need {args.num_samples * args.seq_len}. "
                  f"Padding with random tokens.")
            pad = torch.randint(0, tokenizer.vocab_size,
                                (args.num_samples * args.seq_len - len(all_ids),))
            all_ids = all_ids + pad.tolist()

        input_ids = torch.tensor(all_ids).reshape(args.num_samples, args.seq_len)
    else:
        # Random tokens for quick calibration
        print("No dataset specified, using random tokens for calibration.")
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size,
                                  (args.num_samples, args.seq_len))

    return input_ids


def main() -> None:
    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model}")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    print("Preparing calibration data...")
    input_ids = get_calibration_tokens(args, tokenizer).to(args.device)

    print(f"Collecting activation statistics ({args.num_samples} samples, "
          f"seq_len={args.seq_len})...")
    collector = ActivationStatsCollector(model)

    with torch.no_grad():
        for i in range(args.num_samples):
            model(input_ids[i:i+1])
            if (i + 1) % 32 == 0:
                print(f"  Processed {i + 1}/{args.num_samples} samples")

    collector.remove_hooks()

    print(f"Collected stats for {len(collector.stats)} layers")
    print(f"Saving to: {args.output}")
    collector.save(args.output)

    # Print summary
    print("\nCalibration Summary:")
    print(f"{'Layer':<60} {'Channels':>8} {'Act Max':>10} {'Samples':>8}")
    print("-" * 90)
    for name, stats in sorted(collector.stats.items()):
        act_max = stats.input_absmax.max().item()
        channels = stats.input_absmax.shape[0]
        print(f"{name:<60} {channels:>8} {act_max:>10.4f} {stats.num_samples:>8}")

    print(f"\nDone! Stats saved to {args.output}")


if __name__ == "__main__":
    main()
