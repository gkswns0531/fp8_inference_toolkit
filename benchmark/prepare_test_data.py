#!/usr/bin/env python3
"""
Test Data Preparation Script

Project Gutenberg에서 "War and Peace" 텍스트를 다운로드하고,
지정된 tokenizer로 다양한 길이의 텍스트 파일을 생성합니다.

Requirements:
    pip install transformers requests

Usage:
    # Qwen3-Next tokenizer (생성형 모델용)
    python benchmark/prepare_test_data.py \
        --tokenizer Qwen/Qwen3-Next-80B-A3B-Instruct \
        --lengths 128,256,512,1024,2048,4096,8192,16384,32768,65536,131072

    # Qwen3-Embedding tokenizer (임베딩 모델용)
    python benchmark/prepare_test_data.py \
        --tokenizer Qwen/Qwen3-Embedding-0.6B \
        --lengths 128,256,512,1024,2048,4096,8192

    # BGE-M3 tokenizer
    python benchmark/prepare_test_data.py \
        --tokenizer BAAI/bge-m3 \
        --lengths 128,256,512,1024,2048,4096,8192
"""

import argparse
import json
import os
import re
import time
import urllib.request


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BOOK_URL = "https://www.gutenberg.org/files/2600/2600-0.txt"
DEFAULT_TOKENIZER = "Qwen/Qwen3-Next-80B-A3B-Instruct"
DEFAULT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


# ─────────────────────────────────────────────────────────────────────────────
# Download & Cache
# ─────────────────────────────────────────────────────────────────────────────

def download_book(url: str, cache_path: str) -> str:
    """Download book text from URL, using cache if available."""
    if os.path.exists(cache_path):
        print(f"  Using cached book: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    print(f"  Downloading from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"  Saved to cache: {cache_path} ({len(raw):,} chars)")
    return raw


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg header and footer, keeping only the body text."""
    # Common start markers
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
    ]
    # Common end markers
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Move past the marker line
            newline_idx = text.find("\n", idx)
            if newline_idx != -1:
                start_idx = newline_idx + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    body = text[start_idx:end_idx].strip()

    # Clean up excessive whitespace while preserving paragraph structure
    body = re.sub(r"\n{3,}", "\n\n", body)

    return body


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(tokenizer_id: str):
    """Load tokenizer from HuggingFace."""
    from transformers import AutoTokenizer
    print(f"  Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    return tokenizer


def tokenizer_short_name(tokenizer_id: str) -> str:
    """Create a filesystem-safe short name from the tokenizer ID."""
    name = tokenizer_id.split("/")[-1].lower()
    # Simplify common patterns
    name = re.sub(r"-instruct$", "", name)
    name = re.sub(r"-[0-9]+b(-a[0-9]+b)?$", "", name)
    name = re.sub(r"[^a-z0-9-]", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_test_files(
    tokenizer,
    tokenizer_id: str,
    body_text: str,
    lengths: list[int],
    output_dir: str,
) -> dict:
    """Tokenize body text and generate test files at each requested token length."""
    short_name = tokenizer_short_name(tokenizer_id)
    out_path = os.path.join(output_dir, short_name)
    os.makedirs(out_path, exist_ok=True)

    print(f"\n  Tokenizing full text with {tokenizer_id} ...")
    all_tokens = tokenizer.encode(body_text, add_special_tokens=False)
    total_tokens = len(all_tokens)
    print(f"  Total tokens in book: {total_tokens:,}")

    metadata = {
        "tokenizer": tokenizer_id,
        "short_name": short_name,
        "source_total_tokens": total_tokens,
        "source_total_chars": len(body_text),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {},
    }

    for length in lengths:
        filename = f"{length}_tokens.txt"
        filepath = os.path.join(out_path, filename)

        if length > total_tokens:
            print(f"  SKIP {length:>7,} tokens — exceeds available {total_tokens:,}")
            metadata["files"][filename] = {
                "requested_tokens": length,
                "status": "skipped",
                "reason": f"Exceeds available tokens ({total_tokens:,})",
            }
            continue

        # Slice tokens and decode back to text
        sliced_tokens = all_tokens[:length]
        decoded_text = tokenizer.decode(sliced_tokens, skip_special_tokens=True)

        # Verify actual token count after round-trip
        re_encoded = tokenizer.encode(decoded_text, add_special_tokens=False)
        actual_tokens = len(re_encoded)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(decoded_text)

        metadata["files"][filename] = {
            "requested_tokens": length,
            "actual_tokens": actual_tokens,
            "char_count": len(decoded_text),
            "status": "ok",
        }

        print(f"  OK   {length:>7,} tokens -> {actual_tokens:>7,} actual, {len(decoded_text):>8,} chars  [{filename}]")

    # Save metadata
    meta_path = os.path.join(out_path, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved: {meta_path}")

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare benchmark test data from Project Gutenberg texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"HuggingFace tokenizer model ID (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default=",".join(str(l) for l in DEFAULT_LENGTHS),
        help="Comma-separated token lengths to generate (default: 128,256,...,131072)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--book-url",
        type=str,
        default=DEFAULT_BOOK_URL,
        help="URL of the source text (default: War and Peace from Project Gutenberg)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    lengths.sort()

    print("=" * 70)
    print("Benchmark Test Data Preparation")
    print("=" * 70)
    print(f"  Tokenizer:  {args.tokenizer}")
    print(f"  Lengths:    {lengths}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Book URL:   {args.book_url}")
    print()

    # Step 1: Download / cache book
    cache_path = os.path.join(args.output_dir, "war_and_peace.txt")
    raw_text = download_book(args.book_url, cache_path)

    # Step 2: Strip Gutenberg headers
    body_text = strip_gutenberg_header_footer(raw_text)
    print(f"  Body text: {len(body_text):,} chars (after stripping headers)")

    # Step 3: Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # Step 4: Generate test files
    metadata = generate_test_files(tokenizer, args.tokenizer, body_text, lengths, args.output_dir)

    # Summary
    ok_count = sum(1 for v in metadata["files"].values() if v["status"] == "ok")
    skip_count = sum(1 for v in metadata["files"].values() if v["status"] == "skipped")
    print(f"\nDone! Generated {ok_count} files, skipped {skip_count}")
    print(f"Output: {os.path.join(args.output_dir, tokenizer_short_name(args.tokenizer))}/")


if __name__ == "__main__":
    main()
