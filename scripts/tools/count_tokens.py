# scripts/tools/count_tokens.py
# Count tokens in dataset
#
# Copyright (C) 2025-2026 Bryan K Reinhart & BeySoft
#
# This file is part of Parallax.
#
# Parallax is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Parallax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with Parallax. If not, see <https://www.gnu.org/licenses/>.

import os
import argparse
import numpy as np

# USAGE:
# Count tokens in both train and val:
#    python count_tokens.py
#
# Count tokens in specific files:
#    python count_tokens.py --files data/train.bin data/val.bin
#
# Show epoch estimate for a given config:
#    python count_tokens.py --block_size 512 --batch_size 4 --grad_accum 8 --max_iters 100000

def parse_args():
    parser = argparse.ArgumentParser(
        description="Count tokens in pre-tokenized .bin files and estimate training epochs."
    )
    parser.add_argument("--files",       type=str, nargs='+',
                        default=["data/train.bin", "data/val.bin"],
                        help="One or more .bin files to inspect")
    parser.add_argument("--block_size",  type=int, default=512,
                        help="Context window size (from config.py)")
    parser.add_argument("--batch_size",  type=int, default=4,
                        help="Batch size (from train.py)")
    parser.add_argument("--grad_accum",  type=int, default=8,
                        help="Gradient accumulation steps (from train.py)")
    parser.add_argument("--max_iters",   type=int, default=100000,
                        help="Planned training iterations")
    return parser.parse_args()


def human_readable(n):
    """Format a large integer as a readable string with units."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def main():
    args = parse_args()

    print("=" * 55)
    print("  Parallax Token Counter")
    print("=" * 55)

    train_tokens = None

    for path in args.files:
        if not os.path.isfile(path):
            print(f"  [NOT FOUND] {path}")
            continue

        # np.uint32 must match qtok.py / prepare_data.py output dtype
        data = np.memmap(path, dtype=np.uint32, mode='r')
        n = len(data)
        size_mb = os.path.getsize(path) / 1024 / 1024

        print(f"\n  File    : {path}")
        print(f"  Tokens  : {n:,}  ({human_readable(n)})")
        print(f"  Size    : {size_mb:.1f} MB")
        print(f"  dtype   : {data.dtype}  (uint32 = 4 bytes/token)")

        # Token range sanity check — catch dtype mismatches early
        token_min = int(data.min())
        token_max = int(data.max())
        print(f"  Range   : {token_min} → {token_max}", end="")
        if token_max >= 32000:
            print(f"  ⚠️  WARNING: token id {token_max} exceeds Llama 2 vocab size (32000)")
        else:
            print(f"  ✓  within Llama 2 vocab (32000)")

        # Track train file for epoch estimates
        if "train" in os.path.basename(path).lower():
            train_tokens = n

    # --- Training estimates (based on train.bin) ---
    if train_tokens is not None:
        tokens_per_iter  = args.batch_size * args.grad_accum * args.block_size
        total_tokens     = tokens_per_iter * args.max_iters
        epochs           = total_tokens / train_tokens
        iters_per_epoch  = train_tokens / tokens_per_iter

        print(f"\n{'=' * 55}")
        print(f"  Training Estimates  (train.bin = {human_readable(train_tokens)} tokens)")
        print(f"{'=' * 55}")
        print(f"  Tokens per iteration : {tokens_per_iter:,}  "
              f"(batch {args.batch_size} × accum {args.grad_accum} × block {args.block_size})")
        print(f"  Iterations per epoch : {iters_per_epoch:,.0f}")
        print(f"  Total tokens @ {args.max_iters:,} iters : {human_readable(total_tokens)}")
        print(f"  Epochs covered       : {epochs:.2f}")
        print()

        # Show how many tokens were seen at common checkpoints
        checkpoints = [1000, 2000, 5000, 10000, 20000, 50000, args.max_iters]
        checkpoints = sorted(set(checkpoints))
        print(f"  {'Iter':>8} | {'Tokens Seen':>12} | {'Epoch':>8}")
        print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*8}")
        for cp in checkpoints:
            if cp > args.max_iters:
                continue
            tokens_at = cp * tokens_per_iter
            epoch_at  = tokens_at / train_tokens
            print(f"  {cp:>8,} | {human_readable(tokens_at):>12} | {epoch_at:>8.3f}")
    else:
        print("\n  (No train.bin found — skipping training estimates)")

    print()

if __name__ == "__main__":
    main()
