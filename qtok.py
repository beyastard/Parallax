# qtok.py - Quick TOKenizer

import os
import argparse
import tqdm
import numpy as np
from transformers import AutoTokenizer

# USAGE:
# Tokenize the TinyStories training file:
#    python qtok.py --input data/raw_text/TinyStories-train.txt --output data/train.bin
#
# Tokenize the TinyStories validation file:
#    python qtok.py --input data/raw_text/TinyStories-valid.txt --output data/val.bin
#
# Using a local tokenizer:
#    python qtok.py --input data/raw_text/TinyStories-train.txt --output data/train.bin \
#                   --tokenizer /path/to/llama2/tokenizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quickly tokenize a single text file into a .bin for Parallax training. "
                    "No splitting or shuffling — what goes in is what comes out."
    )
    parser.add_argument("--input",     type=str, required=True,
                        help="Path to the input .txt file")
    parser.add_argument("--output",    type=str, required=True,
                        help="Path to the output .bin file (e.g. data/train.bin)")
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Llama-2-7b-hf",
                        help="HuggingFace tokenizer name or local path")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Number of characters to tokenize per chunk (reduces peak RAM usage)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id
    print(f"Vocab size: {tokenizer.vocab_size}  |  EOS id: {eos_id}")

    # --- Stream and tokenize in chunks to keep RAM low ---
    # We read the file in character-level chunks, tokenize each, and write
    # incrementally so even a 2GB file won't blow out memory.
    input_size = os.path.getsize(args.input)
    print(f"Input file: {args.input}  ({input_size / 1024 / 1024:.1f} MB)")
    print(f"Output file: {args.output}")
    print(f"Tokenizing...")

    total_tokens = 0
    # Use a temporary file so a failed run doesn't leave a corrupt .bin
    tmp_path = args.output + ".tmp"

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(tmp_path, "wb") as fout, \
         tqdm.tqdm(total=input_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:

        remainder = ""
        while True:
            chunk = fin.read(args.chunk_size)
            if not chunk:
                # Flush any remaining text
                if remainder:
                    ids = tokenizer.encode(remainder, add_special_tokens=False)
                    ids.append(eos_id)
                    arr = np.array(ids, dtype=np.uint32)
                    fout.write(arr.tobytes())
                    total_tokens += len(ids)
                break

            # Combine with any leftover from the previous chunk to avoid
            # splitting a word or multi-byte character at the boundary
            text = remainder + chunk

            # Find the last newline so we don't cut mid-sentence
            split_at = text.rfind("\n")
            if split_at == -1:
                # No newline found — process the whole thing and carry nothing over
                remainder = ""
            else:
                remainder = text[split_at + 1:]
                text = text[:split_at + 1]

            ids = tokenizer.encode(text, add_special_tokens=False)
            arr = np.array(ids, dtype=np.uint32)
            fout.write(arr.tobytes())
            total_tokens += len(ids)
            pbar.update(len(chunk.encode("utf-8")))

    # Rename tmp to final output only on clean completion
    os.replace(tmp_path, args.output)

    file_size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nDone.")
    print(f"  Total tokens : {total_tokens:,}")
    print(f"  Output size  : {file_size_mb:.1f} MB  ({args.output})")
    print(f"  Avg tokens/MB: {total_tokens / (input_size / 1024 / 1024):,.0f}")


if __name__ == "__main__":
    main()
