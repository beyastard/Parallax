import os
import glob
import random
import argparse
import tqdm
import numpy as np
from transformers import AutoTokenizer

# USAGE:
# Default (scans data/raw_text/ for .txt files):
#    python prepare_data.py
#
# Custom paths:
#    python prepare_data.py --input_dir data/raw_text/ --output_dir data/ --val_pct 0.05
#
# Using a local tokenizer instead of downloading:
#    python prepare_data.py --tokenizer /path/to/llama2/tokenizer

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize raw text data for Parallax training.")
    parser.add_argument("--input_dir",  type=str, default="data/raw_text/",
                        help="Directory containing .txt files to tokenize")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Directory to write train.bin and val.bin")
    parser.add_argument("--tokenizer",  type=str, default="meta-llama/Llama-2-7b-hf",
                        help="HuggingFace tokenizer name or local path")
    parser.add_argument("--val_pct",    type=float, default=0.05,
                        help="Fraction of documents to use for validation (default: 0.05)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for document-level train/val shuffle")
    args = parser.parse_args()

    # --- 1. Load Llama 2 tokenizer ---
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Llama 2's EOS token (<\s>, id=2) is used as the document separator
    eos_id = tokenizer.eos_token_id
    print(f"Vocab size: {tokenizer.vocab_size}  |  EOS token id: {eos_id}")

    # --- 2. Collect all .txt files ---
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
    if not files:
        print(f"No .txt files found in '{args.input_dir}'. Please add some data first.")
        return
    print(f"Found {len(files)} file(s). Starting tokenization...")

    # --- 3. Tokenize each file into a list of token arrays (one per document) ---
    # We keep documents separate so we can do a clean document-level train/val split
    doc_tokens = []
    for path in tqdm.tqdm(files, desc="Tokenizing files"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # encode() on AutoTokenizer returns a list of ints; add_special_tokens=False
        # avoids prepending a BOS on every file (we only want EOS as separator)
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)  # Document separator
        doc_tokens.append(ids)

    # --- 4. Shuffle at the document level for a representative val split ---
    rng = random.Random(args.seed)
    rng.shuffle(doc_tokens)

    val_count  = max(1, int(len(doc_tokens) * args.val_pct))
    val_docs   = doc_tokens[:val_count]
    train_docs = doc_tokens[val_count:]

    print(f"Split: {len(train_docs)} train docs / {len(val_docs)} val docs")

    # --- 5. Flatten and convert to uint32 numpy arrays ---
    # uint16 caps at 65,535 which technically fits Llama 2's 32,000 vocab,
    # but uint32 is the safe, conventional dtype for token IDs
    def flatten(docs):
        flat = []
        for doc in docs:
            flat.extend(doc)
        return np.array(flat, dtype=np.uint32)

    train_arr = flatten(train_docs)
    val_arr   = flatten(val_docs)

    # --- 6. Write binary files ---
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path   = os.path.join(args.output_dir, "val.bin")

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    total = len(train_arr) + len(val_arr)
    print(f"\nTokenization complete!")
    print(f"  Total tokens : {total:,}")
    print(f"  Train tokens : {len(train_arr):,}  →  {train_path}")
    print(f"  Val tokens   : {len(val_arr):,}  →  {val_path}")

if __name__ == "__main__":
    main()
