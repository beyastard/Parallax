import os
import argparse
import random
import tqdm
from datasets import load_dataset

# USAGE:
# From Hugging Face:
#    python data_collector.py --source "roneneldan/TinyStories" --format hf --max_mb 200
#
# From a Local Parquet file:
#    python data_collector.py --source "./my_data.parquet" --format parquet --column "body_text"
#
# From a Local CSV file:
#    python data_collector.py --source "./stories.csv" --format csv --column "story_content"

# --- CONFIG ---
BUFFER_SIZE = 10000  # Number of rows to keep in memory for reservoir shuffling

def get_text_column(first_example, preferred_name="text"):
    """
    Identifies the best text column from the first row of a streaming dataset.
    Checks the preferred name first, then common candidates, then falls back
    to the first column whose value is a string.
    """
    columns = list(first_example.keys())

    if preferred_name in columns:
        return preferred_name

    candidates = ["content", "body", "story", "document", "raw_text"]
    for cand in candidates:
        if cand in columns:
            print(f"Warning: Column '{preferred_name}' not found. Using '{cand}' instead.")
            return cand

    for col in columns:
        if isinstance(first_example[col], str):
            print(f"Warning: No standard text column found. Falling back to '{col}'.")
            return col

    raise ValueError(f"Could not find a usable text column in: {columns}")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets for Parallax training.")
    parser.add_argument("--source",  type=str, required=True,
                        help="HuggingFace dataset path (e.g. 'roneneldan/TinyStories') or local file path")
    parser.add_argument("--format",  type=str, choices=['hf', 'csv', 'parquet', 'text'], default='hf')
    parser.add_argument("--column",  type=str, default='text',
                        help="Target text column name (auto-detected if not found)")
    parser.add_argument("--output",  type=str, default='data/raw_text/collected_data.txt')
    parser.add_argument("--max_mb",  type=int, default=100,
                        help="Target total collected text size in MB")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="Disable reservoir shuffling (writes data in stream order)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # --- 1. Load the streaming dataset ---
    print(f"Loading data from: {args.source}")
    if args.format == 'hf':
        ds = load_dataset(args.source, split='train', streaming=True)
    elif args.format == 'parquet':
        ds = load_dataset('parquet', data_files=args.source, split='train', streaming=True)
    elif args.format == 'csv':
        ds = load_dataset('csv', data_files=args.source, split='train', streaming=True)
    else:  # plain text
        ds = load_dataset('text', data_files=args.source, split='train', streaming=True)

    iterator = iter(ds)

    # --- 2. Detect the text column from the first example ---
    first_example = next(iterator)
    text_col = get_text_column(first_example, preferred_name=args.column)
    print(f"Using text column: '{text_col}'")

    # --- 3. Pre-fill the reservoir shuffle buffer ---
    target_bytes = args.max_mb * 1024 * 1024
    buffer = [first_example]  # Don't discard the first row we already pulled

    if not args.no_shuffle:
        print(f"Pre-filling shuffle buffer ({BUFFER_SIZE} rows)...")
        try:
            for _ in range(BUFFER_SIZE - 1):
                buffer.append(next(iterator))
        except StopIteration:
            print("Dataset is smaller than the buffer size — shuffling entire dataset in memory.")

    # --- 4. Stream, shuffle (reservoir), and write ---
    print(f"Writing shuffled text to: {args.output}  (target: {args.max_mb} MB)")
    current_bytes = 0

    with open(args.output, "w", encoding="utf-8") as f:
        with tqdm.tqdm(total=args.max_mb, unit="MB", dynamic_ncols=True) as pbar:
            while current_bytes < target_bytes:
                if not buffer:
                    break

                if args.no_shuffle:
                    # Sequential: pop from the front
                    entry = buffer.pop(0)
                else:
                    # Reservoir: pop a random entry, refill from stream
                    idx = random.randrange(len(buffer))
                    entry = buffer.pop(idx)

                text = str(entry.get(text_col, "")) + "\n\n"
                if not text.strip():
                    # Skip empty rows but still try to refill the buffer
                    pass
                else:
                    f.write(text)
                    chunk_mb = len(text.encode("utf-8")) / (1024 * 1024)
                    current_bytes += int(chunk_mb * 1024 * 1024)
                    pbar.update(chunk_mb)

                # Refill the buffer from the stream
                if not args.no_shuffle:
                    try:
                        buffer.append(next(iterator))
                    except StopIteration:
                        pass  # Stream exhausted; drain the remaining buffer

    print(f"\nDone. Collected {current_bytes / 1024 / 1024:.2f} MB → {args.output}")

if __name__ == "__main__":
    main()
