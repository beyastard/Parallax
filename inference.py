import os
import argparse
import torch
from torch.amp import autocast
from safetensors.torch import load_file
from transformers import AutoTokenizer
from model import Parallax
from config import ParallaxConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained Parallax model.")

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/parallax_v1",
                        help="Directory containing model and meta checkpoint files")
    parser.add_argument("--tag",            type=str, default="best",
                        help="Checkpoint tag to load, e.g. 'best' or 'iter_5000'")
    parser.add_argument("--tokenizer",      type=str, default="meta-llama/Llama-2-7b-hf",
                        help="HuggingFace tokenizer name or local path")

    # Generation
    parser.add_argument("--prompt",         type=str,   default="Once upon a time, there was a little bird named")
    parser.add_argument("--max_new_tokens", type=int,   default=200)
    parser.add_argument("--temperature",    type=float, default=0.7,
                        help="Sampling temperature. Lower = more focused, higher = more creative")
    parser.add_argument("--top_k",          type=int,   default=50,
                        help="Top-K candidates to sample from (0 = disabled)")

    return parser.parse_args()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, config, device):
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        x_cond = x[:, -config.block_size:]

        with autocast(device_type=device_type, dtype=torch.float16):
            logits, _ = model(x_cond)

        # Take the logits for the last token position across the full vocab
        logits = logits[:, -1, :] / temperature

        # Top-K filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Stop at EOS
        if next_token.item() == eos_id:
            break

        x = torch.cat((x, next_token), dim=1)

    # Decode only the newly generated tokens (exclude the prompt)
    generated_ids = x[0, len(ids):].tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights_path = os.path.join(args.checkpoint_dir, f"model_{args.tag}.safetensors")
    meta_path    = os.path.join(args.checkpoint_dir, f"meta_{args.tag}.pt")

    # --- 1. Load metadata and reconstruct config ---
    # weights_only=False is required because the meta file contains a plain dict
    # with non-tensor values (iter, loss, optimizer state, config dict)
    meta   = torch.load(meta_path, map_location=device, weights_only=False)
    config = ParallaxConfig(**meta['config'])

    print(f"Loaded checkpoint '{args.tag}'  |  iter {meta['iter']}  |  loss {meta['loss']:.4f}")

    # --- 2. Load tokenizer ---
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # --- 3. Initialise model and load weights ---
    model = Parallax(config).to(device)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Device: {device}")

    # --- 4. Generate ---
    print(f"\n--- Parallax Generation (tag: {args.tag}) ---")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    output = generate(
        model, tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        config=config,
        device=device,
    )

    print(args.prompt + output)
    print("-" * 40)


if __name__ == "__main__":
    main()
