# scripts/inference.py
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
import torch

from torch.amp import autocast
from safetensors.torch import load_file
from transformers import AutoTokenizer

from model.parallax import ParallaxConfig, ParallaxForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained Parallax model.")

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/parallax_v1",
                        help="Directory containing model and meta checkpoint files")
    parser.add_argument("--tag",            type=str, default="best",
                        help="Checkpoint tag to load, e.g. 'best' or 'iter_5000'")
    parser.add_argument("--tokenizer",      type=str, default=None,
                        help="HuggingFace tokenizer name or local path. Defaults to the "
                             "tokenizer saved alongside the checkpoint.")

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
        x_cond = x[:, -config.max_position_embeddings:]

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

    for path, name in [(weights_path, "weights"), (meta_path, "meta")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    # --- 1. Load metadata and reconstruct config ---
    # weights_only=False is required because the meta file contains a plain dict
    # with non-tensor values (iter, loss, optimizer state, config dict).
    # ParallaxConfig accepts both old-style (legacy field names) and new-style
    # (HF field names) meta dicts via its legacy alias kwargs.
    meta   = torch.load(meta_path, map_location=device, weights_only=False)
    config = ParallaxConfig(**meta["config"])

    loss_str = f"{meta['loss']:.4f}" if meta.get("loss") is not None else "n/a"
    print(f"Loaded checkpoint '{args.tag}'  |  iter {meta['iter']}  |  loss {loss_str}")
    print(f"  hidden_size={config.hidden_size}, "
          f"num_hidden_layers={config.num_hidden_layers}, "
          f"num_attention_heads={config.num_attention_heads}, "
          f"num_key_value_heads={config.num_key_value_heads}, "
          f"max_position_embeddings={config.max_position_embeddings}")

    # --- 2. Load tokenizer ---
    # Prefer the tokenizer saved alongside the checkpoint; fall back to --tokenizer
    # argument, then to NousResearch as a last resort.
    saved_tokenizer_path = os.path.join(args.checkpoint_dir, "tokenizer")
    if args.tokenizer is not None:
        tokenizer_source = args.tokenizer
    elif os.path.isdir(saved_tokenizer_path):
        tokenizer_source = saved_tokenizer_path
    else:
        tokenizer_source = "NousResearch/Llama-2-7b-hf"
        print("  Warning: no saved tokenizer found alongside checkpoint.")
        print("  Falling back to NousResearch/Llama-2-7b-hf from HuggingFace.")
    print(f"Loading tokenizer: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    # --- 3. Initialise model and load weights ---
    model = ParallaxForCausalLM(config).to(device)
    state_dict = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        non_freq = [k for k in missing
                    if k not in ("model.freqs_cos", "model.freqs_sin")]
        if non_freq:
            print(f"  Warning: {len(non_freq)} unexpected missing keys:")
            for k in non_freq[:5]:
                print(f"    {k}")
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
