# scripts/init_model.py
# Initialize an "empty" model as a baseline checkpoint.
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
import torch

from safetensors.torch import save_file

from model.parallax import ParallaxConfig, ParallaxForCausalLM


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    # Non-embedding params are often a better measure of "intelligence"
    non_emb_params = total_params - model.model.embed_tokens.weight.numel()
    return total_params, non_emb_params


def estimate_vram_usage(model):
    # Rough estimate: Weights (FP16) + Optimizer States (AdamW FP32: 2 states per param)
    # FP16 = 2 bytes/param, AdamW FP32 states = 8 bytes/param
    total_params = sum(p.numel() for p in model.parameters())
    param_size_mb = total_params * 2 / 1024 / 1024         # FP16 weights
    opt_size_mb   = total_params * 8 / 1024 / 1024         # AdamW optimizer states
    remaining_mb  = 6000 - (param_size_mb + opt_size_mb)

    print(f"--- VRAM Estimate (RTX 3050 6GB) ---")
    print(f"Model Weights (FP16):   ~{param_size_mb:.2f} MB")
    print(f"Optimizer States:       ~{opt_size_mb:.2f} MB")
    print(f"Remaining for Acts/Buf: ~{remaining_mb:.2f} MB")
    print("-" * 37)


def main():
    config = ParallaxConfig()
    model  = ParallaxForCausalLM(config)

    total, non_emb = count_parameters(model)

    print(f"\n[Parallax Configuration]")
    print(f"  hidden_size:             {config.hidden_size}")
    print(f"  num_hidden_layers:       {config.num_hidden_layers}  (per track)")
    print(f"  num_attention_heads:     {config.num_attention_heads}")
    print(f"  num_key_value_heads:     {config.num_key_value_heads}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  vocab_size:              {config.vocab_size}")
    print(f"  num_loops:               {config.num_loops}  (use_swap={config.use_swap})")
    print(f"  Total Parameters:        {total / 1e6:.2f}M")
    print(f"  Non-Embedding Params:    {non_emb / 1e6:.2f}M")
    print()
    estimate_vram_usage(model)

    # Save the blank (untrained) model weights
    init_dir = "checkpoints/parallax_v1/initial"
    os.makedirs(init_dir, exist_ok=True)

    weights_path = os.path.join(init_dir, "model_blank.safetensors")
    save_file(model.state_dict(), weights_path)

    # Use config.to_dict() (PretrainedConfig method) so all canonical HF field
    # names are stored and the meta is consistent with checkpoints from train.py.
    meta = {
        "iter":   0,
        "loss":   0.0,
        "config": config.to_dict(),
    }
    torch.save(meta, os.path.join(init_dir, "meta_blank.pt"))

    print(f"\nSuccessfully saved blank baseline to: {init_dir}")


if __name__ == "__main__":
    main()
