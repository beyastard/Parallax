# Initialize an "empty" model

import os
import torch
from safetensors.torch import save_file
from config import ParallaxConfig
from model import Parallax

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    # Non-embedding params are often a better measure of "intelligence"
    non_emb_params = total_params - model.tok_emb.weight.numel()
    return total_params, non_emb_params

def estimate_vram_usage(model, config):
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
    model = Parallax(config)
    
    total, non_emb = count_parameters(model)
    
    print(f"\n[Parallax Configuration]")
    print(f"Total Parameters:         {total/1e6:.2f}M")
    print(f"Non-Embedding Parameters: {non_emb/1e6:.2f}M")

    estimate_vram_usage(model)

    # Save the blank (untrained) model weights
    init_dir = 'checkpoints/parallax_v1/initial'
    os.makedirs(init_dir, exist_ok=True)

    weights_path = os.path.join(init_dir, "model_blank.safetensors")
    save_file(model.state_dict(), weights_path)
    
    # Save metadata as a plain dict to avoid import-path dependencies at load time
    # Using config.__dict__ means loading this file only needs torch, not the config module
    meta = {
        'iter': 0,
        'loss': 0.0,
        'config': config.__dict__
    }
    torch.save(meta, os.path.join(init_dir, "meta_blank.pt"))

    print(f"\nSuccessfully saved blank baseline to: {init_dir}")

if __name__ == "__main__":
    main()
