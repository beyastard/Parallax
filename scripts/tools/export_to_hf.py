# export_to_hf.py
# Converts a Parallax training checkpoint into a HuggingFace-compatible
# model directory suitable for push_to_hub() or local use with AutoModel.

import os
import argparse
import shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from model.parallax import ParallaxConfig, ParallaxForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", required=True,
    help="Parallax checkpoint directory (contains meta_*.pt and model_*.safetensors)")
parser.add_argument("--tag", default="best",
    help="Checkpoint tag to export (default: best)")
parser.add_argument("--output_dir", required=True,
    help="Output directory for HF-format model")
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--hub_repo", type=str, default=None,
    help="HuggingFace repo ID, e.g. 'username/parallax-65m'")
args = parser.parse_args()

# Load checkpoint
meta = torch.load(
    os.path.join(args.checkpoint_dir, f"meta_{args.tag}.pt"),
    map_location="cpu", weights_only=False
)
config = ParallaxConfig(**meta["config"])
model  = ParallaxForCausalLM(config)
state_dict = load_file(
    os.path.join(args.checkpoint_dir, f"model_{args.tag}.safetensors")
)
model.load_state_dict(state_dict, strict=False)
model.eval()

os.makedirs(args.output_dir, exist_ok=True)

# Save in HF format — this writes config.json and model.safetensors
model.save_pretrained(args.output_dir)

# Copy tokenizer
tok_src = os.path.join(args.checkpoint_dir, "tokenizer")
tokenizer = AutoTokenizer.from_pretrained(tok_src)
tokenizer.save_pretrained(args.output_dir)

print(f"Exported to: {args.output_dir}")
print(f"Contents: {os.listdir(args.output_dir)}")

if args.push_to_hub:
    assert args.hub_repo, "Provide --hub_repo username/model-name"
    model.push_to_hub(args.hub_repo)
    tokenizer.push_to_hub(args.hub_repo)
    print(f"Pushed to HuggingFace: {args.hub_repo}")
