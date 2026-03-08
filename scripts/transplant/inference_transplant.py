# scripts/transplant/inference_transplant,py
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

"""
inference_transplant.py
=======================
Tests a transplanted Parallax model as-is, before any fine-tuning.
Reads the meta_transplant.pt saved by transplant_stream.py to
automatically configure the model and format the prompt correctly.

Supports any donor chat template:
  - Llama 3.x  (<|begin_of_text|><|start_header_id|>...)
  - Llama 2    ([INST] ... [/INST])
  - Phi-3      (<|user|>...<|end|><|assistant|>)
  - ChatML     (<|im_start|>user\n...<|im_end|>)
  - Raw        (no template, plain prompt passthrough)

Usage:
  # Auto-detect template from saved meta
  python inference_transplant.py --checkpoint checkpoints/parallax_nano

  # Override template
  python inference_transplant.py --checkpoint checkpoints/parallax_nano --template llama3

  # Custom system prompt
  python inference_transplant.py --checkpoint checkpoints/parallax_nano
      --system "You are a helpful assistant."
      --prompt "Tell me a short story about a cat."

  # Raw prompt (no template wrapping)
  python inference_transplant.py --checkpoint checkpoints/parallax_nano
      --template raw
      --prompt "Once upon a time"

  # Adjust generation settings
  python inference_transplant.py --checkpoint checkpoints/parallax_nano
      --max_new_tokens 300 --temperature 0.8 --top_k 40 --top_p 0.95

  # Reduce block_size if model is too large for VRAM
  python inference_transplant.py --checkpoint checkpoints/parallax_nano
      --block_size_override 512
"""

import os
import sys
import argparse
import torch

from safetensors.torch import load_file

# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Inference for transplanted Parallax model")
parser.add_argument("--checkpoint", required=True,
    help="Directory containing model_transplant.safetensors and meta_transplant.pt")
parser.add_argument("--prompt", default="Tell me a short story.",
    help="User message (template tokens added automatically)")
parser.add_argument("--system", default="",
    help="System prompt (optional)")
parser.add_argument("--template", default="auto",
    choices=["auto", "llama3", "llama2", "phi3", "chatml", "raw"],
    help="Chat template. 'auto' detects from donor config.")
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--repetition_penalty", type=float, default=1.1,
    help="Penalise recently seen tokens (1.0=off). Helps reduce repetition loops.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--block_size_override", type=int, default=None,
    help="Override block_size from meta — useful to reduce VRAM for large donors")
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Load meta
# ─────────────────────────────────────────────────────────────────────────────
meta_path = os.path.join(args.checkpoint, "meta_transplant.pt")
if not os.path.exists(meta_path):
    print(f"ERROR: No meta_transplant.pt at {args.checkpoint}")
    print("       Run transplant_stream.py first.")
    sys.exit(1)

meta     = torch.load(meta_path, map_location="cpu", weights_only=False)
cfg_dict = meta["config"]
meta["config"]["use_swap"] = False
meta["config"]["num_loops"] = 1

if args.block_size_override:
    cfg_dict["max_position_embeddings"] = args.block_size_override
    print(f"  max_position_embeddings overridden → {args.block_size_override}")

print(f"  Donor:      {meta.get('donor', 'unknown')}")
print(f"  Arch:       {meta.get('donor_arch', 'unknown')}")
print(f"  Parallax:   num_hidden_layers={cfg_dict.get('num_hidden_layers', cfg_dict.get('n_layer'))}, "
      f"hidden_size={cfg_dict.get('hidden_size', cfg_dict.get('n_embd'))}, "
      f"num_attention_heads={cfg_dict.get('num_attention_heads', cfg_dict.get('n_head'))}, "
      f"num_key_value_heads={cfg_dict.get('num_key_value_heads', cfg_dict.get('n_kv_heads'))}")
print(f"  vocab_size: {cfg_dict['vocab_size']},  "
      f"max_position_embeddings: {cfg_dict.get('max_position_embeddings', cfg_dict.get('block_size'))}")

# ─────────────────────────────────────────────────────────────────────────────
# Build Parallax model from transplant config
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.getcwd())
try:
    from model.parallax import ParallaxConfig, ParallaxForCausalLM
except ImportError as e:
    print(f"\nERROR: Cannot import Parallax — run from your project directory.\n({e})")
    sys.exit(1)

# ParallaxConfig accepts both old-style (legacy field names) and new-style
# (HF field names) config dicts via its legacy alias kwargs, so this works
# for both pre-refactor and post-refactor transplant checkpoints.
config = ParallaxConfig(**cfg_dict)

print(f"\n  Building Parallax model...")
model = ParallaxForCausalLM(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params/1e6:.2f}M")

# ─────────────────────────────────────────────────────────────────────────────
# Load transplanted weights
# ─────────────────────────────────────────────────────────────────────────────
weights_path = os.path.join(args.checkpoint, "model_transplant.safetensors")
if not os.path.exists(weights_path):
    print(f"ERROR: No model_transplant.safetensors at {args.checkpoint}")
    sys.exit(1)

print(f"  Loading weights...")
transplant_sd = load_file(weights_path, device="cpu")
result = model.load_state_dict(transplant_sd, strict=False)
model  = model.to(device)
model.eval()

total_keys = len(list(model.state_dict().keys()))
missing    = len(result.missing_keys)
loaded     = total_keys - missing

print(f"\n  Weight loading report:")
print(f"    Total keys:          {total_keys}")
print(f"    Loaded from transplant: {loaded}  ({100*loaded/total_keys:.1f}%)")
print(f"    Missing (random init):  {missing}")
print(f"    Unexpected (ignored):   {len(result.unexpected_keys)}")

if missing > 0:
    print(f"\n  Missing keys (first 10 shown):")
    for k in result.missing_keys[:10]:
        print(f"    {k}")
    if missing > 10:
        print(f"    ... and {missing-10} more")
    print("\n  ⚠  Missing keys = random weights for those layers.")
    print("     Verify config.py matches the transplant config.")
    print("     If many keys are missing, key names in transplant_stream.py")
    print("     need adjustment to match your model.py.")
else:
    print(f"    Status: ✓ Perfect — all keys matched")

# ─────────────────────────────────────────────────────────────────────────────
# Load tokenizer
# ─────────────────────────────────────────────────────────────────────────────
tok_path = os.path.join(args.checkpoint, "tokenizer")
if not os.path.exists(tok_path):
    print(f"\nERROR: No tokenizer directory at {tok_path}")
    sys.exit(1)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
print(f"\n  Tokenizer: vocab={tokenizer.vocab_size}, "
      f"eos_id={tokenizer.eos_token_id} ({tokenizer.eos_token!r})")

# ─────────────────────────────────────────────────────────────────────────────
# Detect chat template
# ─────────────────────────────────────────────────────────────────────────────
def detect_template(donor_arch, tokenizer):
    arch = (donor_arch or "").lower()
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        tmpl = tokenizer.chat_template
        if "begin_of_text" in tmpl or "start_header_id" in tmpl:
            return "llama3"
        if "im_start" in tmpl:
            return "chatml"
        if "<|user|>" in tmpl:
            return "phi3"
        if "[INST]" in tmpl:
            return "llama2"
    if "llama" in arch:
        return "llama3" if tokenizer.vocab_size > 100000 else "llama2"
    if "phi" in arch:
        return "phi3"
    if "mistral" in arch or "mixtral" in arch:
        return "llama2"
    return "chatml"

template = args.template
if template == "auto":
    template = detect_template(meta.get("donor_arch", ""), tokenizer)
    print(f"  Template auto-detected: {template}")

# ─────────────────────────────────────────────────────────────────────────────
# Format prompt
# ─────────────────────────────────────────────────────────────────────────────
def format_prompt(user_msg, system_msg, template):
    if template == "llama3":
        bos  = "<|begin_of_text|>"
        sh   = "<|start_header_id|>"
        eh   = "<|end_header_id|>"
        eot  = "<|eot_id|>"
        out  = [bos]
        if system_msg:
            out.append(f"{sh}system{eh}\n\n{system_msg}{eot}")
        out.append(f"{sh}user{eh}\n\n{user_msg}{eot}")
        out.append(f"{sh}assistant{eh}\n\n")
        return "".join(out)

    elif template == "llama2":
        if system_msg:
            return (f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
                    f"{user_msg} [/INST]")
        return f"<s>[INST] {user_msg} [/INST]"

    elif template == "phi3":
        out = []
        if system_msg:
            out.append(f"<|system|>\n{system_msg}<|end|>")
        out.append(f"<|user|>\n{user_msg}<|end|>")
        out.append("<|assistant|>\n")
        return "\n".join(out)

    elif template == "chatml":
        out = []
        if system_msg:
            out.append(f"<|im_start|>system\n{system_msg}<|im_end|>")
        out.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
        out.append("<|im_start|>assistant\n")
        return "\n".join(out)

    else:  # raw
        return user_msg

formatted = format_prompt(args.prompt, args.system, template)

# ─────────────────────────────────────────────────────────────────────────────
# Stop token set
# ─────────────────────────────────────────────────────────────────────────────
stop_ids = set()
if tokenizer.eos_token_id is not None:
    stop_ids.add(tokenizer.eos_token_id)

template_stop_strings = {
    "llama3": ["<|eot_id|>", "<|end_of_text|>"],
    "phi3":   ["<|end|>", "<|endoftext|>"],
    "chatml": ["<|im_end|>"],
    "llama2": ["</s>"],
}
for s in template_stop_strings.get(template, []):
    tid = tokenizer.convert_tokens_to_ids(s)
    if tid is not None and tid != tokenizer.unk_token_id:
        stop_ids.add(tid)

# ─────────────────────────────────────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  Template:           {template}")
if args.system:
    print(f"  System:             {args.system!r}")
print(f"  User prompt:        {args.prompt!r}")
print(f"  temperature={args.temperature}, top_k={args.top_k}, "
      f"top_p={args.top_p}, rep_penalty={args.repetition_penalty}")
print(f"{'─'*70}")
print(f"\n[Formatted prompt sent to model]\n{formatted}")
print(f"\n{'─'*70}")
print("[Generated output — streaming token by token]")
print(f"{'─'*70}\n")

input_ids  = tokenizer.encode(formatted, return_tensors="pt").to(device)
prompt_len = input_ids.shape[1]
print(f"(prompt length: {prompt_len} tokens)\n")

if prompt_len >= config.max_position_embeddings:
    print(f"⚠  Prompt ({prompt_len} tokens) >= max_position_embeddings ({config.max_position_embeddings}).")
    print(f"   Use --block_size_override N to increase, or shorten the prompt.")

generated     = input_ids.clone()
recent_tokens = []

with torch.no_grad():
    for step in range(args.max_new_tokens):
        ctx = generated[:, -config.max_position_embeddings:]

        with torch.amp.autocast(
            device_type=device.split(":")[0],
            dtype=torch.float16,
            enabled=(device == "cuda")
        ):
            logits, _ = model(ctx)

        logits_last = logits[:, -1, :].float()

        # Temperature scaling
        if args.temperature > 0:
            logits_last /= args.temperature

        # Repetition penalty over recent context
        if args.repetition_penalty != 1.0 and recent_tokens:
            for tid in set(recent_tokens):
                val = logits_last[0, tid]
                logits_last[0, tid] = val / args.repetition_penalty if val > 0 \
                                      else val * args.repetition_penalty

        # Top-k filtering
        if args.top_k > 0:
            kvals, _ = torch.topk(logits_last, min(args.top_k, logits_last.size(-1)))
            logits_last[logits_last < kvals[:, -1:]] = float("-inf")

        # Top-p (nucleus) filtering
        if args.top_p < 1.0:
            sorted_l, sorted_i = torch.sort(logits_last, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
            sorted_l[cum_probs > args.top_p] = float("-inf")
            logits_last.scatter_(1, sorted_i, sorted_l)

        probs      = torch.softmax(logits_last, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tok_id     = next_token.item()

        generated = torch.cat([generated, next_token], dim=1)
        recent_tokens.append(tok_id)
        if len(recent_tokens) > 128:
            recent_tokens.pop(0)

        # Stream decoded token
        tok_str = tokenizer.decode(next_token[0], skip_special_tokens=False)
        print(tok_str, end="", flush=True)

        if tok_id in stop_ids:
            break

new_tokens    = generated.shape[1] - prompt_len
response_ids  = generated[0, prompt_len:]
response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

print(f"\n\n{'─'*70}")
print(f"  Generated: {new_tokens} tokens")
print(f"\n[Response with special tokens stripped]\n{response_text}")
print(f"\n{'─'*70}")
print("\nNOTE: Pre-fine-tuning output quality reflects the donor's knowledge")
print("arranged through Parallax's interleaved dual-track structure.")
print("Content should be recognisable from the donor, fluency may vary.")
print("Fine-tune with fine_tune.py on domain data for best results.")
