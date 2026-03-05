"""
gen_inference.py
============
General-purpose inference script for Parallax models.
Handles both transplant checkpoints (meta_transplant.pt)
and fine-tuned checkpoints (meta_best.pt / meta_final.pt).

Auto-detects checkpoint type and reconstructs config accordingly.

Usage:
  # Fine-tuned checkpoint (best weights)
  python gen_inference.py --checkpoint checkpoints/parallax_amd_linear_ft

  # Specific meta/weights file
  python gen_inference.py --checkpoint checkpoints/parallax_amd_linear_ft ^
      --meta meta_best.pt --weights model_best.safetensors

  # Transplant checkpoint (pre-fine-tune)
  python gen_inference.py --checkpoint checkpoints/parallax_amd_linear

  # Raw prompt (no chat template)
  python gen_inference.py --checkpoint checkpoints/parallax_amd_linear_ft ^
      --template raw --prompt "Once upon a time"

  # Interactive mode
  python gen_inference.py --checkpoint checkpoints/parallax_amd_linear_ft --interactive
"""

import os
import sys
import argparse
import torch

from safetensors.torch import load_file

# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Parallax inference — works with any checkpoint")
parser.add_argument("--checkpoint", required=True,
    help="Directory containing model weights and meta file")
parser.add_argument("--weights", default=None,
    help="Weights filename inside checkpoint dir. "
         "Auto-detected if not specified (prefers model_best over model_final over model_transplant)")
parser.add_argument("--meta", default=None,
    help="Meta filename inside checkpoint dir. Auto-detected if not specified.")
parser.add_argument("--prompt", default="Once upon a time",
    help="Prompt text (template tokens added automatically unless --template raw)")
parser.add_argument("--system", default="",
    help="System prompt (only used with chat templates)")
parser.add_argument("--template", default="auto",
    choices=["auto", "llama3", "llama2", "phi3", "chatml", "raw"],
    help="Chat template. 'auto' detects from tokenizer. "
         "Fine-tuned TinyStories models should use 'raw'.")
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--interactive", action="store_true",
    help="Interactive prompt loop — enter prompts one at a time, 'quit' to exit")
parser.add_argument("--block_size_override", type=int, default=None)
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect weights and meta files
# ─────────────────────────────────────────────────────────────────────────────
def find_file(directory, candidates):
    """Return the first candidate filename that exists in directory."""
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None

if args.weights:
    weights_path = os.path.join(args.checkpoint, args.weights)
else:
    weights_path = find_file(args.checkpoint, [
        "model_best.safetensors",
        "model_final.safetensors",
        "model_transplant.safetensors",
    ])

if args.meta:
    meta_path = os.path.join(args.checkpoint, args.meta)
else:
    meta_path = find_file(args.checkpoint, [
        "meta_best.pt",
        "meta_final.pt",
        "meta_transplant.pt",
    ])

if not weights_path or not os.path.exists(weights_path):
    print(f"ERROR: No weights file found in {args.checkpoint}")
    print("  Expected one of: model_best.safetensors, model_final.safetensors, model_transplant.safetensors")
    sys.exit(1)

if not meta_path or not os.path.exists(meta_path):
    print(f"ERROR: No meta file found in {args.checkpoint}")
    print("  Expected one of: meta_best.pt, meta_final.pt, meta_transplant.pt")
    sys.exit(1)

print(f"\n  Weights: {os.path.basename(weights_path)}")
print(f"  Meta:    {os.path.basename(meta_path)}")

# ─────────────────────────────────────────────────────────────────────────────
# Load meta — handle both fine-tune and transplant formats
# ─────────────────────────────────────────────────────────────────────────────
meta = torch.load(meta_path, map_location="cpu", weights_only=False)

# Detect checkpoint type
is_transplant = "donor" in meta
is_finetune   = "optimizer" in meta

if is_transplant:
    checkpoint_type = "transplant"
    cfg_dict = meta["config"]
    donor_arch = meta.get("donor_arch", "llama")
    print(f"  Type:    transplant (donor: {meta.get('donor', 'unknown')})")
elif is_finetune:
    checkpoint_type = "finetune"
    cfg_dict = meta["config"]
    donor_arch = "raw"  # fine-tuned models use raw prompting by default
    loss_str = f"{meta['loss']:.4f}" if meta.get('loss') is not None else "n/a"
    print(f"  Type:    fine-tuned (iter {meta.get('iter', '?')}, loss {loss_str})")
else:
    # Fallback — treat config dict directly
    checkpoint_type = "unknown"
    cfg_dict = meta.get("config", meta)
    donor_arch = "raw"
    print(f"  Type:    unknown format — attempting direct config load")

if args.block_size_override:
    cfg_dict["block_size"] = args.block_size_override
    print(f"  block_size overridden → {args.block_size_override}")

# ─────────────────────────────────────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.getcwd())
try:
    from model.parallax.modeling_parallax import Parallax
    from model.parallax.configuration_parallax import ParallaxConfig
except ImportError as e:
    print(f"\nERROR: Cannot import Parallax — run from your project directory.\n({e})")
    sys.exit(1)

valid_fields = ParallaxConfig.__dataclass_fields__.keys()
config = ParallaxConfig(**{k: v for k, v in cfg_dict.items() if k in valid_fields})

print(f"\n  Config:  n_layer={config.n_layer}, n_embd={config.n_embd}, "
      f"n_head={config.n_head}, n_kv_heads={config.n_kv_heads}")
print(f"           vocab={config.vocab_size}, block_size={config.block_size}, "
      f"ffn_dim={config.ffn_dim}")

model = Parallax(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Params:  {total_params/1e6:.2f}M")

# ─────────────────────────────────────────────────────────────────────────────
# Load weights
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Loading weights...")
state_dict = load_file(weights_path, device="cpu")
result = model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

loaded  = len(model.state_dict()) - len(result.missing_keys)
total   = len(model.state_dict())
print(f"  Loaded:  {loaded}/{total} keys ({100*loaded/total:.1f}%)")
if result.missing_keys:
    non_freq = [k for k in result.missing_keys
                if k not in ("freqs_cos", "freqs_sin")]
    if non_freq:
        print(f"  WARNING: {len(non_freq)} unexpected missing keys:")
        for k in non_freq[:5]:
            print(f"    {k}")
    else:
        print(f"  Note: freqs_cos/sin computed at runtime — expected")

# ─────────────────────────────────────────────────────────────────────────────
# Load tokenizer
# ─────────────────────────────────────────────────────────────────────────────
tok_path = os.path.join(args.checkpoint, "tokenizer")
if not os.path.exists(tok_path):
    print(f"\nERROR: No tokenizer directory at {tok_path}")
    sys.exit(1)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
print(f"  Tokenizer: vocab={tokenizer.vocab_size}, "
      f"eos={tokenizer.eos_token_id} ({tokenizer.eos_token!r})")

# ─────────────────────────────────────────────────────────────────────────────
# Template detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_template(donor_arch, tokenizer, checkpoint_type):
    # Fine-tuned models on plain text datasets → raw
    if checkpoint_type == "finetune":
        return "raw"
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
    return "raw"

template = args.template
if template == "auto":
    template = detect_template(donor_arch, tokenizer, checkpoint_type)
    print(f"  Template: {template} (auto-detected)")
else:
    print(f"  Template: {template}")

# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────────────────────
def format_prompt(user_msg, system_msg, template):
    if template == "llama3":
        bos = "<|begin_of_text|>"
        sh  = "<|start_header_id|>";  eh = "<|end_header_id|>";  eot = "<|eot_id|>"
        out = [bos]
        if system_msg:
            out.append(f"{sh}system{eh}\n\n{system_msg}{eot}")
        out.append(f"{sh}user{eh}\n\n{user_msg}{eot}")
        out.append(f"{sh}assistant{eh}\n\n")
        return "".join(out)
    elif template == "llama2":
        if system_msg:
            return f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]"
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

# ─────────────────────────────────────────────────────────────────────────────
# Stop tokens
# ─────────────────────────────────────────────────────────────────────────────
stop_ids = set()
if tokenizer.eos_token_id is not None:
    stop_ids.add(tokenizer.eos_token_id)
for s in {"llama3": ["<|eot_id|>", "<|end_of_text|>"],
           "phi3":   ["<|end|>"],
           "chatml": ["<|im_end|>"],
           "llama2": ["</s>"]}.get(template, []):
    tid = tokenizer.convert_tokens_to_ids(s)
    if tid is not None and tid != tokenizer.unk_token_id:
        stop_ids.add(tid)

# ─────────────────────────────────────────────────────────────────────────────
# Generation function
# ─────────────────────────────────────────────────────────────────────────────
def generate(prompt_text, system_text=""):
    formatted   = format_prompt(prompt_text, system_text, template)
    input_ids   = tokenizer.encode(formatted, return_tensors="pt").to(device)
    prompt_len  = input_ids.shape[1]

    if prompt_len >= config.block_size:
        print(f"  WARNING: prompt ({prompt_len} tokens) >= block_size ({config.block_size})")

    generated     = input_ids.clone()
    recent_tokens = []

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            ctx = generated[:, -config.block_size:]
            with torch.amp.autocast(device_type=device.split(":")[0],
                                    dtype=torch.float16,
                                    enabled=(device == "cuda")):
                logits, _ = model(ctx)

            logits_last = logits[:, -1, :].float()

            if args.temperature > 0:
                logits_last /= args.temperature

            if args.repetition_penalty != 1.0 and recent_tokens:
                for tid in set(recent_tokens):
                    v = logits_last[0, tid]
                    logits_last[0, tid] = v / args.repetition_penalty if v > 0 \
                                          else v * args.repetition_penalty

            if args.top_k > 0:
                kv, _ = torch.topk(logits_last, min(args.top_k, logits_last.size(-1)))
                logits_last[logits_last < kv[:, -1:]] = float("-inf")

            if args.top_p < 1.0:
                sv, si = torch.sort(logits_last, descending=True)
                cp = torch.cumsum(torch.softmax(sv, dim=-1), dim=-1)
                sv[cp > args.top_p] = float("-inf")
                logits_last.scatter_(1, si, sv)

            probs      = torch.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tok_id     = next_token.item()

            generated = torch.cat([generated, next_token], dim=1)
            recent_tokens.append(tok_id)
            if len(recent_tokens) > 128:
                recent_tokens.pop(0)

            tok_str = tokenizer.decode(next_token[0], skip_special_tokens=False)
            print(tok_str, end="", flush=True)

            if tok_id in stop_ids:
                break

    print()
    response_ids  = generated[0, prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  temperature={args.temperature}, top_k={args.top_k}, "
      f"top_p={args.top_p}, rep_penalty={args.repetition_penalty}")
print(f"{'─'*70}\n")

if args.interactive:
    print("Interactive mode — type a prompt and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue
        print("\n[Output]\n")
        generate(prompt, args.system)
        print()
else:
    print(f"[Prompt]\n{args.prompt}\n")
    print("[Output]\n")
    generate(args.prompt, args.system)
    print(f"\n{'─'*70}")
