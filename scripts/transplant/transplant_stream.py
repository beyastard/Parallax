# scripts/transplant/transplant_stream.py
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
transplant_stream.py
====================
Generalised streaming weight transplant from any Llama-family donor
into a Parallax model. Works with local paths or HuggingFace model IDs,
single safetensors files or sharded models.

Auto-detects donor architecture from config.json. Processes one layer
at a time to minimise RAM usage (~500 MB peak per layer).

Supported donor architectures:
  - Llama 3.x (hidden=2048/3072/4096, vocab=128256, GQA)
  - Llama 2.x (hidden=2048/4096, vocab=32000, GQA)
  - Mistral (hidden=4096, vocab=32000, sliding window — treated as standard)
  - Any model with standard HF LlamaForCausalLM key naming

Interleave strategy:
  Donor layer 0  → Track A layer 0
  Donor layer 1  → Track B layer 0
  Donor layer 2  → Track A layer 1
  ...
  Donor layer N-1 → Track B/A layer (N-1)//2

  If donor has odd layer count, final layer goes to Track A,
  and Track B's last layer is a copy of Track A's last layer.

Usage:
  # Local model
  python transplant_stream.py --donor D:/models/Nano_Imp_1B --output checkpoints/parallax_nano

  # HuggingFace model
  python transplant_stream.py --donor SicariusSicariiStuff/Nano_Imp_1B --output checkpoints/parallax_nano

  # Dry run (no weights loaded, just shows plan)
  python transplant_stream.py --donor D:/models/Nano_Imp_1B --dry_run

  # Override Parallax key prefix if your model.py uses different names
  python transplant_stream.py --donor D:/models/Nano_Imp_1B --track_prefix tracks

Dependencies:
  pip install transformers safetensors huggingface_hub torch
"""

import os
import sys
import gc
import json
import glob
import argparse
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generalised streaming Llama-family → Parallax transplant"
)
parser.add_argument("--donor", required=True,
    help="Local path to model directory OR HuggingFace model ID (e.g. 'SicariusSicariiStuff/Nano_Imp_1B')")
parser.add_argument("--output", required=True,
    help="Output directory for transplanted Parallax checkpoint")
parser.add_argument("--cache_dir", default="./hf_cache",
    help="HuggingFace download cache (only used when --donor is a HF model ID)")
parser.add_argument("--track_prefix", default=None,
    help="Override auto-detected Parallax track key prefix. "
         "Auto-detected by inspecting model.py state dict. "
         "Set manually if auto-detection fails (e.g. 'tracks' if keys are 'tracks.0.layers...')")
parser.add_argument("--num_loops", type=int, default=2,
    help="Parallax num_loops (default 2)")
#parser.add_argument("--use_swap", type=lambda x: x.lower() != 'false', default=True,
#    help="Parallax use_swap (default True)")
parser.add_argument("--use_swap",
    type=lambda x: x.lower() not in ("false", "0", "no"),
    default=None,
    metavar="BOOL",
    help="Enable track swap (default: True). Pass false/0/no to disable.")
parser.add_argument("--transplant_mode", default="interleave",
    choices=["interleave", "linear"],
    help="interleave: odd→A, even→B. linear: first half→A, second half→B")
parser.add_argument("--dry_run", action="store_true",
    help="Print plan without loading or saving any weights")
parser.add_argument("--copy_odd_to_even", action="store_true",
    help="If donor has odd layer count, copy final Track A layer to Track B "
         "instead of leaving Track B's last layer uninitialised")
args = parser.parse_args()

is_local = os.path.isdir(args.donor)

print("=" * 70)
print("  Parallax Generalised Streaming Transplant")
print("=" * 70)
print(f"  Donor source:  {'LOCAL' if is_local else 'HuggingFace'}")
print(f"  Donor path:    {args.donor}")
print(f"  Output:        {args.output}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load donor config.json to auto-detect architecture
# ─────────────────────────────────────────────────────────────────────────────
print("  Step 1: Reading donor config.json...")

def load_donor_config(donor, cache_dir, is_local):
    if is_local:
        cfg_path = os.path.join(donor, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"No config.json found at {cfg_path}")
        with open(cfg_path) as f:
            return json.load(f)
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(
            repo_id=donor, filename="config.json", cache_dir=cache_dir
        )
        with open(cfg_path) as f:
            return json.load(f)

donor_cfg = load_donor_config(args.donor, args.cache_dir, is_local)

# Extract architecture parameters
DONOR_HIDDEN       = donor_cfg.get("hidden_size")
DONOR_LAYERS       = donor_cfg.get("num_hidden_layers")
DONOR_HEADS        = donor_cfg.get("num_attention_heads")
DONOR_KV_HEADS     = donor_cfg.get("num_key_value_heads", DONOR_HEADS)
DONOR_INTERMEDIATE = donor_cfg.get("intermediate_size")
DONOR_VOCAB        = donor_cfg.get("vocab_size")
DONOR_HEAD_DIM     = DONOR_HIDDEN // DONOR_HEADS
DONOR_NORM_EPS     = donor_cfg.get("rms_norm_eps", 1e-5)
DONOR_ROPE_BASE    = donor_cfg.get("rope_theta", 10000.0)
DONOR_ARCH         = donor_cfg.get("model_type", "llama")

# Detect QKV storage format
# Llama 3.x / Mistral: separate q_proj, k_proj, v_proj
# Some older models: combined qkv_proj
DONOR_COMBINED_QKV = donor_cfg.get("_qkv_combined", False)  # manual override if needed

# Parallax layer count: half of donor (interleaved)
PARALLAX_N_LAYER = DONOR_LAYERS // 2
ODD_LAYERS       = DONOR_LAYERS % 2 == 1

print(f"  Detected donor architecture: {DONOR_ARCH}")
print(f"  hidden_size:          {DONOR_HIDDEN}")
print(f"  num_hidden_layers:    {DONOR_LAYERS}")
print(f"  num_attention_heads:  {DONOR_HEADS}  (head_dim={DONOR_HEAD_DIM})")
print(f"  num_key_value_heads:  {DONOR_KV_HEADS}")
print(f"  intermediate_size:    {DONOR_INTERMEDIATE}")
print(f"  vocab_size:           {DONOR_VOCAB}")
print(f"  rms_norm_eps:         {DONOR_NORM_EPS}")
print(f"  rope_theta:           {DONOR_ROPE_BASE}")
print()
print(f"  → Parallax num_hidden_layers per track: {PARALLAX_N_LAYER}")
if ODD_LAYERS:
    print(f"  → Odd donor layer count: final layer goes to Track A only")
    if args.copy_odd_to_even:
        print(f"    (will be copied to Track B as well via --copy_odd_to_even)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build weight file index
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Step 2: Indexing weight files...")

def build_weight_index(donor, cache_dir, is_local):
    """Returns (weight_map, shard_resolver_fn).
    weight_map: {tensor_name: absolute_local_path}
    """
    def local_shard_path(filename):
        return os.path.join(donor, filename)

    def hf_shard_path(filename):
        # Check if already cached
        model_id_safe = donor.replace("/", "--")
        pattern = os.path.join(
            os.path.expanduser(cache_dir),
            f"models--{model_id_safe}", "snapshots", "*", filename
        )
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        # Download on demand
        from huggingface_hub import hf_hub_download
        print(f"    Downloading: {filename}")
        return hf_hub_download(repo_id=donor, filename=filename, cache_dir=cache_dir)

    resolver = local_shard_path if is_local else hf_shard_path

    # Try sharded index first
    if is_local:
        index_path = os.path.join(donor, "model.safetensors.index.json")
    else:
        try:
            from huggingface_hub import hf_hub_download
            index_path = hf_hub_download(
                repo_id=donor,
                filename="model.safetensors.index.json",
                cache_dir=cache_dir
            )
        except Exception:
            index_path = None

    if index_path and os.path.exists(index_path if index_path else ""):
        with open(index_path) as f:
            index = json.load(f)
        # Map tensor name → absolute local path
        weight_map = {
            k: resolver(v)
            for k, v in index["weight_map"].items()
        }
        print(f"  Sharded model: {len(set(weight_map.values()))} shards, {len(weight_map)} tensors")
        return weight_map

    # Single file fallback
    if is_local:
        single = os.path.join(donor, "model.safetensors")
    else:
        single = hf_shard_path("model.safetensors")

    if not os.path.exists(single):
        raise FileNotFoundError(f"Cannot find safetensors weights at {donor}")

    weight_map = {}
    with safe_open(single, framework="pt", device="cpu") as f:
        for key in f.keys():
            weight_map[key] = single
    print(f"  Single-file model: {len(weight_map)} tensors in {os.path.basename(single)}")
    return weight_map

weight_map = build_weight_index(args.donor, args.cache_dir, is_local)

def get_tensor(name):
    """Load a single tensor by name — minimal RAM, lazy from shard."""
    path = weight_map.get(name)
    if path is None:
        raise KeyError(f"Tensor not in weight map: {name!r}\n"
                       f"Available keys matching pattern: "
                       f"{[k for k in weight_map if name.split('.')[-1] in k][:5]}")
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(torch.float32)

# Detect QKV format by inspecting keys
sample_layer_keys = [k for k in weight_map if "layers.0.self_attn" in k]
has_combined_qkv  = any("qkv_proj" in k for k in sample_layer_keys)
has_separate_qkv  = any("q_proj" in k for k in sample_layer_keys)
print(f"  QKV format: {'combined qkv_proj' if has_combined_qkv else 'separate q/k/v_proj'}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Detect Parallax key naming by direct probe inspection
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Step 3: Detecting Parallax key naming...")

# Defaults — overridden by probe or manual flag below
TRACK_A_PREFIX = "model.track_a"
TRACK_B_PREFIX = "model.track_b"
ATTN_KEY       = "attn"
FFN_KEY        = "ffn"
WQ_KEY = "wq";  WK_KEY = "wk";  WV_KEY = "wv";  WO_KEY = "wo"
W1_KEY = "w1";  W2_KEY = "w2";  W3_KEY = "w3"
NORM_ATTN_KEY  = "norm1"
NORM_FFN_KEY   = "norm2"
EMBED_KEY       = "model.embed_tokens.weight"   # token embedding key in Parallax
OUTPUT_NORM_KEY = "model.output_norm.weight"     # final pre-head norm key in Parallax

if args.track_prefix:
    TRACK_A_PREFIX = f"{args.track_prefix}.0"
    TRACK_B_PREFIX = f"{args.track_prefix}.1"
    print(f"  Manual override: track_a={TRACK_A_PREFIX}, track_b={TRACK_B_PREFIX}")
else:
    try:
        sys.path.insert(0, os.getcwd())
        from model.parallax import ParallaxConfig, ParallaxForCausalLM

        probe_cfg = ParallaxConfig(
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            hidden_size=8, max_position_embeddings=8, vocab_size=32
        )
        probe      = ParallaxForCausalLM(probe_cfg)
        probe_keys = list(probe.state_dict().keys())
        del probe
        gc.collect()

        print(f"  Probe keys (first 25):")
        for k in probe_keys[:25]:
            print(f"    {k}")

        # ── detect embedding key ──────────────────────────────────────────────
        for candidate in ["model.embed_tokens.weight", "tok_emb.weight",
                          "embedding.weight", "wte.weight"]:
            if candidate in probe_keys:
                EMBED_KEY = candidate
                break

        # ── detect output norm key ────────────────────────────────────────────
        for candidate in ["model.output_norm.weight", "output_norm.weight",
                          "model.norm.weight", "final_norm.weight", "ln_f.weight"]:
            if candidate in probe_keys:
                OUTPUT_NORM_KEY = candidate
                break

        # ── find all keys that contain a layer index digit ────────────────────
        # Key pattern: <prefix>.<digit>.<submod>.<wname>.weight
        # e.g. track_a.0.attn.wq.weight  or  track_b.3.ffn.w2.weight
        import re
        layer_re = re.compile(r'^(.+?)\.(\d+)\.(.+?)\.(.+?)\.weight$')

        track_a_layers = []
        track_b_layers = []

        for k in probe_keys:
            m = layer_re.match(k)
            if not m:
                continue
            prefix, layer_idx, submod, wname = m.groups()
            if not track_a_layers:
                # First match → Track A
                TRACK_A_PREFIX = prefix
            if prefix == TRACK_A_PREFIX:
                track_a_layers.append((submod, wname))
            elif not track_b_layers or prefix == TRACK_B_PREFIX:
                TRACK_B_PREFIX = prefix
                track_b_layers.append((submod, wname))

        # ── extract submodule and weight key names from Track A layers ─────────
        for submod, wname in track_a_layers:
            s = submod.lower()
            w = wname.lower()
            if any(x in s for x in ["attn", "attention", "self_attn"]):
                ATTN_KEY = submod
                if w in ("wq", "q_proj", "query"):   WQ_KEY = wname
                elif w in ("wk", "k_proj", "key"):   WK_KEY = wname
                elif w in ("wv", "v_proj", "value"): WV_KEY = wname
                elif w in ("wo", "o_proj", "out"):   WO_KEY = wname
            elif any(x in s for x in ["ffn", "mlp", "feed_forward"]):
                FFN_KEY = submod
                if w in ("w1", "gate_proj", "gate"): W1_KEY = wname
                elif w in ("w2", "down_proj", "down"): W2_KEY = wname
                elif w in ("w3", "up_proj", "up"):   W3_KEY = wname
            elif "norm" in s:
                # norm1/norm2 or attn_norm/ffn_norm etc.
                # First norm encountered = attn norm, second = ffn norm
                if NORM_ATTN_KEY == "norm1" and submod != NORM_ATTN_KEY:
                    NORM_ATTN_KEY = submod
                elif submod != NORM_ATTN_KEY:
                    NORM_FFN_KEY  = submod

        # ── verify all keys were found ─────────────────────────────────────────
        missing_detection = [name for name, val in [
            ("WQ_KEY", WQ_KEY), ("WK_KEY", WK_KEY), ("WV_KEY", WV_KEY),
            ("WO_KEY", WO_KEY), ("W1_KEY", W1_KEY), ("W2_KEY", W2_KEY),
            ("W3_KEY", W3_KEY),
        ] if val is None]
        if missing_detection:
            print(f"  WARNING: Could not auto-detect: {missing_detection}")
            print(f"  Using defaults for missing entries.")

    except ImportError:
        print("  WARNING: Cannot import Parallax — run from project directory.")
        print("  Using hardcoded defaults.")

print(f"  track_a prefix:  {TRACK_A_PREFIX}")
print(f"  track_b prefix:  {TRACK_B_PREFIX}")
print(f"  embed key:       {EMBED_KEY}")
print(f"  output norm key: {OUTPUT_NORM_KEY}")
print(f"  attn submodule:  {ATTN_KEY}   Q={WQ_KEY} K={WK_KEY} V={WV_KEY} O={WO_KEY}")
print(f"  ffn  submodule:  {FFN_KEY}    W1={W1_KEY} W2={W2_KEY} W3={W3_KEY}")
print(f"  norm keys:       attn={NORM_ATTN_KEY}  ffn={NORM_FFN_KEY}")

# ─────────────────────────────────────────────────────────────────────────────
# Key builder helpers
# ─────────────────────────────────────────────────────────────────────────────
def pkey(track, layer_idx, submod, weight):
    """Build Parallax state dict key: <prefix>.<layer>.<submod>.<weight>.weight"""
    prefix = TRACK_A_PREFIX if track == "A" else TRACK_B_PREFIX
    return f"{prefix}.{layer_idx}.{submod}.{weight}.weight"

def norm_key(track, layer_idx, which):
    """Build Parallax norm key. which='attn' or 'ffn'."""
    prefix = TRACK_A_PREFIX if track == "A" else TRACK_B_PREFIX
    nk = NORM_ATTN_KEY if which == "attn" else NORM_FFN_KEY
    return f"{prefix}.{layer_idx}.{nk}.weight"

def dkey(layer_idx, component):
    """Build donor state dict key for a given layer/component."""
    base = f"model.layers.{layer_idx}"
    mapping = {
        "q":         f"{base}.self_attn.q_proj.weight",
        "k":         f"{base}.self_attn.k_proj.weight",
        "v":         f"{base}.self_attn.v_proj.weight",
        "qkv":       f"{base}.self_attn.qkv_proj.weight",
        "o":         f"{base}.self_attn.o_proj.weight",
        "gate":      f"{base}.mlp.gate_proj.weight",
        "up":        f"{base}.mlp.up_proj.weight",
        "down":      f"{base}.mlp.down_proj.weight",
        "attn_norm": f"{base}.input_layernorm.weight",
        "ffn_norm":  f"{base}.post_attention_layernorm.weight",
    }
    return mapping[component]

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Print interleave plan
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Interleave plan ({DONOR_LAYERS} donor layers → {PARALLAX_N_LAYER} per track):")
print(f"  {'Donor':>6} │ {'Track':>7} │ {'Parallax Layer':>14}")
print(f"  {'─'*6}─┼─{'─'*7}─┼─{'─'*14}")
for i in range(DONOR_LAYERS):
    if args.transplant_mode == "interleave":
        track  = "A" if i % 2 == 0 else "B"
        player = i // 2
    else:
        track  = "A" if i < DONOR_LAYERS // 2 else "B"
        player = i if track == "A" else i - (DONOR_LAYERS // 2)
    print(f"  {i:>6} │ {'Track '+track:>7} │ {player:>14}")
if ODD_LAYERS and args.copy_odd_to_even:
    print(f"  (final Track A layer {PARALLAX_N_LAYER-1} also copied to Track B {PARALLAX_N_LAYER-1})")

if args.dry_run:
    print("\n  [dry_run] Exiting without loading weights.")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Save tokenizer
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Step 4: Saving tokenizer...")
os.makedirs(args.output, exist_ok=True)
tok_out = os.path.join(args.output, "tokenizer")
os.makedirs(tok_out, exist_ok=True)
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.donor,
        cache_dir=None if is_local else args.cache_dir,
        trust_remote_code=True,
        local_files_only=is_local,
    )
    tokenizer.save_pretrained(tok_out)
    print(f"  Tokenizer saved: vocab_size={tokenizer.vocab_size}, {tok_out}")
except Exception as e:
    print(f"  WARNING: Could not save tokenizer ({e})")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Streaming layer transplant
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Step 5: Streaming transplant (one donor layer at a time)...")
q_dim = DONOR_HEAD_DIM * DONOR_HEADS
k_dim = DONOR_HEAD_DIM * DONOR_KV_HEADS
v_dim = DONOR_HEAD_DIM * DONOR_KV_HEADS

new_sd = {}

# 6a. Shared tensors (small, load once)
print("  [shared] Embedding, final norm, LM head...")
embed_key = "model.embed_tokens.weight"
norm_key_donor = "model.norm.weight"
head_key = "lm_head.weight"

new_sd[EMBED_KEY] = get_tensor(embed_key).to(torch.float16)
new_sd[OUTPUT_NORM_KEY] = get_tensor(norm_key_donor).to(torch.float16)

if head_key in weight_map:
    new_sd["lm_head.weight"] = get_tensor(head_key).to(torch.float16)
else:
    print("  [shared] lm_head not found — using tied embedding (transposed)")
    new_sd["lm_head.weight"] = new_sd[EMBED_KEY].clone()

gc.collect()

# 6b. Layer-by-layer
last_a_layer = {}  # stores final Track A layer tensors for odd-layer copy

for donor_layer in range(DONOR_LAYERS):
    if args.transplant_mode == "interleave":
        track  = "A" if donor_layer % 2 == 0 else "B"
        player = donor_layer // 2
    else:   # linear
        track  = "A" if donor_layer < DONOR_LAYERS // 2 else "B"
        player = donor_layer if track == "A" else donor_layer - (DONOR_LAYERS // 2)

    print(f"  [L{donor_layer:02d}→{track}{player}] ", end="", flush=True)

    # Load Q, K, V (handling both separate and combined formats)
    if has_combined_qkv:
        qkv = get_tensor(dkey(donor_layer, "qkv"))
        wq, wk, wv = torch.split(qkv, [q_dim, k_dim, v_dim], dim=0)
        del qkv
    else:
        wq = get_tensor(dkey(donor_layer, "q"))
        wk = get_tensor(dkey(donor_layer, "k"))
        wv = get_tensor(dkey(donor_layer, "v"))

    wo   = get_tensor(dkey(donor_layer, "o"))
    gate = get_tensor(dkey(donor_layer, "gate"))
    up   = get_tensor(dkey(donor_layer, "up"))
    down = get_tensor(dkey(donor_layer, "down"))
    an   = get_tensor(dkey(donor_layer, "attn_norm"))
    fn   = get_tensor(dkey(donor_layer, "ffn_norm"))

    layer_data = {
        pkey(track, player, ATTN_KEY, WQ_KEY): wq.to(torch.float16),
        pkey(track, player, ATTN_KEY, WK_KEY): wk.to(torch.float16),
        pkey(track, player, ATTN_KEY, WV_KEY): wv.to(torch.float16),
        pkey(track, player, ATTN_KEY, WO_KEY): wo.to(torch.float16),
        pkey(track, player, FFN_KEY,  W1_KEY): gate.to(torch.float16),
        pkey(track, player, FFN_KEY,  W2_KEY): up.to(torch.float16),
        pkey(track, player, FFN_KEY,  W3_KEY): down.to(torch.float16),
        norm_key(track, player, "attn"):        an.to(torch.float16),
        norm_key(track, player, "ffn"):         fn.to(torch.float16),
    }
    new_sd.update(layer_data)

    # Keep last Track A layer for potential odd-layer copy
    if track == "A" and ODD_LAYERS and donor_layer == DONOR_LAYERS - 1:
        last_a_layer = {k.replace(f".{player}.", f".{player}."): v.clone()
                        for k, v in layer_data.items()
                        if TRACK_A_PREFIX in k}

    del wq, wk, wv, wo, gate, up, down, an, fn
    gc.collect()
    print(f"done")

# 6c. Handle odd donor layer count
if ODD_LAYERS:
    if args.copy_odd_to_even:
        print(f"  [odd]  Copying final Track A layer to Track B layer {PARALLAX_N_LAYER-1}...")
        for k, v in last_a_layer.items():
            new_key = k.replace(TRACK_A_PREFIX, TRACK_B_PREFIX)
            new_sd[new_key] = v.clone()
    else:
        print(f"  [odd]  Track B layer {PARALLAX_N_LAYER-1} will use random init "
              f"(pass --copy_odd_to_even to mirror Track A)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Save output
# ─────────────────────────────────────────────────────────────────────────────
out_weights = os.path.join(args.output, "model_transplant.safetensors")
print(f"\n  Step 6: Saving {len(new_sd)} tensors → {out_weights}")
save_file(new_sd, out_weights)

# Save meta / config for inference_transplant.py and fine_tune.py
meta = {
    "donor":             args.donor,
    "donor_arch":        DONOR_ARCH,
    "donor_layers":      DONOR_LAYERS,
    "parallax_n_layer":  PARALLAX_N_LAYER,
    "config": {
        "max_position_embeddings": min(donor_cfg.get("max_position_embeddings", 2048), 2048),
        "vocab_size":              DONOR_VOCAB,
        "num_hidden_layers":       PARALLAX_N_LAYER,
        "num_attention_heads":     DONOR_HEADS,
        "num_key_value_heads":     DONOR_KV_HEADS,
        "hidden_size":             DONOR_HIDDEN,
        "intermediate_size":       DONOR_INTERMEDIATE,
        "rope_theta":              DONOR_ROPE_BASE,
        "attention_dropout":       0.0,
        "attention_bias":          False,
        "num_loops":               args.num_loops,
        "use_swap":                args.use_swap,
        "hidden_act":              "swiglu",
        "rms_norm_eps":            DONOR_NORM_EPS,
    },
    "iter":  0,
    "loss":  None,
    "key_map": {
        "track_a_prefix":  TRACK_A_PREFIX,
        "track_b_prefix":  TRACK_B_PREFIX,
        "attn_key":        ATTN_KEY,
        "ffn_key":         FFN_KEY,
        "wq": WQ_KEY, "wk": WK_KEY, "wv": WV_KEY, "wo": WO_KEY,
        "w1": W1_KEY, "w2": W2_KEY, "w3": W3_KEY,
        "norm_attn": NORM_ATTN_KEY, "norm_ffn": NORM_FFN_KEY,
    }
}
torch.save(meta, os.path.join(args.output, "meta_transplant.pt"))
print(f"  Meta saved.")

print("\n" + "=" * 70)
print("  Transplant complete.")
print()
print("  Parallax config for this transplant:")
cfg = meta["config"]
for k, v in cfg.items():
    print(f"    {k:30s} = {v}")
print()
print("  Verify key loading:")
print("    python scripts/tools/gen_inference.py \\")
print(f"      --checkpoint {args.output} \\")
print(f'      --prompt "Once upon a time"')
print()
print("  Fine-tune:")
print("    python scripts/fine_tune.py \\")
print(f"      --base_checkpoint {args.output}/model_transplant.safetensors \\")
print(f"      --base_meta {args.output}/meta_transplant.pt \\")
print("      --train_path data/train.bin --val_path data/val.bin \\")
print("      --lr 1e-4 --warmup_iters 200 --max_iters 5000")
print("=" * 70)
