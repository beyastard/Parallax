# Parallax

> A dual-track transformer language model with cross-pollinating attention passes.

Copyright (C) 2025-2026 Bryan K Reinhart & BeySoft  
Licensed under the [GNU Affero General Public License v3.0](LICENSE)

---

Parallax is an experimental language model built around a novel architecture: instead of a
single linear stack of transformer layers, two independent tracks process the same input in
parallel, exchange their representations via a swap mechanism, then fuse at the end. This
allows each track to refine its understanding using the other track's contextualised view of
the same sequence — a mechanism inspired by the concept of parallax itself, where two offset
vantage points reveal depth that neither could resolve alone.

---

## Architecture

A standard transformer passes tokens through a single sequential stack of attention and
feed-forward layers. Parallax replaces this with the following structure:

```
             [ Input Tokens ]
                    |
          [ Token Embedding (x) ]
            |                  |
    ________|________   _______|________
   |                 | |                |
   |    TRACK A      | |    TRACK B     |
   |  n_layer Blocks | |  n_layer Blocks|
   |  GQA Attention  | |  GQA Attention |
   |  SwiGLU FFN     | |  SwiGLU FFN    |
   |  RMSNorm        | |  RMSNorm       |
   |_________________| |________________|
          |  out_a               |  out_b
          |                      |
   =========== THE SWAP ===========
          |                      |
          +--> in_b = out_a + x  |
          |        in_a = out_b -+
          |                + x
   =========== THE SWAP ===========
          |                      |
   _______|_________   __________|______
   |                 | |                |
   |    TRACK A      | |    TRACK B     |
   |  (Pass 2,       | |  (Pass 2,      |
   | shared weights) | | shared weights)|
   |_________________| |________________|
          |  out_a               |  out_b
          |                      |
          \__________  __________/
                     \/
              [ out_a + out_b ]
                     |
               [ RMSNorm ]
                     |
          [ Linear Projection ]
                     |
       [ Logits over Vocab (32k) ]
```

### Key Properties

- **Two parallel tracks** — Track A and Track B are separate `nn.ModuleList` instances with
  independent weights. They process the same token embeddings simultaneously on Pass 1.
- **The Swap** — After Pass 1, the output of Track A becomes the input to Track B's second
  pass, and vice versa. A residual connection back to the original embedding `x` is added to
  each swapped input to stabilise gradients.
- **Weight-tied passes** — Both Pass 1 and Pass 2 use the same layer weights within each
  track. This gives Parallax a recurrent character — layers are reused across depth rather
  than being unique per pass.
- **Configurable loop count** — `num_loops` in the config controls how many passes are
  performed. At `num_loops=2` (default), one swap event occurs. At `num_loops=3`, two swaps
  occur. At `num_loops=1`, no swap occurs and the model behaves as a parallel ensemble with
  no cross-track interaction until the final fusion.
- **Final fusion** — The outputs of both tracks are summed, normalised with RMSNorm, and
  projected to vocabulary logits.

### Components

| Component | Implementation |
|-----------|----------------|
| Attention | Grouped Query Attention (GQA) with `num_attention_heads` query heads and `num_key_value_heads` KV heads |
| Position encoding | Rotary Position Embeddings (RoPE), applied to Q and K inside each attention block |
| Feed-forward | SwiGLU (`W3(SiLU(W1(x)) * W2(x))`) |
| Normalisation | RMSNorm (pre-norm, applied before attention and FFN in each block) |
| Precision | FP16 via `torch.amp.autocast` during training and inference |
| Attention kernel | Flash Attention 2 via `torch.nn.functional.scaled_dot_product_attention` |

### HuggingFace Integration

Parallax is implemented as a first-class HuggingFace model:

- `ParallaxConfig` subclasses `PretrainedConfig`
- `ParallaxForCausalLM` subclasses `PreTrainedModel`
- Returns `CausalLMOutputWithPast` from forward pass
- Compatible with `model.save_pretrained()` and `model.push_to_hub()`
- Compatible with PEFT/LoRA, BitsAndBytes quantization, and TRL

---

## Default Configuration

The default configuration targets the NVIDIA RTX 3050 (6 GB VRAM) as a minimum development
platform and represents the smallest configuration considered non-trivial for architecture
research:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 512 | Embedding dimension |
| `num_hidden_layers` | 6 | Transformer blocks per track (12 total) |
| `num_attention_heads` | 8 | Query attention heads (head_dim = 64) |
| `num_key_value_heads` | 2 | Key/Value heads — 4:1 GQA ratio |
| `intermediate_size` | 1408 | SwiGLU FFN intermediate dimension (auto-computed if 0) |
| `max_position_embeddings` | 1024 | Context window (tokens) |
| `vocab_size` | 32000 | Llama 2 tokenizer vocabulary size |
| `rope_theta` | 10000.0 | RoPE base frequency |
| `rms_norm_eps` | 1e-5 | RMSNorm epsilon |
| `attention_dropout` | 0.0 | Dropout rate (0.0 recommended for initial runs) |
| `num_loops` | 2 | Number of passes through each track |
| `use_swap` | True | Enable cross-track swap between passes |

This yields approximately **65.8M total parameters** (~50M non-embedding).

**Optimal configuration for larger hardware** (198.47M total, 165.70M non-embedding):

```python
from model.parallax import ParallaxConfig, ParallaxForCausalLM

config = ParallaxConfig(
    hidden_size=1024,
    num_hidden_layers=6,
    num_attention_heads=16,
    num_key_value_heads=4,
    intermediate_size=2752,
    max_position_embeddings=1024,
    vocab_size=32000,
)
model = ParallaxForCausalLM(config)
```

---

## Repository Structure

```
Parallax/
├── model/
│   └── parallax/
│       ├── __init__.py                     # Package exports
│       ├── configuration_parallax.py       # ParallaxConfig (PretrainedConfig subclass)
│       └── modeling_parallax.py            # ParallaxForCausalLM (PreTrainedModel subclass)
├── scripts/
│   ├── dataset/
│   │   ├── data_collector.py               # Stream and shuffle datasets from HuggingFace or local files
│   │   └── prepare_data.py                 # Tokenize and split a corpus from scratch
│   ├── tools/
│   │   ├── count_tokens.py                 # Simple script to count tokens in pre-tokenized dataset
│   │   ├── export_to_hf.py                 # Simple script to export model to Hugging Face format
│   │   ├── gen_inference.py                # General-purpose inference — handles all checkpoint types
│   │   ├── inspect_checkpoint.py           # Inspect meta_*.pt config fields
│   │   ├── qtest.py                        # Quick model key structure verification
│   │   └── qtok.py                         # Quickly tokenize a text file such as Tiny Stories
│   ├── transplant/
│   │   ├── inference_transplant.py         # Inference for transplant checkpoints (pre-fine-tune)
│   │   └── transplant_stream.py            # Streaming weight transplant from any Llama-family donor
│   ├── fine_tune.py                        # Fine-tune a pre-trained or transplanted checkpoint
│   ├── inference.py                        # Text generation with temperature and top-k/p sampling
│   ├── init_model.py                       # Initialise and save a blank untrained baseline model
│   └── train.py                            # Training loop with TensorBoard logging
├── whl/
│   └── flash_attn-2.8.3+...whl             # Flash Attention 2 wheel (CUDA 13.0, PyTorch 2.10, Python 3.13)
├── data/                                   # Data directory (contents supplied by user)
│   ├── raw_text/                           # Place .txt source files here
│   ├── train.bin                           # Tokenized training data (uint32 numpy array)
│   └── val.bin                             # Tokenized validation data
├── checkpoints/                            # Training checkpoints (generated at runtime)
├── logs/                                   # TensorBoard event files (generated at runtime)
├── pyproject.toml                          # Package metadata and build configuration
├── setup.bat                               # Windows environment setup script
├── LICENSE                                 # GNU Affero General Public License v3.0
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/beyastard/Parallax.git
cd Parallax
```

### 2. Set up the environment

**Windows (recommended):** Run the provided setup script which creates a virtual environment,
installs all pinned dependencies, and registers the `model` package:

```bat
setup.bat
```

**Manual setup:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate.bat
# Linux/macOS
source .venv/bin/activate

# Install PyTorch with CUDA 13.0
pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 \
    -i https://download.pytorch.org/whl/cu130

# Install dependencies
pip install transformers safetensors tensorboard numpy datasets \
    huggingface_hub accelerate peft trl

# Install Flash Attention 2 (Ampere+ GPUs only)
pip install whl/flash_attn-2.8.3+cu130torch2.10.0cxx11abiTRUE-cp313-cp313-win_amd64.whl

# Register the model package
pip install -e .
```

> **Note:** Exact dependency versions are pinned in `requirements.txt` for reproducibility.
> The `pip install -e .` step is required — without it scripts will raise
> `ModuleNotFoundError: No module named 'model'`.

### 3. Prepare data (TinyStories example)

Download `TinyStories-train.txt` and `TinyStories-valid.txt` from
[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
and place them in `data/raw_text/`. Then tokenize:

```bash
python scripts/dataset/prepare_data.py \
    --input data/raw_text/TinyStories-train.txt \
    --output data/train.bin

python scripts/dataset/prepare_data.py \
    --input data/raw_text/TinyStories-valid.txt \
    --output data/val.bin
```

### 4. Initialise a blank baseline model

```bash
python scripts/init_model.py
```

This saves a blank untrained checkpoint to `checkpoints/parallax_v1/initial/` and prints
a VRAM estimate for your hardware.

### 5. Train

```bash
python scripts/train.py \
    --checkpoint_dir checkpoints/parallax_v1 \
    --log_dir logs/parallax_v1 \
    --max_iters 5000 \
    --warmup_iters 100
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir=logs
```

### 6. Generate text

```bash
# Using the dedicated inference script
python scripts/inference.py \
    --checkpoint_dir checkpoints/parallax_v1 \
    --prompt "Once upon a time" \
    --max_new_tokens 200 \
    --temperature 0.7

# Using the general-purpose inference script (auto-detects checkpoint type)
python scripts/tools/gen_inference.py \
    --checkpoint checkpoints/parallax_v1 \
    --prompt "Once upon a time" \
    --template raw
```

---

## Weight Transplant

Parallax supports initialising from the weights of any Llama-family donor model via a
streaming transplant that processes one layer at a time (~500 MB peak RAM per layer).
This provides a significantly better starting point than random initialisation.

**Tested donors:**
- AMD-Llama-135m (12 layers, hidden=2048, GQA disabled)
- Any model with standard HuggingFace `LlamaForCausalLM` key naming

**Transplant modes:**
- `interleave` — donor layers alternate between Track A and Track B (0→A0, 1→B0, 2→A1, ...)
- `linear` — first half of donor layers → Track A, second half → Track B

Experimental results on TinyStories (5000 fine-tuning iterations):

| Initialisation | Best Val Loss | vs. Random Init |
|----------------|--------------|-----------------|
| Random init | 1.4128 | baseline |
| AMD interleaved transplant | 1.3452 | +4.8% |
| AMD linear transplant | **1.2649** | **+10.5%** |

```bash
# Transplant from a local donor
python scripts/transplant/transplant_stream.py \
    --donor D:/models/AMD-Llama-135m \
    --output checkpoints/parallax_amd_linear \
    --transplant_mode linear

# Fine-tune the transplanted model
python scripts/fine_tune.py \
    --base_checkpoint checkpoints/parallax_amd_linear/model_transplant.safetensors \
    --base_meta      checkpoints/parallax_amd_linear/meta_transplant.pt \
    --train_path     data/train.bin \
    --val_path       data/val.bin \
    --max_iters      5000 \
    --lr             1e-4
```

---

## Fine-tuning

Fine-tune any Parallax checkpoint (pre-trained or transplanted) on a new dataset:

```bash
python scripts/fine_tune.py \
    --base_checkpoint checkpoints/parallax_v1/model_best.safetensors \
    --base_meta      checkpoints/parallax_v1/meta_best.pt \
    --train_path     data/new_dataset/train.bin \
    --val_path       data/new_dataset/val.bin \
    --lr             1e-4 \
    --max_iters      20000
```

**Catastrophic forgetting mitigation** — mix original pre-training data into fine-tuning
batches to preserve previously learned representations:

```bash
python scripts/fine_tune.py ... \
    --anchor_path data/train.bin \
    --mix_ratio   0.3
```

---

## Hardware Requirements

| Use case | Min VRAM | Notes |
|----------|----------|-------|
| Training (65M default) | 6 GB | RTX 3050 or equivalent |
| Training (198M config) | 12–16 GB | RTX 3080/3090 recommended |
| Inference (65M) | 2 GB | CPU inference also supported |

Developed and tested on an NVIDIA RTX 3050 (6 GB VRAM) under Windows 11 with Python 3.13
and CUDA 13.0. Flash Attention 2 is enabled automatically on Ampere (RTX 30xx) and newer
GPUs.

---

## Relationship to Other Architectures

Parallax is not related to [Paraformer](https://arxiv.org/abs/2206.08317) (Gao et al., 2022),
which is a non-autoregressive speech recognition model. The names are superficially similar
but the architectures and domains are entirely distinct.

---

## Status

This is an experimental architecture under active development. The codebase has been
structured for compatibility with the HuggingFace ecosystem (Transformers, PEFT, TRL,
BitsAndBytes) from the ground up. Current benchmarks are val loss and generation quality
on the TinyStories dataset. No claims are made about performance relative to standard
transformer baselines at this stage — rigorous controlled experiments are ongoing.

---

## License

Copyright (C) 2025-2026 Bryan K Reinhart & BeySoft

Parallax is free software: you can redistribute it and/or modify it under the terms of the
GNU Affero General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

See [LICENSE](LICENSE) for the full license text.
