# Parallax

> A dual-track transformer language model with cross-pollinating attention passes.

Parallax is an experimental small language model built around a novel architecture idea: instead of a single linear stack of transformer layers, two independent tracks process the same input in parallel, then exchange their representations before a second pass. This "swap" allows each track to refine its understanding using the other track's contextualized view of the same sequence — a mechanism inspired loosely by the concept of parallax itself, where two offset vantage points reveal depth that neither could resolve alone.

---

## Architecture

A standard transformer passes tokens through a single sequential stack of attention and feed-forward layers. Parallax replaces this with the following structure:

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
   ________|________   ___________|_____
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

**Key properties:**

- **Two parallel tracks** — Track A and Track B are separate `nn.ModuleList` instances with independent weights. They process the same token embeddings simultaneously on Pass 1.
- **The Swap** — After Pass 1, the output of Track A becomes the input to Track B's second pass, and vice versa. A residual connection back to the original embedding `x` is added to each swapped input to stabilise gradients.
- **Weight-tied passes** — Both Pass 1 and Pass 2 use the same layer weights within each track. This gives Parallax a recurrent character — layers are reused across depth rather than being unique per pass.
- **Configurable loop count** — `num_loops` in config controls how many passes are performed. At `num_loops = 2` (default), one swap event occurs. At `num_loops = 3`, two swaps occur. At `num_loops = 1`, no swap occurs and the model behaves as a parallel ensemble.
- **Final fusion** — The outputs of both tracks are summed, normalised with RMSNorm, and projected to vocabulary logits.

### Components

| Component | Implementation |
|-----------|---------------|
| Attention | Grouped Query Attention (GQA) with `n_head` query heads and `n_kv_heads` KV heads |
| Position encoding | Rotary Position Embeddings (RoPE), applied to Q and K inside each attention block |
| Feed-forward | SwiGLU (`W3(SiLU(W1(x)) * W2(x))`) |
| Normalisation | RMSNorm (pre-norm, applied before attention and FFN) |
| Precision | FP16 via `torch.amp.autocast` during training and inference |

---

## Repository Structure

```
parallax/
├── config.py            # ParallaxConfig dataclass
├── model.py             # Parallax model definition
├── train.py             # Training loop with TensorBoard logging
├── inference.py         # Text generation with temperature and top-k sampling
├── prepare_data.py      # Tokenize and split a corpus from scratch
├── qtok.py              # Quickly tokenize a single pre-split file (e.g. TinyStories)
├── data_collector.py    # Stream and shuffle datasets from HuggingFace or local files
├── init_model.py        # Initialise and save a blank untrained model
├── data/
│   ├── raw_text/        # Place .txt source files here
│   ├── train.bin        # Tokenized training data (uint32 numpy array)
│   └── val.bin          # Tokenized validation data
├── checkpoints/
│   └── parallax_v1/
│       ├── model_best.safetensors
│       └── meta_best.pt
└── logs/                # TensorBoard event files
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch transformers safetensors tensorboard numpy tqdm datasets
```

Flash Attention 2 is used automatically via `torch.nn.functional.scaled_dot_product_attention` when running on a CUDA device with a compatible GPU.

### 2. Prepare data (TinyStories example)

Download `TinyStories-train.txt` and `TinyStories-valid.txt` from
[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
and place them in `data/raw_text/`. Then tokenize each file:

```bash
python qtok.py --input data/raw_text/TinyStories-train.txt --output data/train.bin
python qtok.py --input data/raw_text/TinyStories-valid.txt --output data/val.bin
```

### 3. Train

```bash
python train.py
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir=logs
```

### 4. Generate text

```bash
python inference.py --prompt "Once upon a time" --max_new_tokens 200 --temperature 0.7
```

---

## Configuration

All model hyperparameters are set in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 512 | Context window (tokens) |
| `vocab_size` | 32000 | Llama 2 tokenizer vocabulary size |
| `n_layer` | 6 | Transformer blocks per track |
| `n_head` | 4 | Query attention heads |
| `n_kv_heads` | 2 | Key/Value heads (GQA) |
| `n_embd` | 512 | Embedding dimension (~40-60M params) |
| `dropout` | 0.0 | Dropout rate (0.0 recommended for initial runs) |
| `num_loops` | 2 | Number of passes through each track |
| `use_swap` | True | Enable cross-track swap between passes |
| `norm_eps` | 1e-5 | RMSNorm epsilon |

---

## Relationship to Other Architectures

Parallax is not related to [Paraformer](https://arxiv.org/abs/2206.08317) (Gao et al., 2022), which is a non-autoregressive speech recognition model developed by Alibaba. The names are superficially similar but the architectures and domains are entirely distinct.

---

## Hardware

Developed and tested on an NVIDIA RTX 3050 (6GB VRAM). The default configuration is tuned to fit within this constraint. Larger `n_layer`, `n_embd`, or `num_loops` values will increase VRAM usage proportionally.

---

## Status

This is an experimental architecture under active development. Loss curves and generation quality on TinyStories are the current benchmarks. No claims are made about performance relative to standard transformer baselines at this stage.
