# Configuration

from dataclasses import dataclass

@dataclass
class ParallaxConfig:
    block_size: int = 512        # Context window
    vocab_size: int = 32000      # Llama2 default
    n_layer: int = 6             # Layers PER track
    n_head: int = 4
    n_kv_heads: int = 2          # GQA: Grouped Query Attention
    n_embd: int = 512            # ~40-60M params at 6 layers per track
    dropout: float = 0.0         # Start at 0.0; introduce if val loss diverges from train
    bias: bool = False           # Llama-style: no bias in Linear layers (informational)
    num_loops: int = 2           # Passes through each track; swap fires on loops > 1
    use_swap: bool = True        # Cross-pollinates track outputs between passes: A->B, B->A
    # Architecture choices
    activation: str = "swiglu"   # Hardcoded in model.py; informational only
    norm_eps: float = 1e-5       # RMSNorm epsilon; consistent with Llama 2
