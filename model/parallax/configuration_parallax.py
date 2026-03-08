# model/parallax/configuration_parallax.py
# HuggingFace-compatible configuration for the Parallax architecture.
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

from transformers import PretrainedConfig


class ParallaxConfig(PretrainedConfig):
    r"""
    Configuration class for the Parallax dual-track language model.

    Parallax replaces the standard single transformer stack with two independent
    tracks (Track A and Track B) that process the same input in parallel, exchange
    their representations via a swap operation, then fuse at the end.

    Args:
        vocab_size (int): Vocabulary size. Default: 32000 (Llama 2 tokenizer).
        hidden_size (int): Embedding dimension and hidden state size. Default: 512.
        num_hidden_layers (int): Number of transformer blocks *per track*. Default: 6.
        num_attention_heads (int): Number of query attention heads. Default: 8.
        num_key_value_heads (int): Number of key/value heads for GQA. Default: 4.
            Set equal to num_attention_heads for standard MHA.
        intermediate_size (int): FFN intermediate (hidden) dimension. Default: 0.
            When 0, computed automatically as int(2/3 * 4 * hidden_size) (SwiGLU default).
        max_position_embeddings (int): Maximum sequence length (context window). Default: 512.
        rope_theta (float): RoPE base frequency. Default: 10000.0.
        attention_dropout (float): Dropout probability in attention. Default: 0.0.
        attention_bias (bool): Whether to use bias in attention projections. Default: False.
        hidden_act (str): Activation function identifier. Currently only "swiglu" is
            implemented. Informational only — not used to select the activation at
            runtime. Default: "swiglu".
        rms_norm_eps (float): Epsilon for RMSNorm layers. Default: 1e-5.
        num_loops (int): Number of passes through each track. The swap fires once per
            loop after the first. Default: 2 (one swap event).
        use_swap (bool): Whether to cross-pollinate track outputs between loops.
            When False the model behaves as a parallel ensemble with no interaction
            between tracks until the final fusion. Default: True.
        tie_word_embeddings (bool): Whether to tie input and output embeddings.
            Default: False.

    Example (198.47M total parameters, 165.70M non-embedding)::

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
    """

    model_type = "parallax"

    def __init__(
        self,
        vocab_size: int               = 32000,
        hidden_size: int              = 512,
        num_hidden_layers: int        = 6,
        num_attention_heads: int      = 8,
        num_key_value_heads: int      = 2,
        intermediate_size: int        = 0,
        max_position_embeddings: int  = 512,
        rope_theta: float             = 5.0,
        attention_dropout: float      = 0.0,
        attention_bias: bool          = False,
        hidden_act: str               = "swiglu",
        rms_norm_eps: float           = 1e-5,
        num_loops: int                = 2,
        use_swap: bool                = True,
        tie_word_embeddings: bool     = False,
        # ── Legacy aliases ────────────────────────────────────────────────────
        # These are accepted for backwards-compatibility with checkpoints that
        # were saved before the HF rename.  They are silently mapped to the
        # canonical HF names and are NOT stored as separate attributes.
        block_size: int               = None,
        n_embd: int                   = None,
        n_layer: int                  = None,
        n_head: int                   = None,
        n_kv_heads: int               = None,
        ffn_dim: int                  = None,
        norm_eps: float               = None,
        dropout: float                = None,
        bias: bool                    = None,
        activation: str               = None,
        **kwargs,
    ):
        # ── Apply legacy aliases (old name wins only when new name is still at
        #    its default, so explicit new-name kwargs always take precedence) ──
        if block_size is not None:
            max_position_embeddings = block_size
        if n_embd is not None:
            hidden_size = n_embd
        if n_layer is not None:
            num_hidden_layers = n_layer
        if n_head is not None:
            num_attention_heads = n_head
        if n_kv_heads is not None:
            num_key_value_heads = n_kv_heads
        if ffn_dim is not None:
            intermediate_size = ffn_dim
        if norm_eps is not None:
            rms_norm_eps = norm_eps
        if dropout is not None:
            attention_dropout = dropout
        if bias is not None:
            attention_bias = bias
        if activation is not None:
            hidden_act = activation
        # Auto-compute intermediate_size if not specified (SwiGLU standard: 2/3 * 4 * hidden)
        if intermediate_size == 0 or intermediate_size is None:
            intermediate_size = int(2 / 3 * 4 * hidden_size)
            intermediate_size = (intermediate_size + 63) // 64 * 64  # round to multiple of 64

        # ── Store canonical attributes ─────────────────────────────────────
        self.vocab_size              = vocab_size
        self.hidden_size             = hidden_size
        self.num_hidden_layers       = num_hidden_layers
        self.num_attention_heads     = num_attention_heads
        self.num_key_value_heads     = num_key_value_heads
        self.intermediate_size       = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta              = rope_theta
        self.attention_dropout       = attention_dropout
        self.attention_bias          = attention_bias
        self.hidden_act              = hidden_act
        self.rms_norm_eps            = rms_norm_eps
        self.num_loops               = num_loops
        self.use_swap                = use_swap

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # ── Read-only shims so existing code that reads the old names still works ─
    @property
    def block_size(self) -> int:
        return self.max_position_embeddings

    @property
    def n_embd(self) -> int:
        return self.hidden_size

    @property
    def n_layer(self) -> int:
        return self.num_hidden_layers

    @property
    def n_head(self) -> int:
        return self.num_attention_heads

    @property
    def n_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def ffn_dim(self) -> int:
        return self.intermediate_size

    @property
    def norm_eps(self) -> float:
        return self.rms_norm_eps

    @property
    def dropout(self) -> float:
        return self.attention_dropout

    @property
    def bias(self) -> bool:
        return self.attention_bias

    @property
    def activation(self) -> str:
        return self.hidden_act
