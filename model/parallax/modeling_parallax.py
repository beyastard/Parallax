# modeling_parallax.py
# HuggingFace-compatible model definition for the Parallax architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_parallax import ParallaxConfig


# ─────────────────────────────────────────────────────────────────────────────
# RoPE helpers
# ─────────────────────────────────────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (B, S, num_attention_heads, head_dim).
        k: Key tensor of shape (B, S, num_key_value_heads, head_dim).
        cos: Cosine frequencies of shape (S, head_dim).
        sin: Sine frequencies of shape (S, head_dim).

    Returns:
        Tuple of rotated (q, k) tensors with the same shapes as input.
    """
    # Reshape to (1, S, 1, head_dim) to broadcast across batch and heads.
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────────────

class ParallaxRMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    Equivalent to Llama's RMSNorm.  Uses float32 internally for numerical
    stability then casts back to the input dtype.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU feed-forward network
# ─────────────────────────────────────────────────────────────────────────────

class ParallaxMLP(nn.Module):
    """SwiGLU feed-forward network as used in Llama 2+.

    Computes: W3( SiLU(W1(x)) ⊙ W2(x) )

    Weight naming follows the convention established during pre-training:
        w1 — gate projection  (hidden_size → intermediate_size)
        w2 — up projection    (hidden_size → intermediate_size)
        w3 — down projection  (intermediate_size → hidden_size)
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ─────────────────────────────────────────────────────────────────────────────
# Grouped Query Attention
# ─────────────────────────────────────────────────────────────────────────────

class ParallaxAttention(nn.Module):
    """Grouped Query Attention (GQA) with Rotary Position Embeddings.

    Supports full MHA (num_key_value_heads == num_attention_heads) and GQA
    (num_key_value_heads < num_attention_heads).  Uses
    ``torch.nn.functional.scaled_dot_product_attention``, which dispatches to
    Flash Attention 2 when available.
    """

    def __init__(self, config: ParallaxConfig):
        super().__init__()
        self.num_attention_heads  = config.num_attention_heads
        self.num_key_value_heads  = config.num_key_value_heads
        self.head_dim             = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout    = config.attention_dropout

        self.wq = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.wk = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.wv = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.wo = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        b, s, _ = hidden_states.shape

        q = self.wq(hidden_states)
        k = self.wk(hidden_states)
        v = self.wv(hidden_states)

        q = q.view(b, s, self.num_attention_heads,  self.head_dim)
        k = k.view(b, s, self.num_key_value_heads,  self.head_dim)
        v = v.view(b, s, self.num_key_value_heads,  self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match query head count for GQA.
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=2)
            v = v.repeat_interleave(self.num_key_value_groups, dim=2)

        # (B, S, H, D) → (B, H, S, D) for SDPA.
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.wo(attn_out)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder block
# ─────────────────────────────────────────────────────────────────────────────

class ParallaxDecoderLayer(nn.Module):
    """Single Parallax transformer block (pre-norm, GQA + SwiGLU)."""

    def __init__(self, config: ParallaxConfig):
        super().__init__()
        intermediate_size = (
            config.intermediate_size
            if config.intermediate_size > 0
            else int(2 / 3 * 4 * config.hidden_size)
        )
        self.attn  = ParallaxAttention(config)
        self.ffn   = ParallaxMLP(config.hidden_size, intermediate_size)
        self.norm1 = ParallaxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = ParallaxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# Core Parallax model (no LM head)
# ─────────────────────────────────────────────────────────────────────────────

class ParallaxModel(PreTrainedModel):
    """The bare Parallax dual-track transformer outputting raw hidden states.

    Use ``ParallaxForCausalLM`` for text generation.
    """

    config_class = ParallaxConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def __init__(self, config: ParallaxConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.track_a = nn.ModuleList(
            [ParallaxDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.track_b = nn.ModuleList(
            [ParallaxDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.output_norm = ParallaxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        head_dim = config.hidden_size // config.num_attention_heads
        cos, sin = self._precompute_freqs(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

        self.post_init()

    # ------------------------------------------------------------------
    # RoPE precomputation
    # ------------------------------------------------------------------

    @staticmethod
    def _precompute_freqs(
        head_dim: int,
        seq_len: int,
        rope_theta: float = 10000.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)              # (S, head_dim/2)
        full_freqs = torch.cat((freqs, freqs), dim=-1)  # (S, head_dim)
        return full_freqs.cos(), full_freqs.sin()

    # ------------------------------------------------------------------
    # PreTrainedModel required overrides
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token indices of shape (batch_size, sequence_length).

        Returns:
            Hidden states of shape (batch_size, sequence_length, hidden_size).
        """
        b, s = input_ids.shape
        x = self.embed_tokens(input_ids)

        cos = self.freqs_cos[:s]
        sin = self.freqs_sin[:s]

        out_a = x
        out_b = x

        for loop_idx in range(self.config.num_loops):
            # Swap: cross-pollinate track outputs before every pass after the first.
            if self.config.use_swap and loop_idx > 0:
                new_a = out_b.clone() + x
                new_b = out_a.clone() + x
                del out_a, out_b
                out_a, out_b = new_a, new_b

            for layer_a, layer_b in zip(self.track_a, self.track_b):
                out_a = layer_a(out_a, cos, sin)
                out_b = layer_b(out_b, cos, sin)

        return self.output_norm(out_a + out_b)


# ─────────────────────────────────────────────────────────────────────────────
# Parallax for causal language modelling
# ─────────────────────────────────────────────────────────────────────────────

class ParallaxForCausalLM(PreTrainedModel):
    """Parallax dual-track transformer with a causal language modelling head.

    This is the primary class for text generation and training.  It wraps
    ``ParallaxModel`` and adds a linear projection from hidden states to
    vocabulary logits.

    Example::

        from model.parallax import ParallaxConfig, ParallaxForCausalLM

        config = ParallaxConfig()
        model  = ParallaxForCausalLM(config)
        logits, loss = model(input_ids, labels=input_ids)
    """

    config_class      = ParallaxConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]

    def __init__(self, config: ParallaxConfig):
        super().__init__(config)
        self.model   = ParallaxModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # ------------------------------------------------------------------
    # Standard PreTrainedModel interface
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        self.model._init_weights(module)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: ParallaxModel) -> None:
        self.model = decoder

    def get_decoder(self) -> ParallaxModel:
        return self.model

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor                    = None,
        labels: torch.LongTensor                       = None,
        # The following kwargs are accepted for HF-pipeline compatibility
        # but are not used internally (Parallax has no KV cache yet).
        attention_mask: torch.Tensor                   = None,
        position_ids: torch.LongTensor                 = None,
        past_key_values                                = None,
        inputs_embeds: torch.FloatTensor               = None,
        use_cache: bool                                = None,
        output_attentions: bool                        = None,
        output_hidden_states: bool                     = None,
        return_dict: bool                              = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids: Token indices of shape (batch_size, sequence_length).
            labels: Target token indices of shape (batch_size, sequence_length).
                Positions with value -100 are ignored in the loss.  When
                ``None``, no loss is computed.

        Returns:
            ``CausalLMOutputWithPast`` with fields:
                - ``loss`` (optional): Scalar cross-entropy loss.
                - ``logits``: Unnormalised scores of shape
                  (batch_size, sequence_length, vocab_size).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.model(input_ids)
        logits        = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    # ------------------------------------------------------------------
    # Convenience method — matches the old Parallax(config) call signature
    # so existing training / inference scripts work without changes until
    # they are updated to use the HF forward signature.
    # ------------------------------------------------------------------

    def __call__(self, tokens, targets=None, **kwargs):
        if targets is not None:
            out = super().__call__(input_ids=tokens, labels=targets, **kwargs)
        else:
            out = super().__call__(input_ids=tokens, **kwargs)
        # Return (logits, loss) to match the old API used by train.py etc.
        return out.logits, out.loss
