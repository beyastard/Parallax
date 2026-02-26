# Model File

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ParallaxConfig

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k shape: (B, S, H, D)
    # cos, sin shape: (S, D)
    # Reshape to (1, S, 1, D) to broadcast across Batch and Heads
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(d_model, intermediate_size, bias=False)
        self.w2 = nn.Linear(d_model, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class GQAAttention(nn.Module):
    def __init__(self, config: ParallaxConfig):
        super().__init__()
        self.n_heads = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.num_groups = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout  # Store for use in forward

        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(b, s, self.n_heads, self.head_dim)
        k = k.view(b, s, self.n_kv_heads, self.head_dim)
        v = v.view(b, s, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV for Grouped Query Attention
        k = k.repeat_interleave(self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_groups, dim=2)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Scaled dot-product attention — uses Flash Attention kernel when available
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.wo(out)

class ParallaxBlock(nn.Module):
    def __init__(self, config: ParallaxConfig):
        super().__init__()
        self.attn = GQAAttention(config)
        self.ffn = SwiGLU(config.n_embd, int(2/3 * 4 * config.n_embd))
        self.norm1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x

class Parallax(nn.Module):
    def __init__(self, config: ParallaxConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Two parallel tracks
        self.track_a = nn.ModuleList([ParallaxBlock(config) for _ in range(config.n_layer)])
        self.track_b = nn.ModuleList([ParallaxBlock(config) for _ in range(config.n_layer)])

        self.output_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Precompute rotary freqs and register as buffers so they move
        # automatically with model.to(device) / model.cuda()
        cos, sin = self._precompute_freqs(config.n_embd // config.n_head, config.block_size)
        self.register_buffer("freqs_cos", cos)
        self.register_buffer("freqs_sin", sin)

    def _precompute_freqs(self, dim, seq_len):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len)
        freqs = torch.outer(t, inv_freq)       # (S, D/2)
        full_freqs = torch.cat((freqs, freqs), dim=-1)  # (S, D)
        return full_freqs.cos(), full_freqs.sin()

    def forward(self, tokens, targets=None):
        b, s = tokens.shape
        x = self.tok_emb(tokens)

        # Slice to current sequence length — already on the correct device
        # because freqs_cos/sin are registered buffers
        cos = self.freqs_cos[:s]
        sin = self.freqs_sin[:s]

        # --- PASS 1 ---
        # Run both tracks in parallel for num_loops passes total.
        # On the first pass they each start from the token embeddings.
        out_a = x
        out_b = x

        for _ in range(self.config.num_loops):
            # If swap is enabled, cross the track outputs before each new pass
            # (skip the swap on the very first iteration so Pass 1 is independent)
            if self.config.use_swap and _ > 0:
                # Output of A feeds B and vice versa; residual from x stabilises gradients
                new_a = out_b.clone() + x
                new_b = out_a.clone() + x
                del out_a, out_b          # free memory early on the 3050
                out_a, out_b = new_a, new_b

            for layer_a, layer_b in zip(self.track_a, self.track_b):
                out_a = layer_a(out_a, cos, sin)
                out_b = layer_b(out_b, cos, sin)

        # --- Final Fusion ---
        combined = self.output_norm(out_a + out_b)
        logits = self.lm_head(combined)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
