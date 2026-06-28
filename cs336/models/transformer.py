"""Transformer language model built from scratch (CS336 Assignment 1).

A Llama-style decoder: RMSNorm pre-norm, rotary position embeddings (RoPE),
multi-head causal self-attention, and a SwiGLU feed-forward network.
"""

from __future__ import annotations

import torch
from torch import nn


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax along ``dim``."""
    x = x - x.amax(dim=dim, keepdim=True)
    exp = torch.exp(x)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    ``q,k,v`` are ``(..., seq, d)``. ``mask`` (if given) is a boolean tensor
    broadcastable to the score shape where ``True`` means *attend*.
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-1, -2) / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ v


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embeddings (RoPE), rotating adjacent dim pairs."""

    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires an even head dimension"
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, d_k/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[positions]  # (seq, d_k/2)
        sin = self.sin[positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class Linear(nn.Module):
    """Linear layer without bias: ``y = x @ W.T``."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """Token embedding lookup table."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        if positions is None:
            positions = torch.arange(t, device=x.device)

        def split_heads(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, t, self.num_heads, self.d_head).transpose(1, 2)

        q = split_heads(self.q_proj(x))  # (b, heads, t, d_head)
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        if self.rope is not None:
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask=causal)  # (b, heads, t, d_head)
        out = out.transpose(1, 2).reshape(b, t, self.num_heads * self.d_head)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: ``W2( SiLU(W1 x) * W3 x )``."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        silu = a * torch.sigmoid(a)
        return self.w2(silu * self.w3(x))


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no mean subtraction, no bias)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x / rms * self.weight
        return out.to(in_dtype)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: x + attn(norm(x)), then x + ffn(norm(x))."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        rope = RotaryPositionalEmbedding(d_model // num_heads, context_length, theta)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope=rope) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy. ``logits`` is ``(..., vocab)``, ``targets`` is ``(...,)``.

    Computed in a numerically stable way (subtract the max before exponentiating).
    """
    logits = logits - logits.amax(dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.exp(logits).sum(dim=-1))
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()
