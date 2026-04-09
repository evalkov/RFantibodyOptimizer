"""
Attention modules for Protenix-Mini-Flow, ported to MLX.

Implements:
  - AttentionPairBias (AF3 Algorithm 24): self-attention with additive pair bias
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class AttentionPairBias(nn.Module):
    """Self-attention with additive pair bias (AF3 Algorithm 24).

    Computes multi-head self-attention on single embeddings with an
    additive bias derived from pair embeddings, plus sigmoid gating.

    Args:
        c_s: single embedding dimension (default 384)
        c_z: pair embedding dimension (default 128)
        n_head: number of attention heads (default 16)
        c_hidden: per-head hidden dimension (default c_s // n_head)
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        n_head: int = 16,
        c_hidden: int | None = None,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.n_head = n_head
        self.c_hidden = c_hidden if c_hidden is not None else c_s // n_head
        self.scale = 1.0 / math.sqrt(self.c_hidden)

        # Normalization
        self.layer_norm_s = nn.LayerNorm(c_s)
        self.layer_norm_z = nn.LayerNorm(c_z)

        # QKV projections
        self.to_q = nn.Linear(c_s, n_head * self.c_hidden, bias=False)
        self.to_k = nn.Linear(c_s, n_head * self.c_hidden, bias=False)
        self.to_v = nn.Linear(c_s, n_head * self.c_hidden, bias=False)

        # Pair bias projection: pair -> per-head bias
        self.linear_z = nn.Linear(c_z, n_head, bias=False)

        # Gating and output
        self.to_g = nn.Linear(c_s, n_head * self.c_hidden, bias=False)
        self.to_out = nn.Linear(n_head * self.c_hidden, c_s, bias=False)

    def __call__(
        self,
        s: mx.array,
        z: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            s: [B, N, c_s] single embeddings
            z: [B, N, N, c_z] pair embeddings
            mask: [B, N] sequence mask (optional)
        Returns:
            [B, N, c_s] updated single embeddings
        """
        B, N, _ = s.shape

        # Normalize
        s_norm = self.layer_norm_s(s)
        z_norm = self.layer_norm_z(z)

        # QKV
        q = self.to_q(s_norm).reshape(B, N, self.n_head, self.c_hidden)
        k = self.to_k(s_norm).reshape(B, N, self.n_head, self.c_hidden)
        v = self.to_v(s_norm).reshape(B, N, self.n_head, self.c_hidden)
        g = mx.sigmoid(self.to_g(s_norm)).reshape(B, N, self.n_head, self.c_hidden)

        # Transpose to (B, H, N, D)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))

        # Pair bias: [B, N, N, c_z] -> [B, N, N, H] -> [B, H, N, N]
        pair_bias = self.linear_z(z_norm)
        pair_bias = mx.transpose(pair_bias, axes=(0, 3, 1, 2))

        # Mask bias
        if mask is not None:
            # mask: [B, N] -> [B, 1, 1, N]
            mask_bias = 1e9 * (mask.astype(mx.float32) - 1.0)
            mask_bias = mask_bias[:, None, None, :]
            pair_bias = pair_bias + mask_bias

        # Scaled dot-product attention with pair bias as additive mask
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=pair_bias
        )

        # out: (B, H, N, D) -> (B, N, H, D)
        out = mx.transpose(out, axes=(0, 2, 1, 3)).reshape(B, N, -1)

        # Sigmoid gating
        g = g.reshape(B, N, -1)
        out = g * out

        return self.to_out(out)
