"""
MLX attention modules for RFdiffusion.

All 8 attention variants from Attention_module.py.
Dropout is no-op at inference.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, r_ff, p_drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * r_ff)
        self.linear2 = nn.Linear(d_model * r_ff, d_model)

    def __call__(self, src):
        src = self.norm(src)
        return self.linear2(nn.relu(self.linear1(src)))


class Attention(nn.Module):
    def __init__(self, d_query, d_key, n_head, d_hidden, d_out, p_drop=0.1):
        super().__init__()
        self.h = n_head
        self.dim = d_hidden
        self.to_q = nn.Linear(d_query, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_key, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_key, n_head * d_hidden, bias=False)
        self.to_out = nn.Linear(n_head * d_hidden, d_out)
        self.scaling = 1.0 / math.sqrt(d_hidden)

    def __call__(self, query, key, value):
        B, Q = query.shape[:2]
        K = key.shape[1]
        query = self.to_q(query).reshape(B, Q, self.h, self.dim)
        key = self.to_k(key).reshape(B, K, self.h, self.dim)
        value = self.to_v(value).reshape(B, K, self.h, self.dim)

        # Transpose to (B, h, T, d) for mx.fast.scaled_dot_product_attention
        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))

        out = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scaling)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, Q, self.h * self.dim)
        return self.to_out(out)


class AttentionWithBias(nn.Module):
    def __init__(self, d_in=256, d_bias=128, n_head=8, d_hidden=32):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_in)
        self.norm_bias = nn.LayerNorm(d_bias)
        self.to_q = nn.Linear(d_in, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_in, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_in, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False)
        self.to_g = nn.Linear(d_in, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_in)
        self.scaling = 1.0 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

    def __call__(self, x, bias):
        B, L = x.shape[:2]
        x = self.norm_in(x)
        bias = self.norm_bias(bias)

        query = self.to_q(x).reshape(B, L, self.h, self.dim)
        key = self.to_k(x).reshape(B, L, self.h, self.dim)
        value = self.to_v(x).reshape(B, L, self.h, self.dim)
        bias = self.to_b(bias)  # (B, L, L, h)
        gate = mx.sigmoid(self.to_g(x))

        # Transpose to (B, h, L, d) for SDPA
        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))
        # bias: (B, L, L, h) -> (B, h, L, L) as additive mask
        mask = mx.transpose(bias, (0, 3, 1, 2))

        out = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scaling, mask=mask)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, L, -1)
        out = gate * out
        return self.to_out(out)


class SequenceWeight(nn.Module):
    def __init__(self, d_msa, n_head, d_hidden, p_drop=0.1):
        super().__init__()
        self.h = n_head
        self.dim = d_hidden
        self.scale = 1.0 / math.sqrt(d_hidden)
        self.to_query = nn.Linear(d_msa, n_head * d_hidden)
        self.to_key = nn.Linear(d_msa, n_head * d_hidden)

    def __call__(self, msa):
        B, N, L = msa.shape[:3]
        tar_seq = msa[:, 0]  # (B, L, d)
        q = self.to_query(tar_seq).reshape(B, 1, L, self.h, self.dim)
        k = self.to_key(msa).reshape(B, N, L, self.h, self.dim)
        q = q * self.scale
        attn = mx.einsum('bqihd,bkihd->bkihq', q, k)
        return mx.softmax(attn, axis=1)


class MSARowAttentionWithBias(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32):
        super().__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_msa, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_msa)
        self.scaling = 1.0 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

    def __call__(self, msa, pair):
        B, N, L = msa.shape[:3]
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)

        seq_weight = self.seq_weight(msa)  # (B, N, L, h, 1)
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        bias = self.to_b(pair)  # (B, L, L, h)
        gate = mx.sigmoid(self.to_g(msa))

        query = query * mx.broadcast_to(seq_weight, query.shape)
        key = key * self.scaling
        attn = mx.einsum('bsqhd,bskhd->bqkh', query, key)
        attn = attn + bias
        attn = mx.softmax(attn, axis=-2)

        out = mx.einsum('bqkh,bskhd->bsqhd', attn, value).reshape(B, N, L, -1)
        out = gate * out
        return self.to_out(out)


class MSAColAttention(nn.Module):
    def __init__(self, d_msa=256, n_head=8, d_hidden=32):
        super().__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.to_q = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_msa)
        self.scaling = 1.0 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

    def __call__(self, msa):
        B, N, L = msa.shape[:3]
        msa = self.norm_msa(msa)
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        gate = mx.sigmoid(self.to_g(msa))

        # Reshape to (B*L, h, N, d) for column-wise attention over sequences
        query = mx.transpose(query, (0, 2, 3, 1, 4)).reshape(B * L, self.h, N, self.dim)
        key = mx.transpose(key, (0, 2, 3, 1, 4)).reshape(B * L, self.h, N, self.dim)
        value = mx.transpose(value, (0, 2, 3, 1, 4)).reshape(B * L, self.h, N, self.dim)

        out = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scaling)
        out = out.reshape(B, L, self.h, N, self.dim)
        out = mx.transpose(out, (0, 3, 1, 2, 4)).reshape(B, N, L, -1)
        out = gate * out
        return self.to_out(out)


class MSAColGlobalAttention(nn.Module):
    def __init__(self, d_msa=64, n_head=8, d_hidden=8):
        super().__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.to_q = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_msa)
        self.scaling = 1.0 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

    def __call__(self, msa):
        B, N, L = msa.shape[:3]
        msa = self.norm_msa(msa)
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        query = mx.mean(query, axis=1)  # (B, L, h, dim)
        key = self.to_k(msa)  # (B, N, L, dim)
        value = self.to_v(msa)  # (B, N, L, dim)
        gate = mx.sigmoid(self.to_g(msa))  # (B, N, L, h*dim)

        query = query * self.scaling
        attn = mx.einsum('bihd,bkid->bihk', query, key)  # (B, L, h, N)
        attn = mx.softmax(attn, axis=-1)

        out = mx.einsum('bihk,bkid->bihd', attn, value).reshape(B, 1, L, -1)
        out = gate * out
        return self.to_out(out)


class BiasedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super().__init__()
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)
        self.to_q = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False)
        self.to_g = nn.Linear(d_pair, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_pair)
        self.scaling = 1.0 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

    def __call__(self, pair, bias):
        B, L = pair.shape[:2]
        if self.is_row:
            pair = mx.transpose(pair, (0, 2, 1, 3))
            bias = mx.transpose(bias, (0, 2, 1, 3))

        pair = self.norm_pair(pair)
        bias = self.norm_bias(bias)

        query = self.to_q(pair).reshape(B, L, L, self.h, self.dim)
        key = self.to_k(pair).reshape(B, L, L, self.h, self.dim)
        value = self.to_v(pair).reshape(B, L, L, self.h, self.dim)
        bias = self.to_b(bias)  # (B, L, L, h)
        gate = mx.sigmoid(self.to_g(pair))

        # Attention scores: contract over (n, k) dims via batched matmul
        # Fold both scales into a single combined_scale to avoid 2 pointwise ops
        combined_scale = self.scaling / math.sqrt(L)
        # query (B, n, i, h, k) -> (B*h, i, n*k)
        q = mx.transpose(query, (0, 3, 2, 1, 4)).reshape(B * self.h, L, L * self.dim)
        # key (B, n, j, h, k) -> (B*h, n*k, j)
        k = mx.transpose(key, (0, 3, 1, 4, 2)).reshape(B * self.h, L * self.dim, L)
        attn = (q @ k) * combined_scale  # (B*h, i, j)
        attn = attn.reshape(B, self.h, L, L)
        attn = mx.transpose(attn, (0, 2, 3, 1))  # (B, i, j, h)
        attn = attn + bias
        attn = mx.softmax(attn, axis=-2)

        out = mx.einsum('bijh,bkjhd->bikhd', attn, value).reshape(B, L, L, -1)
        out = gate * out
        out = self.to_out(out)

        if self.is_row:
            out = mx.transpose(out, (0, 2, 1, 3))
        return out
