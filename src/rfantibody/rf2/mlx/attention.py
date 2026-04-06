"""
RF2 attention modules ported to MLX.

Port of src/rfantibody/rf2/network/Attention_module.py.
Omits stride/striping parameters (MLX processes full tensors).
"""
import math

import mlx.core as mx
import mlx.nn as nn


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, r_ff, p_drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * r_ff)
        self.linear2 = nn.Linear(d_model * r_ff, d_model)

    def __call__(self, src):
        src = self.norm(src)
        return self.linear2(nn.relu(self.linear1(src)))


class Attention(nn.Module):
    def __init__(self, d_query, d_key, n_head, d_hidden, d_out, p_drop=0.0):
        super().__init__()
        self.h = n_head
        self.dim = d_hidden
        self.d_out = d_out
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


class SequenceWeight(nn.Module):
    def __init__(self, d_msa, n_head, d_hidden, p_drop=0.0):
        super().__init__()
        self.h = n_head
        self.dim = d_hidden
        self.scale = 1.0 / math.sqrt(d_hidden)
        self.to_query = nn.Linear(d_msa, n_head * d_hidden)
        self.to_key = nn.Linear(d_msa, n_head * d_hidden)

    def __call__(self, msa, MSANorm):
        B, N, L = msa.shape[:3]
        tar_seq = MSANorm(msa[:, 0])
        q = self.to_query(tar_seq).reshape(B, 1, L, self.h, self.dim)
        q = q * self.scale
        k = self.to_key(MSANorm(msa)).reshape(B, N, L, self.h, self.dim)
        attn = mx.einsum('bqihd,bkihd->bkihq', q, k)
        attn = mx.softmax(attn, axis=1)
        return attn


class MSARowAttentionWithBias(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32):
        super().__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden)
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
        seq_weight = self.seq_weight(msa, self.norm_msa)

        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)

        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        bias = self.to_b(pair)
        gate = mx.sigmoid(self.to_g(msa))

        query = query * mx.broadcast_to(
            seq_weight, query.shape[:3] + (self.h, self.dim))
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
        key = self.to_k(msa)
        value = self.to_v(msa)
        gate = mx.sigmoid(self.to_g(msa))
        query = query * self.scaling
        attn = mx.einsum('bihd,bkid->bihk', query, key)
        attn = mx.softmax(attn, axis=-1)
        out = mx.einsum('bihk,bkid->bihd', attn, value).reshape(B, 1, L, -1)
        out = gate * out
        return self.to_out(out)


class BiasedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.0, is_row=True):
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
            pair = mx.transpose(pair, axes=(0, 2, 1, 3))
            bias = mx.transpose(bias, axes=(0, 2, 1, 3))

        pair = self.norm_pair(pair)
        bias = self.norm_bias(bias)

        query = self.to_q(pair).reshape(B, L, L, self.h, self.dim)
        key = self.to_k(pair).reshape(B, L, L, self.h, self.dim)
        value = self.to_v(pair).reshape(B, L, L, self.h, self.dim)
        bias = self.to_b(bias)
        gate = mx.sigmoid(self.to_g(pair))

        query = query * self.scaling
        key = key / L  # normalize for tied attention
        attn = mx.einsum('bnihk,bnjhk->bijh', query, key)
        attn = attn + bias
        attn = mx.softmax(attn, axis=-2)

        out = mx.einsum('bijh,bnjhd->bnihd', attn, value).reshape(B, L, L, -1)
        out = gate * out
        out = self.to_out(out)

        if self.is_row:
            out = mx.transpose(out, axes=(0, 2, 1, 3))
        return out


class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hidden=128, outgoing=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_pair)
        self.left_proj = nn.Linear(d_pair, d_hidden)
        self.right_proj = nn.Linear(d_pair, d_hidden)
        self.left_gate = nn.Linear(d_pair, d_hidden)
        self.right_gate = nn.Linear(d_pair, d_hidden)
        self.gate = nn.Linear(d_pair, d_pair)
        self.norm_out = nn.LayerNorm(d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_pair)
        self.d_hidden = d_hidden
        self.outgoing = outgoing

    def __call__(self, pair):
        B, L = pair.shape[:2]
        pair = self.norm(pair)

        left = mx.sigmoid(self.left_gate(pair)) * self.left_proj(pair)
        right = mx.sigmoid(self.right_gate(pair)) * self.right_proj(pair)

        if self.outgoing:
            out = mx.einsum('bikd,bjkd->bijd', left, right / float(L))
        else:
            out = mx.einsum('bkid,bkjd->bijd', left, right / float(L))

        out = self.norm_out(out)
        out = self.out_proj(out)
        gate = mx.sigmoid(self.gate(pair))
        return gate * out
