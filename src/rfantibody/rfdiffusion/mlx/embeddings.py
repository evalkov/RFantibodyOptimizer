"""
MLX embedding modules for RFdiffusion.

Timestep, MSA, Extra, Template, and Recycling embeddings.
"""

import math

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .attention import Attention, AttentionWithBias, FeedForwardLayer
from .util_module import rbf, cdist, cross


def get_timestep_embedding(timesteps: mx.array, embedding_dim: int,
                           max_positions: int = 10000) -> mx.array:
    """Sinusoidal timestep embedding."""
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = mx.exp(mx.arange(half_dim).astype(mx.float32) * -emb)
    emb = timesteps.astype(mx.float32).reshape(-1, 1) * emb.reshape(1, -1)
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class Timestep_emb(nn.Module):
    def __init__(self, input_size, output_size, T, use_motif_timestep=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.T = T

        # Precompute embeddings for all timesteps + zero
        self.source_embeddings = get_timestep_embedding(
            mx.arange(T + 1), input_size)

        self.node_embedder = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False),
            nn.ReLU(),
            nn.Linear(output_size, output_size, bias=True),
            nn.LayerNorm(output_size),
        )

    def get_init_emb(self, t, L, motif_mask):
        t_idx = t.reshape(-1).astype(mx.int32)
        t_emb = self.source_embeddings[t_idx[0]]
        zero_emb = self.source_embeddings[0]

        timestep_embedding = mx.broadcast_to(
            mx.expand_dims(t_emb, 0), (L,) + t_emb.shape)

        # Slice in motif zero timestep at motif positions
        timestep_embedding = mx.where(
            mx.expand_dims(motif_mask, -1),
            mx.broadcast_to(mx.expand_dims(zero_emb, 0), timestep_embedding.shape),
            timestep_embedding)
        return timestep_embedding

    def __call__(self, L, t, motif_mask):
        emb_in = self.get_init_emb(t, L, motif_mask)
        return self.node_embedder(emb_in)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, minpos=-32, maxpos=32, p_drop=0.1):
        super().__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos) + maxpos + 1
        self.emb = nn.Embedding(self.nbin, d_model)

    def __call__(self, x, idx):
        bins = mx.arange(self.minpos, self.maxpos)
        seqsep = mx.expand_dims(idx, 1) - mx.expand_dims(idx, 2)  # (B, L, L)

        # Bucketize: find bin index for each seqsep value
        ib = mx.zeros(seqsep.shape, dtype=mx.int32)
        for i, b in enumerate(range(self.minpos, self.maxpos)):
            ib = mx.where(seqsep > b, i + 1, ib)
        ib = mx.clip(ib, 0, self.nbin - 1)

        emb = self.emb(ib)
        return x + emb


class MSA_emb(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=32,
                 d_init=22 + 22 + 2 + 2, minpos=-32, maxpos=32,
                 p_drop=0.1, input_seq_onehot=False):
        super().__init__()
        self.emb = nn.Linear(d_init, d_msa)
        self.emb_q = nn.Embedding(22, d_msa)
        self.emb_left = nn.Embedding(22, d_pair)
        self.emb_right = nn.Embedding(22, d_pair)
        self.emb_state = nn.Embedding(22, d_state)
        self.pos = PositionalEncoding2D(d_pair, minpos=minpos,
                                        maxpos=maxpos, p_drop=p_drop)

    def __call__(self, msa, seq, idx):
        N = msa.shape[1]
        msa = self.emb(msa)

        # Sergey's one hot trick: seq @ embedding_weight
        tmp = mx.expand_dims(seq @ self.emb_q.weight, 1)
        msa = msa + mx.broadcast_to(tmp, msa.shape)

        left = mx.expand_dims(seq @ self.emb_left.weight, 1)
        right = mx.expand_dims(seq @ self.emb_right.weight, 2)
        pair = left + right
        pair = self.pos(pair, idx)

        state = seq @ self.emb_state.weight
        return msa, pair, state


class Extra_emb(nn.Module):
    def __init__(self, d_msa=256, d_init=22 + 1 + 2, p_drop=0.1,
                 input_seq_onehot=False):
        super().__init__()
        self.emb = nn.Linear(d_init, d_msa)
        self.emb_q = nn.Embedding(22, d_msa)

    def __call__(self, msa, seq, idx):
        N = msa.shape[1]
        msa = self.emb(msa)
        seq_emb = mx.expand_dims(seq @ self.emb_q.weight, 1)
        msa = msa + mx.broadcast_to(seq_emb, msa.shape)
        return msa


class PairStr2Pair(nn.Module):
    """Imported here to avoid circular deps — used by TemplatePairStack."""
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        super().__init__()
        from .attention import BiasedAxialAttention
        self.emb_rbf = nn.Linear(d_rbf, d_hidden)
        self.proj_rbf = nn.Linear(d_hidden, d_pair)
        self.row_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden,
                                             p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden,
                                             p_drop=p_drop, is_row=False)
        self.ff = FeedForwardLayer(d_pair, 2)

    def __call__(self, pair, rbf_feat):
        rbf_feat = self.proj_rbf(nn.relu(self.emb_rbf(rbf_feat)))
        pair = pair + self.row_attn(pair, rbf_feat)
        pair = pair + self.col_attn(pair, rbf_feat)
        pair = pair + self.ff(pair)
        return pair


class TemplatePairStack(nn.Module):
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.25):
        super().__init__()
        self.n_block = n_block
        self.block = [
            PairStr2Pair(d_pair=d_templ, n_head=n_head,
                         d_hidden=d_hidden, p_drop=p_drop)
            for _ in range(n_block)
        ]
        self.norm = nn.LayerNorm(d_templ)

    def __call__(self, templ, rbf_feat, use_checkpoint=False):
        B, T, L = templ.shape[:3]
        templ = templ.reshape(B * T, L, L, -1)
        for i_block in range(self.n_block):
            templ = self.block[i_block](templ, rbf_feat)
        return self.norm(templ).reshape(B, T, L, L, -1)


class TemplateTorsionStack(nn.Module):
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.15):
        super().__init__()
        self.n_block = n_block
        self.proj_pair = nn.Linear(d_templ + 36, d_templ)
        self.row_attn = [
            AttentionWithBias(d_in=d_templ, d_bias=d_templ,
                              n_head=n_head, d_hidden=d_hidden)
            for _ in range(n_block)
        ]
        self.ff = [FeedForwardLayer(d_templ, 4, p_drop=p_drop)
                    for _ in range(n_block)]
        self.norm = nn.LayerNorm(d_templ)

    def __call__(self, tors, pair, rbf_feat, use_checkpoint=False):
        B, T, L = tors.shape[:3]
        tors = tors.reshape(B * T, L, -1)
        pair = pair.reshape(B * T, L, L, -1)
        pair = mx.concatenate([pair, rbf_feat], axis=-1)
        pair = self.proj_pair(pair)

        for i_block in range(self.n_block):
            tors = tors + self.row_attn[i_block](tors, pair)
            tors = tors + self.ff[i_block](tors)
        return self.norm(tors).reshape(B, T, L, -1)


class Templ_emb(nn.Module):
    def __init__(self, d_t1d=21 + 1 + 1, d_t2d=43 + 1, d_tor=30,
                 d_pair=128, d_state=32, n_block=2, d_templ=64,
                 n_head=4, d_hidden=16, p_drop=0.25):
        super().__init__()
        self.emb = nn.Linear(d_t1d * 2 + d_t2d, d_templ)
        self.templ_stack = TemplatePairStack(n_block=n_block, d_templ=d_templ,
                                             n_head=n_head, d_hidden=d_hidden,
                                             p_drop=p_drop)
        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_pair,
                              p_drop=p_drop)
        self.emb_t1d = nn.Linear(d_t1d + d_tor, d_templ)
        self.proj_t1d = nn.Linear(d_templ, d_templ)
        self.attn_tor = Attention(d_state, d_templ, n_head, d_hidden, d_state,
                                  p_drop=p_drop)

    def __call__(self, t1d, t2d, alpha_t, xyz_t, pair, state,
                 use_checkpoint=False):
        B, T, L, _ = t1d.shape

        left = mx.broadcast_to(
            mx.expand_dims(t1d, 3), (B, T, L, L, t1d.shape[-1]))
        right = mx.broadcast_to(
            mx.expand_dims(t1d, 2), (B, T, L, L, t1d.shape[-1]))
        templ = mx.concatenate([t2d, left, right], axis=-1)
        templ = self.emb(templ)

        xyz_t_flat = xyz_t.reshape(B * T, L, -1, 3)
        ca = xyz_t_flat[:, :, 1]
        rbf_feat = rbf(cdist(ca, ca))
        templ = self.templ_stack(templ, rbf_feat, use_checkpoint=use_checkpoint)

        t1d_cat = mx.concatenate([t1d, alpha_t], axis=-1)
        t1d_proj = self.proj_t1d(nn.relu(self.emb_t1d(t1d_cat)))

        # Mix query state with template state
        state_flat = state.reshape(B * L, 1, -1)
        t1d_perm = mx.transpose(t1d_proj, (0, 2, 1, 3)).reshape(B * L, T, -1)
        out = self.attn_tor(state_flat, t1d_perm, t1d_perm).reshape(B, L, -1)
        state = state.reshape(B, L, -1) + out

        # Mix query pair with template info
        pair_flat = pair.reshape(B * L * L, 1, -1)
        templ_perm = mx.transpose(templ, (0, 2, 3, 1, 4)).reshape(
            B * L * L, T, -1)
        out = self.attn(pair_flat, templ_perm, templ_perm).reshape(B, L, L, -1)
        pair = pair.reshape(B, L, L, -1) + out

        return pair, state


class Recycling(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=32):
        super().__init__()
        self.proj_dist = nn.Linear(36 + d_state * 2, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_msa = nn.LayerNorm(d_msa)

    def __call__(self, seq, msa, pair, xyz, state):
        B, L = pair.shape[:2]
        state = self.norm_state(state)

        left = mx.broadcast_to(
            mx.expand_dims(state, 2), (B, L, L, state.shape[-1]))
        right = mx.broadcast_to(
            mx.expand_dims(state, 1), (B, L, L, state.shape[-1]))

        N = xyz[:, :, 0]
        Ca = xyz[:, :, 1]
        C = xyz[:, :, 2]

        b = Ca - N
        c = C - Ca
        a = cross(b, c)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        dist = rbf(cdist(Cb, Cb))
        dist = mx.concatenate([dist, left, right], axis=-1)
        dist = self.proj_dist(dist)
        pair = dist + self.norm_pair(pair)
        msa = self.norm_msa(msa)
        return msa, pair, state
