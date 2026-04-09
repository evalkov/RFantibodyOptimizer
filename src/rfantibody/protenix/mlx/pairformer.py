"""
Pairformer stack for Protenix-Mini-Flow, ported to MLX.

Implements:
  - OuterProductMean  (AF3 Algorithm 10): MSA -> pair update
  - Transition        (AF3 Algorithm 11 variant): SwiGLU feed-forward
  - PairformerBlock   (AF3 Algorithm 17 lines 2-8)
  - PairformerStack   (AF3 Algorithm 17)
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from rfantibody.protenix.mlx.triangle import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from rfantibody.protenix.mlx.attention import AttentionPairBias


# ---------------------------------------------------------------------------
# Outer Product Mean
# ---------------------------------------------------------------------------

class OuterProductMean(nn.Module):
    """AF3 Algorithm 10: Outer product mean.

    Projects MSA rows, computes outer products, averages over sequences,
    and projects to pair embedding space.

    Args:
        c_m: MSA embedding dimension
        c_z: pair embedding dimension
        c_hidden: hidden dimension for the outer product
        eps: epsilon for numerical stability in normalization
    """

    def __init__(
        self,
        c_m: int = 256,
        c_z: int = 128,
        c_hidden: int = 32,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden, bias=False)
        self.linear_2 = nn.Linear(c_m, c_hidden, bias=False)
        self.linear_out = nn.Linear(c_hidden * c_hidden, c_z, bias=False)

    def __call__(
        self,
        m: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            m: [*, N_seq, N_res, c_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask
        Returns:
            [*, N_res, N_res, c_z] pair embedding update
        """
        # Normalize
        ln = self.layer_norm(m)

        # Project
        a = self.linear_1(ln)  # [*, N_seq, N_res, c_hidden]
        b = self.linear_2(ln)  # [*, N_seq, N_res, c_hidden]

        if mask is not None:
            mask_expanded = mx.expand_dims(mask, axis=-1)
            a = a * mask_expanded
            b = b * mask_expanded

        # Transpose seq and res dims: [*, N_res, N_seq, c_hidden]
        a = mx.swapaxes(a, -2, -3)
        b = mx.swapaxes(b, -2, -3)

        # Outer product averaged over sequences
        # a: [*, N_res_i, N_seq, C], b: [*, N_res_j, N_seq, C]
        # outer: [*, N_res_i, N_res_j, C, C]
        outer = mx.einsum("...iac,...jac->...ijce", a, b)
        # Note: the 'e' label is the second c_hidden from b.
        # We actually want: outer_ij = sum_s a[i,s,:] outer b[j,s,:]
        # = einsum("...sic,...sjc->...ijce" after transpose)
        # Let me correct: after transpose a is [*, N_res, N_seq, C]
        # We want outer[i,j,c1,c2] = sum_s a[i,s,c1] * b[j,s,c2]
        # = einsum("...isc,...jsd->...ijcd", a, b)
        outer = mx.einsum("...isc,...jsd->...ijcd", a, b)

        # Flatten last two dims: [*, N_res, N_res, C*C]
        shape = outer.shape[:-2] + (self.c_hidden * self.c_hidden,)
        outer = outer.reshape(shape)

        # Project to pair dimension
        outer = self.linear_out(outer)

        # Normalize by number of sequences
        if mask is not None:
            # mask: [*, N_seq, N_res] -> transpose -> [*, N_res, N_seq]
            mask_t = mx.swapaxes(mask, -1, -2)
            # norm[i,j] = sum_s mask[s,i] * mask[s,j]
            norm = mx.einsum("...is,...js->...ij", mask_t, mask_t)
            norm = mx.expand_dims(norm, axis=-1) + self.eps
            outer = outer / norm
        else:
            n_seq = m.shape[-3]
            outer = outer / float(n_seq)

        return outer


# ---------------------------------------------------------------------------
# Transition (SwiGLU)
# ---------------------------------------------------------------------------

class Transition(nn.Module):
    """SwiGLU transition block (AF3 Algorithm 11 variant).

    LayerNorm -> Linear(c, n*c) producing two halves ->
    SiLU(first_half) * second_half -> Linear(n*c/2, c)

    The "split" approach: project to 2*n*c_in, split in half, apply
    SiLU gating, then project back.

    Args:
        c_in: input/output dimension
        n: expansion factor (default 4, so hidden = 4 * c_in)
    """

    def __init__(self, c_in: int, n: int = 4):
        super().__init__()
        self.c_in = c_in
        self.n = n
        c_hidden = n * c_in

        self.layer_norm = nn.LayerNorm(c_in)
        # Project to 2 * c_hidden (for SwiGLU split)
        self.linear_in = nn.Linear(c_in, 2 * c_hidden, bias=False)
        self.linear_out = nn.Linear(c_hidden, c_in, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [*, N, c_in]
        Returns:
            [*, N, c_in]
        """
        x = self.layer_norm(x)
        ab = self.linear_in(x)
        a, b = mx.split(ab, 2, axis=-1)
        x = nn.silu(a) * b
        return self.linear_out(x)


# ---------------------------------------------------------------------------
# PairformerBlock
# ---------------------------------------------------------------------------

class PairformerBlock(nn.Module):
    """Single Pairformer block (AF3 Algorithm 17, lines 2-8).

    Applies in sequence:
      1. TriangleMultiplicationOutgoing  (pair)
      2. TriangleMultiplicationIncoming  (pair)
      3. TriangleAttentionStartingNode   (pair)
      4. TriangleAttentionEndingNode     (pair)
      5. Transition                      (pair)
      6. AttentionPairBias               (single, conditioned on pair) -- optional
      7. Transition                      (single) -- optional

    Args:
        c_z: pair embedding dimension (default 128)
        c_s: single embedding dimension (default 384, set to 0 to disable)
        c_hidden_mul: hidden dim for triangle multiplication (default 128)
        c_hidden_pair_att: per-head hidden dim for triangle attention (default 32)
        n_head_pair: number of heads for triangle attention (default 4)
        n_head_single: number of heads for pair-bias attention (default 16)
        n_transition: expansion factor for Transition (default 4)
    """

    def __init__(
        self,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        n_head_pair: int = 4,
        n_head_single: int = 16,
        n_transition: int = 4,
    ):
        super().__init__()
        self.c_s = c_s

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z=c_z, c_hidden=c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=c_z, c_hidden=c_hidden_mul)
        self.tri_att_start = TriangleAttentionStartingNode(
            c_in=c_z, c_hidden=c_hidden_pair_att, n_head=n_head_pair
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_in=c_z, c_hidden=c_hidden_pair_att, n_head=n_head_pair
        )
        self.pair_transition = Transition(c_in=c_z, n=n_transition)

        if c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                c_s=c_s, c_z=c_z, n_head=n_head_single
            )
            self.single_transition = Transition(c_in=c_s, n=n_transition)

    def __call__(
        self,
        s: mx.array | None,
        z: mx.array,
        pair_mask: mx.array | None = None,
    ) -> tuple[mx.array | None, mx.array]:
        """
        Args:
            s: [B, N, c_s] single embeddings (or None if c_s == 0)
            z: [B, N, N, c_z] pair embeddings
            pair_mask: [B, N, N] pair mask (optional)
        Returns:
            (s_updated, z_updated)
        """
        # Triangle updates on pair representation (residual connections)
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask)
        z = z + self.tri_att_end(z, mask=pair_mask)
        z = z + self.pair_transition(z)

        # Single representation updates (optional)
        if self.c_s > 0 and s is not None:
            s = s + self.attention_pair_bias(s=s, z=z)
            s = s + self.single_transition(s)

        return s, z


# ---------------------------------------------------------------------------
# PairformerStack
# ---------------------------------------------------------------------------

class PairformerStack(nn.Module):
    """Stack of PairformerBlocks (AF3 Algorithm 17).

    Periodically calls mx.eval() to bound memory usage on long stacks.

    Args:
        n_blocks: number of Pairformer blocks (default 16)
        c_z: pair embedding dimension (default 128)
        c_s: single embedding dimension (default 384)
        c_hidden_mul: hidden dim for triangle multiplication (default 128)
        c_hidden_pair_att: per-head hidden dim for triangle attention (default 32)
        n_head_pair: number of heads for triangle attention (default 4)
        n_head_single: number of heads for pair-bias attention (default 16)
        n_transition: expansion factor for Transition (default 4)
        eval_stride: call mx.eval() every this many blocks (default 4)
    """

    def __init__(
        self,
        n_blocks: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        n_head_pair: int = 4,
        n_head_single: int = 16,
        n_transition: int = 4,
        eval_stride: int = 4,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.eval_stride = eval_stride

        self.blocks = [
            PairformerBlock(
                c_z=c_z,
                c_s=c_s,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                n_head_pair=n_head_pair,
                n_head_single=n_head_single,
                n_transition=n_transition,
            )
            for _ in range(n_blocks)
        ]

    def __call__(
        self,
        s: mx.array | None,
        z: mx.array,
        pair_mask: mx.array | None = None,
    ) -> tuple[mx.array | None, mx.array]:
        """
        Args:
            s: [B, N, c_s] single embeddings (or None)
            z: [B, N, N, c_z] pair embeddings
            pair_mask: [B, N, N] pair mask (optional)
        Returns:
            (s_updated, z_updated)
        """
        for i, block in enumerate(self.blocks):
            s, z = block(s, z, pair_mask=pair_mask)

            # Periodic evaluation to bound lazy graph size and memory
            if (i + 1) % self.eval_stride == 0:
                eval_list = [z]
                if s is not None:
                    eval_list.append(s)
                mx.eval(*eval_list)

        return s, z
