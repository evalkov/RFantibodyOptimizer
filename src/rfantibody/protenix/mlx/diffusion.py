"""
Diffusion module for Protenix-Mini-Flow on MLX (Apple Silicon).

Implements:
  - FourierEmbedding: noise level embedding (AF3 Algorithm 21, line 8)
  - DiffusionConditioning: combines noise + trunk conditioning (Algorithm 21)
  - DiffusionTransformerBlock: self-attention + transition (Algorithm 23)
  - AtomAttentionEncoder: atom-level encoder with transformer (Algorithm 5)
  - AtomAttentionDecoder: atom-level decoder with transformer (Algorithm 6)
  - DiffusionModule: full diffusion network (Algorithm 20)
  - FlowMatchingODESampler: 5-step ODE sampler with TeaCache hook

Ported from:
  - Protenix  (protenix/model/modules/diffusion.py)
  - Protenix  (protenix/model/modules/transformer.py)
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class LinearNoBias(nn.Module):
    """Linear layer without bias, matching Protenix convention."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        scale = 1.0 / math.sqrt(in_features)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(out_features, in_features)
        )

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


class SwiGLUTransition(nn.Module):
    """SwiGLU feed-forward transition block (conditioned variant).

    gate = swish(Linear(x))
    value = Linear(x)
    output = Linear(gate * value)
    """

    def __init__(self, c_in: int, n: int = 2):
        super().__init__()
        c_hidden = c_in * n
        self.linear_gate = LinearNoBias(c_in, c_hidden)
        self.linear_value = LinearNoBias(c_in, c_hidden)
        self.linear_out = LinearNoBias(c_hidden, c_in)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.linear_gate(x))
        value = self.linear_value(x)
        return self.linear_out(gate * value)


class AdaptiveLayerNorm(nn.Module):
    """Layer norm conditioned on single representation s.

    Computes standard LN on a, then modulates with learned
    scale/shift derived from s.
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.ln = nn.LayerNorm(c_a, affine=False)
        self.linear = LinearNoBias(c_s, 2 * c_a)

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        a_norm = self.ln(a)
        params = self.linear(s)
        gamma, beta = mx.split(params, 2, axis=-1)
        return a_norm * (1 + gamma) + beta


# ---------------------------------------------------------------------------
# FourierEmbedding
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """Random Fourier feature embedding for noise levels.

    Maps a scalar noise level to a fixed-dim embedding via:
        embed = [sin(2*pi*W*x), cos(2*pi*W*x)]
    where W is drawn from N(0, 1) at init and frozen.

    Args:
        c: output dimension (must be even; c//2 frequencies).
    """

    def __init__(self, c: int = 256, seed: int = 42):
        super().__init__()
        # Frozen random frequencies and biases (Algorithm 22)
        # Protenix uses cos-only: output dim = c (not 2c)
        key = mx.random.key(seed)
        self.w = mx.random.normal(shape=(c,), key=key)
        key2 = mx.random.key(seed + 1)
        self.b = mx.random.normal(shape=(c,), key=key2)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [...] arbitrary-shape noise level (already log-scaled)
        Returns:
            [..., c] Fourier embedding
        """
        # x: [...] -> [..., 1]
        x = mx.expand_dims(x, axis=-1)
        # cos(2pi(x*w + b)) -- cosine-only, matching Protenix
        return mx.cos(2.0 * math.pi * (x * self.w + self.b))


# ---------------------------------------------------------------------------
# DiffusionConditioning  (Algorithm 21)
# ---------------------------------------------------------------------------

class DiffusionConditioning(nn.Module):
    """Noise-conditioned single/pair embedding.

    Implements AF3 Algorithm 21 (simplified for Mini-Flow):
      1. Combine s_trunk + s_inputs -> project to c_s
      2. Fourier-embed noise level, project to c_s, add to single
      3. Two SwiGLU transitions on single
      4. Pair conditioning: z_trunk through two SwiGLU transitions

    Args:
        sigma_data: data standard deviation constant (16.0).
        c_s: single representation dim.
        c_z: pair representation dim.
        c_s_inputs: input embedding dim from InputFeatureEmbedder.
        c_noise_emb: Fourier embedding dim.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        c_noise_emb: int = 256,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_s = c_s
        self.c_z = c_z

        # Single conditioning
        self.ln_s = nn.LayerNorm(c_s + c_s_inputs)
        self.linear_s = LinearNoBias(c_s + c_s_inputs, c_s)

        # Noise embedding
        self.fourier_emb = FourierEmbedding(c=c_noise_emb)
        self.ln_n = nn.LayerNorm(c_noise_emb)
        self.linear_n = LinearNoBias(c_noise_emb, c_s)

        # Single transitions
        self.transition_s1 = SwiGLUTransition(c_s, n=2)
        self.transition_s2 = SwiGLUTransition(c_s, n=2)

        # Pair transitions
        self.transition_z1 = SwiGLUTransition(c_z, n=2)
        self.transition_z2 = SwiGLUTransition(c_z, n=2)

    def __call__(
        self,
        t: mx.array,
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Args:
            t: [...] noise level sigma
            s_inputs: [..., N_tokens, c_s_inputs]
            s_trunk: [..., N_tokens, c_s]
            z_trunk: [..., N_tokens, N_tokens, c_z]

        Returns:
            s: [..., N_tokens, c_s] conditioned single
            z: [..., N_tokens, N_tokens, c_z] conditioned pair
        """
        # --- Single conditioning ---
        s = mx.concatenate([s_trunk, s_inputs], axis=-1)
        s = self.linear_s(self.ln_s(s))

        # Noise embedding: log(t / sigma_data) / 4
        n = mx.log(t / self.sigma_data) / 4.0
        n = self.fourier_emb(n)  # [..., c_noise_emb]
        n = self.linear_n(self.ln_n(n))  # [..., c_s]

        # Broadcast noise embedding to token dim
        # n: [..., c_s] -> [..., 1, c_s]
        s = s + mx.expand_dims(n, axis=-2)

        s = s + self.transition_s1(s)
        s = s + self.transition_s2(s)

        # --- Pair conditioning ---
        z = z_trunk
        z = z + self.transition_z1(z)
        z = z + self.transition_z2(z)

        return s, z


# ---------------------------------------------------------------------------
# AttentionPairBias (simplified, self-contained)
# ---------------------------------------------------------------------------

class AttentionPairBias(nn.Module):
    """Multi-head self-attention with pair bias and adaptive layer norm.

    Implements the attention component of AF3 Algorithm 23.

    Args:
        c_a: token activation dim (c_token).
        c_s: single embedding dim (for AdaLN conditioning).
        c_z: pair embedding dim (for attention bias).
        n_head: number of attention heads.
    """

    def __init__(self, c_a: int, c_s: int, c_z: int, n_head: int):
        super().__init__()
        self.c_a = c_a
        self.n_head = n_head
        self.d_head = c_a // n_head
        assert c_a % n_head == 0

        # Adaptive layer norm
        self.ada_ln = AdaptiveLayerNorm(c_a, c_s)

        # QKV projections (no bias)
        self.w_q = LinearNoBias(c_a, c_a)
        self.w_k = LinearNoBias(c_a, c_a)
        self.w_v = LinearNoBias(c_a, c_a)

        # Pair bias: normalize pair rep then project to per-head bias
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_bias = LinearNoBias(c_z, n_head)

        # Gating
        self.linear_gate = nn.Linear(c_a, c_a)

        # Output projection
        self.linear_out = LinearNoBias(c_a, c_a)

    def __call__(
        self,
        a: mx.array,
        s: mx.array,
        z: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            a: [..., N, c_a] token embeddings
            s: [..., N, c_s] single embeddings (for AdaLN)
            z: [..., N, N, c_z] pair embeddings (for attention bias)
            mask: [..., N] token mask

        Returns:
            [..., N, c_a] attention output (residual-ready)
        """
        N = a.shape[-2]

        # AdaLN conditioning
        a_norm = self.ada_ln(a, s)

        # QKV
        q = self.w_q(a_norm)
        k = self.w_k(a_norm)
        v = self.w_v(a_norm)

        # Reshape for multi-head: [..., N, n_head, d_head]
        q = q.reshape(*q.shape[:-1], self.n_head, self.d_head)
        k = k.reshape(*k.shape[:-1], self.n_head, self.d_head)
        v = v.reshape(*v.shape[:-1], self.n_head, self.d_head)

        # Transpose to [..., n_head, N, d_head]
        q = mx.transpose(q, axes=(*range(len(q.shape) - 3), -2, -3, -1))
        k = mx.transpose(k, axes=(*range(len(k.shape) - 3), -2, -3, -1))
        v = mx.transpose(v, axes=(*range(len(v.shape) - 3), -2, -3, -1))

        # Pair bias: normalize then project [..., N, N, c_z] -> [..., N, N, n_head] -> [..., n_head, N, N]
        pair_bias = self.linear_bias(self.layer_norm_z(z))
        pair_bias = mx.transpose(
            pair_bias,
            axes=(*range(len(pair_bias.shape) - 3), -1, -3, -2),
        )

        # Scaled dot-product attention with pair bias
        scale = math.sqrt(self.d_head)
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / scale, mask=pair_bias
        )

        # Transpose back: [..., n_head, N, d_head] -> [..., N, n_head, d_head]
        attn_out = mx.transpose(
            attn_out,
            axes=(*range(len(attn_out.shape) - 3), -2, -3, -1),
        )
        attn_out = attn_out.reshape(*attn_out.shape[:-2], self.c_a)

        # Gating
        gate = mx.sigmoid(self.linear_gate(a_norm))
        attn_out = gate * attn_out

        return self.linear_out(attn_out)


# ---------------------------------------------------------------------------
# ConditionedTransition
# ---------------------------------------------------------------------------

class ConditionedTransition(nn.Module):
    """SwiGLU transition conditioned on single embedding s.

    Uses AdaLN for conditioning before the transition.
    """

    def __init__(self, c_a: int, c_s: int, n: int = 2):
        super().__init__()
        self.ada_ln = AdaptiveLayerNorm(c_a, c_s)
        self.transition = SwiGLUTransition(c_a, n=n)

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        return self.transition(self.ada_ln(a, s))


# ---------------------------------------------------------------------------
# DiffusionTransformerBlock  (Algorithm 23)
# ---------------------------------------------------------------------------

class DiffusionTransformerBlock(nn.Module):
    """Single block of the diffusion transformer.

    Implements AF3 Algorithm 23:
      a = a + AttentionPairBias(a, s, z)
      a = a + ConditionedTransition(a, s)

    Args:
        c_token: token activation dim (768).
        c_s: single embedding dim (384).
        c_z: pair embedding dim (128).
        n_head: number of attention heads (16).
        n_transition: transition expansion factor (2).
    """

    def __init__(
        self,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        n_head: int = 16,
        n_transition: int = 2,
    ):
        super().__init__()
        self.attention = AttentionPairBias(
            c_a=c_token, c_s=c_s, c_z=c_z, n_head=n_head
        )
        self.transition = ConditionedTransition(
            c_a=c_token, c_s=c_s, n=n_transition
        )

    def __call__(
        self,
        a: mx.array,
        s: mx.array,
        z: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            a: [..., N, c_token] token-level embedding
            s: [..., N, c_s] conditioned single embedding
            z: [..., N, N, c_z] conditioned pair embedding
            mask: [..., N] token mask

        Returns:
            [..., N, c_token] updated token embedding
        """
        a = a + self.attention(a, s, z, mask)
        a = a + self.transition(a, s)
        return a


# ---------------------------------------------------------------------------
# Atom-level Attention Modules (Algorithms 5, 6, 7)
# ---------------------------------------------------------------------------

class AtomAdaptiveLayerNorm(nn.Module):
    """Protenix-style AdaptiveLayerNorm used in atom attention.

    The Protenix checkpoint stores:
      - layernorm_s.weight: [c_a] for the LN itself
      - linear_nobias_s.weight: [c_a, c_s] for scale (no bias)
      - linear_s.weight: [c_a, c_s] + linear_s.bias: [c_a] for shift

    This differs from the main diffusion AdaLN which combines them.
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.layernorm_s = nn.LayerNorm(c_a, affine=True)
        self.linear_nobias_s = LinearNoBias(c_s, c_a)
        self.linear_s = nn.Linear(c_s, c_a)

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        a_norm = self.layernorm_s(a)
        scale = self.linear_nobias_s(s)
        shift = self.linear_s(s)
        return a_norm * (1 + scale) + shift


class AtomAttentionPairBias(nn.Module):
    """AttentionPairBias for atom-level transformers.

    Matches the Protenix checkpoint structure with:
      - layernorm_a (AtomAdaptiveLayerNorm) for query
      - layernorm_kv (AtomAdaptiveLayerNorm) for key/value (cross_attention_mode)
      - linear_a_last: gating by single embedding s
      - Standard attention: linear_q (with bias), linear_k, linear_v, linear_g, linear_o
      - layernorm_z + linear_nobias_z for pair bias

    Args:
        c_a: atom embedding dim (128).
        c_s: atom single embedding dim (128 = c_atom).
        c_z: atom pair dim (16 = c_atompair).
        n_heads: number of attention heads (4).
    """

    def __init__(self, c_a: int, c_s: int, c_z: int, n_heads: int):
        super().__init__()
        self.c_a = c_a
        self.n_heads = n_heads
        self.d_head = c_a // n_heads
        assert c_a % n_heads == 0

        # AdaLN for query and key/value (cross_attention_mode=True)
        self.layernorm_a = AtomAdaptiveLayerNorm(c_a, c_s)
        self.layernorm_kv = AtomAdaptiveLayerNorm(c_a, c_s)

        # Attention projections (matching Protenix naming)
        self.attention = _AtomAttention(c_a, n_heads)

        # Pair bias
        self.layernorm_z = nn.LayerNorm(c_z)
        self.linear_nobias_z = LinearNoBias(c_z, n_heads)

        # Output gating by single embedding (adaLN-Zero)
        self.linear_a_last = nn.Linear(c_s, c_a)

    def __call__(
        self,
        a: mx.array,
        s: mx.array,
        z: mx.array,
    ) -> mx.array:
        """
        Args:
            a: [..., N, c_a] atom embeddings
            s: [..., N, c_s] atom single conditioning
            z: [..., N, N, c_z] atom pair embeddings

        Returns:
            [..., N, c_a] attention output (residual-ready)
        """
        # AdaLN conditioning
        q_input = self.layernorm_a(a, s)
        kv_input = self.layernorm_kv(a, s)

        # Pair bias: [..., N, N, c_z] -> [..., N, N, n_heads] -> [..., n_heads, N, N]
        pair_bias = self.linear_nobias_z(self.layernorm_z(z))
        pair_bias = mx.transpose(
            pair_bias,
            axes=(*range(len(pair_bias.shape) - 3), -1, -3, -2),
        )

        # Attention with gating
        out = self.attention(q_input, kv_input, pair_bias)

        # Output gating by single embedding (adaLN-Zero)
        out = mx.sigmoid(self.linear_a_last(s)) * out

        return out


class _AtomAttention(nn.Module):
    """Low-level multi-head attention for atom transformers.

    Matches Protenix checkpoint structure:
      - linear_q (with bias), linear_k, linear_v, linear_g, linear_o
    """

    def __init__(self, c_a: int, n_heads: int):
        super().__init__()
        self.c_a = c_a
        self.n_heads = n_heads
        self.d_head = c_a // n_heads

        self.linear_q = nn.Linear(c_a, c_a)  # has bias
        self.linear_k = LinearNoBias(c_a, c_a)
        self.linear_v = LinearNoBias(c_a, c_a)
        self.linear_g = LinearNoBias(c_a, c_a)
        self.linear_o = LinearNoBias(c_a, c_a)

    def __call__(
        self,
        q_input: mx.array,
        kv_input: mx.array,
        pair_bias: mx.array,
    ) -> mx.array:
        """
        Args:
            q_input: [..., N, c_a] query input (after AdaLN)
            kv_input: [..., N, c_a] key/value input (after AdaLN)
            pair_bias: [..., n_heads, N, N] attention bias

        Returns:
            [..., N, c_a] attention output
        """
        q = self.linear_q(q_input)
        k = self.linear_k(kv_input)
        v = self.linear_v(kv_input)

        # Reshape for multi-head: [..., N, n_heads, d_head]
        q = q.reshape(*q.shape[:-1], self.n_heads, self.d_head)
        k = k.reshape(*k.shape[:-1], self.n_heads, self.d_head)
        v = v.reshape(*v.shape[:-1], self.n_heads, self.d_head)

        # Transpose to [..., n_heads, N, d_head]
        perm = (*range(len(q.shape) - 3), -2, -3, -1)
        q = mx.transpose(q, axes=perm)
        k = mx.transpose(k, axes=perm)
        v = mx.transpose(v, axes=perm)

        # Scaled dot-product attention with pair bias
        scale = math.sqrt(self.d_head)
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / scale, mask=pair_bias
        )

        # Transpose back: [..., n_heads, N, d_head] -> [..., N, n_heads, d_head]
        perm_back = (*range(len(attn_out.shape) - 3), -2, -3, -1)
        attn_out = mx.transpose(attn_out, axes=perm_back)
        attn_out = attn_out.reshape(*attn_out.shape[:-2], self.c_a)

        # Gating
        gate = mx.sigmoid(self.linear_g(q_input))
        attn_out = gate * attn_out

        return self.linear_o(attn_out)


class AtomConditionedTransitionBlock(nn.Module):
    """Conditioned transition block for atom transformers (Algorithm 25).

    Matches Protenix checkpoint structure with separate adaln, linear_nobias_a1/a2,
    linear_nobias_b, and linear_s for output gating.

    Args:
        c_a: atom embedding dim (128).
        c_s: atom single embedding dim (128).
        n: expansion factor (2).
    """

    def __init__(self, c_a: int, c_s: int, n: int = 2):
        super().__init__()
        self.adaln = AtomAdaptiveLayerNorm(c_a, c_s)
        self.linear_nobias_a1 = LinearNoBias(c_a, n * c_a)
        self.linear_nobias_a2 = LinearNoBias(c_a, n * c_a)
        self.linear_nobias_b = LinearNoBias(n * c_a, c_a)
        self.linear_s = nn.Linear(c_s, c_a)

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        a_normed = self.adaln(a, s)
        b = nn.silu(self.linear_nobias_a1(a_normed)) * self.linear_nobias_a2(a_normed)
        # Output gating by single (adaLN-Zero)
        return mx.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)


class AtomDiffusionTransformerBlock(nn.Module):
    """Single block of atom-level diffusion transformer.

    Combines AtomAttentionPairBias + AtomConditionedTransitionBlock.
    """

    def __init__(self, c_a: int, c_s: int, c_z: int, n_heads: int):
        super().__init__()
        self.attention_pair_bias = AtomAttentionPairBias(
            c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads
        )
        self.conditioned_transition_block = AtomConditionedTransitionBlock(
            c_a=c_a, c_s=c_s, n=2
        )

    def __call__(self, a: mx.array, s: mx.array, z: mx.array) -> mx.array:
        a = a + self.attention_pair_bias(a, s, z)
        a = a + self.conditioned_transition_block(a, s)
        return a


class AtomTransformer(nn.Module):
    """Atom-level transformer (Algorithm 7).

    Uses DiffusionTransformer blocks with cross_attention_mode=True
    at atom resolution with atom pair bias.

    For protein-only (1:1 atom-to-token mapping), the local windowing
    is not needed -- we run standard full attention at atom resolution
    since N_atoms = N_tokens.

    Args:
        c_atom: atom embedding dim (128).
        c_atompair: atom pair dim (16).
        n_blocks: number of transformer blocks.
        n_heads: number of attention heads (4).
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 1,
        n_heads: int = 4,
    ):
        super().__init__()
        self.diffusion_transformer = _AtomDiffusionTransformer(
            c_a=c_atom, c_s=c_atom, c_z=c_atompair,
            n_blocks=n_blocks, n_heads=n_heads,
        )

    def __call__(
        self,
        q: mx.array,
        c: mx.array,
        p: mx.array,
    ) -> mx.array:
        """
        Args:
            q: [..., N_atom, c_atom] atom query embedding
            c: [..., N_atom, c_atom] atom conditioning embedding
            p: [..., N_atom, N_atom, c_atompair] atom pair embedding

        Returns:
            [..., N_atom, c_atom] updated atom embedding
        """
        return self.diffusion_transformer(a=q, s=c, z=p)


class _AtomDiffusionTransformer(nn.Module):
    """Stack of AtomDiffusionTransformerBlocks.

    This is the inner 'diffusion_transformer' inside AtomTransformer.
    """

    def __init__(self, c_a: int, c_s: int, c_z: int, n_blocks: int, n_heads: int):
        super().__init__()
        self.blocks = [
            AtomDiffusionTransformerBlock(c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads)
            for _ in range(n_blocks)
        ]

    def __call__(self, a: mx.array, s: mx.array, z: mx.array) -> mx.array:
        for block in self.blocks:
            a = block(a, s, z)
        return a


# ---------------------------------------------------------------------------
# AtomAttentionEncoder (Algorithm 5)
# ---------------------------------------------------------------------------

class AtomAttentionEncoder(nn.Module):
    """Atom attention encoder for the diffusion module.

    Projects atom features and noisy coordinates to atom space, runs an atom
    transformer, then aggregates atom features to token-level features.

    For protein-only prediction with 1:1 atom-to-token mapping (Ca atoms),
    atom_to_token_idx is simply range(N), and aggregation is identity.

    The encoder creates:
      - c_l: atom single conditioning from ref_pos, ref_charge, features
      - q_l: atom query = c_l + s_trunk_broadcast + noisy_r
      - p_lm: atom pair embedding from distances, inv distances, validity, z_pair_broadcast
      - After atom transformer: aggregate atom features to token level

    Args:
        c_atom: atom feature dim (128).
        c_atompair: atom pair dim (16).
        c_token: output token dim (768).
        c_s: single embedding dim (384).
        c_z: pair embedding dim (128).
        n_blocks: number of atom transformer blocks.
        n_heads: number of attention heads.
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 1,
        n_heads: int = 4,
    ):
        super().__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token

        # --- Atom single conditioning projections ---
        self.linear_no_bias_ref_pos = LinearNoBias(3, c_atom)
        self.linear_no_bias_ref_charge = LinearNoBias(1, c_atom)
        # Features: ref_mask(1) + ref_element(128) + ref_atom_name_chars(256) = 385
        self.linear_no_bias_f = LinearNoBias(385, c_atom)

        # --- Atom pair projections ---
        self.linear_no_bias_d = LinearNoBias(3, c_atompair)
        self.linear_no_bias_invd = LinearNoBias(1, c_atompair)
        self.linear_no_bias_v = LinearNoBias(1, c_atompair)

        # --- Conditioning from trunk ---
        self.layernorm_s = nn.LayerNorm(c_s, affine=True)
        self.linear_no_bias_s = LinearNoBias(c_s, c_atom)
        self.layernorm_z = nn.LayerNorm(c_z, affine=True)
        self.linear_no_bias_z = LinearNoBias(c_z, c_atompair)

        # --- Noisy coordinates ---
        self.linear_no_bias_r = LinearNoBias(3, c_atom)

        # --- Pair refinement ---
        self.linear_no_bias_cl = LinearNoBias(c_atom, c_atompair)
        self.linear_no_bias_cm = LinearNoBias(c_atom, c_atompair)

        # Small MLP on pair: ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear
        # Protenix stores as small_mlp.{1,3,5}.weight (indices 0,2,4 are ReLU)
        self.small_mlp = _SmallMLP(c_atompair)

        # --- Atom transformer ---
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom, c_atompair=c_atompair,
            n_blocks=n_blocks, n_heads=n_heads,
        )

        # --- Aggregate to token level ---
        self.linear_no_bias_q = LinearNoBias(c_atom, c_token)

    def __call__(
        self,
        r_noisy: mx.array,
        s_trunk: mx.array,
        z_pair: mx.array,
        ref_pos: mx.array,
        ref_charge: mx.array,
        ref_mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Args:
            r_noisy: [..., N_atoms, 3] noisy coordinates (already scaled by c_in)
            s_trunk: [..., N_tokens, c_s] trunk single embedding
            z_pair: [..., N_tokens, N_tokens, c_z] trunk pair embedding
            ref_pos: [..., N_atoms, 3] reference positions (Ca coords)
            ref_charge: [..., N_atoms] reference charges
            ref_mask: [..., N_atoms] reference mask

        Returns:
            a_token: [..., N_tokens, c_token] token-level features
            q_l: [..., N_atoms, c_atom] atom query (skip connection for decoder)
            c_l: [..., N_atoms, c_atom] atom conditioning (skip for decoder)
            p_lm: [..., N_atoms, N_atoms, c_atompair] pair embedding (skip for decoder)
        """
        N_atoms = r_noisy.shape[-2]

        # --- Build atom single conditioning c_l ---
        ref_charge_expanded = mx.expand_dims(ref_charge, axis=-1)  # [..., N, 1]
        ref_mask_expanded = mx.expand_dims(ref_mask, axis=-1)  # [..., N, 1]

        c_l = (
            self.linear_no_bias_ref_pos(ref_pos)
            + self.linear_no_bias_ref_charge(mx.arcsinh(ref_charge_expanded))
        )
        # For protein-only: features are just ref_mask (1 dim).
        # ref_element and ref_atom_name_chars are zeros for proteins.
        # We still need the right input dim (385).
        features = mx.concatenate([
            ref_mask_expanded,
            mx.zeros((*ref_mask.shape, 128)),  # ref_element placeholder
            mx.zeros((*ref_mask.shape, 256)),   # ref_atom_name_chars placeholder
        ], axis=-1)
        c_l = c_l + self.linear_no_bias_f(features.astype(c_l.dtype))
        c_l = c_l * ref_mask_expanded

        # --- Build atom pair embedding p_lm ---
        # Pairwise distance vectors: d_ij = ref_pos_i - ref_pos_j
        d = mx.expand_dims(ref_pos, axis=-2) - mx.expand_dims(ref_pos, axis=-3)
        # [..., N, N, 3]

        # Validity mask: v_ij = mask_i * mask_j
        v = mx.expand_dims(ref_mask, axis=-1) * mx.expand_dims(ref_mask, axis=-2)
        v = mx.expand_dims(v, axis=-1)  # [..., N, N, 1]

        p_lm = self.linear_no_bias_d(d) * v
        inv_d = 1.0 / (1.0 + mx.sum(d ** 2, axis=-1, keepdims=True))
        p_lm = p_lm + self.linear_no_bias_invd(inv_d) * v
        p_lm = p_lm + self.linear_no_bias_v(v.astype(p_lm.dtype))

        # Add z_pair broadcast (token pair -> atom pair, 1:1 mapping)
        p_lm = p_lm + self.linear_no_bias_z(self.layernorm_z(z_pair))

        # Add trunk single conditioning to atoms (1:1 mapping for proteins)
        c_l = c_l + self.linear_no_bias_s(self.layernorm_s(s_trunk))

        # Add noisy coordinates
        q_l = c_l + self.linear_no_bias_r(r_noisy)

        # --- Refine pair embedding with atom conditioning ---
        c_l_expanded_q = mx.expand_dims(c_l, axis=-2)  # [..., N, 1, c_atom]
        c_l_expanded_k = mx.expand_dims(c_l, axis=-3)  # [..., 1, N, c_atom]
        p_lm = (
            p_lm
            + self.linear_no_bias_cl(nn.relu(c_l_expanded_q))
            + self.linear_no_bias_cm(nn.relu(c_l_expanded_k))
        )
        p_lm = p_lm + self.small_mlp(p_lm)

        # --- Run atom transformer ---
        q_l = self.atom_transformer(q_l, c_l, p_lm)

        # --- Aggregate to token level (1:1 for proteins: just project) ---
        a_token = nn.relu(self.linear_no_bias_q(q_l))

        return a_token, q_l, c_l, p_lm


class _SmallMLP(nn.Module):
    """Small MLP for pair refinement in AtomAttentionEncoder.

    Structure: ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear
    Protenix checkpoint: small_mlp.{1,3,5}.weight
    (indices 0,2,4 are ReLU activations)

    We expose .layers as a list to match the 1-indexed naming.
    """

    def __init__(self, c: int):
        super().__init__()
        # We name these to match checkpoint: small_mlp.1, small_mlp.3, small_mlp.5
        # But MLX lists are 0-indexed, so we use indices 0, 1, 2 internally
        # and handle mapping in weight_converter
        self.layer_1 = LinearNoBias(c, c)
        self.layer_3 = LinearNoBias(c, c)
        self.layer_5 = LinearNoBias(c, c)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(x)
        x = self.layer_1(x)
        x = nn.relu(x)
        x = self.layer_3(x)
        x = nn.relu(x)
        x = self.layer_5(x)
        return x


# ---------------------------------------------------------------------------
# AtomAttentionDecoder (Algorithm 6)
# ---------------------------------------------------------------------------

class AtomAttentionDecoder(nn.Module):
    """Atom attention decoder for the diffusion module.

    Projects token features back to atom space, adds skip connection from
    encoder, runs atom transformer, then projects to coordinate updates.

    Args:
        c_token: input token dim (768).
        c_atom: atom feature dim (128).
        c_atompair: atom pair dim (16).
        n_blocks: number of atom transformer blocks.
        n_heads: number of attention heads.
    """

    def __init__(
        self,
        c_token: int = 768,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 1,
        n_heads: int = 4,
    ):
        super().__init__()
        self.linear_no_bias_a = LinearNoBias(c_token, c_atom)
        self.layernorm_q = nn.LayerNorm(c_atom, affine=True)
        self.linear_no_bias_out = LinearNoBias(c_atom, 3)
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom, c_atompair=c_atompair,
            n_blocks=n_blocks, n_heads=n_heads,
        )

    def __call__(
        self,
        a_token: mx.array,
        q_skip: mx.array,
        c_skip: mx.array,
        p_skip: mx.array,
    ) -> mx.array:
        """
        Args:
            a_token: [..., N_tokens, c_token] token-level features
            q_skip: [..., N_atoms, c_atom] skip connection from encoder
            c_skip: [..., N_atoms, c_atom] conditioning skip from encoder
            p_skip: [..., N_atoms, N_atoms, c_atompair] pair skip from encoder

        Returns:
            [..., N_atoms, 3] coordinate updates
        """
        # Broadcast token to atom (1:1 for proteins) + skip connection
        q = self.linear_no_bias_a(a_token) + q_skip

        # Run atom transformer
        q = self.atom_transformer(q, c_skip, p_skip)

        # Project to coordinate update
        r = self.linear_no_bias_out(self.layernorm_q(q))
        return r


# ---------------------------------------------------------------------------
# DiffusionModule  (Algorithm 20)
# ---------------------------------------------------------------------------

class DiffusionModule(nn.Module):
    """Complete diffusion module for structure prediction.

    Implements AF3 Algorithm 20:
      1. AtomAttentionEncoder: project atoms to token space with atom transformer
      2. DiffusionTransformer stack: full self-attention at token level
      3. AtomAttentionDecoder: project back to atom space with atom transformer

    The EDM parameterization is used:
      - c_in  = 1 / sqrt(sigma_data^2 + sigma^2)
      - c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
      - c_out  = sigma * sigma_data / sqrt(sigma_data^2 + sigma^2)
      - D(x, sigma) = c_skip * x + c_out * F(c_in * x, c_noise(sigma))

    Args:
        sigma_data: data std dev constant.
        c_atom: atom feature dim.
        c_atompair: atom pair dim.
        c_token: token activation dim.
        c_s: single embedding dim.
        c_z: pair embedding dim.
        c_s_inputs: input embedding dim.
        c_noise_emb: noise Fourier embedding dim.
        n_atom_encoder_blocks: number of atom encoder transformer blocks.
        n_transformer_blocks: number of main diffusion transformer blocks.
        n_atom_decoder_blocks: number of atom decoder transformer blocks.
        n_head: number of transformer heads.
        n_atom_head: number of atom transformer heads.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        c_noise_emb: int = 256,
        n_atom_encoder_blocks: int = 1,
        n_transformer_blocks: int = 8,
        n_atom_decoder_blocks: int = 1,
        n_head: int = 16,
        n_atom_head: int = 4,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_token = c_token
        self.c_atom = c_atom

        # Conditioning
        self.conditioning = DiffusionConditioning(
            sigma_data=sigma_data,
            c_s=c_s,
            c_z=c_z,
            c_s_inputs=c_s_inputs,
            c_noise_emb=c_noise_emb,
        )

        # --- AtomAttentionEncoder ---
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
            n_blocks=n_atom_encoder_blocks,
            n_heads=n_atom_head,
        )

        # Single conditioning projection (Alg20 line 4)
        self.ln_s = nn.LayerNorm(c_s, affine=False)
        self.linear_s_to_token = LinearNoBias(c_s, c_token)

        # Diffusion transformer stack
        self.transformer_blocks = [
            DiffusionTransformerBlock(
                c_token=c_token,
                c_s=c_s,
                c_z=c_z,
                n_head=n_head,
                n_transition=2,
            )
            for _ in range(n_transformer_blocks)
        ]

        self.ln_a = nn.LayerNorm(c_token, affine=False)

        # --- AtomAttentionDecoder ---
        self.atom_attention_decoder = AtomAttentionDecoder(
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_blocks=n_atom_decoder_blocks,
            n_heads=n_atom_head,
        )

    def f_forward(
        self,
        r_noisy: mx.array,
        sigma: mx.array,
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: mx.array,
        ref_pos: mx.array,
        ref_charge: mx.array,
        ref_mask: mx.array,
    ) -> mx.array:
        """Raw network F_theta(c_in * x, c_noise(sigma)).

        Args:
            r_noisy: [..., N_atoms, 3] scaled noisy coordinates
            sigma: [...] noise level
            s_inputs: [..., N_tokens, c_s_inputs]
            s_trunk: [..., N_tokens, c_s]
            z_trunk: [..., N_tokens, N_tokens, c_z]
            ref_pos: [..., N_atoms, 3] reference positions
            ref_charge: [..., N_atoms] reference charges
            ref_mask: [..., N_atoms] atom mask

        Returns:
            [..., N_atoms, 3] predicted coordinate update
        """
        # Conditioning
        s_cond, z_cond = self.conditioning(sigma, s_inputs, s_trunk, z_trunk)

        # Atom attention encoder
        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            r_noisy=r_noisy,
            s_trunk=s_cond,
            z_pair=z_cond,
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_mask=ref_mask,
        )

        # Add conditioned single embedding
        a_token = a_token + self.linear_s_to_token(self.ln_s(s_cond))

        # Diffusion transformer stack
        for block in self.transformer_blocks:
            a_token = block(a_token, s_cond, z_cond)

        a_token = self.ln_a(a_token)

        # Atom attention decoder
        r_update = self.atom_attention_decoder(
            a_token=a_token,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
        )
        return r_update

    def __call__(
        self,
        x_noisy: mx.array,
        sigma: mx.array,
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: mx.array,
        ref_pos: Optional[mx.array] = None,
        ref_charge: Optional[mx.array] = None,
        ref_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """One-step denoise: x_noisy, sigma -> x_denoised.

        Uses EDM parameterization:
          c_in   = 1 / sqrt(sigma_data^2 + sigma^2)
          c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
          c_out  = sigma * sigma_data / sqrt(sigma_data^2 + sigma^2)
          D(x, sigma) = c_skip * x + c_out * F(c_in * x, c_noise(sigma))

        Args:
            x_noisy: [..., N_atoms, 3] noisy coordinates
            sigma: [...] noise level (scalar or batch)
            s_inputs: [..., N_tokens, c_s_inputs]
            s_trunk: [..., N_tokens, c_s]
            z_trunk: [..., N_tokens, N_tokens, c_z]
            ref_pos: [..., N_atoms, 3] reference positions (defaults to zeros)
            ref_charge: [..., N_atoms] reference charges (defaults to zeros)
            ref_mask: [..., N_atoms] atom mask (defaults to ones)

        Returns:
            [..., N_atoms, 3] denoised coordinates
        """
        N_atoms = x_noisy.shape[-2]

        # Default reference features for protein-only mode
        if ref_pos is None:
            ref_pos = mx.zeros_like(x_noisy)
        if ref_charge is None:
            ref_charge = mx.zeros(x_noisy.shape[:-1])
        if ref_mask is None:
            ref_mask = mx.ones(x_noisy.shape[:-1])

        sd2 = self.sigma_data ** 2
        s2 = sigma ** 2

        # c_in scaling
        c_in = 1.0 / mx.sqrt(sd2 + s2)
        r_noisy = x_noisy * _expand_scalar(c_in, x_noisy.ndim - 1)

        # Forward pass
        r_update = self.f_forward(
            r_noisy, sigma, s_inputs, s_trunk, z_trunk,
            ref_pos, ref_charge, ref_mask,
        )

        # EDM recombination
        c_skip = sd2 / (sd2 + s2)
        c_out = sigma * self.sigma_data / mx.sqrt(sd2 + s2)

        x_denoised = (
            _expand_scalar(c_skip, x_noisy.ndim - 1) * x_noisy
            + _expand_scalar(c_out, x_noisy.ndim - 1) * r_update
        )
        return x_denoised


# ---------------------------------------------------------------------------
# FlowMatchingODESampler
# ---------------------------------------------------------------------------

class FlowMatchingODESampler:
    """5-step ODE sampler for flow-matching diffusion.

    Noise schedule: sigma from s_max (160.0) to s_min (4e-4)
    using the EDM power schedule:
        sigma(t) = sigma_data * (s_max^(1/p) + t*(s_min^(1/p) - s_max^(1/p)))^p

    Integration: Euler method with 5 steps.
    gamma_0 = 0 (no stochastic noise injection), eta = 1.0.

    This is where TeaCache hooks in: at each step, the TeaCache
    can decide to skip the expensive transformer forward pass.

    Args:
        diffusion_module: the DiffusionModule to evaluate.
        n_steps: number of ODE steps (default 5).
        sigma_data: data std dev (16.0).
        s_max: maximum noise level (160.0).
        s_min: minimum noise level (4e-4).
        p: exponent for noise schedule (7.0).
        gamma_0: stochastic noise injection (0.0 = deterministic ODE).
        eta: step size multiplier (1.0).
    """

    def __init__(
        self,
        diffusion_module: DiffusionModule,
        n_steps: int = 5,
        sigma_data: float = 16.0,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        p: float = 7.0,
        gamma_0: float = 0.0,
        eta: float = 1.0,
        tea_cache=None,
    ):
        self.model = diffusion_module
        self.n_steps = n_steps
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.p = p
        self.gamma_0 = gamma_0
        self.eta = eta
        self.tea_cache = tea_cache

    def _noise_schedule(self) -> list[float]:
        """Compute noise levels at each step boundary.

        Returns list of n_steps+1 sigma values from s_max to s_min.
        """
        sigmas = []
        for i in range(self.n_steps + 1):
            t = i / self.n_steps
            sigma = self.sigma_data * (
                self.s_max ** (1.0 / self.p)
                + t * (self.s_min ** (1.0 / self.p) - self.s_max ** (1.0 / self.p))
            ) ** self.p
            sigmas.append(sigma)
        return sigmas

    def sample(
        self,
        x_init: mx.array,
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: mx.array,
        ref_pos: Optional[mx.array] = None,
        ref_charge: Optional[mx.array] = None,
        ref_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Run the ODE sampler.

        Args:
            x_init: [..., N_atoms, 3] initial noisy coordinates
                    (sampled from N(0, s_max^2 * I))
            s_inputs: [..., N_tokens, c_s_inputs] input embeddings
            s_trunk: [..., N_tokens, c_s] trunk single embeddings
            z_trunk: [..., N_tokens, N_tokens, c_z] trunk pair embeddings
            ref_pos: [..., N_atoms, 3] reference positions (optional)
            ref_charge: [..., N_atoms] reference charges (optional)
            ref_mask: [..., N_atoms] atom mask (optional)

        Returns:
            [..., N_atoms, 3] denoised coordinates
        """
        sigmas = self._noise_schedule()
        x = x_init

        for step_idx in range(self.n_steps):
            sigma_cur = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]
            sigma_arr = mx.array(sigma_cur)

            # --- TeaCache hook ---
            skip = False
            if self.tea_cache is not None:
                skip = not self.tea_cache.should_compute(
                    x, sigma_arr, step_idx
                )

            if skip:
                # Reuse cached velocity
                velocity = self.tea_cache.get_cached()
            else:
                # Full model evaluation: D(x, sigma)
                x_denoised = self.model(
                    x, sigma_arr, s_inputs, s_trunk, z_trunk,
                    ref_pos, ref_charge, ref_mask,
                )
                # Flow-matching velocity: v = (x_denoised - x) / sigma
                velocity = (x_denoised - x) / max(sigma_cur, 1e-8)

                if self.tea_cache is not None:
                    self.tea_cache.cache_result(velocity)

            # Euler step: x_{t+1} = x_t + (sigma_next - sigma_cur) * velocity
            dt = sigma_next - sigma_cur
            x = x + dt * velocity

            # Force evaluation for memory management
            mx.eval(x)

        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_scalar(s: mx.array, n_trailing: int) -> mx.array:
    """Expand a scalar/batch array with n_trailing dimensions of size 1.

    E.g. s of shape [] with n_trailing=2 -> shape [1, 1]
         s of shape [B] with n_trailing=2 -> shape [B, 1, 1]
    """
    for _ in range(n_trailing):
        s = mx.expand_dims(s, axis=-1)
    return s
