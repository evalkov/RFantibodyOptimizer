"""
Diffusion module for Protenix-Mini-Flow on MLX (Apple Silicon).

Implements:
  - FourierEmbedding: noise level embedding (AF3 Algorithm 21, line 8)
  - DiffusionConditioning: combines noise + trunk conditioning (Algorithm 21)
  - DiffusionTransformerBlock: self-attention + transition (Algorithm 23)
  - DiffusionModule: full diffusion network (Algorithm 20)
  - FlowMatchingODESampler: 5-step ODE sampler with TeaCache hook

Ported from:
  - Protenix  (protenix/model/modules/diffusion.py)
  - OpenFold3 (openfold3/core/model/layers/diffusion_transformer.py)
  - OpenFold3 (openfold3/core/model/layers/diffusion_conditioning.py)
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
        # cos(2π(x·w + b)) — cosine-only, matching Protenix
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
# DiffusionModule  (Algorithm 20)
# ---------------------------------------------------------------------------

class DiffusionModule(nn.Module):
    """Complete diffusion module for structure prediction.

    Implements AF3 Algorithm 20 (simplified for Mini-Flow):
      1. AtomAttentionEncoder (simplified: Linear projections)
      2. Stack of DiffusionTransformerBlocks
      3. AtomAttentionDecoder (simplified: Linear projections)

    The EDM parameterization is used:
      - c_in  = 1 / sqrt(sigma_data^2 + sigma^2)
      - c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
      - c_out  = sigma * sigma_data / sqrt(sigma_data^2 + sigma^2)
      - D(x, sigma) = c_skip * x + c_out * F(c_in * x, c_noise(sigma))

    For flow matching, the output is interpreted as a velocity field.

    Args:
        sigma_data: data std dev constant.
        c_atom: atom feature dim.
        c_token: token activation dim.
        c_s: single embedding dim.
        c_z: pair embedding dim.
        c_s_inputs: input embedding dim.
        c_noise_emb: noise Fourier embedding dim.
        n_encoder_blocks: number of atom attention encoder blocks (simplified).
        n_transformer_blocks: number of diffusion transformer blocks.
        n_decoder_blocks: number of atom attention decoder blocks (simplified).
        n_head: number of transformer heads.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        c_noise_emb: int = 256,
        n_encoder_blocks: int = 3,
        n_transformer_blocks: int = 8,
        n_decoder_blocks: int = 3,
        n_head: int = 16,
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

        # --- AtomAttentionEncoder (simplified: linear projections) ---
        # In full AF3 this is a sequence-local atom transformer.
        # Here we use stacked linear layers as a placeholder.
        encoder_layers = []
        # coords (3) -> c_atom
        encoder_layers.append(LinearNoBias(3, c_atom))
        for _ in range(n_encoder_blocks - 1):
            encoder_layers.append(LinearNoBias(c_atom, c_atom))
        self.encoder_layers = encoder_layers
        self.encoder_ln = nn.LayerNorm(c_atom)
        # Aggregate atoms to tokens: c_atom -> c_token
        self.encoder_proj = LinearNoBias(c_atom, c_token)

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

        # --- AtomAttentionDecoder (simplified: linear projections) ---
        self.decoder_proj = LinearNoBias(c_token, c_atom)
        decoder_layers = []
        for _ in range(n_decoder_blocks - 1):
            decoder_layers.append(LinearNoBias(c_atom, c_atom))
        decoder_layers.append(LinearNoBias(c_atom, 3))
        self.decoder_layers = decoder_layers

    def _encode_atoms(self, r: mx.array) -> mx.array:
        """Simplified atom encoder: coords -> token features.

        In the full model, this would be AtomAttentionEncoder with
        sequence-local atom attention. Here we use linear projections.

        Args:
            r: [..., N_atoms, 3] coordinates (already scaled by c_in)

        Returns:
            [..., N_tokens, c_token] token-level activations
            (assuming 1:1 atom-to-token mapping for simplicity)
        """
        h = r
        for layer in self.encoder_layers:
            h = nn.silu(layer(h))
        h = self.encoder_ln(h)
        return self.encoder_proj(h)

    def _decode_atoms(self, a: mx.array) -> mx.array:
        """Simplified atom decoder: token features -> coord updates.

        Args:
            a: [..., N_tokens, c_token]

        Returns:
            [..., N_atoms, 3] coordinate updates
        """
        h = self.decoder_proj(a)
        for layer in self.decoder_layers[:-1]:
            h = nn.silu(layer(h))
        return self.decoder_layers[-1](h)

    def f_forward(
        self,
        r_noisy: mx.array,
        sigma: mx.array,
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: mx.array,
    ) -> mx.array:
        """Raw network F_theta(c_in * x, c_noise(sigma)).

        Args:
            r_noisy: [..., N_atoms, 3] scaled noisy coordinates
            sigma: [...] noise level
            s_inputs: [..., N_tokens, c_s_inputs]
            s_trunk: [..., N_tokens, c_s]
            z_trunk: [..., N_tokens, N_tokens, c_z]

        Returns:
            [..., N_atoms, 3] predicted coordinate update
        """
        # Conditioning
        s_cond, z_cond = self.conditioning(sigma, s_inputs, s_trunk, z_trunk)

        # Encode atoms to token level
        a = self._encode_atoms(r_noisy)

        # Add conditioned single embedding
        a = a + self.linear_s_to_token(self.ln_s(s_cond))

        # Diffusion transformer stack
        for block in self.transformer_blocks:
            a = block(a, s_cond, z_cond)

        a = self.ln_a(a)

        # Decode back to atom coordinates
        r_update = self._decode_atoms(a)
        return r_update

    def __call__(
        self,
        x_noisy: mx.array,
        sigma: mx.array,
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: mx.array,
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

        Returns:
            [..., N_atoms, 3] denoised coordinates
        """
        sd2 = self.sigma_data ** 2
        s2 = sigma ** 2

        # c_in scaling
        c_in = 1.0 / mx.sqrt(sd2 + s2)
        r_noisy = x_noisy * _expand_scalar(c_in, x_noisy.ndim - 1)

        # Forward pass
        r_update = self.f_forward(r_noisy, sigma, s_inputs, s_trunk, z_trunk)

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
    ) -> mx.array:
        """Run the ODE sampler.

        Args:
            x_init: [..., N_atoms, 3] initial noisy coordinates
                    (sampled from N(0, s_max^2 * I))
            s_inputs: [..., N_tokens, c_s_inputs] input embeddings
            s_trunk: [..., N_tokens, c_s] trunk single embeddings
            z_trunk: [..., N_tokens, N_tokens, c_z] trunk pair embeddings

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
                    x, sigma_arr, s_inputs, s_trunk, z_trunk
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
