"""
Triangle operations for Pairformer, ported to MLX from OpenFold3-MLX / Protenix.

Implements:
  - TriangleMultiplicationOutgoing  (AF3 Algorithm 11)
  - TriangleMultiplicationIncoming  (AF3 Algorithm 12)
  - TriangleAttentionStartingNode   (AF3 Algorithm 13)
  - TriangleAttentionEndingNode     (AF3 Algorithm 14)
"""

import math

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Triangle Multiplicative Update
# ---------------------------------------------------------------------------

class TriangleMultiplicativeUpdate(nn.Module):
    """Base triangle multiplicative update (AF3 Algorithms 11-12).

    For outgoing: contracts over the shared k-index  z_ij = sum_k a_ik * b_jk
    For incoming: contracts over the shared k-index  z_ij = sum_k a_ki * b_kj
    """

    def __init__(
        self,
        c_z: int = 128,
        c_hidden: int = 128,
        _outgoing: bool = True,
    ):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        # Input projections
        self.layer_norm_in = nn.LayerNorm(c_z)

        self.linear_a_p = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_a_g = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_b_p = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_b_g = nn.Linear(c_z, c_hidden, bias=False)

        # Output
        self.layer_norm_out = nn.LayerNorm(c_hidden)
        self.linear_z = nn.Linear(c_hidden, c_z, bias=False)
        self.linear_g = nn.Linear(c_z, c_z, bias=False)

    def __call__(
        self,
        z: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            z: [*, N, N, c_z] pair representation
            mask: [*, N, N] pair mask (unused at inference, kept for API compat)
        Returns:
            [*, N, N, c_z] update
        """
        z_in = self.layer_norm_in(z)

        # Gated projections: sigmoid(gate) * projection
        a = mx.sigmoid(self.linear_a_g(z_in)) * self.linear_a_p(z_in)
        b = mx.sigmoid(self.linear_b_g(z_in)) * self.linear_b_p(z_in)

        if mask is not None:
            mask_expanded = mx.expand_dims(mask, axis=-1)
            a = a * mask_expanded
            b = b * mask_expanded

        # Contract over the shared index k
        # outgoing: z_ij = sum_k a_ik * b_jk  ->  einsum("...ikd,...jkd->...ijd")
        # incoming: z_ij = sum_k a_ki * b_kj  ->  einsum("...kid,...kjd->...ijd")
        if self._outgoing:
            x = mx.einsum("...ikd,...jkd->...ijd", a, b)
        else:
            x = mx.einsum("...kid,...kjd->...ijd", a, b)

        # Output projection with gating
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = mx.sigmoid(self.linear_g(z_in))
        x = x * g

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """AF3 Algorithm 11: Triangle multiplication (outgoing edges)."""

    def __init__(self, c_z: int = 128, c_hidden: int = 128):
        super().__init__(c_z=c_z, c_hidden=c_hidden, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """AF3 Algorithm 12: Triangle multiplication (incoming edges)."""

    def __init__(self, c_z: int = 128, c_hidden: int = 128):
        super().__init__(c_z=c_z, c_hidden=c_hidden, _outgoing=False)


# ---------------------------------------------------------------------------
# Triangle Attention
# ---------------------------------------------------------------------------

class TriangleAttention(nn.Module):
    """Triangle attention (AF3 Algorithms 13-14).

    Performs self-attention along one axis of the pair representation,
    with an additive bias computed from the pair representation itself.

    For starting node (Algorithm 13): attention along j for each i
    For ending node (Algorithm 14): transpose, attend, transpose back
    """

    def __init__(
        self,
        c_in: int = 128,
        c_hidden: int = 32,
        n_head: int = 4,
        starting: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.n_head = n_head
        self.starting = starting
        self.inf = inf
        self.scale = 1.0 / math.sqrt(c_hidden)

        self.layer_norm = nn.LayerNorm(c_in)

        # Bias projection: pair -> per-head bias
        self.linear_z = nn.Linear(c_in, n_head, bias=False)

        # QKV + gating + output
        self.to_q = nn.Linear(c_in, n_head * c_hidden, bias=False)
        self.to_k = nn.Linear(c_in, n_head * c_hidden, bias=False)
        self.to_v = nn.Linear(c_in, n_head * c_hidden, bias=False)
        self.to_g = nn.Linear(c_in, n_head * c_hidden, bias=False)
        self.to_out = nn.Linear(n_head * c_hidden, c_in, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            x: [*, I, J, c_in] pair representation
            mask: [*, I, J] pair mask
        Returns:
            [*, I, J, c_in] output
        """
        if not self.starting:
            # Ending node: transpose spatial dims, attend, transpose back
            x = mx.swapaxes(x, -2, -3)
            if mask is not None:
                mask = mx.swapaxes(mask, -1, -2)

        # Shape: [*, I, J, c_in]
        x_norm = self.layer_norm(x)

        # Triangle bias: [*, I, J, n_head] -> [*, I, n_head, J] via permute
        # Then unsqueeze for broadcast: [*, I, n_head, 1, J]
        triangle_bias = self.linear_z(x_norm)
        # triangle_bias shape: [*, I, J, H]
        # We need it as [*, I, H, 1, J] for additive bias in attention
        # Permute last three dims: (..., I, J, H) -> (..., I, H, J)
        triangle_bias = mx.moveaxis(triangle_bias, -1, -2)
        # Now [*, I, H, J] -> [*, I, H, 1, J]
        triangle_bias = mx.expand_dims(triangle_bias, axis=-2)

        # Mask bias
        if mask is not None:
            # mask: [*, I, J] -> [*, I, 1, 1, J]
            mask_bias = self.inf * (mask.astype(mx.float32) - 1.0)
            mask_bias = mx.expand_dims(mx.expand_dims(mask_bias, axis=-2), axis=-2)
        else:
            mask_bias = None

        # Compute Q, K, V per row i
        # x_norm: [*, I, J, c_in]
        batch_shape = x_norm.shape[:-3]
        I, J, _ = x_norm.shape[-3:]

        q = self.to_q(x_norm).reshape(*batch_shape, I, J, self.n_head, self.c_hidden)
        k = self.to_k(x_norm).reshape(*batch_shape, I, J, self.n_head, self.c_hidden)
        v = self.to_v(x_norm).reshape(*batch_shape, I, J, self.n_head, self.c_hidden)
        g = mx.sigmoid(self.to_g(x_norm)).reshape(
            *batch_shape, I, J, self.n_head, self.c_hidden
        )

        # Reshape for attention: group by row i
        # q,k,v: [*, I, J, H, D] -> [*, I, H, J, D]
        q = mx.moveaxis(q, -2, -3)
        k = mx.moveaxis(k, -2, -3)
        v = mx.moveaxis(v, -2, -3)

        # Combine triangle_bias and mask_bias
        # triangle_bias: [*, I, H, 1, J]
        # mask_bias: [*, I, 1, 1, J]
        if mask_bias is not None:
            bias = triangle_bias + mask_bias
        else:
            bias = triangle_bias

        # mx.fast.scaled_dot_product_attention requires rank 4: (B, H, T, D)
        # q,k,v: [*, I, H, J, D] -> fold batch+I into one dim
        flat_B = 1
        for s in batch_shape:
            flat_B *= s
        flat_B *= I

        q_4d = q.reshape(flat_B, self.n_head, J, self.c_hidden)
        k_4d = k.reshape(flat_B, self.n_head, J, self.c_hidden)
        v_4d = v.reshape(flat_B, self.n_head, J, self.c_hidden)

        if bias is not None:
            # bias: [*, I, H, 1, J] -> fold batch+I -> (flat_B, H, 1, J)
            # SDPA will broadcast the 1 over the query dimension
            bias_4d = bias.reshape(flat_B, self.n_head, -1, J)
        else:
            bias_4d = None

        out = mx.fast.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, scale=self.scale, mask=bias_4d
        )

        # Unfold: (flat_B, H, J, D) -> [*, I, H, J, D] -> [*, I, J, H, D]
        out = out.reshape(*batch_shape, I, self.n_head, J, self.c_hidden)
        out = mx.moveaxis(out, -3, -2)
        out = out.reshape(*batch_shape, I, J, self.n_head * self.c_hidden)

        # Gating
        g = g.reshape(*batch_shape, I, J, self.n_head * self.c_hidden)
        out = g * out

        out = self.to_out(out)

        if not self.starting:
            out = mx.swapaxes(out, -2, -3)

        return out


class TriangleAttentionStartingNode(TriangleAttention):
    """AF3 Algorithm 13: Triangle attention (starting node)."""

    def __init__(self, c_in: int = 128, c_hidden: int = 32, n_head: int = 4):
        super().__init__(c_in=c_in, c_hidden=c_hidden, n_head=n_head, starting=True)


class TriangleAttentionEndingNode(TriangleAttention):
    """AF3 Algorithm 14: Triangle attention (ending node)."""

    def __init__(self, c_in: int = 128, c_hidden: int = 32, n_head: int = 4):
        super().__init__(c_in=c_in, c_hidden=c_hidden, n_head=n_head, starting=False)
