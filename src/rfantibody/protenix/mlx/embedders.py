"""
Input feature embedders for Protenix-Mini-Flow on MLX (Apple Silicon).

Implements:
  - InputFeatureEmbedder (AF3 Algorithm 2, simplified):
    Token-level embeddings from sequence one-hot + profile + deletion_mean.
    In Mini-Flow, the AtomAttentionEncoder is simplified to a linear
    projection from c_token to c_token (no full atom transformer).
  - RelativePositionEncoding (AF3 Algorithm 3):
    Relative position, token index, and chain identity features
    projected to pair embedding space.

Ported from: protenix/model/modules/embedders.py
"""

from __future__ import annotations

import math

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


# ---------------------------------------------------------------------------
# InputFeatureEmbedder  (Algorithm 2 simplified)
# ---------------------------------------------------------------------------

class InputFeatureEmbedder(nn.Module):
    """Token-level input embedding for Protenix-Mini-Flow.

    Simplified version of AF3 Algorithm 2 for Mini-Flow:
      - No full AtomAttentionEncoder (that requires atom-level features)
      - Instead: sequence one-hot (32 token types) concatenated with
        profile (32-dim) and deletion_mean (1-dim) = 65 total
      - For antibody design: we use a simple linear projection from
        the token feature dimension to c_token
      - Optional: add ESM embeddings when available

    The output is s_inputs of shape [..., N_token, c_s_inputs]
    where c_s_inputs = c_token + 32 + 32 + 1 = 449 (matching AF3 convention).

    For the simplified Mini-Flow, we embed sequence tokens directly:
      - restype one-hot: 32 dims
      - profile: 32 dims (MSA-derived, or zeros if unavailable)
      - deletion_mean: 1 dim
    Then optionally project through a learned layer.

    Args:
        c_token: token embedding dimension (default 384, for atom attention output).
        n_token_types: number of residue/token types (default 32).
        c_s_inputs: total input feature dim = c_token + 32 + 32 + 1 = 449.
        esm_dim: ESM embedding dimension if enabled (default 0 = disabled).
    """

    def __init__(
        self,
        c_token: int = 384,
        n_token_types: int = 32,
        c_s_inputs: int = 449,
        esm_dim: int = 0,
    ):
        super().__init__()
        self.c_token = c_token
        self.n_token_types = n_token_types
        self.c_s_inputs = c_s_inputs
        self.esm_dim = esm_dim

        # Project token one-hot (32) to c_token via a simple linear
        # This replaces the full AtomAttentionEncoder
        self.proj_token = LinearNoBias(n_token_types, c_token)

        # Optional ESM embedding projection
        if esm_dim > 0:
            self.linear_esm = LinearNoBias(esm_dim, c_s_inputs)
            # Zero-init for ESM projection (additive)
            self.linear_esm.weight = mx.zeros_like(self.linear_esm.weight)

    def __call__(
        self,
        restype: mx.array,
        profile: mx.array | None = None,
        deletion_mean: mx.array | None = None,
        esm_embeddings: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            restype: [..., N_token] integer token types (0..31)
                     or [..., N_token, 32] one-hot encoded
            profile: [..., N_token, 32] MSA profile features (optional)
            deletion_mean: [..., N_token, 1] deletion mean (optional)
            esm_embeddings: [..., N_token, esm_dim] ESM embeddings (optional)

        Returns:
            [..., N_token, c_s_inputs] token-level input embeddings
        """
        # One-hot encode if integer input
        if restype.ndim == 1 or (restype.ndim >= 2 and restype.shape[-1] != self.n_token_types):
            restype_oh = mx.zeros((*restype.shape, self.n_token_types))
            # Clamp to valid range
            restype_clamped = mx.clip(restype, 0, self.n_token_types - 1)
            restype_oh = mx.zeros((*restype.shape, self.n_token_types))
            indices = mx.expand_dims(restype_clamped, axis=-1)
            restype_oh = mx.zeros((*restype.shape, self.n_token_types))
            # Scatter one-hot
            for i in range(self.n_token_types):
                restype_oh = mx.where(
                    mx.expand_dims(restype_clamped == i, axis=-1),
                    mx.concatenate([
                        mx.zeros((*restype.shape, i)),
                        mx.ones((*restype.shape, 1)),
                        mx.zeros((*restype.shape, self.n_token_types - i - 1)),
                    ], axis=-1),
                    restype_oh,
                )
        else:
            restype_oh = restype

        # Project residue type to token embedding
        a = self.proj_token(restype_oh)  # [..., N_token, c_token]

        # Build profile features (defaults to zeros if not provided)
        batch_shape = restype_oh.shape[:-1]
        if profile is None:
            profile = mx.zeros((*batch_shape, 32))
        if deletion_mean is None:
            deletion_mean = mx.zeros((*batch_shape, 1))
        if deletion_mean.ndim == len(batch_shape):
            deletion_mean = mx.expand_dims(deletion_mean, axis=-1)

        # Concatenate: [atom_emb, restype_oh, profile, deletion_mean]
        # = [c_token, 32, 32, 1] = 449 total (c_s_inputs)
        s_inputs = mx.concatenate(
            [a, restype_oh, profile, deletion_mean],
            axis=-1,
        )

        # Add ESM embeddings if available
        if esm_embeddings is not None and self.esm_dim > 0:
            s_inputs = s_inputs + self.linear_esm(esm_embeddings)

        return s_inputs


# ---------------------------------------------------------------------------
# RelativePositionEncoding  (Algorithm 3)
# ---------------------------------------------------------------------------

class RelativePositionEncoding(nn.Module):
    """Relative position encoding for pair embeddings (AF3 Algorithm 3).

    Encodes relative sequence positions, token indices, and chain
    identity into pair features and projects to c_z.

    Features computed:
      - d_residue: one-hot relative residue index (clipped to +/- r_max),
        with out-of-chain positions mapped to a separate bin.
        Size: 2*(r_max + 1) = 66
      - d_token: one-hot relative token index (within same residue),
        same structure. Size: 2*(r_max + 1) = 66
      - b_same_entity: binary indicator (same entity). Size: 1
      - d_chain: one-hot relative chain/symmetry index (clipped to +/- s_max).
        Size: 2*(s_max + 1) = 6

    Total input features: 66 + 66 + 1 + 6 = 139
    Output: projected to c_z via LinearNoBias.

    Args:
        r_max: clip range for relative position (default 32).
        s_max: clip range for relative chain index (default 2).
        c_z: pair embedding dimension (default 128).
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        n_features = 4 * r_max + 2 * s_max + 7
        self.linear = LinearNoBias(n_features, c_z)

    def _one_hot(self, indices: mx.array, n_classes: int) -> mx.array:
        """Create one-hot encoding.

        Args:
            indices: integer array of class indices
            n_classes: number of classes

        Returns:
            one-hot encoded array with n_classes in last dimension
        """
        return (mx.expand_dims(indices, axis=-1) == mx.arange(n_classes)).astype(mx.float32)

    def encode(
        self,
        residue_index: mx.array,
        chain_id: mx.array,
        entity_id: mx.array | None = None,
        token_index: mx.array | None = None,
        sym_id: mx.array | None = None,
    ) -> mx.array:
        """Compute relative position features.

        Args:
            residue_index: [..., N_token] residue indices
            chain_id: [..., N_token] chain/asymmetric unit IDs
            entity_id: [..., N_token] entity IDs (default = chain_id)
            token_index: [..., N_token] token indices (default = residue_index)
            sym_id: [..., N_token] symmetry IDs (default = chain_id)

        Returns:
            [..., N_token, N_token, n_features] relative position features
        """
        if entity_id is None:
            entity_id = chain_id
        if token_index is None:
            token_index = residue_index
        if sym_id is None:
            sym_id = chain_id

        # Same-chain indicator: [..., N, N]
        b_same_chain = (
            mx.expand_dims(chain_id, axis=-1) == mx.expand_dims(chain_id, axis=-2)
        ).astype(mx.int32)

        # Same-residue indicator
        b_same_residue = (
            mx.expand_dims(residue_index, axis=-1) == mx.expand_dims(residue_index, axis=-2)
        ).astype(mx.int32)

        # Same-entity indicator
        b_same_entity = (
            mx.expand_dims(entity_id, axis=-1) == mx.expand_dims(entity_id, axis=-2)
        ).astype(mx.int32)

        # Relative residue position (clipped)
        d_res = mx.clip(
            mx.expand_dims(residue_index, axis=-1) - mx.expand_dims(residue_index, axis=-2) + self.r_max,
            0, 2 * self.r_max,
        )
        # Out-of-chain positions get special bin
        d_res = d_res * b_same_chain + (1 - b_same_chain) * (2 * self.r_max + 1)
        a_rel_pos = self._one_hot(d_res, 2 * (self.r_max + 1))

        # Relative token position (clipped, within same residue)
        d_tok = mx.clip(
            mx.expand_dims(token_index, axis=-1) - mx.expand_dims(token_index, axis=-2) + self.r_max,
            0, 2 * self.r_max,
        )
        d_tok = d_tok * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (2 * self.r_max + 1)
        a_rel_token = self._one_hot(d_tok, 2 * (self.r_max + 1))

        # Relative chain/symmetry index
        d_chain = mx.clip(
            mx.expand_dims(sym_id, axis=-1) - mx.expand_dims(sym_id, axis=-2) + self.s_max,
            0, 2 * self.s_max,
        )
        d_chain = d_chain * b_same_entity + (1 - b_same_entity) * (2 * self.s_max + 1)
        a_rel_chain = self._one_hot(d_chain, 2 * (self.s_max + 1))

        # Concatenate all features
        relp = mx.concatenate(
            [a_rel_pos, a_rel_token, mx.expand_dims(b_same_entity.astype(mx.float32), axis=-1), a_rel_chain],
            axis=-1,
        )
        return relp

    def __call__(self, relp_features: mx.array) -> mx.array:
        """Project pre-computed relative position features to c_z.

        Args:
            relp_features: [..., N_token, N_token, n_features] from encode()

        Returns:
            [..., N_token, N_token, c_z]
        """
        return self.linear(relp_features)
