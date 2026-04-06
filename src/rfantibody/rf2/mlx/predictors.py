"""
RF2 prediction head modules ported to MLX.

Port of src/rfantibody/rf2/network/AuxiliaryPredictor.py.
Classes: DistanceNetwork, MaskedTokenNetwork, LDDTNetwork,
PAENetwork, v2_BinderNetwork, ExpResolvedNetwork.
"""

import mlx.core as mx
import mlx.nn as nn


class DistanceNetwork(nn.Module):
    """Predicts inter-residue distance and orientation distributions.

    Outputs 4 distribution heads:
        - dist (37 bins, symmetric)
        - omega (37 bins, symmetric)
        - theta (37 bins, asymmetric)
        - phi (19 bins, asymmetric)
    """

    def __init__(self, d_pair: int = 128, p_drop: float = 0.0):
        super().__init__()
        self.proj_symm = nn.Linear(d_pair, 37 * 2)
        self.proj_asymm = nn.Linear(d_pair, 37 + 19)

    def __call__(self, pair: mx.array) -> tuple:
        B, L = pair.shape[:2]

        # Asymmetric predictions: theta (37) + phi (19)
        logits_asymm = self.proj_asymm(pair)
        logits_theta = mx.transpose(logits_asymm[:, :, :, :37], axes=(0, 3, 1, 2))
        logits_phi = mx.transpose(logits_asymm[:, :, :, 37:], axes=(0, 3, 1, 2))

        # Symmetric predictions: dist (37) + omega (37)
        logits_symm = self.proj_symm(pair)
        logits_symm = logits_symm + mx.transpose(logits_symm, axes=(0, 2, 1, 3))
        logits_dist = mx.transpose(logits_symm[:, :, :, :37], axes=(0, 3, 1, 2))
        logits_omega = mx.transpose(logits_symm[:, :, :, 37:], axes=(0, 3, 1, 2))

        return logits_dist, logits_omega, logits_theta, logits_phi


class MaskedTokenNetwork(nn.Module):
    """Predicts amino acid identity from MSA representation."""

    def __init__(self, d_msa: int = 256, p_drop: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_msa, 21)

    def __call__(self, msa: mx.array) -> mx.array:
        B, N, L = msa.shape[:3]
        logits = mx.transpose(self.proj(msa), axes=(0, 3, 1, 2))
        return logits.reshape(B, -1, N * L)


class LDDTNetwork(nn.Module):
    """Predicts per-residue lDDT confidence scores."""

    def __init__(self, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d_state)
        self.linear_1 = nn.Linear(d_state, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.proj = nn.Linear(128, 50)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.linear_1(self.norm(x)))
        x = nn.relu(self.linear_2(x))
        logits = self.proj(x)  # (B, L, 50)
        return mx.transpose(logits, axes=(0, 2, 1))


class PAENetwork(nn.Module):
    """Predicts pairwise alignment error."""

    def __init__(self, d_pair: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_pair, 64)

    def __call__(self, pair: mx.array) -> mx.array:
        logits = self.proj(pair)
        return mx.transpose(logits, axes=(0, 3, 1, 2))


class v2_BinderNetwork(nn.Module):
    """Predicts binding probability from pair, RBF, and state features.

    Architecture:
        1. rbf2attn: softmax attention weights from RBF features
        2. downsample: Linear(pair + left_state + right_state -> 1)
        3. Weighted sum of inter-chain logits -> sigmoid
    """

    def __init__(self, d_pair: int = 128, d_state: int = 32, d_rbf: int = 64):
        super().__init__()
        self.rbf2attn = nn.Linear(d_rbf, 1)
        self.downsample = nn.Linear(d_pair + 2 * d_state, 1)

    def __call__(self, pair: mx.array, rbf_feat: mx.array,
                 state: mx.array, same_chain: mx.array) -> mx.array:
        B, L = pair.shape[:2]

        # Attention from RBF
        attn = self.rbf2attn(rbf_feat)  # (B, L, L, 1)

        # Combine pair + state outer product
        left = mx.broadcast_to(
            mx.expand_dims(state, 2), (B, L, L, state.shape[-1]))
        right = mx.broadcast_to(
            mx.expand_dims(state, 1), (B, L, L, state.shape[-1]))
        logits = self.downsample(
            mx.concatenate([pair, left, right], axis=-1))  # (B, L, L, 1)

        # Select inter-chain pairs via masking (MLX lacks boolean indexing)
        inter_mask = mx.expand_dims(
            mx.expand_dims((same_chain == 0), 0), -1)  # (1, L, L, 1)
        n_inter = mx.sum(inter_mask)

        if n_inter == 0:
            # Single chain — attend over all pairs
            attn_weights = mx.softmax(attn.reshape(-1), axis=0)
            logits_inter = mx.sum(logits.reshape(-1) * attn_weights, axis=0)
        else:
            # Mask intra-chain attn to -inf so softmax gives them 0 weight
            attn_masked = mx.where(inter_mask, attn, mx.array(float('-inf')))
            attn_weights = mx.softmax(attn_masked.reshape(-1), axis=0)
            logits_inter = mx.sum(logits.reshape(-1) * attn_weights, axis=0)

        prob = mx.sigmoid(logits_inter)
        return prob


class ExpResolvedNetwork(nn.Module):
    """Predicts per-residue experimental resolution."""

    def __init__(self, d_msa: int = 256, d_state: int = 16):
        super().__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj = nn.Linear(d_msa + d_state, 1)

    def __call__(self, msa: mx.array, state: mx.array) -> mx.array:
        B, L = msa.shape[:2]
        msa = self.norm_msa(msa)
        state = self.norm_state(state)
        feat = mx.concatenate([msa, state], axis=-1)
        logits = self.proj(feat)
        return logits.reshape(B, L)
