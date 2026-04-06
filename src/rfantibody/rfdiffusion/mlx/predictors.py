"""
MLX prediction heads for RFdiffusion.

Simple linear prediction modules: DistanceNetwork, MaskedTokenNetwork,
LDDTNetwork, ExpResolvedNetwork.
"""

import mlx.core as mx
import mlx.nn as nn


class DistanceNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super().__init__()
        self.proj_symm = nn.Linear(n_feat, 37 * 2)
        self.proj_asymm = nn.Linear(n_feat, 37 + 19)

    def __call__(self, x):
        logits_asymm = self.proj_asymm(x)
        logits_theta = mx.transpose(logits_asymm[:, :, :, :37], (0, 3, 1, 2))
        logits_phi = mx.transpose(logits_asymm[:, :, :, 37:], (0, 3, 1, 2))

        logits_symm = self.proj_symm(x)
        logits_symm = logits_symm + mx.transpose(logits_symm, (0, 2, 1, 3))
        logits_dist = mx.transpose(logits_symm[:, :, :, :37], (0, 3, 1, 2))
        logits_omega = mx.transpose(logits_symm[:, :, :, 37:], (0, 3, 1, 2))

        return logits_dist, logits_omega, logits_theta, logits_phi


class MaskedTokenNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, 21)

    def __call__(self, x):
        B, N, L = x.shape[:3]
        logits = mx.transpose(self.proj(x), (0, 3, 1, 2)).reshape(B, -1, N * L)
        return logits


class LDDTNetwork(nn.Module):
    def __init__(self, n_feat, n_bin_lddt=50):
        super().__init__()
        self.proj = nn.Linear(n_feat, n_bin_lddt)

    def __call__(self, x):
        logits = self.proj(x)  # (B, L, 50)
        return mx.transpose(logits, (0, 2, 1))


class ExpResolvedNetwork(nn.Module):
    def __init__(self, d_msa, d_state, p_drop=0.1):
        super().__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj = nn.Linear(d_msa + d_state, 1)

    def __call__(self, seq, state):
        B, L = seq.shape[:2]
        seq = self.norm_msa(seq)
        state = self.norm_state(state)
        feat = mx.concatenate([seq, state], axis=-1)
        logits = self.proj(feat)
        return logits.reshape(B, L)
