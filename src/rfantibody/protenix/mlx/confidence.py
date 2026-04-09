"""
Confidence head for Protenix-Mini-Flow on MLX (Apple Silicon).

Implements:
  - ConfidenceHead: Algorithm 31 from AF3
    - pLDDT prediction (per-atom, 50 bins)
    - PAE prediction (per-pair, 64 bins)
    - PDE prediction (per-pair, 64 bins)
    - Resolution prediction (per-atom, 2 bins)
  - compute_iptm: interface pTM from PAE logits

Ported from:
  - Protenix (protenix/model/modules/confidence.py)
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from rfantibody.protenix.mlx.embedders import LinearNoBias
from rfantibody.protenix.mlx.pairformer import PairformerStack


# ---------------------------------------------------------------------------
# ConfidenceHead  (Algorithm 31)
# ---------------------------------------------------------------------------

class ConfidenceHead(nn.Module):
    """Confidence prediction head.

    Implements AF3 Algorithm 31:
      1. Embed predicted distances into pair representation
      2. Run PairformerStack (4 blocks) on (s_trunk, z_trunk + distance_embed)
      3. Predict:
         - pLDDT: per-atom local distance difference test (50 bins)
         - PAE: per-pair predicted aligned error (64 bins)
         - PDE: per-pair predicted distance error (64 bins)
         - resolution: per-atom resolved/unresolved (2 bins)
      4. Compute pTM/ipTM from PAE logits

    Uses the SAME PairformerStack as the trunk (same architecture, different
    weights) to match the Protenix checkpoint structure.

    Args:
        c_s: single embedding dim.
        c_z: pair embedding dim.
        c_s_inputs: input embedding dim.
        n_blocks: number of Pairformer blocks.
        b_plddt: number of pLDDT bins.
        b_pae: number of PAE bins.
        b_pde: number of PDE bins.
        b_resolved: number of resolution bins.
        n_head_pair: number of pair attention heads.
        n_head_single: number of single attention heads.
        distance_bin_start: start of distance binning range.
        distance_bin_end: end of distance binning range.
        distance_bin_step: step size for distance bins.
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        n_blocks: int = 4,
        b_plddt: int = 50,
        b_pae: int = 64,
        b_pde: int = 64,
        b_resolved: int = 2,
        n_head_pair: int = 4,
        n_head_single: int = 16,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.b_plddt = b_plddt
        self.b_pae = b_pae
        self.b_pde = b_pde
        self.b_resolved = b_resolved

        # Distance bins for embedding predicted coordinates
        lower = []
        val = distance_bin_start
        while val < distance_bin_end:
            lower.append(val)
            val += distance_bin_step
        self.n_dist_bins = len(lower)
        self.lower_bins = mx.array(lower)
        upper = lower[1:] + [1e6]
        self.upper_bins = mx.array(upper)

        # s_inputs -> pair init
        self.linear_s1 = LinearNoBias(c_s_inputs, c_z)
        self.linear_s2 = LinearNoBias(c_s_inputs, c_z)

        # Distance embedding -> pair
        self.linear_d = LinearNoBias(self.n_dist_bins, c_z)
        self.linear_d_raw = LinearNoBias(1, c_z)

        # Input normalization
        self.ln_s_trunk = nn.LayerNorm(c_s)

        # Pairformer stack -- SAME architecture as trunk
        # (uses the same PairformerBlock with tri_mul, tri_att, attention_pair_bias, transitions)
        self.pairformer_stack = PairformerStack(
            n_blocks=n_blocks,
            c_z=c_z,
            c_s=c_s,
            n_head_pair=n_head_pair,
            n_head_single=n_head_single,
            n_transition=4,  # matches checkpoint expansion factor
            eval_stride=n_blocks,  # no intermediate eval for small stack
        )

        # Output heads (zero-initialized)
        self.ln_pae = nn.LayerNorm(c_z)
        self.linear_pae = LinearNoBias(c_z, b_pae)

        self.ln_pde = nn.LayerNorm(c_z)
        self.linear_pde = LinearNoBias(c_z, b_pde)

        self.ln_plddt = nn.LayerNorm(c_s)
        self.linear_plddt = LinearNoBias(c_s, b_plddt)
        # Multi-head pLDDT weight: [n_atom_type, c_s, b_plddt] -- loaded from checkpoint
        self.plddt_weight = mx.zeros((24, c_s, b_plddt))

        self.ln_resolved = nn.LayerNorm(c_s)
        self.linear_resolved = LinearNoBias(c_s, b_resolved)
        # Multi-head resolved weight: [n_atom_type, c_s, b_resolved]
        self.resolved_weight = mx.zeros((24, c_s, b_resolved))

        # Zero-init output projections
        self.linear_pae.weight = mx.zeros_like(self.linear_pae.weight)
        self.linear_pde.weight = mx.zeros_like(self.linear_pde.weight)
        self.linear_plddt.weight = mx.zeros_like(self.linear_plddt.weight)
        self.linear_resolved.weight = mx.zeros_like(self.linear_resolved.weight)

    def _distance_one_hot(self, distances: mx.array) -> mx.array:
        """Bin distances into one-hot representation.

        Args:
            distances: [..., N, N] pairwise distances

        Returns:
            [..., N, N, n_dist_bins] one-hot encoded
        """
        d = mx.expand_dims(distances, axis=-1)
        lower = self.lower_bins
        upper = self.upper_bins
        # one_hot: 1 where lower <= d < upper
        mask = (d >= lower) & (d < upper)
        return mask.astype(mx.float32)

    def __call__(
        self,
        x_pred: mx.array,
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: mx.array,
    ) -> dict:
        """
        Args:
            x_pred: [..., N_atoms, 3] predicted coordinates
            s_inputs: [..., N_tokens, c_s_inputs] input embeddings
            s_trunk: [..., N_tokens, c_s] trunk single embeddings
            z_trunk: [..., N_tokens, N_tokens, c_z] trunk pair embeddings

        Returns:
            dict with keys:
                plddt_logits: [..., N_tokens, b_plddt]
                pae_logits:   [..., N_tokens, N_tokens, b_pae]
                pde_logits:   [..., N_tokens, N_tokens, b_pde]
                resolved_logits: [..., N_tokens, b_resolved]
                plddt: [..., N_tokens] predicted pLDDT scores (0-1)
                ptm:   scalar predicted TM-score
        """
        # Normalize trunk single
        s = self.ln_s_trunk(s_trunk)

        # Init pair from s_inputs outer product
        z_init = (
            mx.expand_dims(self.linear_s1(s_inputs), axis=-2)
            + mx.expand_dims(self.linear_s2(s_inputs), axis=-3)
        )
        z = z_trunk + z_init

        # Embed predicted distances
        # x_pred: [..., N, 3]
        # For simplicity: assuming 1:1 atom-to-token mapping
        diff = mx.expand_dims(x_pred, axis=-2) - mx.expand_dims(x_pred, axis=-3)
        distances = mx.sqrt(mx.sum(diff ** 2, axis=-1) + 1e-8)

        z = z + self.linear_d(self._distance_one_hot(distances))
        z = z + self.linear_d_raw(mx.expand_dims(distances, axis=-1))

        # Pairformer
        s, z = self.pairformer_stack(s, z)

        # --- Output heads ---
        # PAE: per-pair aligned error
        pae_logits = self.linear_pae(self.ln_pae(z))

        # PDE: per-pair distance error (symmetrized)
        z_sym = z + mx.transpose(z, axes=(*range(len(z.shape) - 3), -2, -3, -1))
        pde_logits = self.linear_pde(self.ln_pde(z_sym))

        # pLDDT: per-token confidence (multi-head: einsum with [n_atom_type, c_s, n_bins])
        s_plddt = self.ln_plddt(s)
        if hasattr(self, 'plddt_weight'):
            # Multi-head: plddt_weight is [n_atom_type, c_s, b_plddt]
            # For proteins, use atom_type=1 (CA) or average over types
            # Simple approach: use first atom type (backbone representative)
            plddt_logits = s_plddt @ self.plddt_weight[0]  # [*, N, b_plddt]
        else:
            plddt_logits = self.linear_plddt(s_plddt)

        # Resolution: per-token resolved/unresolved
        s_resolved = self.ln_resolved(s)
        if hasattr(self, 'resolved_weight'):
            resolved_logits = s_resolved @ self.resolved_weight[0]
        else:
            resolved_logits = self.linear_resolved(s_resolved)

        # Compute pLDDT scores from logits
        plddt_probs = mx.softmax(plddt_logits, axis=-1)
        # Bins centered at (0.5/50, 1.5/50, ..., 49.5/50)
        bin_centers = mx.arange(0, self.b_plddt).astype(mx.float32)
        bin_centers = (bin_centers + 0.5) / self.b_plddt
        plddt = mx.sum(plddt_probs * bin_centers, axis=-1)

        # Compute pTM from PAE
        ptm = compute_ptm(pae_logits)

        return {
            "plddt_logits": plddt_logits,
            "pae_logits": pae_logits,
            "pde_logits": pde_logits,
            "resolved_logits": resolved_logits,
            "plddt": plddt,
            "ptm": ptm,
        }


# ---------------------------------------------------------------------------
# pTM / ipTM computation
# ---------------------------------------------------------------------------

def compute_ptm(
    pae_logits: mx.array,
    max_bin: float = 31.75,
    n_bins: int = 64,
    asym_id: Optional[mx.array] = None,
) -> mx.array:
    """Compute predicted TM-score from PAE logits.

    The TM-score is computed as:
        pTM = max_i sum_j P(PAE_ij < d) * TM_weight(d, N)

    where TM_weight(d, N) = 1 / (1 + (d / d0)^2) with
    d0 = 1.24 * (N - 15)^(1/3) - 1.8.

    Args:
        pae_logits: [..., N, N, n_bins] PAE logits
        max_bin: maximum PAE bin edge.
        n_bins: number of PAE bins.
        asym_id: if provided, restrict to inter-chain pairs (for ipTM).

    Returns:
        scalar pTM score
    """
    N = pae_logits.shape[-2]

    # PAE bin centers
    bin_edges = mx.linspace(0, max_bin, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # TM-score weight per bin
    d0 = 1.24 * max((N - 15), 1) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    tm_weights = 1.0 / (1.0 + (bin_centers / d0) ** 2)

    # P(PAE) from logits
    pae_probs = mx.softmax(pae_logits, axis=-1)  # [..., N, N, n_bins]

    # Expected TM-score per pair
    # [..., N, N]
    pair_tm = mx.sum(pae_probs * tm_weights, axis=-1)

    if asym_id is not None:
        # Mask for inter-chain pairs
        inter_mask = mx.expand_dims(asym_id, -1) != mx.expand_dims(asym_id, -2)
        inter_mask = inter_mask.astype(mx.float32)
        # Weighted average over j for each alignment residue i
        pair_tm = pair_tm * inter_mask
        per_residue = mx.sum(pair_tm, axis=-1) / (
            mx.sum(inter_mask, axis=-1) + 1e-8
        )
    else:
        per_residue = mx.mean(pair_tm, axis=-1)

    # pTM = max over alignment residues
    ptm = mx.max(per_residue, axis=-1)
    return ptm


def compute_iptm(
    pae_logits: mx.array,
    chain_ids: mx.array,
    max_bin: float = 31.75,
    n_bins: int = 64,
) -> mx.array:
    """Compute interface pTM (ipTM) from PAE logits.

    Restricts the TM-score computation to inter-chain residue pairs,
    measuring confidence in the predicted interface geometry.

    Args:
        pae_logits: [..., N, N, n_bins] PAE logits
        chain_ids: [..., N] integer chain IDs per residue
        max_bin: maximum PAE bin edge.
        n_bins: number of PAE bins.

    Returns:
        scalar ipTM score
    """
    return compute_ptm(
        pae_logits, max_bin=max_bin, n_bins=n_bins, asym_id=chain_ids
    )
