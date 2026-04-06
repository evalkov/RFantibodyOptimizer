"""
RF2 three-track architecture ported to MLX.

Port of src/rfantibody/rf2/network/Track_module.py.
Omits stride/striping, symmetry, dropout (inference only).
Uses mx.eval() between blocks to bound memory.
"""

import mlx.core as mx
import mlx.nn as nn

from rfantibody.rf2.mlx.attention import (
    BiasedAxialAttention,
    FeedForwardLayer,
    MSAColAttention,
    MSAColGlobalAttention,
    MSARowAttentionWithBias,
    TriangleMultiplication,
)
from rfantibody.rfdiffusion.mlx.se3.wrapper import SE3TransformerWrapper
from rfantibody.rfdiffusion.mlx.util_module import make_topk_graph


# ---------------------------------------------------------------------------
# RF2-specific RBF (64 bins, sigma=0.5) -- differs from RFdiffusion (36 bins)
# ---------------------------------------------------------------------------

def rbf(D: mx.array, D_min: float = 0.0, D_count: int = 64,
        D_sigma: float = 0.5) -> mx.array:
    """Distance radial basis function (RF2 parameterisation)."""
    D_max = D_min + (D_count - 1) * D_sigma
    D_mu = mx.linspace(D_min, D_max, D_count).reshape(1, -1)
    D_expand = mx.expand_dims(D, -1)
    return mx.exp(-((D_expand - D_mu) / D_sigma) ** 2)


def cdist(a: mx.array, b: mx.array) -> mx.array:
    """Pairwise Euclidean distance: a (B,M,D), b (B,N,D) -> (B,M,N)."""
    a_sq = mx.sum(a * a, axis=-1, keepdims=True)
    b_sq = mx.sum(b * b, axis=-1, keepdims=True)
    ab = a @ mx.transpose(b, (0, 2, 1))
    dist_sq = a_sq - 2 * ab + mx.transpose(b_sq, (0, 2, 1))
    return mx.sqrt(mx.maximum(dist_sq, 1e-12))


def get_seqsep(idx: mx.array) -> mx.array:
    """Sequence separation feature with sign. (B,L) -> (B,L,L,1)."""
    seqsep = mx.expand_dims(idx, 1) - mx.expand_dims(idx, 2)
    sign = mx.sign(seqsep)
    neigh = mx.abs(seqsep).astype(mx.float32)
    neigh = mx.where(neigh > 1, 0.0, neigh)
    neigh = sign * neigh
    return mx.expand_dims(neigh, -1)


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def normQ(Q: mx.array) -> mx.array:
    """Normalize quaternion to unit length."""
    return Q / mx.sqrt(mx.sum(Q * Q, axis=-1, keepdims=True) + 1e-8)


def Qs2Rs(Q: mx.array) -> mx.array:
    """Convert unit quaternion (w,x,y,z) to 3x3 rotation matrix."""
    q0 = Q[..., 0]
    q1 = Q[..., 1]
    q2 = Q[..., 2]
    q3 = Q[..., 3]

    R00 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    R01 = 2 * q1 * q2 - 2 * q0 * q3
    R02 = 2 * q1 * q3 + 2 * q0 * q2
    R10 = 2 * q1 * q2 + 2 * q0 * q3
    R11 = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    R12 = 2 * q2 * q3 - 2 * q0 * q1
    R20 = 2 * q1 * q3 - 2 * q0 * q2
    R21 = 2 * q2 * q3 + 2 * q0 * q1
    R22 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    row0 = mx.stack([R00, R01, R02], axis=-1)
    row1 = mx.stack([R10, R11, R12], axis=-1)
    row2 = mx.stack([R20, R21, R22], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


# ---------------------------------------------------------------------------
# SeqSep -- positional encoding via nn.Embedding + bucketize
# ---------------------------------------------------------------------------

class SeqSep(nn.Module):
    """Relative positional encoding for pair features."""

    def __init__(self, d_model: int, minpos: int = -32, maxpos: int = 32):
        super().__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos) + maxpos + 1
        self.emb = nn.Embedding(self.nbin, d_model)

    def __call__(self, idx: mx.array, idx2: mx.array = None) -> mx.array:
        """
        Args:
            idx:  (B, L1) residue indices
            idx2: (B, L2) residue indices (defaults to idx)
        Returns:
            (1, L1, L2, d_model) positional embeddings
        """
        if idx2 is None:
            idx2 = idx

        # seqsep: (B, L1, L2)
        seqsep = mx.expand_dims(idx2, 1) - mx.expand_dims(idx, 2)

        # Bucketize: clip to [minpos, maxpos-1] then shift to [0, nbin-1]
        ib = mx.clip(seqsep, self.minpos, self.maxpos - 1) - self.minpos
        ib = ib.astype(mx.int32)

        # Embedding lookup -- take first batch element (no oligo)
        emb = self.emb(ib[0:1])  # (1, L1, L2, d_model)
        return emb


# ---------------------------------------------------------------------------
# MSAPairStr2MSA -- updates MSA with pair + structure info
# ---------------------------------------------------------------------------

class MSAPairStr2MSA(nn.Module):
    """Update MSA features with pair and structure information."""

    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16,
                 d_rbf=64, d_hidden=32, p_drop=0.0, use_global_attn=False):
        super().__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.emb_rbf = nn.Linear(d_rbf, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.row_attn = MSARowAttentionWithBias(
            d_msa=d_msa, d_pair=d_pair, n_head=n_head, d_hidden=d_hidden)
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(
                d_msa=d_msa, n_head=n_head, d_hidden=d_hidden)
        else:
            self.col_attn = MSAColAttention(
                d_msa=d_msa, n_head=n_head, d_hidden=d_hidden)
        self.ff = FeedForwardLayer(d_msa, 4)

    def __call__(self, msa: mx.array, pair: mx.array,
                 rbf_feat: mx.array, state: mx.array) -> mx.array:
        """
        Args:
            msa:      (B, N, L, d_msa)
            pair:     (B, L, L, d_pair)
            rbf_feat: (B, L, L, d_rbf)
            state:    (B, L, d_state)
        Returns:
            msa: updated (B, N, L, d_msa)
        """
        B, N, L, _ = msa.shape

        # Combine pair and RBF coordinate info
        pair_rbf = self.norm_pair(pair) + self.emb_rbf(rbf_feat)

        # Update query sequence (first row of MSA) with structure state feedback
        state_proj = self.proj_state(self.norm_state(state)).reshape(B, 1, L, -1)
        msa = msa.at[:, 0:1].add(state_proj)

        # Attention updates
        msa = msa + self.row_attn(msa, pair_rbf)
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)
        return msa


# ---------------------------------------------------------------------------
# PairStr2Pair -- updates pair features
# ---------------------------------------------------------------------------

class PairStr2Pair(nn.Module):
    """Update pair features with structure information."""

    def __init__(self, d_pair=128, n_head=4, d_hidden=32,
                 d_hidden_state=16, d_rbf=64, d_state=32, p_drop=0.0):
        super().__init__()
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_left = nn.Linear(d_state, d_hidden_state)
        self.proj_right = nn.Linear(d_state, d_hidden_state)
        self.to_gate = nn.Linear(d_hidden_state * d_hidden_state, d_pair)
        self.emb_rbf = nn.Linear(d_rbf, d_pair)

        self.tri_mul_out = TriangleMultiplication(d_pair, d_hidden=d_hidden, outgoing=True)
        self.tri_mul_in = TriangleMultiplication(d_pair, d_hidden=d_hidden, outgoing=False)
        self.row_attn = BiasedAxialAttention(
            d_pair, d_pair, n_head, d_hidden, is_row=True)
        self.col_attn = BiasedAxialAttention(
            d_pair, d_pair, n_head, d_hidden, is_row=False)
        self.ff = FeedForwardLayer(d_pair, 2)

    def __call__(self, pair: mx.array, rbf_feat: mx.array,
                 state: mx.array) -> mx.array:
        """
        Args:
            pair:     (B, L, L, d_pair)
            rbf_feat: (B, L, L, d_rbf)
            state:    (B, L, d_state)
        Returns:
            pair: updated (B, L, L, d_pair)
        """
        B, L = pair.shape[:2]

        # Gate RBF features with state outer product
        state_normed = self.norm_state(state)
        left = self.proj_left(state_normed)   # (B, L, d_hs)
        right = self.proj_right(state_normed)  # (B, L, d_hs)

        # Outer product: (B, L, d_hs) x (B, L, d_hs) -> (B, L, L, d_hs*d_hs)
        gate = mx.einsum('bli,bmj->blmij', left, right)
        gate = gate.reshape(B, L, L, -1)
        gate = mx.sigmoid(self.to_gate(gate))

        rbf_feat = gate * self.emb_rbf(rbf_feat)

        # Triangle multiplication and attention updates
        pair = pair + self.tri_mul_out(pair)
        pair = pair + self.tri_mul_in(pair)
        pair = pair + self.row_attn(pair, rbf_feat)
        pair = pair + self.col_attn(pair, rbf_feat)
        pair = pair + self.ff(pair)
        return pair


# ---------------------------------------------------------------------------
# MSA2Pair -- outer product from MSA to pair
# ---------------------------------------------------------------------------

class MSA2Pair(nn.Module):
    """Extract coevolution signal from MSA to update pair features."""

    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden * d_hidden, d_pair)

    def __call__(self, msa: mx.array, pair: mx.array) -> mx.array:
        """
        Args:
            msa:  (B, N, L, d_msa)
            pair: (B, L, L, d_pair)
        Returns:
            pair: updated (B, L, L, d_pair)
        """
        B, N, L, _ = msa.shape

        msa_normed = self.norm(msa)
        left = self.proj_left(msa_normed)     # (B, N, L, d_h)
        right = self.proj_right(msa_normed)   # (B, N, L, d_h)
        right = right / float(N)

        # Outer product: (B, N, L, d_h) x (B, N, L, d_h) -> (B, L, L, d_h*d_h)
        out = mx.einsum('bsli,bsmj->blmij', left, right)
        out = out.reshape(B, L, L, -1)
        out = self.proj_out(out)

        return pair + out


# ---------------------------------------------------------------------------
# SCPred -- side-chain torsion predictor
# ---------------------------------------------------------------------------

class SCPred(nn.Module):
    """Predict side-chain torsion angles."""

    def __init__(self, d_msa=256, d_state=32, d_hidden=128, p_drop=0.0):
        super().__init__()
        self.norm_s0 = nn.LayerNorm(d_msa)
        self.norm_si = nn.LayerNorm(d_state)
        self.linear_s0 = nn.Linear(d_msa, d_hidden)
        self.linear_si = nn.Linear(d_state, d_hidden)

        # ResNet layers
        self.linear_1 = nn.Linear(d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.linear_3 = nn.Linear(d_hidden, d_hidden)
        self.linear_4 = nn.Linear(d_hidden, d_hidden)

        # Final output
        self.linear_out = nn.Linear(d_hidden, 20)

    def __call__(self, seq: mx.array, state: mx.array) -> mx.array:
        """
        Args:
            seq:   (B, L, d_msa) -- query sequence hidden embeddings
            state: (B, L, d_state) -- state feature from SE3 layer
        Returns:
            si: (B, L, 10, 2) predicted torsion angles
        """
        B, L = seq.shape[:2]
        seq = self.norm_s0(seq)
        state = self.norm_si(state)
        si = self.linear_s0(seq) + self.linear_si(state)

        si = si + self.linear_2(nn.relu(self.linear_1(nn.relu(si))))
        si = si + self.linear_4(nn.relu(self.linear_3(nn.relu(si))))

        si = self.linear_out(nn.relu(si))
        return si.reshape(B, L, 10, 2)


# ---------------------------------------------------------------------------
# Str2Str -- structure-to-structure update via SE3
# ---------------------------------------------------------------------------

class Str2Str(nn.Module):
    """Structure update using SE(3)-Transformer."""

    def __init__(self, d_msa=256, d_pair=128, d_state=16, d_rbf=64,
                 SE3_param=None, p_drop=0.0):
        super().__init__()
        if SE3_param is None:
            SE3_param = {
                'l0_in_features': 32,
                'l0_out_features': 16,
                'num_edge_features': 32,
            }

        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)

        self.n_node = SE3_param['l0_in_features']
        self.n_edge = SE3_param['num_edge_features']

        self.embed_node1 = nn.Linear(d_msa, SE3_param['l0_in_features'])
        self.embed_node2 = nn.Linear(d_state, SE3_param['l0_in_features'])
        self.ff_node = FeedForwardLayer(SE3_param['l0_in_features'], 2)
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])

        self.embed_edge1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_edge2 = nn.Linear(d_rbf + 1, SE3_param['num_edge_features'])
        self.ff_edge = FeedForwardLayer(SE3_param['num_edge_features'], 2)
        self.norm_edge = nn.LayerNorm(SE3_param['num_edge_features'])

        self.se3 = SE3TransformerWrapper(
            **SE3_param, final_layer='lin', populate_edge='arcsin')
        self.sc_predictor = SCPred(
            d_msa=d_msa, d_state=SE3_param['l0_out_features'])

    def __call__(self, msa: mx.array, pair: mx.array,
                 R_in: mx.array, T_in: mx.array,
                 xyz: mx.array, state: mx.array,
                 idx_in: mx.array, top_k: int = 64) -> tuple:
        """
        Args:
            msa:    (B, N, L, d_msa)
            pair:   (B, L, L, d_pair)
            R_in:   (B, L, 3, 3)
            T_in:   (B, L, 3)
            xyz:    (B, L, 3, 3)  -- N/CA/C coordinates
            state:  (B, L, d_state)
            idx_in: (B, L) residue indices
            top_k:  number of graph neighbors
        Returns:
            (R, T, state, alpha)
        """
        B, N, L = msa.shape[:3]

        # Node features: embed query sequence + state
        seq = self.norm_msa(msa[:, 0])
        state = self.norm_state(state)
        node = self.embed_node1(seq) + self.embed_node2(state)
        node = node + self.ff_node(node)
        node = self.norm_node(node)
        node = node.reshape(B * L, -1, 1)

        # Edge features: pair + RBF(Ca-Ca distance) + seqsep
        seqsep = get_seqsep(idx_in)  # (B, L, L, 1)
        pair_normed = self.norm_pair(pair)
        rbf_feat = rbf(cdist(xyz[:, :, 1], xyz[:, :, 1])).reshape(B, L, L, -1)
        rbf_feat = mx.concatenate([rbf_feat, seqsep], axis=-1)  # (B, L, L, d_rbf+1)
        edge = self.embed_edge1(pair_normed) + self.embed_edge2(rbf_feat)
        edge = edge + self.ff_edge(edge)
        edge = self.norm_edge(edge)

        # Build top-k graph
        G, edge_feats = make_topk_graph(
            mx.stop_gradient(xyz[:, :, 1, :]), edge, idx_in, top_k=top_k)

        # L1 features: CA-N and CA-C vectors
        l1_feats = mx.stack([xyz[:, :, 0, :], xyz[:, :, 2, :]], axis=-2)
        l1_feats = l1_feats - mx.expand_dims(xyz[:, :, 1, :], 2)
        l1_feats = l1_feats.reshape(B * L, -1, 3).astype(mx.float32)

        # Apply SE(3)-Transformer
        shift = self.se3(G, node, l1_feats, edge_feats)

        # Update state with l0 output
        state = state + shift['0'].reshape(B, L, -1)

        # Extract rotation and translation from l1 output
        offset = shift['1'].reshape(B, L, 2, 3)
        Ts = offset[:, :, 0, :] * 10.0  # translation
        Qs = offset[:, :, 1, :]          # rotation (as quaternion xyz)

        # Convert to full quaternion (w, x, y, z) with w=1 then normalize
        ones = mx.ones((B, L, 1))
        Qs = mx.concatenate([ones, Qs], axis=-1)
        Qs = normQ(Qs)
        Rs = Qs2Rs(Qs)

        # Side-chain torsion prediction
        seqfull = msa[:, 0]
        alpha = self.sc_predictor(seqfull, state)

        # Compose rotation and translation with input
        Rs = mx.einsum('bnij,bnjk->bnik', Rs, R_in)
        Ts = Ts + T_in

        return Rs, Ts, state, alpha


# ---------------------------------------------------------------------------
# IterBlock -- wraps all four track modules + SeqSep
# ---------------------------------------------------------------------------

class IterBlock(nn.Module):
    """One iteration block combining MSA, pair, and structure tracks."""

    def __init__(self, d_msa=256, d_pair=128, d_rbf=64,
                 n_head_msa=8, n_head_pair=4,
                 use_global_attn=False,
                 d_hidden=32, d_hidden_msa=None, p_drop=0.0,
                 SE3_param=None):
        super().__init__()
        if SE3_param is None:
            SE3_param = {
                'l0_in_features': 32,
                'l0_out_features': 16,
                'num_edge_features': 32,
            }
        if d_hidden_msa is None:
            d_hidden_msa = d_hidden

        self.d_rbf = d_rbf

        self.pos = SeqSep(d_rbf)
        self.msa2msa = MSAPairStr2MSA(
            d_msa=d_msa, d_pair=d_pair,
            n_head=n_head_msa,
            d_state=SE3_param['l0_out_features'],
            use_global_attn=use_global_attn,
            d_hidden=d_hidden_msa)
        self.msa2pair = MSA2Pair(
            d_msa=d_msa, d_pair=d_pair,
            d_hidden=d_hidden // 2)
        self.pair2pair = PairStr2Pair(
            d_pair=d_pair, n_head=n_head_pair,
            d_state=SE3_param['l0_out_features'],
            d_hidden=d_hidden)
        self.str2str = Str2Str(
            d_msa=d_msa, d_pair=d_pair,
            d_state=SE3_param['l0_out_features'],
            SE3_param=SE3_param)

    def __call__(self, msa: mx.array, pair: mx.array,
                 R_in: mx.array, T_in: mx.array,
                 xyz: mx.array, state: mx.array,
                 idx: mx.array, topk: int = 0,
                 skip_se3: bool = False) -> tuple:
        """
        Args:
            msa:   (B, N, L, d_msa)
            pair:  (B, L, L, d_pair)
            R_in:  (B, L, 3, 3)
            T_in:  (B, L, 3)
            xyz:   (B, L, 3, 3)  -- N/CA/C
            state: (B, L, d_state)
            idx:   (B, L)
            topk:  k for SE3 graph (0 = use default 64)
            skip_se3: skip structural update (for SE3 stride)
        Returns:
            (msa, pair, R, T, state, alpha)
        """
        B, L = pair.shape[:2]

        # Compute RBF features + positional encoding
        rbf_feat = rbf(cdist(xyz[:, :, 1], xyz[:, :, 1])).reshape(B, L, L, -1)
        rbf_feat = rbf_feat + self.pos(idx)  # (B, L, L, d_rbf)

        # Track 1: MSA update
        msa = self.msa2msa(msa, pair, rbf_feat, state)

        # Track 3 (MSA -> Pair): outer product update
        pair = self.msa2pair(msa, pair)

        # Track 2: Pair update
        pair = self.pair2pair(pair, rbf_feat, state)

        # Track 4: Structure update
        if skip_se3:
            return msa, pair, R_in, T_in, state, mx.zeros((B, L, 10, 2))

        eff_topk = topk if topk > 0 else 64
        R, T, state, alpha = self.str2str(
            msa, pair, R_in, T_in, xyz, state, idx, top_k=eff_topk)

        return msa, pair, R, T, state, alpha


# ---------------------------------------------------------------------------
# IterativeSimulator -- manages extra / main / ref block sequences
# ---------------------------------------------------------------------------

class IterativeSimulator(nn.Module):
    """Full iterative simulator with extra, main, and refinement blocks."""

    def __init__(self, n_extra_block=4, n_main_block=36, n_ref_block=4,
                 d_msa=256, d_msa_full=64, d_pair=128, d_hidden=32,
                 n_head_msa=8, n_head_pair=4,
                 SE3_param_full=None, SE3_param_topk=None,
                 p_drop=0.0):
        super().__init__()
        if SE3_param_full is None:
            SE3_param_full = {
                'l0_in_features': 32,
                'l0_out_features': 16,
                'num_edge_features': 32,
            }
        if SE3_param_topk is None:
            SE3_param_topk = {
                'l0_in_features': 32,
                'l0_out_features': 16,
                'num_edge_features': 32,
            }

        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        self.n_ref_block = n_ref_block

        # Project state between topk and full SE3 feature dimensions
        self.proj_state = nn.Linear(
            SE3_param_topk['l0_out_features'],
            SE3_param_full['l0_out_features'])

        # Extra blocks (global attention, smaller MSA dim)
        if n_extra_block > 0:
            self.extra_block = [
                IterBlock(
                    d_msa=d_msa_full, d_pair=d_pair,
                    n_head_msa=n_head_msa, n_head_pair=n_head_pair,
                    d_hidden_msa=8, d_hidden=d_hidden,
                    use_global_attn=True,
                    SE3_param=SE3_param_full)
                for _ in range(n_extra_block)
            ]

        # Main blocks
        if n_main_block > 0:
            self.main_block = [
                IterBlock(
                    d_msa=d_msa, d_pair=d_pair,
                    n_head_msa=n_head_msa, n_head_pair=n_head_pair,
                    d_hidden=d_hidden,
                    use_global_attn=False,
                    SE3_param=SE3_param_full)
                for _ in range(n_main_block)
            ]

        # Project state back for refinement
        self.proj_state2 = nn.Linear(
            SE3_param_full['l0_out_features'],
            SE3_param_topk['l0_out_features'])

        # Refinement blocks (Str2Str only, topk SE3 params)
        if n_ref_block > 0:
            self.str_refiner = Str2Str(
                d_msa=d_msa, d_pair=d_pair,
                d_state=SE3_param_topk['l0_out_features'],
                SE3_param=SE3_param_topk)

    def __call__(self, seq: mx.array, msa: mx.array, msa_full: mx.array,
                 pair: mx.array, xyz_in: mx.array, state: mx.array,
                 idx: mx.array, topk_crop: int = -1) -> tuple:
        """
        Args:
            seq:      (B, L) query sequence
            msa:      (B, N, L, d_msa) seed MSA embeddings
            msa_full: (B, N, L, d_msa_full) extra MSA embeddings
            pair:     (B, L, L, d_pair) initial pair features
            xyz_in:   (B, L, 3, 3) initial BB coordinates (N/CA/C)
            state:    (B, L, d_state) initial state features
            idx:      (B, L) residue indices
            topk_crop: k for SE3 graph (-1 = default)
        Returns:
            (msa, pair, R_s, T_s, alpha_s, state)
        """
        B, _, L = msa.shape[:3]

        # Apply top_k override if set
        if topk_crop <= 0 and hasattr(self, 'top_k_override') and self.top_k_override > 0:
            topk_crop = self.top_k_override

        # Initialize rotation as identity, translation as CA position
        R_in = mx.broadcast_to(
            mx.eye(3).reshape(1, 1, 3, 3),
            (B, L, 3, 3))
        T_in = xyz_in[:, :, 1].astype(mx.float32)  # CA coords
        xyz_in = xyz_in - mx.expand_dims(T_in, -2)

        # Project state to full SE3 dimension
        state = self.proj_state(state)

        R_s = []
        T_s = []
        alpha_s = []

        eval_stride = getattr(self, 'eval_stride', 1)

        # Extra blocks
        for i_m in range(self.n_extra_block):
            R_in = mx.stop_gradient(R_in)
            T_in = mx.stop_gradient(T_in)

            # Reconstruct current BB from rigid transform
            xyz = mx.einsum('bnij,bnaj->bnai', R_in, xyz_in) + mx.expand_dims(T_in, -2)

            msa_full, pair, R_in, T_in, state, alpha = self.extra_block[i_m](
                msa_full, pair, R_in, T_in, xyz, state, idx,
                topk=topk_crop)

            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

            if (i_m + 1) % eval_stride == 0 or i_m == self.n_extra_block - 1:
                mx.eval(msa_full, pair, R_in, T_in, state, alpha)

        # Main blocks
        se3_stride = getattr(self, 'se3_stride', 1)
        for i_m in range(self.n_main_block):
            R_in = mx.stop_gradient(R_in)
            T_in = mx.stop_gradient(T_in)

            xyz = mx.einsum('bnij,bnaj->bnai', R_in, xyz_in) + mx.expand_dims(T_in, -2)

            skip = (se3_stride > 1 and i_m % se3_stride != 0)
            msa, pair, R_in, T_in, state, alpha = self.main_block[i_m](
                msa, pair, R_in, T_in, xyz, state, idx,
                topk=topk_crop, skip_se3=skip)

            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

            if (i_m + 1) % eval_stride == 0 or i_m == self.n_main_block - 1:
                mx.eval(msa, pair, R_in, T_in, state, alpha)

        # Refinement blocks (Str2Str only)
        state = self.proj_state2(state)
        for i_m in range(self.n_ref_block):
            R_in = mx.stop_gradient(R_in)
            T_in = mx.stop_gradient(T_in)

            xyz = mx.einsum('bnij,bnaj->bnai', R_in, xyz_in) + mx.expand_dims(T_in, -2)

            R_in, T_in, state, alpha = self.str_refiner(
                msa, pair, R_in, T_in, xyz, state, idx, top_k=64)

            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

            if (i_m + 1) % eval_stride == 0 or i_m == self.n_ref_block - 1:
                mx.eval(R_in, T_in, state, alpha)

        R_s = mx.stack(R_s, axis=0)
        T_s = mx.stack(T_s, axis=0)
        alpha_s = mx.stack(alpha_s, axis=0)

        return msa, pair, R_s, T_s, alpha_s, state
