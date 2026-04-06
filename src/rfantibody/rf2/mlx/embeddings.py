"""
RF2 embedding modules ported to MLX.

Port of src/rfantibody/rf2/network/Embeddings.py.
Omits stride/striping parameters and dropout (MLX inference processes full tensors).
"""
import mlx.core as mx
import mlx.nn as nn

from rfantibody.rf2.mlx.attention import Attention
from rfantibody.rf2.mlx.track import PairStr2Pair, rbf, cdist


def get_Cb(xyz: mx.array) -> mx.array:
    """Reconstruct Cb from N, Ca, C coordinates.

    Parameters
    ----------
    xyz : mx.array
        Shape (B, L, 3, 3) — the first three backbone atoms [N, Ca, C].

    Returns
    -------
    mx.array
        Shape (B, L, 3) — virtual Cb coordinates.
    """
    N = xyz[:, :, 0]
    Ca = xyz[:, :, 1]
    C = xyz[:, :, 2]
    b = Ca - N
    c = C - Ca
    # Cross product (MLX has no mx.cross): a = b x c
    a = mx.concatenate([
        (b[..., 1:2] * c[..., 2:3] - b[..., 2:3] * c[..., 1:2]),
        (b[..., 2:3] * c[..., 0:1] - b[..., 0:1] * c[..., 2:3]),
        (b[..., 0:1] * c[..., 1:2] - b[..., 1:2] * c[..., 0:1]),
    ], axis=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
    return Cb


# ---------------------------------------------------------------------------
# PositionalEncoding2D
# ---------------------------------------------------------------------------

class PositionalEncoding2D(nn.Module):
    """Relative positional encoding added to pair features."""

    def __init__(self, d_model: int, minpos: int = -32, maxpos: int = 32):
        super().__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos) + maxpos + 1
        self.emb = nn.Embedding(self.nbin, d_model)
        self.d_out = d_model

    def __call__(self, idx: mx.array, nc_cycle: bool = False) -> mx.array:
        """
        Parameters
        ----------
        idx : mx.array
            Residue indices, shape (B, L).
        nc_cycle : bool
            Whether to apply cyclic wrapping of sequence separation.

        Returns
        -------
        mx.array
            Positional pair embedding, shape (B, L, L, d_model).
        """
        B, L = idx.shape[:2]

        bins = mx.arange(self.minpos, self.maxpos)  # (nbin-1,)

        # Compute pairwise sequence separation
        seqsep = mx.full((B, L, L), 100)
        seqsep_0 = mx.expand_dims(idx[0], 0) - mx.expand_dims(idx[0], 1)  # (L, L)
        if nc_cycle:
            seqsep_0 = (seqsep_0 + L // 2) % L - L // 2
        # Place batch-0 separation (single-batch inference)
        seqsep = mx.expand_dims(seqsep_0, 0)  # (1, L, L)
        if B > 1:
            rest = mx.full((B - 1, L, L), 100)
            seqsep = mx.concatenate([seqsep, rest], axis=0)

        # Bucketize: find bin index for each separation value
        # Use > (not >=) to match torch.bucketize(right=False) semantics
        ib = mx.sum(mx.expand_dims(seqsep, -1) > mx.reshape(bins, (1, 1, 1, -1)),
                     axis=-1).astype(mx.int32)

        emb = self.emb(ib)  # (B, L, L, d_model)
        return emb


# ---------------------------------------------------------------------------
# MSA_emb
# ---------------------------------------------------------------------------

class MSA_emb(nn.Module):
    """Initial seed MSA, pair, and state embeddings."""

    def __init__(self, d_msa: int = 256, d_pair: int = 128, d_state: int = 32,
                 d_init: int = 22 + 22 + 2 + 2, minpos: int = -32,
                 maxpos: int = 32, p_drop: float = 0.0):
        super().__init__()
        self.emb = nn.Linear(d_init, d_msa)
        self.emb_q = nn.Embedding(22, d_msa)
        self.emb_left = nn.Embedding(22, d_pair)
        self.emb_right = nn.Embedding(22, d_pair)
        self.emb_state = nn.Embedding(22, d_state)
        self.pos = PositionalEncoding2D(d_pair, minpos=minpos, maxpos=maxpos)

        self.d_init = d_init
        self.d_msa = d_msa

    def __call__(self, msa: mx.array, seq: mx.array, idx: mx.array,
                 nc_cycle: bool = False) -> tuple:
        """
        Parameters
        ----------
        msa : mx.array
            Input MSA features, shape (B, N, L, d_init).
        seq : mx.array
            Input sequence indices, shape (B, L).
        idx : mx.array
            Residue indices, shape (B, L).
        nc_cycle : bool
            Cyclic chain wrapping flag.

        Returns
        -------
        tuple of mx.array
            (msa, pair, state) embeddings.
        """
        B, N, L = msa.shape[:3]

        # MSA embedding
        msa = self.emb(msa)  # (B, N, L, d_msa)
        tmp = mx.expand_dims(self.emb_q(seq), 1)  # (B, 1, L, d_msa)
        msa = msa + mx.broadcast_to(tmp, (B, N, L, self.d_msa))

        # Pair embedding
        left = mx.expand_dims(self.emb_left(seq), 1)    # (B, 1, L, d_pair)
        right = mx.expand_dims(self.emb_right(seq), 2)  # (B, L, 1, d_pair)
        pair = left + right  # (B, L, L, d_pair)
        pair = pair + self.pos(idx, nc_cycle)

        # State embedding
        state = self.emb_state(seq)  # (B, L, d_state)

        return msa, pair, state


# ---------------------------------------------------------------------------
# Extra_emb
# ---------------------------------------------------------------------------

class Extra_emb(nn.Module):
    """Extra MSA embedding (for extra sequences beyond the seed MSA)."""

    def __init__(self, d_msa: int = 256, d_init: int = 22 + 1 + 2,
                 p_drop: float = 0.0):
        super().__init__()
        self.emb = nn.Linear(d_init, d_msa)
        self.emb_q = nn.Embedding(22, d_msa)

        self.d_init = d_init
        self.d_msa = d_msa

    def __call__(self, msa: mx.array, seq: mx.array,
                 idx: mx.array) -> mx.array:
        """
        Parameters
        ----------
        msa : mx.array
            Extra MSA features, shape (B, N, L, d_init).
        seq : mx.array
            Input sequence indices, shape (B, L).
        idx : mx.array
            Residue indices, shape (B, L).

        Returns
        -------
        mx.array
            Embedded extra MSA, shape (B, N, L, d_msa).
        """
        B, N, L = msa.shape[:3]

        msa = self.emb(msa)  # (B, N, L, d_msa)
        seq_emb = mx.expand_dims(self.emb_q(seq), 1)  # (B, 1, L, d_msa)
        msa = msa + mx.broadcast_to(seq_emb, (B, N, L, self.d_msa))

        return msa


# ---------------------------------------------------------------------------
# TemplatePairStack
# ---------------------------------------------------------------------------

class TemplatePairStack(nn.Module):
    """Process template pairwise features with structure-biased attention."""

    def __init__(self, n_block: int = 2, d_templ: int = 64, n_head: int = 4,
                 d_hidden: int = 16, d_t1d: int = 22, d_state: int = 32,
                 p_drop: float = 0.0):
        super().__init__()
        self.n_block = n_block
        self.proj_t1d = nn.Linear(d_t1d, d_state)
        self.block = [
            PairStr2Pair(d_pair=d_templ, n_head=n_head, d_hidden=d_hidden,
                         d_state=d_state)
            for _ in range(n_block)
        ]
        self.norm = nn.LayerNorm(d_templ)
        self.d_out = d_templ

    def __call__(self, templ: mx.array, rbf_feat: mx.array,
                 t1d: mx.array) -> mx.array:
        """
        Parameters
        ----------
        templ : mx.array
            Template pair features, shape (B, T, L, L, d_templ).
        rbf_feat : mx.array
            Template RBF features, shape (B*T, L, L, d_rbf).
        t1d : mx.array
            Template 1D features, shape (B, T, L, d_t1d).

        Returns
        -------
        mx.array
            Processed template features, shape (B, T, L, L, d_templ).
        """
        B, T, L = templ.shape[:3]

        templ = templ.reshape(B * T, L, L, -1)
        t1d = t1d.reshape(B * T, L, -1)
        state = self.proj_t1d(t1d)

        for i_block in range(self.n_block):
            templ = self.block[i_block](templ, rbf_feat, state)

        out = self.norm(templ)
        return out.reshape(B, T, L, L, -1)


# ---------------------------------------------------------------------------
# Templ_emb
# ---------------------------------------------------------------------------

class Templ_emb(nn.Module):
    """Template embedding with attention-based mixing into pair and state.

    Template features:
      - t2d: 37 distogram bins + 6 orientations + 1 mask = 44
      - t1d: 21 AA + 1 confidence = 22
    """

    def __init__(self, d_t1d: int = 22, d_t2d: int = 44, d_tor: int = 30,
                 d_pair: int = 128, d_state: int = 32, n_block: int = 2,
                 d_templ: int = 64, n_head: int = 4, d_hidden: int = 16,
                 p_drop: float = 0.0):
        super().__init__()
        # 2D template embedding
        self.emb = nn.Linear(d_t1d * 2 + d_t2d, d_templ)
        self.templ_stack = TemplatePairStack(n_block=n_block, d_templ=d_templ,
                                             n_head=n_head, d_hidden=d_hidden,
                                             d_t1d=d_t1d)

        # Template -> pair attention
        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_pair)

        # Torsion angle projection + template -> state attention
        self.proj_t1d = nn.Linear(d_t1d + d_tor, d_templ)
        self.attn_tor = Attention(d_state, d_templ, n_head, d_hidden, d_state)

        self.d_rbf = 64
        self.d_templ = d_templ

    def _get_templ_emb(self, t1d: mx.array, t2d: mx.array) -> mx.array:
        """Build 2D template embedding from 1D and 2D features.

        Parameters
        ----------
        t1d : mx.array
            Shape (B, T, L, d_t1d).
        t2d : mx.array
            Shape (B, T, L, L, d_t2d).

        Returns
        -------
        mx.array
            Shape (B, T, L, L, d_templ).
        """
        B, T, L, _ = t1d.shape
        left = mx.broadcast_to(
            mx.expand_dims(t1d, 3), (B, T, L, L, t1d.shape[-1]))
        right = mx.broadcast_to(
            mx.expand_dims(t1d, 2), (B, T, L, L, t1d.shape[-1]))
        templ = mx.concatenate([t2d, left, right], axis=-1)  # (B, T, L, L, 2*d_t1d+d_t2d)
        return self.emb(templ)

    def _get_templ_rbf(self, xyz_t: mx.array,
                       mask_t: mx.array) -> mx.array:
        """Compute RBF features from template Ca coordinates.

        Parameters
        ----------
        xyz_t : mx.array
            Template Ca coordinates, shape (B, T, L, 3).
        mask_t : mx.array
            Template pair mask, shape (B, T, L, L).

        Returns
        -------
        mx.array
            RBF features, shape (B*T, L, L, d_rbf).
        """
        B, T, L = xyz_t.shape[:3]
        xyz_t = xyz_t.reshape(B * T, L, 3)
        mask_t = mask_t.reshape(B * T, L, L)

        dist = cdist(xyz_t, xyz_t)  # (B*T, L, L)
        rbf_feat = rbf(dist)        # (B*T, L, L, d_rbf)
        rbf_feat = rbf_feat * mx.expand_dims(mask_t, -1)

        return rbf_feat

    def __call__(self, t1d: mx.array, t2d: mx.array, alpha_t: mx.array,
                 xyz_t: mx.array, mask_t: mx.array, pair: mx.array,
                 state: mx.array) -> tuple:
        """
        Parameters
        ----------
        t1d : mx.array
            1D template info, shape (B, T, L, d_t1d).
        t2d : mx.array
            2D template info, shape (B, T, L, L, d_t2d).
        alpha_t : mx.array
            Torsion angle info, shape (B, T, L, d_tor).
        xyz_t : mx.array
            Template CA coordinates, shape (B, T, L, 3).
        mask_t : mx.array
            Valid residue pair mask, shape (B, T, L, L).
        pair : mx.array
            Query pair features, shape (B, L, L, d_pair).
        state : mx.array
            Query state features, shape (B, L, d_state).

        Returns
        -------
        tuple of mx.array
            (pair, state) — updated pair and state features.
        """
        B, T, L, _ = t1d.shape

        templ = self._get_templ_emb(t1d, t2d)
        rbf_feat = self._get_templ_rbf(xyz_t, mask_t)

        # Process template pairs through the stack
        templ = self.templ_stack(templ, rbf_feat, t1d)  # (B, T, L, L, d_templ)

        # Prepare 1D template torsion angle features
        t1d_tor = mx.concatenate([t1d, alpha_t], axis=-1)  # (B, T, L, d_t1d+d_tor)
        t1d_tor = self.proj_t1d(t1d_tor)

        # Mix query state features with template state features via attention
        # state: (B, L, d_state) -> (B*L, 1, d_state)
        state_q = state.reshape(B * L, 1, -1)
        # t1d_tor: (B, T, L, d_templ) -> permute to (B, L, T, d_templ) -> (B*L, T, d_templ)
        t1d_tor_perm = mx.transpose(t1d_tor, (0, 2, 1, 3)).reshape(B * L, T, -1)
        out_state = self.attn_tor(state_q, t1d_tor_perm, t1d_tor_perm)
        out_state = out_state.reshape(B, L, -1)
        state = state + out_state

        # Mix query pair features with template information via attention
        # pair: (B, L, L, d_pair) -> (B*L*L, 1, d_pair)
        pair_q = pair.reshape(B * L * L, 1, -1)
        # templ: (B, T, L, L, d_templ) -> permute to (B, L, L, T, d_templ) -> (B*L*L, T, d_templ)
        templ_perm = mx.transpose(templ, (0, 2, 3, 1, 4)).reshape(B * L * L, T, -1)
        out_pair = self.attn(pair_q, templ_perm, templ_perm)
        out_pair = out_pair.reshape(B, L, L, -1)
        pair = pair + out_pair

        return pair, state


# ---------------------------------------------------------------------------
# Recycling
# ---------------------------------------------------------------------------

class Recycling(nn.Module):
    """Recycle previous predictions into the next iteration.

    Combines normalized previous MSA/pair/state with RBF of Cb-Cb distances
    and projected state features.
    """

    def __init__(self, d_msa: int = 256, d_pair: int = 128,
                 d_state: int = 32, d_rbf: int = 64):
        super().__init__()
        self.proj_dist = nn.Linear(d_rbf + d_state * 2, d_pair)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_state = nn.LayerNorm(d_state)

        self.d_pair = d_pair
        self.d_msa = d_msa
        self.d_rbf = d_rbf

    def __call__(self, seq: mx.array, msa: mx.array, pair: mx.array,
                 state: mx.array, xyz: mx.array,
                 mask_recycle: mx.array = None) -> tuple:
        """
        Parameters
        ----------
        seq : mx.array
            Sequence indices (unused, kept for API compatibility).
        msa : mx.array
            Previous MSA features, shape (B, L, d_msa).
        pair : mx.array
            Previous pair features, shape (B, L, L, d_pair).
        state : mx.array
            Previous state features, shape (B, L, d_state).
        xyz : mx.array
            Previous coordinates, shape (B, L, >=3, 3) — N, Ca, C atoms.
        mask_recycle : mx.array, optional
            Recycling mask, shape (B, L, L).

        Returns
        -------
        tuple of mx.array
            (msa, pair, state) — normalized and updated features.
        """
        B, L = msa.shape[:2]

        state = self.norm_state(state)
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)

        # Expand state into left/right pair contributions
        left = mx.broadcast_to(
            mx.expand_dims(state, 2), (B, L, L, state.shape[-1]))  # (B, L, L, d_state)
        right = mx.broadcast_to(
            mx.expand_dims(state, 1), (B, L, L, state.shape[-1]))  # (B, L, L, d_state)

        # Reconstruct Cb from N, Ca, C
        Cb = get_Cb(xyz[:, :, :3])  # (B, L, 3)

        # Cb-Cb RBF distance features
        dist_CB = rbf(cdist(Cb, Cb))  # (B, L, L, d_rbf)

        if mask_recycle is not None:
            dist_CB = mx.expand_dims(mask_recycle, -1) * dist_CB

        # Concatenate RBF with state outer products
        dist_CB = mx.concatenate([dist_CB, left, right], axis=-1)  # (B, L, L, d_rbf+2*d_state)

        pair = pair + self.proj_dist(dist_CB)

        return msa, pair, state
