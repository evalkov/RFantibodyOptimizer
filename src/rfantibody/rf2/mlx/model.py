"""
RF2 RoseTTAFoldModule ported to MLX.

Port of src/rfantibody/rf2/network/RoseTTAFoldModel.py.
Assembles embeddings, simulator, and prediction heads.
"""

import mlx.core as mx
import mlx.nn as nn

from rfantibody.rf2.mlx.embeddings import (
    Extra_emb,
    MSA_emb,
    Recycling,
    Templ_emb,
)
from rfantibody.rf2.mlx.predictors import (
    DistanceNetwork,
    ExpResolvedNetwork,
    LDDTNetwork,
    MaskedTokenNetwork,
    PAENetwork,
    v2_BinderNetwork,
)
from rfantibody.rf2.mlx.track import IterativeSimulator, rbf, cdist


class RoseTTAFoldModule(nn.Module):
    """MLX port of RF2's RoseTTAFoldModule.

    Architecture:
        1. Embedding (MSA + Extra + Template + Recycling)
        2. IterativeSimulator (extra blocks + main blocks + refinement)
        3. Prediction heads (distance, AA, LDDT, PAE, binding, exp resolved)
    """

    def __init__(
        self,
        n_extra_block=4,
        n_main_block=8,
        n_ref_block=4,
        d_msa=256,
        d_msa_full=64,
        d_pair=128,
        d_templ=64,
        n_head_msa=8,
        n_head_pair=4,
        n_head_templ=4,
        d_hidden=32,
        d_hidden_templ=64,
        p_drop=0.0,
        d_t1d=22,
        d_t2d=44,
        SE3_param_full=None,
        SE3_param_topk=None,
    ):
        super().__init__()

        if SE3_param_full is None:
            SE3_param_full = {
                'l0_in_features': 32, 'l0_out_features': 16,
                'num_edge_features': 32,
            }
        if SE3_param_topk is None:
            SE3_param_topk = {
                'l0_in_features': 32, 'l0_out_features': 16,
                'num_edge_features': 32,
            }

        d_state = SE3_param_topk['l0_out_features']

        # Input embeddings
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, d_state=d_state,
                                   p_drop=p_drop)
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=25, p_drop=p_drop)
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ,
                                    d_state=d_state, n_head=n_head_templ,
                                    d_hidden=d_hidden_templ, p_drop=p_drop,
                                    d_t1d=d_t1d, d_t2d=d_t2d)
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)

        # Iterative simulator
        self.simulator = IterativeSimulator(
            n_extra_block=n_extra_block,
            n_main_block=n_main_block,
            n_ref_block=n_ref_block,
            d_msa=d_msa, d_msa_full=d_msa_full,
            d_pair=d_pair, d_hidden=d_hidden,
            n_head_msa=n_head_msa, n_head_pair=n_head_pair,
            SE3_param_full=SE3_param_full,
            SE3_param_topk=SE3_param_topk,
            p_drop=p_drop)

        # Prediction heads
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)
        self.lddt_pred = LDDTNetwork(d_state)
        self.exp_pred = ExpResolvedNetwork(d_msa, d_state)
        self.pae_pred = PAENetwork(d_pair)
        self.bind_pred = v2_BinderNetwork(d_pair=d_pair, d_state=d_state)

    def __call__(
        self,
        msa_latent,
        msa_full,
        seq,
        xyz_prev,
        idx,
        t1d,
        t2d,
        xyz_t,
        alpha_t,
        mask_t,
        same_chain,
        msa_prev=None,
        pair_prev=None,
        state_prev=None,
        mask_recycle=None,
        return_raw=False,
        nc_cycle=False,
        topk_crop=-1,
    ):
        """Forward pass.

        Args:
            msa_latent: (B, N, L, d_init) seed MSA features
            msa_full:   (B, N', L, d_init_full) extra MSA features
            seq:        (B, L) query sequence indices
            xyz_prev:   (B, L, n_atom, 3) previous coordinates
            idx:        (B, L) residue indices
            t1d:        (B, T, L, d_t1d) 1D template features
            t2d:        (B, T, L, L, d_t2d) 2D template features
            xyz_t:      (B, T, L, 3) template CA coordinates
            alpha_t:    (B, T, L, d_tor) template torsion angles
            mask_t:     (B, T, L, L) template mask
            same_chain: (B, L, L) same-chain indicator
            msa_prev:   (B, L, d_msa) previous MSA (for recycling)
            pair_prev:  (B, L, L, d_pair) previous pair (for recycling)
            state_prev: (B, L, d_state) previous state (for recycling)
            mask_recycle: (B, L, L) recycling mask
            return_raw: if True, return raw features
            nc_cycle:   N-C cyclization flag
            topk_crop:  top-k for SE3 graph

        Returns:
            If return_raw:
                (msa_out, pair, state, xyz_last, alpha_last, None)
            Else:
                (logits, logits_aa, logits_exp, logits_pae, p_bind,
                 xyz, alpha, None, lddt, msa_out, pair, state)
        """
        B, N, L = msa_latent.shape[:3]

        # Get embeddings
        msa_latent, pair, state = self.latent_emb(
            msa_latent, seq, idx, nc_cycle=nc_cycle)
        msa_full_emb = self.full_emb(msa_full, seq, idx)

        # Recycling
        if msa_prev is None:
            msa_prev = mx.zeros((B, L, msa_latent.shape[-1]))
            pair_prev = mx.zeros_like(pair)
            state_prev = mx.zeros_like(state)

        msa_recycle, pair_recycle, state_recycle = self.recycle(
            seq, msa_prev, pair_prev, state_prev, xyz_prev,
            mask_recycle)

        msa_latent = msa_latent.at[:, 0].add(msa_recycle)
        pair = pair + pair_recycle
        state = state + state_recycle

        # Template embedding
        pair, state = self.templ_emb(
            t1d, t2d, alpha_t, xyz_t, mask_t, pair, state)

        # Run simulator
        msa, pair, R, T, alpha, state = self.simulator(
            seq, msa_latent, msa_full_emb, pair,
            xyz_prev[:, :, :3], state, idx,
            topk_crop=topk_crop)

        # Get last structure
        xyz_last = mx.einsum(
            'blij,blaj->blai',
            R[-1],
            xyz_prev - mx.expand_dims(xyz_prev[:, :, 1], -2)
        ) + mx.expand_dims(T[-1], -2)

        if return_raw:
            return msa[:, 0], pair, state, xyz_last, alpha[-1], None

        # Prediction heads
        logits_aa = self.aa_pred(msa)
        logits = self.c6d_pred(pair)
        logits_pae = self.pae_pred(pair)

        # Binding prediction
        rbf_feat = rbf(cdist(
            xyz_last[:, :, 1, :], xyz_last[:, :, 1, :]))
        p_bind = self.bind_pred(pair, rbf_feat, state, same_chain)

        # LDDT prediction
        lddt = self.lddt_pred(state)

        # Experimentally resolved prediction
        logits_exp = self.exp_pred(msa[:, 0], state)

        # Get all intermediate BB structures
        xyz = mx.einsum(
            'rblij,blaj->rblai',
            R,
            xyz_prev - mx.expand_dims(xyz_prev[:, :, 1], -2),
        ) + mx.expand_dims(T, -2)

        return (logits, logits_aa, logits_exp, logits_pae, p_bind,
                xyz, alpha, None, lddt, msa[:, 0], pair, state)

    def enable_mixed_precision(self):
        """Enable float16 for bandwidth-bound pair attention layers."""
        import mlx.nn as mnn
        from rfantibody.rfdiffusion.mlx.weight_converter import _set_params_from_flat

        def _convert_module_fp16(module):
            flat = dict(mnn.utils.tree_flatten(module.parameters()))
            fp16 = {k: v.astype(mx.float16) for k, v in flat.items()}
            _set_params_from_flat(module, fp16)

        for block_list in [self.simulator.extra_block,
                           self.simulator.main_block]:
            for block in block_list:
                _convert_module_fp16(block.pair2pair)
                block.pair2pair._use_fp16 = True

        mx.eval(self.parameters())

    def set_topk_graph(self, top_k: int):
        """Override SE3 graph to use k-NN instead of full graph."""
        self.simulator.top_k_override = top_k

    def set_se3_stride(self, stride: int):
        """Skip SE3 in alternating main blocks."""
        self.simulator.se3_stride = stride

    def set_n_main_block(self, n: int):
        """Reduce main blocks for faster inference (default: 36)."""
        self.simulator.n_main_block_infer = min(n, self.simulator.n_main_block)

    def enable_fused_kernels(self):
        """Enable fused Metal kernels for SE3 convolutions."""
        sim = self.simulator
        for block in getattr(sim, 'extra_block', []):
            block.str2str.se3.enable_fused_kernels()
        for block in getattr(sim, 'main_block', []):
            block.str2str.se3.enable_fused_kernels()
        if hasattr(sim, 'str_refiner'):
            sim.str_refiner.se3.enable_fused_kernels()
