"""
MLX RoseTTAFoldModule for RFdiffusion.

Top-level model that assembles embeddings, iterative simulator,
and prediction heads.
"""

import mlx.core as mx
import mlx.nn as nn

from .embeddings import MSA_emb, Extra_emb, Templ_emb, Timestep_emb, Recycling
from .predictors import DistanceNetwork, MaskedTokenNetwork, LDDTNetwork, ExpResolvedNetwork
from .track import IterativeSimulator


class RoseTTAFoldModule(nn.Module):
    def __init__(self,
                 n_extra_block, n_main_block, n_ref_block,
                 d_msa, d_msa_full, d_pair, d_templ,
                 n_head_msa, n_head_pair, n_head_templ,
                 d_hidden, d_hidden_templ, p_drop,
                 d_t1d, d_t2d,
                 d_time_emb, d_time_emb_proj,
                 T, use_motif_timestep,
                 freeze_track_motif, use_selfcond_emb,
                 SE3_param_full=None, SE3_param_topk=None,
                 input_seq_onehot=False):
        super().__init__()
        if SE3_param_full is None:
            SE3_param_full = {'l0_in_features': 32, 'l0_out_features': 16,
                              'num_edge_features': 32}
        if SE3_param_topk is None:
            SE3_param_topk = {'l0_in_features': 32, 'l0_out_features': 16,
                              'num_edge_features': 32}

        self.freeze_track_motif = freeze_track_motif

        d_state = SE3_param_topk['l0_out_features']
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, d_state=d_state,
                                  p_drop=p_drop, input_seq_onehot=input_seq_onehot)
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=25, p_drop=p_drop,
                                  input_seq_onehot=input_seq_onehot)
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ,
                                   d_state=d_state, n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25,
                                   d_t1d=d_t1d, d_t2d=d_t2d)

        if use_selfcond_emb:
            self.selfcond_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ,
                                          d_state=d_state, n_head=n_head_templ,
                                          d_hidden=d_hidden_templ, p_drop=0.25,
                                          d_t1d=d_t1d, d_t2d=d_t2d)

        if d_time_emb:
            self.timestep_embedder = Timestep_emb(
                input_size=d_time_emb, output_size=d_time_emb_proj,
                T=T, use_motif_timestep=use_motif_timestep)

        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        self.simulator = IterativeSimulator(
            n_extra_block=n_extra_block, n_main_block=n_main_block,
            n_ref_block=n_ref_block, d_msa=d_msa, d_msa_full=d_msa_full,
            d_pair=d_pair, d_hidden=d_hidden,
            n_head_msa=n_head_msa, n_head_pair=n_head_pair,
            SE3_param_full=SE3_param_full, SE3_param_topk=SE3_param_topk,
            p_drop=p_drop)

        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)
        self.lddt_pred = LDDTNetwork(d_state)
        self.exp_pred = ExpResolvedNetwork(d_msa, d_state)

    def enable_mixed_precision(self):
        """Enable float16 for bandwidth-bound attention layers.

        Converts PairStr2Pair weights to float16 and enables input casting.
        Gives ~30% speedup on pair attention with negligible quality impact.
        """
        import mlx.nn as mnn

        def _convert_module_fp16(module):
            flat = dict(mnn.utils.tree_flatten(module.parameters()))
            fp16 = {k: v.astype(mx.float16) for k, v in flat.items()}
            from .weight_converter import _set_params_from_flat
            _set_params_from_flat(module, fp16)

        # Convert pair2pair in all IterBlocks
        for block_list in [self.simulator.extra_block,
                           self.simulator.main_block]:
            for block in block_list:
                _convert_module_fp16(block.pair2pair)
                block.pair2pair._use_fp16 = True

        mx.eval(self.parameters())

    def set_topk_graph(self, top_k: int):
        """Override graph construction to use k-NN instead of full graph.

        Using top_k=64 gives ~2.5x speedup on SE3 convolutions at the cost
        of approximate (rather than exact) message passing. The model was
        trained with full graphs, so this trades quality for speed.

        Args:
            top_k: Number of nearest neighbors. 0 = full graph (default).
                   64 is a good starting point for fast inference.
        """
        self.simulator.top_k_override = top_k

    def set_se3_stride(self, stride: int):
        """Skip SE3 structural updates in main blocks.

        With stride=2, SE3 runs on blocks 0, 2, 4, ... and is skipped
        on blocks 1, 3, 5, ... (pair2pair and msa2msa still run).
        Gives ~25% speedup at the cost of structural accuracy.

        Args:
            stride: Run SE3 every N blocks. 1 = every block (default).
        """
        self.simulator.se3_stride = stride

    def set_n_main_block(self, n: int):
        """Reduce the number of main blocks during inference.

        The model has 32 main blocks by default. Running fewer blocks
        trades structural refinement quality for speed.

        Args:
            n: Number of main blocks to run. Must be <= n_main_block.
        """
        self.simulator.n_main_block_infer = min(n, self.simulator.n_main_block)

    def enable_fused_kernels(self):
        """Enable fused Metal kernels for SE3 convolutions.

        Replaces VersatileConvSE3's 3-step pipeline (MLP + matmul + matmul)
        with single Metal kernels for configs where it's faster.
        """
        sim = self.simulator
        # Extra blocks
        for block in getattr(sim, 'extra_block', []):
            block.str2str.se3.enable_fused_kernels()
        # Main blocks
        for block in getattr(sim, 'main_block', []):
            block.str2str.se3.enable_fused_kernels()
        # Ref block (str_refiner is a Str2Str, not IterBlock)
        if hasattr(sim, 'str_refiner'):
            sim.str_refiner.se3.enable_fused_kernels()

    def __call__(self, msa_latent, msa_full, seq, xyz, idx, t,
                 t1d=None, t2d=None, xyz_t=None, alpha_t=None,
                 sc2d=None, xyz_sc=None,
                 msa_prev=None, pair_prev=None, state_prev=None,
                 return_raw=False, return_full=False, return_infer=False,
                 return_w_msa_prev=False,
                 use_checkpoint=False, motif_mask=None,
                 i_cycle=None, n_cycle=None):

        B, N, L = msa_latent.shape[:3]
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx)
        msa_full = self.full_emb(msa_full, seq, idx)

        if msa_prev is None:
            msa_prev = mx.zeros_like(msa_latent[:, 0])
        if pair_prev is None:
            pair_prev = mx.zeros_like(pair)
        if state_prev is None:
            state_prev = mx.zeros_like(state)

        msa_recycle, pair_recycle, state_recycle = self.recycle(
            seq, msa_prev, pair_prev, xyz, state_prev)
        msa_latent = msa_latent.at[:, 0].add(msa_recycle.reshape(B, L, -1))
        pair = pair + pair_recycle
        state = state + state_recycle

        # Timestep embedding
        if hasattr(self, 'timestep_embedder'):
            time_emb = self.timestep_embedder(L, t, motif_mask)
            n_tmpl = t1d.shape[1]
            time_emb_tiled = mx.broadcast_to(
                time_emb.reshape(1, 1, L, -1),
                (1, n_tmpl, L, time_emb.shape[-1]))
            t1d = mx.concatenate([t1d, time_emb_tiled], axis=-1)

        pair, state = self.templ_emb(
            t1d, t2d, alpha_t, xyz_t, pair, state,
            use_checkpoint=use_checkpoint)

        if hasattr(self, 'selfcond_emb'):
            alpha_sc = mx.zeros_like(alpha_t)
            pair, state = self.selfcond_emb(
                t1d, sc2d, alpha_sc, xyz_sc, pair, state,
                use_checkpoint=use_checkpoint)

        is_frozen = motif_mask if self.freeze_track_motif else mx.zeros_like(motif_mask).astype(mx.bool_)
        msa, pair, R, T, alpha_s, state = self.simulator(
            seq, msa_latent, msa_full, pair, xyz[:, :, :3],
            state, idx, use_checkpoint=use_checkpoint,
            motif_mask=is_frozen)

        if return_raw:
            xyz_out = mx.einsum(
                'bnij,bnaj->bnai', R[-1],
                xyz[:, :, :3] - mx.expand_dims(xyz[:, :, 1], -2)
            ) + mx.expand_dims(T[-1], -2)
            return msa[:, 0], pair, xyz_out, state, alpha_s[-1]

        logits_aa = self.aa_pred(msa)
        lddt = self.lddt_pred(state)

        if return_infer:
            xyz_out = mx.einsum(
                'bnij,bnaj->bnai', R[-1],
                xyz[:, :, :3] - mx.expand_dims(xyz[:, :, 1], -2)
            ) + mx.expand_dims(T[-1], -2)

            nbin = lddt.shape[1]
            bin_step = 1.0 / nbin
            lddt_bins = mx.linspace(bin_step, 1.0, nbin)
            pred_lddt = mx.softmax(lddt, axis=1)
            pred_lddt = mx.sum(
                lddt_bins.reshape(1, -1, 1) * pred_lddt, axis=1)

            return (msa[:, 0], pair, xyz_out, state, alpha_s[-1],
                    mx.transpose(logits_aa, (0, 2, 1)), pred_lddt)

        logits = self.c6d_pred(pair)
        logits_exp = self.exp_pred(msa[:, 0], state)

        xyz_out = mx.einsum(
            'rbnij,bnaj->rbnai', R,
            xyz[:, :, :3] - mx.expand_dims(xyz[:, :, 1], -2)
        ) + mx.expand_dims(T, -2)

        if return_w_msa_prev:
            return (logits, logits_aa, logits_exp, xyz_out,
                    alpha_s, lddt, msa[:, 0], pair, state)

        return logits, logits_aa, logits_exp, xyz_out, alpha_s, lddt
