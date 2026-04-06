"""
Batched diffusion design for MLX.

Runs B independent diffusion trajectories simultaneously by batching the
model forward pass while keeping pre/postprocessing per-design.

The model forward pass takes ~2000ms while preprocessing takes ~1ms,
so batching gives near-linear throughput scaling: B designs in ~1x time.

Usage:
    from rfantibody.rfdiffusion.mlx.batched_design import BatchedDesigner
    designer = BatchedDesigner(sampler, batch_size=4)
    results = designer.run(num_steps=50, seed=0)
    # results: list of B DesignResult objects
"""

import copy
import logging
import time

import numpy as np
import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


class DesignState:
    """Mutable per-design state tracked across timesteps."""

    def __init__(self, x_t, seq_t, seq_init, denoiser):
        self.x_t = x_t            # (L, 14, 3)
        self.seq_t = seq_t        # (L, 22)
        self.seq_init = seq_init  # (L, 22)
        self.denoiser = denoiser
        self.prev_pred = None     # (L, 14, 3) — structural self-conditioning
        self.msa_prev = None      # (1, 1, L, d_msa) — sequence self-conditioning
        # Trajectory stacks
        self.px0_stack = []
        self.xyz_stack = []
        self.seq_stack = []
        self.plddt_stack = []


class DesignResult:
    """Final output of one design trajectory."""

    def __init__(self, state, step_times):
        self.denoised_xyz = torch.stack(state.xyz_stack)   # (T, L, 14, 3)
        self.px0_xyz = torch.stack(state.px0_stack)        # (T, L, 14, 3)
        self.plddt = torch.stack(state.plddt_stack)        # (T, L)
        self.final_seq = state.seq_t                       # (L, 22)
        self.seq_init = state.seq_init                     # (L, 22)
        self.step_times = step_times                       # shared across batch


class BatchedDesigner:
    """Run B diffusion trajectories with batched model forward passes.

    Strategy:
    - Pre/postprocessing: per-design (fast, ~1ms each on CPU)
    - Model forward pass: batched B designs in one call (slow, ~2000ms on GPU)
    - Self-conditioning: tracked per-design in DesignState objects

    This gives near-linear throughput: B designs in approximately 1x wall time.
    """

    def __init__(self, sampler, batch_size=4):
        """
        Args:
            sampler: MLXAbSampler instance (already initialized with conf).
            batch_size: Number of independent designs to run simultaneously.
        """
        self.sampler = sampler
        self.batch_size = batch_size
        # L is set after first sample_init() call
        self.L = None

    def _init_designs(self, base_seed=0):
        """Initialize B independent design states with different random noise.

        Each design gets a different random seed for the initial diffusion,
        producing different starting conformations from the same target.
        """
        states = []
        for i in range(self.batch_size):
            # Set unique seed for each design's initial noise
            seed = base_seed + i
            torch.manual_seed(seed)
            np.random.seed(seed)

            # sample_init() generates randomly noised starting structure
            # This also sets sampler.L, sampler.denoiser, etc.
            x_init, seq_init = self.sampler.sample_init()

            if self.L is None:
                self.L = self.sampler.L

            states.append(DesignState(
                x_t=torch.clone(x_init),
                seq_t=torch.clone(seq_init),
                seq_init=torch.clone(seq_init),
                denoiser=self.sampler.denoiser,
            ))

        # After init, all designs share the same denoiser config but
        # need independent decode schedules for sequence reveal.
        # Reconstruct per-design denoisers.
        for i in range(1, self.batch_size):
            states[i].denoiser = self.sampler.construct_denoiser(
                self.L, visible=self.sampler.diffusion_mask)

        return states

    def _preprocess_one(self, state, t):
        """Preprocess one design for model input. ~1ms on CPU."""
        s = self.sampler

        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = \
            s._preprocess(state.seq_t, state.x_t, t)

        B, N, L = xyz_t.shape[:3]

        # Sequence self-conditioning
        if (t < s.diffuser.T) and (t != s.diffuser_conf.partial_T) \
                and s.preprocess_conf.selfcondition_msaprev \
                and s.preprocess_conf.msaprev_bugfix:
            msa_prev = state.msa_prev
        else:
            msa_prev = None

        # Structural self-conditioning
        from rfantibody.rfdiffusion.inference.model_runners import (
            process_selfcond, process_init_selfcond, correct_selfcond)

        if (t < s.diffuser.T) and (t != s.diffuser_conf.partial_T) \
                and state.prev_pred is not None:
            xyz_t, t2d, xyz_sc, sc2d = process_selfcond(
                state.prev_pred, t2d, xyz_t, s.ab_conf, xyz_t.device)
            t2d = correct_selfcond(
                t2d, s.ab_conf,
                s.ab_item.inputs.xyz_true,
                s.ab_item.loop_mask,
                s.ab_item.target_mask,
                s.ab_item.interchain_mask)
        else:
            sc2d, xyz_sc = process_init_selfcond(t2d, xyz_t, s.ab_conf, xyz_t.device)

        return {
            'msa_masked': msa_masked,
            'msa_full': msa_full,
            'seq_in': seq_in,
            'xt_in': xt_in,
            'idx_pdb': idx_pdb,
            't1d': t1d,
            't2d': t2d,
            'xyz_t': xyz_t,
            'alpha_t': alpha_t,
            'sc2d': sc2d,
            'xyz_sc': xyz_sc,
            'msa_prev': msa_prev,
        }

    def _batch_model_call(self, preprocessed_list, t):
        """Stack B preprocessed inputs and run model once.

        This is where all the GPU time goes (~2000ms for B=1..8).
        """
        s = self.sampler
        B = len(preprocessed_list)

        # Stack all inputs along batch dimension
        # Each preprocessed input has shape (1, ...) — concatenate to (B, ...)
        def stack_field(field):
            vals = [p[field] for p in preprocessed_list]
            if vals[0] is None:
                return None
            return torch.cat(vals, dim=0)

        msa_masked = stack_field('msa_masked')  # (B, 1, L, d)
        msa_full = stack_field('msa_full')
        seq_in = stack_field('seq_in')
        xt_in = stack_field('xt_in')             # (B, L, 14, 3)
        idx_pdb = stack_field('idx_pdb')
        t1d = stack_field('t1d')
        t2d = stack_field('t2d')
        xyz_t = stack_field('xyz_t')
        alpha_t = stack_field('alpha_t')
        sc2d = stack_field('sc2d')
        xyz_sc = stack_field('xyz_sc')
        msa_prev = stack_field('msa_prev')

        # Run model with full batch
        px0 = xt_in
        for rec in range(s.recycle_schedule[t - 1]):
            msa_prev_out, pair_prev, px0, state_prev, alpha, logits, plddt = s.model(
                msa_masked, msa_full, seq_in, px0, idx_pdb,
                t1d=t1d, t2d=t2d, sc2d=sc2d, xyz_sc=xyz_sc,
                xyz_t=xyz_t, alpha_t=alpha_t,
                msa_prev=msa_prev, pair_prev=None, state_prev=None,
                t=torch.tensor(t), return_infer=True,
                motif_mask=s.diffusion_mask.squeeze().to(s.device))

            if rec < s.recycle_schedule[t - 1] - 1:
                from rfantibody.rfdiffusion.inference.model_runners import xyz_to_t2d
                zeros = torch.zeros(B, 1, self.L, 24, 3).float().to(xyz_t.device)
                xyz_t = torch.cat((px0.unsqueeze(1), zeros), dim=-2)
                t2d = xyz_to_t2d(xyz_t)
                px0 = xt_in

        # Return per-design outputs by splitting batch dim
        return {
            'msa_prev': msa_prev_out,  # (B, 1, L, d)
            'px0': px0,                # (B, L, 14, 3)
            'alpha': alpha,            # (B, L, ...)
            'logits': logits,          # (B, L, 22)
            'plddt': plddt,            # (B, L)
            'seq_in': seq_in,          # (B, L, 22)
        }

    def _postprocess_one(self, state, model_out_i, t, final_step):
        """Postprocess one design's model output. ~1ms on CPU."""
        s = self.sampler

        # Save self-conditioning state for next step
        state.prev_pred = torch.clone(model_out_i['px0'])
        state.msa_prev = torch.clone(model_out_i['msa_prev'])

        # Compute all-atom prediction
        seq_in_i = model_out_i['seq_in']  # (1, L, 22)
        px0_i = model_out_i['px0']        # (1, L, 14, 3)
        alpha_i = model_out_i['alpha']    # (1, L, ...)

        _, px0_full = s.allatom(torch.argmax(seq_in_i, dim=-1), px0_i, alpha_i)
        px0_full = px0_full.squeeze()[:, :14]

        # Decode sequence
        logits_i = model_out_i['logits'].squeeze()
        seq_probs = torch.nn.Softmax(dim=-1)(logits_i / s.inf_conf.softmax_T)
        sampled_seq = torch.multinomial(seq_probs, 1).squeeze()

        pseq_0 = torch.nn.functional.one_hot(
            sampled_seq, num_classes=22).to(s.device).float()
        pseq_0[s.mask_seq.squeeze()] = state.seq_init[s.mask_seq.squeeze()].to(s.device)

        plddt_i = model_out_i['plddt']  # (1, L) or similar

        if t > final_step:
            x_t_1, seq_t_1, tors_t_1, px0_full = state.denoiser.get_next_pose(
                xt=state.x_t,
                px0=px0_full,
                t=t,
                diffusion_mask=s.mask_str.squeeze(),
                seq_diffusion_mask=s.mask_seq.squeeze(),
                seq_t=state.seq_t,
                pseq0=pseq_0,
                diffuse_sidechains=s.preprocess_conf.sidechain_input,
                align_motif=s.inf_conf.align_motif,
                include_motif_sidechains=s.preprocess_conf.motif_sidechain_input)
        else:
            x_t_1 = torch.clone(px0_full).to(state.x_t.device)
            seq_t_1 = pseq_0

        # Update state
        state.x_t = x_t_1
        state.seq_t = seq_t_1
        state.px0_stack.append(px0_full)
        state.xyz_stack.append(x_t_1)
        state.seq_stack.append(seq_t_1)
        state.plddt_stack.append(plddt_i[0] if plddt_i.dim() > 1 else plddt_i)

    def run(self, seed=0):
        """Run B diffusion trajectories with batched model calls.

        Returns:
            list of DesignResult, one per design.
        """
        s = self.sampler
        B = self.batch_size
        final_step = getattr(s.inf_conf, 'final_step', 1)

        _log.info(f'Batched design: B={B}, T={s.t_step_input}, L={self.L}')

        # Initialize B designs with different random seeds
        t0 = time.time()
        states = self._init_designs(base_seed=seed)
        _log.info(f'Initialized {B} designs in {time.time() - t0:.1f}s')

        step_times = []

        for t in range(int(s.t_step_input), final_step - 1, -1):
            step_t0 = time.time()

            # 1. Preprocess each design independently (~1ms each)
            preprocessed = []
            for i, state in enumerate(states):
                preprocessed.append(self._preprocess_one(state, t))

            # 2. Batched model forward pass (~2000ms total, regardless of B)
            model_out = self._batch_model_call(preprocessed, t)

            # 3. Postprocess each design independently (~1ms each)
            for i, state in enumerate(states):
                # Slice batch dimension for this design
                out_i = {}
                for key, val in model_out.items():
                    if val is None:
                        out_i[key] = None
                    elif val.dim() > 0 and val.shape[0] == B:
                        out_i[key] = val[i:i+1]  # keep batch dim as size 1
                    else:
                        out_i[key] = val
                self._postprocess_one(state, out_i, t, final_step)

            step_dt = time.time() - step_t0
            step_times.append(step_dt)

            # Print progress
            avg_so_far = np.mean(step_times)
            print(f"  t={t:>3d}: {step_dt*1000:>7.0f}ms  "
                  f"({B} designs, {step_dt/B*1000:.0f}ms/design)")

        results = [DesignResult(state, step_times) for state in states]
        return results
