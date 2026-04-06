"""
MLX-accelerated sampler for RFdiffusion antibody design.

Replaces the PyTorch model forward pass with MLX while keeping all
preprocessing, denoising, and postprocessing in PyTorch/numpy.

Usage:
    from rfantibody.rfdiffusion.mlx.sampler import MLXAbSampler

    sampler = MLXAbSampler(conf)
    xT, seq_T = sampler.sample_init()
    for t in range(sampler.t_step_input, 0, -1):
        px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
            t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_T,
            final_step=sampler.inf_conf.final_step)
"""

import logging
import os

import torch
from omegaconf import DictConfig, OmegaConf

from rfantibody.rfdiffusion.inference.model_runners import AbSampler

_log = logging.getLogger(__name__)


class MLXAbSampler(AbSampler):
    """AbSampler that uses MLX for the model forward pass.

    Overrides only load_model() to build an MLX model instead of PyTorch.
    Everything else (preprocessing, denoising, diffusion) stays in PyTorch.
    """

    def load_model(self):
        """Create MLX RoseTTAFold model from preloaded PyTorch checkpoint."""
        from .model_wrapper import MLXModelWrapper

        self.d_t1d = self._conf.preprocess.d_t1d
        self.d_t2d = self._conf.preprocess.d_t2d
        self.use_selfcond_emb = self._conf.preprocess.use_selfcond_emb

        _log.info('Loading model with MLX backend')

        wrapper, ckpt = MLXModelWrapper.from_checkpoint(
            ckpt_path=self.ckpt_path,
            model_conf=self._conf.model,
            d_t1d=self.d_t1d,
            d_t2d=self.d_t2d,
            use_selfcond_emb=self.use_selfcond_emb,
            T=self._conf.diffuser.T,
            use_final_state=self._conf.inference.final_state,
            torch_device=self.device,
        )

        # Apply performance optimizations
        mlx_opts = getattr(self._conf, 'mlx', None)
        if mlx_opts is not None:
            if getattr(mlx_opts, 'mixed_precision', False):
                wrapper.enable_mixed_precision()
            topk = getattr(mlx_opts, 'topk_graph', 0)
            if topk > 0:
                wrapper.set_topk_graph(topk)

        _log.info('MLX model loaded successfully')

        return wrapper

    def sample_init(self):
        """Clear L-dependent caches before each design (lengths may vary)."""
        self.model.clear_caches()
        return super().sample_init()

    def initialize(self, conf: DictConfig):
        """Override to use CPU as the device (MLX handles GPU internally)."""
        self._log = logging.getLogger(__name__)

        # For MLX, we use CPU for torch tensors since MLX handles Metal internally
        self.device = torch.device('cpu')

        needs_model_reload = not getattr(self, 'initialized', False) or \
            conf.inference.ckpt_override_path != getattr(self, '_conf', conf).inference.ckpt_override_path

        self._conf = conf

        # Checkpoint path selection (same as parent)
        if conf.inference.ckpt_override_path is not None:
            self.ckpt_path = conf.inference.ckpt_override_path
        else:
            # Default paths (same as parent class)
            if conf.contigmap.inpaint_seq is not None or conf.contigmap.provide_seq is not None:
                if conf.scaffoldguided.scaffoldguided:
                    self.ckpt_path = '/net/databases/diffusion/models/seq_alone_models_FoldConditioned_Jan23/BFF_4.pt'
                else:
                    self.ckpt_path = '/net/databases/diffusion/models/seq_alone_models_Dec2022/BFF_6.pt'
            elif conf.ppi.hotspot_res is not None and conf.scaffoldguided.scaffoldguided is False:
                self.ckpt_path = '/net/databases/diffusion/models/hotspot_models/base_complex_finetuned_BFF_9.pt'
            elif conf.scaffoldguided.scaffoldguided is True:
                self.ckpt_path = '/net/databases/diffusion/models/hotspot_models/base_complex_ss_finetuned_BFF_9.pt'
            else:
                self.ckpt_path = '/net/databases/diffusion/nate_tmp/new_SelfCond_crdscale0.25/models/BFF_4.pt'

        assert self._conf.inference.trb_save_ckpt_path is None
        self._conf['inference']['trb_save_ckpt_path'] = self.ckpt_path

        if needs_model_reload:
            self.load_checkpoint()
            self.assemble_config_from_chk()
            self.model = self.load_model()
        else:
            self.assemble_config_from_chk()

        self.initialized = True

        # Continue with parent's remaining initialization
        self.inf_conf = self._conf.inference
        self.contig_conf = self._conf.contigmap
        self.denoiser_conf = self._conf.denoiser
        self.ppi_conf = self._conf.ppi
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        self.ab_conf = self._conf.antibody

        from rfantibody.rfdiffusion.diffusion import Diffuser
        self.diffuser = Diffuser(**self._conf.diffuser)

        if self._conf.seq_diffuser.seqdiff is None:
            self.seq_diffuser = None
            self.seq_self_cond = self._conf.preprocess.seq_self_cond
        else:
            raise NotImplementedError(
                f'MLX sampler does not support seq_diffuser type: {self._conf.seq_diffuser.seqdiff}')

        from rfantibody.rfdiffusion.inference import symmetry
        if self.inf_conf.symmetry is not None:
            self.symmetry = symmetry.SymGen(
                self.inf_conf.symmetry,
                self.inf_conf.model_only_neighbors,
                self.inf_conf.recenter,
                self.inf_conf.radius,
            )
        else:
            self.symmetry = None

        from rfantibody.rfdiffusion.util_module import ComputeAllAtomCoords
        self.allatom = ComputeAllAtomCoords().to(self.device)

        if not self.ab_design():
            if self.inf_conf.input_pdb is None:
                script_dir = os.path.dirname(os.path.realpath(__file__))
                self.inf_conf.input_pdb = os.path.join(
                    script_dir, '../../benchmark/input/1qys.pdb')
            from rfantibody.rfdiffusion.inference import utils as iu
            self.target_feats = iu.process_target(
                self.inf_conf.input_pdb, parse_hetatom=True, center=False)

        self.chain_idx = None
        self.prev_pred = None
        self.msa_prev = None

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)

        from rfantibody.rfdiffusion.inference import utils as iu
        recycle_schedule = str(self.inf_conf.recycle_schedule) \
            if self.inf_conf.recycle_schedule is not None else None
        self.recycle_schedule = iu.recycle_schedule(
            self.T, recycle_schedule, self.inf_conf.num_recycles)
