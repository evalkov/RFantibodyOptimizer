"""
MLX-accelerated AbPredictor for RF2 inference on Apple Silicon.

Replaces the PyTorch model forward pass with MLX via MLXRF2Wrapper.
Recycling loop with Cα RMSD convergence stays in PyTorch.
"""
from __future__ import annotations

import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from rfantibody.rf2.modules.model_runner import AbPredictor, write_output, get_rmsds
import rfantibody.rf2.modules.pose_util as pu
from rfantibody.rf2.network.predict import Predictor, pae_unbin
from rfantibody.rf2.network.util_module import XYZConverter

_log = logging.getLogger(__name__)


class MLXAbPredictor(AbPredictor):
    """AbPredictor with MLX model backend.

    Overrides model loading to use MLXRF2Wrapper instead of PyTorch
    RoseTTAFoldModule. Everything else (preprocessing, recycling loop,
    post-processing) stays in PyTorch.
    """

    def __init__(self, conf, preprocess_fn, device='cpu'):
        # Don't call super().__init__() which would load PyTorch model
        # Instead, set up manually with MLX model
        self.conf = conf
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.return_rmsds = any([
            var is not None for var in
            [conf.input.pdb, conf.input.pdb_dir, conf.input.quiver]
        ])

        # Load MLX model
        from rfantibody.rf2.mlx.model_wrapper import MLXRF2Wrapper
        model_weights = conf.model.model_weights
        if not os.path.exists(model_weights):
            model_weights = os.path.join(
                os.path.dirname(__file__), '..', 'network', model_weights)

        _log.info(f'Loading RF2 MLX model from {model_weights}')
        self.model = MLXRF2Wrapper.from_checkpoint(
            model_weights, torch_device=device)

        # Apply performance optimizations
        self.model.enable_mixed_precision()
        self.model.set_topk_graph(int(os.environ.get('RF2_TOPK', '64')))
        rf2_eval_stride = int(os.environ.get('RF2_EVAL_STRIDE', '8'))
        self.model.set_eval_stride(rf2_eval_stride)
        rf2_se3_stride = int(os.environ.get('RF2_SE3_STRIDE', '1'))
        if rf2_se3_stride > 1:
            self.model.set_se3_stride(rf2_se3_stride)
        rf2_n_main = int(os.environ.get('RF2_N_MAIN', '0'))
        if rf2_n_main > 0:
            self.model.set_n_main_block(rf2_n_main)
        self.model.enable_fused_kernels()
        _log.info(f'RF2 optimizations: fp16 pair, top_k={os.environ.get("RF2_TOPK", "64")}, '
                   f'eval_stride={rf2_eval_stride}, se3_stride={rf2_se3_stride}, '
                   f'n_main={rf2_n_main or "all"}, fused SE3')

        # Set up utilities from Predictor base class
        from rfantibody.rf2.network import util
        self.l2a = util.long2alt.to(self.device)
        self.aamask = util.allatom_mask.to(self.device)
        self.lddt_bins = torch.linspace(1.0 / 50, 1.0, 50) - 1.0 / 100
        self.xyz_converter = XYZConverter()
        self.xyz_converter.to(self.device)
        self.active_fn = nn.Softmax(dim=1)

        _log.info('MLXAbPredictor initialized successfully')
