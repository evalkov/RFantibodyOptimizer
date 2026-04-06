"""
MLX model wrapper for RFdiffusion inference.

Provides a drop-in replacement for the PyTorch RoseTTAFoldModule that:
  1. Converts torch tensor inputs → mlx arrays
  2. Runs the MLX forward pass
  3. Converts mlx outputs → torch tensors

This allows the existing PyTorch AbSampler/Denoise infrastructure to use
the MLX model without any changes to the denoising or preprocessing code.
"""

import logging
import time

import numpy as np

_log = logging.getLogger(__name__)


def _torch_to_mlx(t):
    """Convert a PyTorch tensor to an MLX array via numpy."""
    import mlx.core as mx
    if t is None:
        return None
    if hasattr(t, 'numpy'):
        try:
            arr = t.detach().cpu().numpy()
        except RuntimeError:
            arr = t.cpu().numpy()
    elif isinstance(t, np.ndarray):
        arr = t
    else:
        return t  # pass through non-tensor values
    return mx.array(arr)


def _mlx_to_torch(m, device=None):
    """Convert an MLX array to a PyTorch tensor via numpy."""
    import torch
    if m is None:
        return None
    arr = np.array(m)
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device)
    return t


class MLXModelWrapper:
    """Drop-in replacement for PyTorch RoseTTAFoldModule.

    Usage:
        wrapper = MLXModelWrapper.from_checkpoint(ckpt_path, conf)
        # Then use exactly like the PyTorch model:
        msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = wrapper(
            msa_masked, msa_full, seq_in, px0, idx_pdb,
            t1d=t1d, t2d=t2d, ..., return_infer=True, ...)
    """

    def __init__(self, mlx_model, torch_device=None):
        """
        Args:
            mlx_model: An MLX RoseTTAFoldModule with loaded weights.
            torch_device: Device to place output torch tensors on (default: cpu).
        """
        self.model = mlx_model
        self.torch_device = torch_device or 'cpu'
        self._mlx_msa_prev = None  # cached MLX array to avoid bridge round-trip
        self._mlx_idx_cache = None       # static: doesn't change between steps
        self._mlx_motif_mask_cache = None # static: doesn't change between steps

    def clear_caches(self):
        """Clear L-dependent caches. Must be called between designs of different lengths."""
        self._mlx_msa_prev = None
        self._mlx_idx_cache = None
        self._mlx_motif_mask_cache = None

    @classmethod
    def from_checkpoint(cls, ckpt_path, model_conf, d_t1d, d_t2d,
                        use_selfcond_emb=False, T=50,
                        use_final_state=False, torch_device=None):
        """Build MLX model from a PyTorch checkpoint.

        Args:
            ckpt_path: Path to PyTorch .pt checkpoint.
            model_conf: OmegaConf model config (from checkpoint's config_dict).
            d_t1d, d_t2d: Template feature dimensions.
            use_selfcond_emb: Whether the model uses self-conditioning embeddings.
            T: Number of diffusion timesteps.
            use_final_state: Load from 'final_state_dict' instead of 'model_state_dict'.
            torch_device: Device for output torch tensors.

        Returns:
            MLXModelWrapper instance.
        """
        from .model import RoseTTAFoldModule
        from .weight_converter import load_checkpoint_to_mlx

        # Build MLX model with same architecture
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(model_conf, DictConfig):
                model_kwargs = OmegaConf.to_container(model_conf, resolve=True)
            else:
                model_kwargs = dict(model_conf)
        except ImportError:
            model_kwargs = dict(model_conf)

        model_kwargs['d_t1d'] = d_t1d
        model_kwargs['d_t2d'] = d_t2d
        model_kwargs['use_selfcond_emb'] = use_selfcond_emb
        model_kwargs['T'] = T

        _log.info('Building MLX RoseTTAFoldModule')
        mlx_model = RoseTTAFoldModule(**model_kwargs)

        # Load weights
        mlx_model, ckpt = load_checkpoint_to_mlx(
            ckpt_path, mlx_model, use_final_state=use_final_state)

        _log.info('MLX model ready')
        return cls(mlx_model, torch_device=torch_device), ckpt

    def __call__(self, msa_latent, msa_full, seq, xyz, idx, t,
                 t1d=None, t2d=None, xyz_t=None, alpha_t=None,
                 sc2d=None, xyz_sc=None,
                 msa_prev=None, pair_prev=None, state_prev=None,
                 return_raw=False, return_full=False, return_infer=False,
                 return_w_msa_prev=False,
                 use_checkpoint=False, motif_mask=None,
                 i_cycle=None, n_cycle=None):
        """Forward pass matching PyTorch RoseTTAFoldModule signature.

        Accepts torch tensors, returns torch tensors.
        """
        import mlx.core as mx

        tick = time.time()

        # Convert inputs to MLX (cache static tensors)
        mx_msa_latent = _torch_to_mlx(msa_latent)
        mx_msa_full = _torch_to_mlx(msa_full)
        mx_seq = _torch_to_mlx(seq)
        mx_xyz = _torch_to_mlx(xyz)
        mx_t = _torch_to_mlx(t)
        mx_t1d = _torch_to_mlx(t1d)
        mx_t2d = _torch_to_mlx(t2d)
        mx_xyz_t = _torch_to_mlx(xyz_t)
        mx_alpha_t = _torch_to_mlx(alpha_t)
        mx_sc2d = _torch_to_mlx(sc2d)
        mx_xyz_sc = _torch_to_mlx(xyz_sc)
        # Cache static tensors that don't change between steps
        if idx is not None and self._mlx_idx_cache is None:
            self._mlx_idx_cache = _torch_to_mlx(idx)
        mx_idx = self._mlx_idx_cache if self._mlx_idx_cache is not None else _torch_to_mlx(idx)
        if motif_mask is not None and self._mlx_motif_mask_cache is None:
            self._mlx_motif_mask_cache = _torch_to_mlx(motif_mask)
        mx_motif_mask = self._mlx_motif_mask_cache if self._mlx_motif_mask_cache is not None else _torch_to_mlx(motif_mask)
        # Use cached MLX msa_prev if available (avoids numpy round-trip)
        if msa_prev is not None and self._mlx_msa_prev is not None:
            mx_msa_prev = self._mlx_msa_prev
        else:
            mx_msa_prev = _torch_to_mlx(msa_prev)
        mx_pair_prev = _torch_to_mlx(pair_prev)
        mx_state_prev = _torch_to_mlx(state_prev)

        # Run MLX model
        result = self.model(
            mx_msa_latent, mx_msa_full, mx_seq, mx_xyz, mx_idx, mx_t,
            t1d=mx_t1d, t2d=mx_t2d, xyz_t=mx_xyz_t, alpha_t=mx_alpha_t,
            sc2d=mx_sc2d, xyz_sc=mx_xyz_sc,
            msa_prev=mx_msa_prev, pair_prev=mx_pair_prev,
            state_prev=mx_state_prev,
            return_raw=return_raw, return_full=return_full,
            return_infer=return_infer,
            return_w_msa_prev=return_w_msa_prev,
            use_checkpoint=use_checkpoint, motif_mask=mx_motif_mask,
            i_cycle=i_cycle, n_cycle=n_cycle)

        # Ensure computation is complete
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)

        elapsed = time.time() - tick
        _log.info(f'MLX forward pass: {elapsed:.2f}s')

        # Convert outputs back to torch
        if isinstance(result, tuple):
            converted = []
            for i, r in enumerate(result):
                if return_infer and i == 0:
                    # Cache msa_prev in MLX to avoid round-trip next call
                    self._mlx_msa_prev = r
                if return_infer and i == 1:
                    # pair_prev is never used by caller (always passes None)
                    converted.append(None)
                else:
                    converted.append(_mlx_to_torch(r, self.torch_device))
            return tuple(converted)
        return _mlx_to_torch(result, self.torch_device)

    def enable_mixed_precision(self):
        """Enable float16 for bandwidth-bound layers (~8% speedup)."""
        self.model.enable_mixed_precision()
        _log.info('MLX mixed precision enabled')

    def set_topk_graph(self, top_k: int):
        """Use k-NN graph instead of full graph for SE3 (~2x speedup).

        Args:
            top_k: Number of nearest neighbors. 0 = full graph (default).
        """
        self.model.set_topk_graph(top_k)
        _log.info(f'MLX top-k graph override: {top_k}')

    def set_se3_stride(self, stride: int):
        """Skip SE3 in alternating main blocks (~25% speedup at stride=2).

        Args:
            stride: Run SE3 every N blocks. 1 = every block (default).
        """
        self.model.set_se3_stride(stride)
        _log.info(f'MLX SE3 stride: {stride}')

    def set_n_main_block(self, n: int):
        """Reduce main blocks for faster inference.

        Args:
            n: Number of main blocks to run (default: 32).
        """
        self.model.set_n_main_block(n)
        _log.info(f'MLX main blocks: {n}')

    def set_eval_stride(self, stride: int):
        """Reduce mx.eval() frequency for better kernel fusion.

        Instead of evaluating after every block, evaluate every N blocks.
        Inspired by SimpleFold's single-eval-per-step pattern.
        Higher stride = more fusion opportunities but more peak memory.

        Args:
            stride: Eval every N blocks. 1 = every block (default), 4 = every 4th.
        """
        self.model.simulator.eval_stride = stride
        _log.info(f'MLX eval stride: {stride}')

    def enable_fused_kernels(self):
        """Enable fused Metal kernels for SE3 convolutions."""
        self.model.enable_fused_kernels()
        _log.info('MLX fused Metal kernels enabled')

    def compile_model(self):
        """Apply mx.compile() to hot-path modules for automatic kernel fusion.

        Must be called AFTER enable_mixed_precision() and set_topk_graph()
        since mx.compile caches Python control flow on first trace.
        """
        import mlx.core as mx
        compiled = 0
        for block_list in [self.model.simulator.extra_block,
                           self.model.simulator.main_block]:
            for block in block_list:
                block.pair2pair.__call__ = mx.compile(block.pair2pair.__call__)
                block.msa2msa.__call__ = mx.compile(block.msa2msa.__call__)
                block.str2str.__call__ = mx.compile(block.str2str.__call__)
                compiled += 1
        _log.info(f'MLX compile: {compiled} IterBlocks compiled')

    def eval(self):
        """No-op for API compatibility with PyTorch model.eval()."""
        return self

    def to(self, device):
        """Store the target device for output conversion."""
        if hasattr(device, 'type'):
            self.torch_device = device
        else:
            import torch
            self.torch_device = torch.device(device)
        return self

    def parameters(self):
        """Return MLX model parameters (for inspection/debugging)."""
        return self.model.parameters()
