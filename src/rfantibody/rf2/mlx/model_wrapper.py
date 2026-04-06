"""
MLX model wrapper for RF2 inference.

Provides a drop-in replacement that accepts torch tensors, runs MLX forward,
and returns torch tensors — matching the PyTorch RoseTTAFoldModule API.
"""
import logging

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
        return t
    return mx.array(arr)


def _mlx_to_torch(m, device=None):
    """Convert an MLX array (or tuple/list of arrays) to PyTorch tensor(s)."""
    import torch
    if m is None:
        return None
    if isinstance(m, (tuple, list)):
        return type(m)(_mlx_to_torch(x, device) for x in m)
    arr = np.array(m)
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device)
    return t


class MLXRF2Wrapper:
    """Drop-in replacement for PyTorch RoseTTAFoldModule.

    Usage:
        wrapper = MLXRF2Wrapper.from_checkpoint(ckpt_path)
        outputs = wrapper(msa_latent=..., msa_full=..., seq=..., ...)
    """

    def __init__(self, mlx_model, torch_device=None):
        self.model = mlx_model
        self.torch_device = torch_device or 'cpu'

    @classmethod
    def from_checkpoint(cls, ckpt_path, model_params=None, torch_device=None):
        from .weight_converter import load_checkpoint_to_mlx
        model, ckpt = load_checkpoint_to_mlx(ckpt_path, model_params=model_params)
        return cls(model, torch_device=torch_device)

    def __call__(
        self,
        msa_latent=None,
        msa_full=None,
        seq=None,
        xyz_prev=None,
        idx=None,
        t1d=None,
        t2d=None,
        xyz_t=None,
        alpha_t=None,
        mask_t=None,
        same_chain=None,
        msa_prev=None,
        pair_prev=None,
        state_prev=None,
        mask_recycle=None,
        return_raw=False,
        return_full=False,
        use_checkpoint=False,
        p2p_crop=-1,
        topk_crop=-1,
        nc_cycle=False,
        symmids=None,
        symmsub=None,
        symmRs=None,
        symmmeta=None,
        striping=None,
        low_vram=False,
    ):
        """Forward pass: converts torch tensors to MLX, runs model, converts back."""
        import mlx.core as mx

        # Convert all inputs to MLX
        mx_msa_latent = _torch_to_mlx(msa_latent)
        mx_msa_full = _torch_to_mlx(msa_full)
        mx_seq = _torch_to_mlx(seq)
        if mx_seq is not None:
            mx_seq = mx_seq.astype(mx.int32)
        mx_xyz_prev = _torch_to_mlx(xyz_prev)
        mx_idx = _torch_to_mlx(idx)
        mx_t1d = _torch_to_mlx(t1d)
        mx_t2d = _torch_to_mlx(t2d)
        mx_xyz_t = _torch_to_mlx(xyz_t)
        mx_alpha_t = _torch_to_mlx(alpha_t)
        mx_mask_t = _torch_to_mlx(mask_t)
        mx_same_chain = _torch_to_mlx(same_chain)
        mx_msa_prev = _torch_to_mlx(msa_prev)
        mx_pair_prev = _torch_to_mlx(pair_prev)
        mx_state_prev = _torch_to_mlx(state_prev)
        mx_mask_recycle = _torch_to_mlx(mask_recycle)

        # Run MLX model
        result = self.model(
            msa_latent=mx_msa_latent,
            msa_full=mx_msa_full,
            seq=mx_seq,
            xyz_prev=mx_xyz_prev,
            idx=mx_idx,
            t1d=mx_t1d,
            t2d=mx_t2d,
            xyz_t=mx_xyz_t,
            alpha_t=mx_alpha_t,
            mask_t=mx_mask_t,
            same_chain=mx_same_chain,
            msa_prev=mx_msa_prev,
            pair_prev=mx_pair_prev,
            state_prev=mx_state_prev,
            mask_recycle=mx_mask_recycle,
            return_raw=return_raw,
            nc_cycle=nc_cycle,
            topk_crop=topk_crop,
        )

        # Evaluate and convert back to torch
        mx.eval(*[r for r in result if r is not None])

        if return_raw:
            # (msa_out, pair, state, xyz_last, alpha_last, None)
            return tuple(
                _mlx_to_torch(r, self.torch_device) if r is not None else None
                for r in result
            )
        else:
            # (logits, logits_aa, logits_exp, logits_pae, p_bind,
            #  xyz, alpha, symmsub, lddt, msa_out, pair, state)
            return tuple(
                _mlx_to_torch(r, self.torch_device) if r is not None else None
                for r in result
            )

    def enable_mixed_precision(self):
        """Enable float16 for bandwidth-bound layers."""
        self.model.enable_mixed_precision()
        _log.info('RF2 MLX mixed precision enabled')

    def set_topk_graph(self, top_k: int):
        """Use k-NN graph instead of full graph for SE3."""
        self.model.set_topk_graph(top_k)
        _log.info(f'RF2 MLX top-k graph: {top_k}')

    def set_se3_stride(self, stride: int):
        """Skip SE3 in alternating main blocks."""
        self.model.set_se3_stride(stride)
        _log.info(f'RF2 MLX SE3 stride: {stride}')

    def set_eval_stride(self, stride: int):
        """Reduce mx.eval() frequency for better kernel fusion."""
        self.model.simulator.eval_stride = stride
        _log.info(f'RF2 MLX eval stride: {stride}')

    def enable_fused_kernels(self):
        """Enable fused Metal kernels for SE3 convolutions."""
        self.model.enable_fused_kernels()
        _log.info('RF2 MLX fused Metal kernels enabled')

    def eval(self):
        return self

    def to(self, device):
        if hasattr(device, 'type'):
            self.torch_device = device
        else:
            import torch
            self.torch_device = torch.device(device)
        return self
