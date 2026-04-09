"""
MLX model wrapper for Protenix-Mini-Flow inference.

Provides a drop-in replacement that accepts torch tensors, runs the MLX
forward pass, and returns torch tensors -- matching the expected API
for integration with PyTorch-based pipelines.

Follows the same pattern as rfantibody.rf2.mlx.model_wrapper.MLXRF2Wrapper.
"""
from __future__ import annotations

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
    """Convert an MLX array (or dict/tuple/list of arrays) to PyTorch tensor(s)."""
    import torch
    if m is None:
        return None
    if isinstance(m, dict):
        return {k: _mlx_to_torch(v, device) for k, v in m.items()}
    if isinstance(m, (tuple, list)):
        return type(m)(_mlx_to_torch(x, device) for x in m)
    arr = np.array(m)
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device)
    return t


class MLXProtenixWrapper:
    """Drop-in replacement for Protenix model with MLX backend.

    Accepts PyTorch tensors, converts to MLX, runs the ProtenixMiniModule,
    and converts results back to PyTorch tensors.

    Usage:
        wrapper = MLXProtenixWrapper.from_checkpoint(ckpt_path)
        result = wrapper(seq=seq_tensor, residue_index=idx, chain_id=chains)
        coords = result["coordinates"]  # PyTorch tensor
        plddt = result["plddt"]         # PyTorch tensor
    """

    def __init__(self, mlx_model, torch_device=None):
        self.model = mlx_model
        self.torch_device = torch_device or 'cpu'

    @classmethod
    def from_checkpoint(cls, ckpt_path, model_params=None, torch_device=None):
        """Load model from a Protenix checkpoint.

        Args:
            ckpt_path: path to .pt checkpoint file
            model_params: optional dict of model hyperparameters
            torch_device: PyTorch device for output tensors

        Returns:
            MLXProtenixWrapper instance
        """
        from .weight_converter import load_checkpoint_to_mlx
        model, _ckpt = load_checkpoint_to_mlx(ckpt_path, model_params=model_params)
        return cls(model, torch_device=torch_device)

    def __call__(
        self,
        seq=None,
        residue_index=None,
        chain_id=None,
        entity_id=None,
        token_index=None,
        token_bonds=None,
        atom_mask=None,
        profile=None,
        deletion_mean=None,
        esm_embeddings=None,
        n_cycle=1,
        run_confidence=True,
        coordinates_override=None,
        ref_pos=None,
    ):
        """Forward pass: converts torch tensors to MLX, runs model, converts back.

        Args:
            seq: [B, N] integer sequence tokens
            residue_index: [B, N] residue indices
            chain_id: [B, N] chain IDs
            entity_id: [B, N] entity IDs (optional)
            token_index: [B, N] token indices (optional)
            token_bonds: [B, N, N] token bonds (optional)
            atom_mask: [B, N] atom mask (optional)
            profile: [B, N, 32] MSA profile (optional)
            deletion_mean: [B, N, 1] deletion mean (optional)
            esm_embeddings: [B, N, esm_dim] ESM embeddings (optional)
            n_cycle: recycling cycles (default 1)
            run_confidence: compute confidence metrics (default True)
            coordinates_override: [B, N, 3] skip diffusion, use these coords

        Returns:
            dict of PyTorch tensors with keys:
                coordinates, plddt, pae_logits, pde_logits, ptm, iptm, ...
        """
        import mlx.core as mx

        # Convert inputs to MLX
        mx_seq = _torch_to_mlx(seq)
        if mx_seq is not None:
            mx_seq = mx_seq.astype(mx.int32)
        mx_residue_index = _torch_to_mlx(residue_index)
        if mx_residue_index is not None:
            mx_residue_index = mx_residue_index.astype(mx.int32)
        mx_chain_id = _torch_to_mlx(chain_id)
        if mx_chain_id is not None:
            mx_chain_id = mx_chain_id.astype(mx.int32)
        mx_entity_id = _torch_to_mlx(entity_id)
        if mx_entity_id is not None:
            mx_entity_id = mx_entity_id.astype(mx.int32)
        mx_token_index = _torch_to_mlx(token_index)
        mx_token_bonds = _torch_to_mlx(token_bonds)
        mx_atom_mask = _torch_to_mlx(atom_mask)
        mx_profile = _torch_to_mlx(profile)
        mx_deletion_mean = _torch_to_mlx(deletion_mean)
        mx_esm = _torch_to_mlx(esm_embeddings)
        mx_coords_override = _torch_to_mlx(coordinates_override)
        mx_ref_pos = _torch_to_mlx(ref_pos)

        # Run MLX model
        result = self.model(
            seq=mx_seq,
            residue_index=mx_residue_index,
            chain_id=mx_chain_id,
            entity_id=mx_entity_id,
            token_index=mx_token_index,
            token_bonds=mx_token_bonds,
            atom_mask=mx_atom_mask,
            profile=mx_profile,
            deletion_mean=mx_deletion_mean,
            esm_embeddings=mx_esm,
            n_cycle=n_cycle,
            coordinates_override=mx_coords_override,
            ref_pos=mx_ref_pos,
            run_confidence=run_confidence,
        )

        # Evaluate all outputs
        mx.eval(*[v for v in result.values() if v is not None])

        # Convert back to PyTorch
        return _mlx_to_torch(result, self.torch_device)

    def enable_mixed_precision(self):
        """Enable float16 for bandwidth-bound layers."""
        self.model.enable_mixed_precision()
        _log.info('Protenix MLX mixed precision enabled')

    def set_eval_stride(self, stride: int):
        """Set mx.eval() frequency in Pairformer trunk."""
        self.model.set_eval_stride(stride)
        _log.info(f'Protenix MLX eval stride: {stride}')

    def set_n_pairformer_blocks(self, n: int):
        """Reduce active Pairformer blocks for faster inference."""
        self.model.set_n_pairformer_blocks(n)
        _log.info(f'Protenix MLX pairformer blocks: {n}')

    def set_n_diffusion_steps(self, n: int):
        """Set number of ODE diffusion steps."""
        self.model.set_n_diffusion_steps(n)
        _log.info(f'Protenix MLX diffusion steps: {n}')

    def enable_tea_cache(self, threshold: float = 0.15):
        """Enable TeaCache for diffusion acceleration."""
        self.model.enable_tea_cache(threshold)
        _log.info(f'Protenix MLX TeaCache enabled: threshold={threshold}')

    def disable_tea_cache(self):
        """Disable TeaCache."""
        self.model.disable_tea_cache()
        _log.info('Protenix MLX TeaCache disabled')

    def eval(self):
        """No-op for API compatibility with PyTorch .eval()."""
        return self

    def to(self, device):
        """Set output PyTorch device (no-op for MLX computation)."""
        if hasattr(device, 'type'):
            self.torch_device = device
        else:
            import torch
            self.torch_device = torch.device(device)
        return self
