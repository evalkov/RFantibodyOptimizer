"""
MLX model wrapper for ProteinMPNN inference.

Provides a drop-in replacement that accepts torch tensors, runs MLX forward,
and returns torch tensors — matching the PyTorch ProteinMPNN API exactly.
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
    """Convert an MLX array to a PyTorch tensor via numpy."""
    import torch
    if m is None:
        return None
    arr = np.array(m)
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device)
    return t


class MLXMPNNWrapper:
    """Drop-in replacement for PyTorch ProteinMPNN.

    Usage:
        wrapper = MLXMPNNWrapper.from_checkpoint(ckpt_path)
        log_probs = wrapper(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
        sample_dict = wrapper.sample(X, randn, S_true, ...)
    """

    def __init__(self, mlx_model, torch_device=None):
        self.model = mlx_model
        self.torch_device = torch_device or 'cpu'

    @classmethod
    def from_checkpoint(cls, ckpt_path, hidden_dim=128, num_layers=3,
                        k_neighbors=48, augment_eps=0.0, torch_device=None):
        from .weight_converter import load_checkpoint_to_mlx
        model, ckpt = load_checkpoint_to_mlx(
            ckpt_path, hidden_dim=hidden_dim, num_layers=num_layers,
            k_neighbors=k_neighbors, augment_eps=augment_eps)
        # Disable augmentation at inference
        model.features.augment_eps = 0.0
        return cls(model, torch_device=torch_device)

    def __call__(self, X, S, mask, chain_M, residue_idx, chain_encoding_all,
                 randn, use_input_decoding_order=False, decoding_order=None):
        """Forward pass: returns log_probs as a torch tensor."""
        import mlx.core as mx

        mx_X = _torch_to_mlx(X)
        mx_S = _torch_to_mlx(S).astype(mx.int32)
        mx_mask = _torch_to_mlx(mask)
        mx_chain_M = _torch_to_mlx(chain_M)
        mx_residue_idx = _torch_to_mlx(residue_idx)
        mx_chain_enc = _torch_to_mlx(chain_encoding_all)
        mx_randn = _torch_to_mlx(randn)

        log_probs = self.model(
            mx_X, mx_S, mx_mask, mx_chain_M, mx_residue_idx,
            mx_chain_enc, mx_randn,
            use_input_decoding_order=use_input_decoding_order,
            decoding_order=_torch_to_mlx(decoding_order) if decoding_order is not None else None)

        mx.eval(log_probs)
        return _mlx_to_torch(log_probs, self.torch_device)

    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all,
               residue_idx, mask=None, temperature=1.0,
               omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None,
               omit_AA_mask=None, pssm_coef=None, pssm_bias=None,
               pssm_multi=None, pssm_log_odds_flag=None,
               pssm_log_odds_mask=None, pssm_bias_flag=None,
               bias_by_res=None):
        """Autoregressive sampling: returns dict with 'S', 'probs', 'decoding_order'."""
        import mlx.core as mx

        result = self.model.sample(
            _torch_to_mlx(X),
            _torch_to_mlx(randn),
            _torch_to_mlx(S_true).astype(mx.int32),
            _torch_to_mlx(chain_mask),
            _torch_to_mlx(chain_encoding_all),
            _torch_to_mlx(residue_idx),
            mask=_torch_to_mlx(mask),
            temperature=temperature,
            omit_AAs_np=omit_AAs_np,
            bias_AAs_np=bias_AAs_np,
            chain_M_pos=_torch_to_mlx(chain_M_pos),
            omit_AA_mask=_torch_to_mlx(omit_AA_mask) if omit_AA_mask is not None else None,
            pssm_coef=_torch_to_mlx(pssm_coef) if pssm_coef is not None else None,
            pssm_bias=_torch_to_mlx(pssm_bias) if pssm_bias is not None else None,
            pssm_multi=pssm_multi,
            pssm_log_odds_flag=pssm_log_odds_flag,
            pssm_log_odds_mask=_torch_to_mlx(pssm_log_odds_mask) if pssm_log_odds_mask is not None else None,
            pssm_bias_flag=pssm_bias_flag,
            bias_by_res=_torch_to_mlx(bias_by_res) if bias_by_res is not None else None,
        )

        mx.eval(result['S'], result['probs'], result['decoding_order'])

        return {
            'S': _mlx_to_torch(result['S'], self.torch_device).long(),
            'probs': _mlx_to_torch(result['probs'], self.torch_device),
            'decoding_order': _mlx_to_torch(result['decoding_order'], self.torch_device),
        }

    def eval(self):
        return self

    def to(self, device):
        if hasattr(device, 'type'):
            self.torch_device = device
        else:
            import torch
            self.torch_device = torch.device(device)
        return self
