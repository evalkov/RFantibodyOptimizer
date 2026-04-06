"""
Convert PyTorch ProteinMPNN checkpoint to MLX model weights.

MPNN has no nn.Sequential containers, so key mapping is straightforward.
Only nn.ModuleList needs .N -> .N (Python list), which works directly.
"""
import logging

import numpy as np

_log = logging.getLogger(__name__)


def map_torch_to_mlx(key: str, value) -> tuple:
    """Remap a single PyTorch state_dict key to MLX conventions."""
    # MPNN uses nn.ModuleList → Python list: encoder_layers.0.W1.weight works as-is
    # No nn.Sequential containers to transform.
    if hasattr(value, 'numpy'):
        try:
            arr = value.numpy()
        except RuntimeError:
            arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.array(value)
    return key, arr


def convert_state_dict(torch_state_dict: dict) -> dict:
    mlx_dict = {}
    for key, value in torch_state_dict.items():
        mlx_key, arr = map_torch_to_mlx(key, value)
        mlx_dict[mlx_key] = arr
    return mlx_dict


def _set_params_from_flat(model, flat_weights):
    """Set model parameters from a flat dict of {dotted_key: mx.array}."""
    import mlx.nn as nn

    for flat_key, value in flat_weights.items():
        parts = flat_key.split('.')
        obj = model
        for i, part in enumerate(parts[:-1]):
            if isinstance(obj, nn.Module):
                obj = getattr(obj, part)
            elif isinstance(obj, dict):
                obj = obj[part]
            elif isinstance(obj, list):
                obj = obj[int(part)]
            else:
                raise ValueError(
                    f"Cannot navigate through {type(obj)} at "
                    f"'{'.'.join(parts[:i+1])}' in key '{flat_key}'")

        leaf = parts[-1]
        if isinstance(obj, nn.Module):
            setattr(obj, leaf, value)
        elif isinstance(obj, dict):
            obj[leaf] = value
        elif isinstance(obj, list):
            obj[int(leaf)] = value
        else:
            raise ValueError(
                f"Cannot set leaf on {type(obj)} for key '{flat_key}'")


def load_checkpoint_to_mlx(ckpt_path: str, hidden_dim=128, num_layers=3,
                            k_neighbors=48, augment_eps=0.0):
    """Load a PyTorch MPNN checkpoint into an MLX ProteinMPNN model.

    Returns: (mlx_model, checkpoint_dict)
    """
    import torch
    import mlx.core as mx
    import mlx.nn as nn
    from .model import ProteinMPNN

    _log.info(f'Loading MPNN checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Extract config from checkpoint
    num_edges = ckpt.get('num_edges', k_neighbors)
    noise_level = ckpt.get('noise_level', 0.0)
    _log.info(f'MPNN config: num_edges={num_edges}, noise_level={noise_level}')

    # Build MLX model
    model = ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        k_neighbors=num_edges,
        augment_eps=augment_eps,
        dropout=0.1,
    )

    # Convert and load weights
    torch_sd = ckpt['model_state_dict']
    _log.info(f'Converting {len(torch_sd)} parameters to MLX format')
    mlx_sd = convert_state_dict(torch_sd)

    # Validate
    model_params = dict(nn.utils.tree_flatten(model.parameters()))
    model_keys = set(model_params.keys())
    ckpt_keys = set(mlx_sd.keys())

    missing_in_model = ckpt_keys - model_keys
    missing_in_ckpt = model_keys - ckpt_keys

    if missing_in_model:
        _log.warning(f'{len(missing_in_model)} checkpoint keys not in model:')
        for k in sorted(missing_in_model)[:10]:
            _log.warning(f'  {k}')
    if missing_in_ckpt:
        _log.warning(f'{len(missing_in_ckpt)} model keys not in checkpoint:')
        for k in sorted(missing_in_ckpt)[:10]:
            _log.warning(f'  {k}')

    # Convert to mx.array and load
    weights = {k: mx.array(v) for k, v in mlx_sd.items()}
    _set_params_from_flat(model, weights)
    mx.eval(model.parameters())

    _log.info(f'MLX MPNN model loaded ({len(model_params)} params)')
    return model, ckpt
