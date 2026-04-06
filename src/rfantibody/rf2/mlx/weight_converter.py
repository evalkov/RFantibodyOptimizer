"""
Convert PyTorch RF2 checkpoint to MLX format.

Reuses the converter pattern from rfantibody.rfdiffusion.mlx.weight_converter.
RF2 uses the same SE3 architecture (with graph_modules/net Sequential containers)
so the same key mapping applies.
"""
import logging

import numpy as np

_log = logging.getLogger(__name__)

# Sequential containers that need .layers. inserted
# Same as RFdiffusion: SE3 uses graph_modules and net
_SEQUENTIAL_PARENTS = {
    'net',              # RadialProfile MLP in SE3 convolution
    'graph_modules',    # GraphSequential in SE3Transformer
}


def map_torch_to_mlx(key: str, value) -> tuple:
    """Remap a single PyTorch state_dict key to MLX conventions."""
    parts = key.split('.')
    new_parts = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p in _SEQUENTIAL_PARENTS and i + 1 < len(parts) and parts[i + 1].isdigit():
            new_parts.append(p)
            new_parts.append('layers')
            new_parts.append(parts[i + 1])
            i += 2
        else:
            new_parts.append(p)
            i += 1

    new_key = '.'.join(new_parts)

    if hasattr(value, 'numpy'):
        try:
            arr = value.numpy()
        except RuntimeError:
            arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.array(value)

    return new_key, arr


def convert_state_dict(torch_state_dict: dict) -> dict:
    """Convert entire PyTorch state_dict to MLX-compatible format."""
    mlx_dict = {}
    for key, value in torch_state_dict.items():
        mlx_key, arr = map_torch_to_mlx(key, value)
        mlx_dict[mlx_key] = arr
    return mlx_dict


def _set_params_from_flat(model, flat_weights):
    """Set model parameters from flat dict, navigating lists/dicts correctly."""
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


def load_checkpoint_to_mlx(ckpt_path: str, model_params=None):
    """Load RF2 PyTorch checkpoint into MLX model.

    Args:
        ckpt_path: Path to RF2 checkpoint (.pt file)
        model_params: Optional dict of model parameters. If None, uses
            default RF2 MODEL_PARAM + SE3 params from predict.py.

    Returns:
        (model, ckpt) — loaded MLX RoseTTAFoldModule and raw checkpoint
    """
    import torch
    import mlx.core as mx
    import mlx.nn as nn
    from rfantibody.rf2.mlx.model import RoseTTAFoldModule

    _log.info(f'Loading RF2 checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state_dict'. "
                       f"Available keys: {list(ckpt.keys())}")

    # Default RF2 model parameters (from predict.py)
    if model_params is None:
        SE3_param_full = {
            'num_layers': 1, 'num_channels': 48, 'num_degrees': 2,
            'l0_in_features': 32, 'l0_out_features': 32,
            'l1_in_features': 2, 'l1_out_features': 2,
            'num_edge_features': 32, 'div': 4, 'n_heads': 4,
        }
        SE3_param_topk = {
            'num_layers': 1, 'num_channels': 128, 'num_degrees': 2,
            'l0_in_features': 64, 'l0_out_features': 64,
            'l1_in_features': 2, 'l1_out_features': 2,
            'num_edge_features': 64, 'div': 4, 'n_heads': 4,
        }
        model_params = {
            'n_extra_block': 4,
            'n_main_block': 36,
            'n_ref_block': 4,
            'd_msa': 256,
            'd_msa_full': 64,
            'd_pair': 128,
            'd_templ': 64,
            'n_head_msa': 8,
            'n_head_pair': 4,
            'n_head_templ': 4,
            'd_hidden': 32,
            'd_hidden_templ': 32,
            'p_drop': 0.0,
            'd_t1d': 23,
            'd_t2d': 44,
            'SE3_param_full': SE3_param_full,
            'SE3_param_topk': SE3_param_topk,
        }

    model = RoseTTAFoldModule(**model_params)

    torch_sd = ckpt['model_state_dict']
    _log.info(f'Converting {len(torch_sd)} parameters to MLX format')
    mlx_sd = convert_state_dict(torch_sd)

    # Validate keys
    model_keys = set(
        k for k, _ in nn.utils.tree_flatten(model.parameters()))

    ckpt_keys = set(mlx_sd.keys())
    missing_in_model = ckpt_keys - model_keys
    missing_in_ckpt = model_keys - ckpt_keys

    if missing_in_model:
        _log.warning(f'{len(missing_in_model)} checkpoint keys not in model:')
        for k in sorted(missing_in_model)[:20]:
            _log.warning(f'  {k}')

    if missing_in_ckpt:
        _log.warning(f'{len(missing_in_ckpt)} model keys not in checkpoint:')
        for k in sorted(missing_in_ckpt)[:20]:
            _log.warning(f'  {k}')

    # Load weights
    weights = {k: mx.array(v) for k, v in mlx_sd.items()}
    _set_params_from_flat(model, weights)
    mx.eval(model.parameters())

    _log.info('RF2 MLX model weights loaded successfully')
    return model, ckpt
