"""
Convert PyTorch RFdiffusion checkpoint state_dict keys to MLX naming conventions.

MLX nn.Module differences from PyTorch:
- nn.Sequential children accessed via .layers.N instead of .N
- ParameterDict stored as plain dicts (keys preserved)
- nn.Linear stores weight as (out, in) — same as PyTorch, no transpose needed
- nn.LayerNorm uses .weight and .bias — same as PyTorch
- Python lists (replacing ModuleList) use direct .N indexing — same as PyTorch
"""
import re
import logging

import numpy as np

_log = logging.getLogger(__name__)

# All nn.Sequential containers in the model that need .layers. inserted.
# Pattern: parent.N.child -> parent.layers.N.child where parent is Sequential
_SEQUENTIAL_PARENTS = {
    'net',              # RadialProfile MLP in SE3 convolution
    'node_embedder',    # Timestep_emb in embeddings
    'graph_modules',    # GraphSequential in SE3Transformer
}


def map_torch_to_mlx(key: str, value) -> tuple:
    """Remap a single PyTorch state_dict (key, value) pair to MLX conventions.

    Args:
        key: PyTorch state_dict key
        value: PyTorch tensor (or numpy array)

    Returns:
        (mlx_key, numpy_array) tuple
    """
    parts = key.split('.')
    new_parts = []
    i = 0
    while i < len(parts):
        p = parts[i]
        # Check if this is a Sequential container followed by an integer index
        if p in _SEQUENTIAL_PARENTS and i + 1 < len(parts) and parts[i + 1].isdigit():
            new_parts.append(p)
            new_parts.append('layers')
            new_parts.append(parts[i + 1])
            i += 2
        else:
            new_parts.append(p)
            i += 1

    new_key = '.'.join(new_parts)

    # Convert value to numpy
    if hasattr(value, 'numpy'):
        try:
            arr = value.numpy()
        except RuntimeError:
            # Handle CUDA tensors: move to CPU first
            arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.array(value)

    return new_key, arr


def convert_state_dict(torch_state_dict: dict) -> dict:
    """Convert an entire PyTorch state_dict to MLX-compatible format.

    Args:
        torch_state_dict: PyTorch model state_dict

    Returns:
        dict of {mlx_key: numpy_array}
    """
    mlx_dict = {}
    for key, value in torch_state_dict.items():
        mlx_key, arr = map_torch_to_mlx(key, value)
        mlx_dict[mlx_key] = arr
    return mlx_dict


def load_checkpoint_to_mlx(ckpt_path: str, model, use_final_state: bool = False):
    """Load a PyTorch checkpoint into an MLX model.

    Args:
        ckpt_path: Path to PyTorch checkpoint (.pt file)
        model: MLX nn.Module to load weights into
        use_final_state: If True, use 'final_state_dict' key instead of 'model_state_dict'

    Returns:
        The model with loaded weights, plus the checkpoint dict (for config access)
    """
    import torch
    import mlx.core as mx
    import mlx.nn as nn

    _log.info(f'Loading PyTorch checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    state_key = 'final_state_dict' if use_final_state else 'model_state_dict'
    if state_key not in ckpt:
        raise KeyError(f"Checkpoint missing '{state_key}'. Available keys: {list(ckpt.keys())}")

    torch_sd = ckpt[state_key]
    _log.info(f'Converting {len(torch_sd)} parameters to MLX format')
    mlx_sd = convert_state_dict(torch_sd)

    # Get MLX model's expected parameter tree for validation
    model_params = dict(nn.utils.tree_flatten(model.parameters()))
    model_keys = set(model_params.keys())

    # Convert numpy arrays to mx.array and build nested dict structure
    weights = {}
    missing_in_model = []
    for key, arr in mlx_sd.items():
        if key not in model_keys:
            missing_in_model.append(key)
        weights[key] = mx.array(arr)

    if missing_in_model:
        _log.warning(f'{len(missing_in_model)} checkpoint keys not found in model:')
        for k in missing_in_model[:20]:
            _log.warning(f'  {k}')
        if len(missing_in_model) > 20:
            _log.warning(f'  ... and {len(missing_in_model) - 20} more')

    missing_in_ckpt = model_keys - set(mlx_sd.keys())
    if missing_in_ckpt:
        _log.warning(f'{len(missing_in_ckpt)} model keys not found in checkpoint:')
        for k in sorted(missing_in_ckpt)[:20]:
            _log.warning(f'  {k}')
        if len(missing_in_ckpt) > 20:
            _log.warning(f'  ... and {len(missing_in_ckpt) - 20} more')

    # Load weights by traversing the model's parameter tree
    # We can't use model.load_weights() because tree_unflatten converts
    # all numeric keys to list indices, but our model has dicts with
    # numeric string keys (conv_in['0'], weights['0']).
    _set_params_from_flat(model, weights)
    mx.eval(model.parameters())
    _log.info('MLX model weights loaded successfully')

    return model, ckpt


def _set_params_from_flat(model, flat_weights):
    """Set model parameters from a flat dict of {dotted_key: mx.array}.

    Navigates the model's actual tree structure (respecting lists vs dicts)
    instead of relying on tree_unflatten.
    """
    import mlx.nn as nn

    # Get the model's parameter tree to understand its structure
    param_tree = model.parameters()

    for flat_key, value in flat_weights.items():
        parts = flat_key.split('.')
        # Navigate to the parent container and set the leaf
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

        # Set the leaf value
        leaf = parts[-1]
        if isinstance(obj, nn.Module):
            # Check if it's a plain attribute (mx.array) or a dict
            current = getattr(obj, leaf, None)
            if isinstance(current, dict):
                raise ValueError(
                    f"Expected leaf but got dict at '{flat_key}'")
            setattr(obj, leaf, value)
        elif isinstance(obj, dict):
            obj[leaf] = value
        elif isinstance(obj, list):
            obj[int(leaf)] = value
        else:
            raise ValueError(
                f"Cannot set leaf on {type(obj)} for key '{flat_key}'")
