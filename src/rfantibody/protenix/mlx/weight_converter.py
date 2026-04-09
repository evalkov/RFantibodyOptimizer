"""
Convert Protenix PyTorch checkpoint to MLX format.

Handles:
  - Protenix checkpoint format: ckpt["model"] with optional "module." prefix
  - nn.Sequential containers (SubstructureEmbedder network) -> .layers. insertion
  - nn.ModuleList containers (PairformerStack blocks, etc.)
  - PyTorch tensor -> numpy array conversion

Follows the same pattern as rfantibody.rf2.mlx.weight_converter.
"""

import logging

import numpy as np

_log = logging.getLogger(__name__)

# Sequential containers in Protenix that need .layers. inserted
# When PyTorch uses nn.Sequential, the state_dict keys look like:
#   parent.0.weight, parent.1.weight, ...
# MLX lists store them as:
#   parent.layers.0.weight, parent.layers.1.weight, ...
_SEQUENTIAL_PARENTS = {
    'network',            # SubstructureEmbedder MLP
    'encoder_layers',     # DiffusionModule simplified encoder
    'decoder_layers',     # DiffusionModule simplified decoder
    'small_mlp',          # AtomAttentionEncoder MLP
}


def map_torch_to_mlx(key: str, value) -> tuple:
    """Remap a single PyTorch state_dict key to MLX conventions.

    Transformations:
      1. Strip 'module.' prefix (from DDP checkpoints)
      2. Insert '.layers.' for Sequential containers
      3. Convert tensor to numpy array

    Args:
        key: PyTorch state_dict key
        value: PyTorch tensor (or numpy array)

    Returns:
        (mlx_key, numpy_array)
    """
    # Strip DDP module. prefix
    if key.startswith('module.'):
        key = key[len('module.'):]

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

    # Convert value to numpy
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
    """Convert entire PyTorch state_dict to MLX-compatible format.

    Args:
        torch_state_dict: PyTorch model state_dict

    Returns:
        dict mapping MLX parameter paths to numpy arrays
    """
    mlx_dict = {}
    for key, value in torch_state_dict.items():
        mlx_key, arr = map_torch_to_mlx(key, value)
        mlx_dict[mlx_key] = arr
    return mlx_dict


def _set_params_from_flat(model, flat_weights):
    """Set model parameters from a flat dict, navigating lists/dicts correctly.

    Handles nn.Module attributes, dict children, and list children
    (nn.ModuleList in PyTorch becomes a Python list in MLX).

    Args:
        model: MLX nn.Module
        flat_weights: dict mapping dotted key paths to mx.array values
    """
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
                    f"'{'.'.join(parts[:i+1])}' in key '{flat_key}'"
                )

        leaf = parts[-1]
        if isinstance(obj, nn.Module):
            setattr(obj, leaf, value)
        elif isinstance(obj, dict):
            obj[leaf] = value
        elif isinstance(obj, list):
            obj[int(leaf)] = value
        else:
            raise ValueError(
                f"Cannot set leaf on {type(obj)} for key '{flat_key}'"
            )


def _remap_protenix_keys(mlx_key: str) -> str | None:
    """Remap full Protenix model keys to Mini-Flow MLX model keys.

    The full Protenix model has many modules we don't use in Mini-Flow
    (atom_attention_encoder, template_embedder, msa_module, etc.).
    This function maps keys we DO need and returns None for keys to skip.

    Args:
        mlx_key: key from converted Protenix state_dict

    Returns:
        Remapped key for Mini-Flow model, or None to skip
    """
    # Direct mappings for modules that exist in both
    key_map = {
        'pairformer_stack.': 'pairformer_stack.',
        'diffusion_module.': 'diffusion_module.',
        'confidence_head.': 'confidence_head.',
        'relative_position_encoding.': 'relative_position_encoding.',
        'linear_no_bias_sinit.': 'linear_s_init.',
        'linear_no_bias_zinit1.': 'linear_z_init1.',
        'linear_no_bias_zinit2.': 'linear_z_init2.',
        'linear_no_bias_token_bond.': 'linear_token_bond.',
        'linear_no_bias_z_cycle.': 'linear_z_cycle.',
        'linear_no_bias_s.': 'linear_s_cycle.',
        'layernorm_z_cycle.': 'ln_z_cycle.',
        'layernorm_s.': 'ln_s_cycle.',
        'distogram_head.linear.': 'linear_distogram.',
    }

    for src_prefix, dst_prefix in key_map.items():
        if mlx_key.startswith(src_prefix):
            return dst_prefix + mlx_key[len(src_prefix):]

    # Modules that Mini-Flow doesn't have -- skip silently
    skip_prefixes = [
        'input_embedder.',
        'template_embedder.',
        'msa_module.',
        'constraint_embedder.',
    ]
    for prefix in skip_prefixes:
        if mlx_key.startswith(prefix):
            return None

    # Unknown key -- return as-is and log warning
    return mlx_key


def load_checkpoint_to_mlx(ckpt_path: str, model_params=None):
    """Load Protenix PyTorch checkpoint into MLX model.

    Protenix checkpoints use the format:
      ckpt["model"] = state_dict
    with optional "module." prefix from DDP training.

    Args:
        ckpt_path: path to Protenix checkpoint (.pt file)
        model_params: optional dict of model parameters. If None,
            uses default Mini-Flow parameters.

    Returns:
        (model, ckpt) -- loaded MLX ProtenixMiniModule and raw checkpoint
    """
    import torch
    import mlx.core as mx
    import mlx.nn as nn
    from rfantibody.protenix.mlx.model import ProtenixMiniModule

    _log.info(f'Loading Protenix checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Protenix uses "model" key (not "model_state_dict" like RF2)
    if 'model' in ckpt:
        torch_sd = ckpt['model']
    elif 'model_state_dict' in ckpt:
        torch_sd = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        torch_sd = ckpt['state_dict']
    else:
        # Assume the checkpoint IS the state dict
        _log.warning(
            f"No 'model' key found. Available keys: {list(ckpt.keys())[:10]}. "
            "Treating checkpoint as raw state_dict."
        )
        torch_sd = ckpt

    _log.info(f'Found {len(torch_sd)} parameters in checkpoint')

    # Default Mini-Flow parameters
    if model_params is None:
        model_params = {
            'c_s': 384,
            'c_z': 128,
            'c_token': 768,
            'c_s_inputs': 449,
            'n_token_types': 32,
            'r_max': 32,
            's_max': 2,
            'n_pairformer_blocks': 16,
            'n_diffusion_blocks': 8,
            'n_confidence_blocks': 4,
            'n_diffusion_steps': 5,
            'n_head_single': 16,
            'n_head_pair': 4,
            'sigma_data': 16.0,
            'eval_stride': 4,
        }

    model = ProtenixMiniModule(**model_params)

    # Convert PyTorch state dict
    mlx_sd = convert_state_dict(torch_sd)

    # Remap Protenix keys to Mini-Flow keys
    remapped = {}
    skipped = 0
    for key, arr in mlx_sd.items():
        new_key = _remap_protenix_keys(key)
        if new_key is None:
            skipped += 1
            continue
        remapped[new_key] = arr

    _log.info(
        f'Remapped {len(remapped)} parameters, skipped {skipped} '
        f'(modules not in Mini-Flow)'
    )

    # Validate against model
    model_keys = set(k for k, _ in nn.utils.tree_flatten(model.parameters()))
    ckpt_keys = set(remapped.keys())

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

    # Load weights (only keys that exist in model)
    loadable = {k: mx.array(v) for k, v in remapped.items() if k in model_keys}
    _log.info(f'Loading {len(loadable)} parameters into model')
    _set_params_from_flat(model, loadable)
    mx.eval(model.parameters())

    _log.info('Protenix MLX model weights loaded successfully')
    return model, ckpt
