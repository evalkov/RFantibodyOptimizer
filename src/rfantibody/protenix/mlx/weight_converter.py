"""
Convert Protenix PyTorch checkpoint to MLX format.

Handles:
  - Protenix checkpoint format: ckpt["model"] with optional "module." prefix
  - Complex key renaming between Protenix and MLX naming conventions
  - Weight concatenation for split SwiGLU gates (linear_no_bias_a + _b -> linear_in)
  - Weight stacking for AdaLN (linear_nobias_s + linear_s -> ada_ln.linear)
  - Skipping modules not present in the simplified MLX model

Follows the same pattern as rfantibody.rf2.mlx.weight_converter.
"""

import logging
import re

import numpy as np

_log = logging.getLogger(__name__)


def _strip_module_prefix(key: str) -> str:
    """Strip DDP 'module.' prefix."""
    return key[7:] if key.startswith('module.') else key


def _to_numpy(value) -> np.ndarray:
    """Convert a PyTorch tensor or value to numpy array."""
    if hasattr(value, 'numpy'):
        try:
            return value.numpy()
        except RuntimeError:
            return value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.array(value)


# ---------------------------------------------------------------------------
# Modules to skip entirely (not in Mini-Flow)
# ---------------------------------------------------------------------------

_SKIP_PREFIXES = (
    'input_embedder.',
    'template_embedder.',
    'msa_blocks.',
    'msa_module.',
    'msa_linear_no_bias_m.',
    'msa_linear_no_bias_s.',
    'diffusion_module.atom_attention_encoder.',
    'diffusion_module.atom_attention_decoder.',
)


def _should_skip(key: str) -> bool:
    """Check if key belongs to a module not in Mini-Flow."""
    for prefix in _SKIP_PREFIXES:
        if key.startswith(prefix):
            return True
    return False


# ---------------------------------------------------------------------------
# Trunk pairformer block remapping (within a block suffix)
# ---------------------------------------------------------------------------

def _remap_trunk_block_suffix(suffix: str) -> str | None:
    """Map a suffix within pairformer_stack.blocks.N.<suffix>."""

    # --- AttentionPairBias ---
    if suffix.startswith('attention_pair_bias.'):
        rest = suffix[len('attention_pair_bias.'):]

        if rest.startswith('layernorm_a.'):
            return 'attention_pair_bias.layer_norm_s.' + rest[len('layernorm_a.'):]
        if rest.startswith('layernorm_z.'):
            return 'attention_pair_bias.layer_norm_z.' + rest[len('layernorm_z.'):]
        if rest == 'linear_nobias_z.weight':
            return 'attention_pair_bias.linear_z.weight'

        if rest.startswith('attention.'):
            m = {
                'attention.linear_q.weight': 'attention_pair_bias.to_q.weight',
                'attention.linear_q.bias': None,  # MLX to_q has no bias
                'attention.linear_k.weight': 'attention_pair_bias.to_k.weight',
                'attention.linear_v.weight': 'attention_pair_bias.to_v.weight',
                'attention.linear_g.weight': 'attention_pair_bias.to_g.weight',
                'attention.linear_o.weight': 'attention_pair_bias.to_out.weight',
            }
            return m.get(rest)
        return None

    # --- Triangle multiplication (same naming in both) ---
    if suffix.startswith('tri_mul_out.') or suffix.startswith('tri_mul_in.'):
        return suffix

    # --- Triangle attention ---
    if suffix.startswith('tri_att_start.') or suffix.startswith('tri_att_end.'):
        prefix, rest = suffix.split('.', 1)
        if rest.startswith('layer_norm.'):
            return suffix
        if rest == 'linear.weight':
            return f'{prefix}.linear_z.weight'
        if rest.startswith('mha.'):
            m = {
                'mha.linear_q.weight': 'to_q.weight',
                'mha.linear_k.weight': 'to_k.weight',
                'mha.linear_v.weight': 'to_v.weight',
                'mha.linear_g.weight': 'to_g.weight',
                'mha.linear_o.weight': 'to_out.weight',
            }
            mapped = m.get(rest)
            return f'{prefix}.{mapped}' if mapped else None
        return None

    # --- Transitions: pair_transition / single_transition ---
    # These need weight combining, handled specially. Return a marker.
    if suffix.startswith('pair_transition.') or suffix.startswith('single_transition.'):
        return '__TRANSITION__'

    return None


# ---------------------------------------------------------------------------
# Confidence head pairformer block remapping
# ---------------------------------------------------------------------------

def _remap_confidence_block_suffix(suffix: str) -> str | None:
    """Map a suffix within confidence_head.pairformer_stack.blocks.N.<suffix>.

    MLX confidence pairformer uses simplified classes with different naming.
    """
    # AttentionPairBias -- not in simplified confidence pairformer
    if suffix.startswith('attention_pair_bias.'):
        return None

    # Triangle multiplication
    if suffix.startswith('tri_mul_out.') or suffix.startswith('tri_mul_in.'):
        prefix, rest = suffix.split('.', 1)
        m = {
            'layer_norm_in.weight': 'ln.weight',
            'layer_norm_in.bias': 'ln.bias',
            'layer_norm_out.weight': 'ln_out.weight',
            'layer_norm_out.bias': 'ln_out.bias',
            'linear_a_p.weight': 'linear_a.weight',
            'linear_b_p.weight': 'linear_b.weight',
            'linear_a_g.weight': 'gate_a.weight',
            'linear_b_g.weight': 'gate_b.weight',
            'linear_z.weight': 'linear_out.weight',
            'linear_g.weight': 'gate_out.weight',
        }
        mapped = m.get(rest)
        return f'{prefix}.{mapped}' if mapped else None

    # Triangle attention
    if suffix.startswith('tri_att_start.') or suffix.startswith('tri_att_end.'):
        prefix, rest = suffix.split('.', 1)
        mlx_prefix = prefix.replace('tri_att_', 'tri_attn_')
        if rest.startswith('layer_norm.'):
            return f'{mlx_prefix}.ln.{rest[len("layer_norm."):]}'
        if rest == 'linear.weight':
            return f'{mlx_prefix}.linear_bias.weight'
        if rest.startswith('mha.'):
            m = {
                'mha.linear_q.weight': 'w_q.weight',
                'mha.linear_k.weight': 'w_k.weight',
                'mha.linear_v.weight': 'w_v.weight',
                'mha.linear_g.weight': 'gate.weight',  # MLX uses nn.Linear (with bias)
                'mha.linear_o.weight': 'linear_out.weight',
            }
            mapped = m.get(rest)
            return f'{mlx_prefix}.{mapped}' if mapped else None
        return None

    # Transitions -- need weight combining
    if suffix.startswith('pair_transition.') or suffix.startswith('single_transition.'):
        return '__TRANSITION__'

    return None


# ---------------------------------------------------------------------------
# Diffusion transformer block remapping
# ---------------------------------------------------------------------------

def _remap_diffusion_block_suffix(suffix: str) -> str | None:
    """Map a suffix within diffusion_module.diffusion_transformer.blocks.N.<suffix>."""

    # --- AttentionPairBias ---
    if suffix.startswith('attention_pair_bias.'):
        rest = suffix[len('attention_pair_bias.'):]

        # AdaLN (needs combining)
        if rest.startswith('layernorm_a.'):
            return '__ADALN_ATTN__'
        if rest == 'layernorm_z.weight':
            return None
        if rest == 'linear_nobias_z.weight':
            return 'attention.linear_bias.weight'
        if rest.startswith('attention.'):
            m = {
                'attention.linear_q.weight': 'attention.w_q.weight',
                'attention.linear_q.bias': None,
                'attention.linear_k.weight': 'attention.w_k.weight',
                'attention.linear_v.weight': 'attention.w_v.weight',
                'attention.linear_g.weight': 'attention.linear_gate.weight',
                'attention.linear_o.weight': 'attention.linear_out.weight',
            }
            return m.get(rest)
        if rest.startswith('linear_a_last.'):
            return None
        return None

    # --- ConditionedTransitionBlock ---
    if suffix.startswith('conditioned_transition_block.'):
        rest = suffix[len('conditioned_transition_block.'):]
        # AdaLN (needs combining)
        if rest.startswith('adaln.'):
            return '__ADALN_TRANS__'
        m = {
            'linear_nobias_a1.weight': 'transition.transition.linear_gate.weight',
            'linear_nobias_a2.weight': 'transition.transition.linear_value.weight',
            'linear_nobias_b.weight': 'transition.transition.linear_out.weight',
            'linear_s.weight': None,
            'linear_s.bias': None,
        }
        return m.get(rest)

    return None


# ---------------------------------------------------------------------------
# Diffusion conditioning remapping
# ---------------------------------------------------------------------------

def _remap_conditioning_suffix(suffix: str) -> str | None:
    """Map a suffix within diffusion_module.diffusion_conditioning.<suffix>."""
    if suffix == 'fourier_embedding.w':
        return 'fourier_emb.w'
    if suffix == 'fourier_embedding.b':
        return None
    if suffix in ('layernorm_n.weight', 'layernorm_s.weight', 'layernorm_z.weight'):
        return None
    if suffix == 'linear_no_bias_n.weight':
        return 'linear_n.weight'
    if suffix == 'linear_no_bias_s.weight':
        return 'linear_s.weight'
    if suffix == 'linear_no_bias_z.weight':
        return None
    if suffix.startswith('relpe.'):
        return None

    # Transitions
    for tname in ('transition_s1', 'transition_s2', 'transition_z1', 'transition_z2'):
        if suffix.startswith(tname + '.'):
            rest = suffix[len(tname) + 1:]
            m = {
                'linear_no_bias_a.weight': 'linear_gate.weight',
                'linear_no_bias_b.weight': 'linear_value.weight',
                'linear_no_bias.weight': 'linear_out.weight',
                'layernorm1.weight': None,
                'layernorm1.bias': None,
            }
            mapped = m.get(rest)
            if mapped is None:
                return None
            return f'{tname}.{mapped}'

    return None


# ---------------------------------------------------------------------------
# Main remapping function
# ---------------------------------------------------------------------------

def _remap_key(key: str) -> str | None:
    """Remap a single checkpoint key (module. stripped) to MLX model key.

    Returns MLX key, or None to skip, or a __MARKER__ for combined weights.
    """
    if _should_skip(key):
        return None

    # --- Top-level init projections ---
    top_renames = {
        'linear_no_bias_sinit.weight': 'linear_s_init.weight',
        'linear_no_bias_zinit1.weight': 'linear_z_init1.weight',
        'linear_no_bias_zinit2.weight': 'linear_z_init2.weight',
        'linear_no_bias_token_bond.weight': 'linear_token_bond.weight',
        'linear_no_bias_z_cycle.weight': 'linear_z_cycle.weight',
        'linear_no_bias_s.weight': 'linear_s_cycle.weight',
    }
    if key in top_renames:
        return top_renames[key]

    # Top-level layer norms
    for ckpt_pref, mlx_pref in [('layernorm_z_cycle.', 'ln_z_cycle.'),
                                  ('layernorm_s.', 'ln_s_cycle.')]:
        if key.startswith(ckpt_pref):
            return mlx_pref + key[len(ckpt_pref):]

    # Relative position encoding
    if key.startswith('relative_position_encoding.linear_no_bias.'):
        return 'relative_position_encoding.linear.' + key[len('relative_position_encoding.linear_no_bias.'):]

    # Distogram
    if key.startswith('distogram_head.linear.'):
        return 'linear_distogram.' + key[len('distogram_head.linear.'):]

    # Distogram layer norm -- not in checkpoint, only in MLX
    # (it's a new addition, will just keep random init)

    # --- Pairformer stack (trunk) ---
    m = re.match(r'pairformer_stack\.blocks\.(\d+)\.(.*)', key)
    if m:
        block_num, suffix = m.group(1), m.group(2)
        mapped = _remap_trunk_block_suffix(suffix)
        if mapped is None:
            return None
        if mapped == '__TRANSITION__':
            return mapped  # handled in combining pass
        return f'pairformer_stack.blocks.{block_num}.{mapped}'

    # --- Confidence head ---
    if key.startswith('confidence_head.'):
        rest = key[len('confidence_head.'):]

        # Simple renames
        # Direct leaf tensors (no dot suffix)
        conf_direct = {
            'plddt_weight': 'plddt_weight',
            'resolved_weight': 'resolved_weight',
        }
        if rest in conf_direct:
            return f'confidence_head.{conf_direct[rest]}'

        conf_renames = {
            'input_strunk_ln.': 'ln_s_trunk.',
            'pae_ln.': 'ln_pae.',
            'pde_ln.': 'ln_pde.',
            'plddt_ln.': 'ln_plddt.',
            'resolved_ln.': 'ln_resolved.',
            'linear_no_bias_d.': 'linear_d.',
            'linear_no_bias_d_wo_onehot.': 'linear_d_raw.',
            'linear_no_bias_pae.': 'linear_pae.',
            'linear_no_bias_pde.': 'linear_pde.',
            'linear_no_bias_s1.': 'linear_s1.',
            'linear_no_bias_s2.': 'linear_s2.',
        }
        for ckpt_pref, mlx_pref in conf_renames.items():
            if rest.startswith(ckpt_pref):
                return 'confidence_head.' + mlx_pref + rest[len(ckpt_pref):]

        # Confidence pairformer blocks
        bm = re.match(r'pairformer_stack\.blocks\.(\d+)\.(.*)', rest)
        if bm:
            block_num, suffix = bm.group(1), bm.group(2)
            mapped = _remap_confidence_block_suffix(suffix)
            if mapped is None:
                return None
            if mapped == '__TRANSITION__':
                return mapped
            return f'confidence_head.pairformer.blocks.{block_num}.{mapped}'

        # Buffers
        if rest in ('lower_bins', 'upper_bins'):
            return key

        return None

    # --- Diffusion module ---
    if key.startswith('diffusion_module.'):
        rest = key[len('diffusion_module.'):]

        # Top-level diffusion layers
        # ln_a and ln_s are affine=False in MLX (no weight params), skip
        if rest in ('layernorm_a.weight', 'layernorm_s.weight'):
            return None
        if rest == 'linear_no_bias_s.weight':
            return 'diffusion_module.linear_s_to_token.weight'

        # Diffusion conditioning
        if rest.startswith('diffusion_conditioning.'):
            cond_rest = rest[len('diffusion_conditioning.'):]
            mapped = _remap_conditioning_suffix(cond_rest)
            if mapped is None:
                return None
            return f'diffusion_module.conditioning.{mapped}'

        # Diffusion transformer blocks
        bm = re.match(r'diffusion_transformer\.blocks\.(\d+)\.(.*)', rest)
        if bm:
            block_num, suffix = bm.group(1), bm.group(2)
            mapped = _remap_diffusion_block_suffix(suffix)
            if mapped is None:
                return None
            if mapped.startswith('__ADALN'):
                return mapped  # handled in combining
            return f'diffusion_module.transformer_blocks.{block_num}.{mapped}'

        return None

    return None


# ---------------------------------------------------------------------------
# Weight combining
# ---------------------------------------------------------------------------

def _combine_trunk_transitions(raw_sd: dict, remapped: dict):
    """Combine split SwiGLU transitions for trunk pairformer blocks.

    Protenix: linear_no_bias_a [c_h, c_in] + linear_no_bias_b [c_h, c_in]
    MLX:      linear_in [2*c_h, c_in] (concatenated)
    """
    pattern = re.compile(
        r'pairformer_stack\.blocks\.(\d+)\.(pair_transition|single_transition)\.(.*)'
    )
    # Group by (block_num, trans_name)
    groups = {}
    for key in raw_sd:
        m = pattern.match(key)
        if m:
            block_num, trans_name, rest = m.group(1), m.group(2), m.group(3)
            gkey = (block_num, trans_name)
            if gkey not in groups:
                groups[gkey] = {}
            groups[gkey][rest] = raw_sd[key]

    for (block_num, trans_name), params in groups.items():
        mlx_prefix = f'pairformer_stack.blocks.{block_num}.{trans_name}.'

        # Combine gate + value -> linear_in
        if 'linear_no_bias_a.weight' in params and 'linear_no_bias_b.weight' in params:
            combined = np.concatenate(
                [params['linear_no_bias_a.weight'], params['linear_no_bias_b.weight']],
                axis=0,
            )
            remapped[mlx_prefix + 'linear_in.weight'] = combined

        # Output
        if 'linear_no_bias.weight' in params:
            remapped[mlx_prefix + 'linear_out.weight'] = params['linear_no_bias.weight']

        # Layer norm
        if 'layernorm1.weight' in params:
            remapped[mlx_prefix + 'layer_norm.weight'] = params['layernorm1.weight']
        if 'layernorm1.bias' in params:
            remapped[mlx_prefix + 'layer_norm.bias'] = params['layernorm1.bias']


def _combine_confidence_transitions(raw_sd: dict, remapped: dict):
    """Combine split SwiGLU transitions for confidence pairformer blocks.

    Confidence pairformer uses SwiGLUTransition (no LN, separate gate/value/out).
    """
    pattern = re.compile(
        r'confidence_head\.pairformer_stack\.blocks\.(\d+)\.(pair_transition|single_transition)\.(.*)'
    )
    groups = {}
    for key in raw_sd:
        m = pattern.match(key)
        if m:
            block_num, trans_name, rest = m.group(1), m.group(2), m.group(3)
            gkey = (block_num, trans_name)
            if gkey not in groups:
                groups[gkey] = {}
            groups[gkey][rest] = raw_sd[key]

    for (block_num, trans_name), params in groups.items():
        # Map pair_transition -> transition_z, single_transition -> transition_s
        mlx_trans = 'transition_z' if trans_name == 'pair_transition' else 'transition_s'
        mlx_prefix = f'confidence_head.pairformer.blocks.{block_num}.{mlx_trans}.'

        if 'linear_no_bias_a.weight' in params:
            remapped[mlx_prefix + 'linear_gate.weight'] = params['linear_no_bias_a.weight']
        if 'linear_no_bias_b.weight' in params:
            remapped[mlx_prefix + 'linear_value.weight'] = params['linear_no_bias_b.weight']
        if 'linear_no_bias.weight' in params:
            remapped[mlx_prefix + 'linear_out.weight'] = params['linear_no_bias.weight']


def _combine_diffusion_adaln(raw_sd: dict, remapped: dict):
    """Combine AdaLN projections for diffusion transformer blocks.

    Protenix: linear_nobias_s [c_a, c_s] + linear_s [c_a, c_s]
    MLX:      ada_ln.linear [2*c_a, c_s] (stacked)
    """
    # Attention AdaLN
    pattern_attn = re.compile(
        r'diffusion_module\.diffusion_transformer\.blocks\.(\d+)\.attention_pair_bias\.layernorm_a\.(.*)'
    )
    groups_attn = {}
    for key in raw_sd:
        m = pattern_attn.match(key)
        if m:
            block_num, rest = m.group(1), m.group(2)
            if block_num not in groups_attn:
                groups_attn[block_num] = {}
            groups_attn[block_num][rest] = raw_sd[key]

    for block_num, params in groups_attn.items():
        if 'linear_nobias_s.weight' in params and 'linear_s.weight' in params:
            combined = np.concatenate(
                [params['linear_nobias_s.weight'], params['linear_s.weight']],
                axis=0,
            )
            mlx_key = f'diffusion_module.transformer_blocks.{block_num}.attention.ada_ln.linear.weight'
            remapped[mlx_key] = combined

    # Transition AdaLN
    pattern_trans = re.compile(
        r'diffusion_module\.diffusion_transformer\.blocks\.(\d+)\.conditioned_transition_block\.adaln\.(.*)'
    )
    groups_trans = {}
    for key in raw_sd:
        m = pattern_trans.match(key)
        if m:
            block_num, rest = m.group(1), m.group(2)
            if block_num not in groups_trans:
                groups_trans[block_num] = {}
            groups_trans[block_num][rest] = raw_sd[key]

    for block_num, params in groups_trans.items():
        if 'linear_nobias_s.weight' in params and 'linear_s.weight' in params:
            combined = np.concatenate(
                [params['linear_nobias_s.weight'], params['linear_s.weight']],
                axis=0,
            )
            mlx_key = f'diffusion_module.transformer_blocks.{block_num}.transition.ada_ln.linear.weight'
            remapped[mlx_key] = combined


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_state_dict(torch_state_dict: dict) -> dict:
    """Convert entire PyTorch state_dict to MLX-compatible format.

    Handles key remapping AND weight combining (for SwiGLU and AdaLN).

    Args:
        torch_state_dict: PyTorch model state_dict

    Returns:
        dict mapping MLX parameter paths to numpy arrays
    """
    # Step 1: Convert all values to numpy and strip module. prefix
    raw_sd = {}
    for key, value in torch_state_dict.items():
        key = _strip_module_prefix(key)
        raw_sd[key] = _to_numpy(value)

    _log.info(f'Processing {len(raw_sd)} checkpoint parameters')

    # Step 2: Simple key remapping (1:1 mappings)
    remapped = {}
    skipped = 0
    deferred = 0

    for key, arr in raw_sd.items():
        new_key = _remap_key(key)
        if new_key is None:
            skipped += 1
        elif new_key.startswith('__'):
            deferred += 1  # handled in combining pass
        else:
            remapped[new_key] = arr

    _log.info(f'Simple remapping: {len(remapped)} mapped, {skipped} skipped, {deferred} deferred for combining')

    # Step 3: Weight combining
    _combine_trunk_transitions(raw_sd, remapped)
    _combine_confidence_transitions(raw_sd, remapped)
    _combine_diffusion_adaln(raw_sd, remapped)

    _log.info(f'After combining: {len(remapped)} total parameters')

    return remapped


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

    # Convert PyTorch state dict with comprehensive remapping
    remapped = convert_state_dict(torch_sd)

    # Validate against model
    model_keys = set(k for k, _ in nn.utils.tree_flatten(model.parameters()))
    ckpt_keys = set(remapped.keys())

    missing_in_model = ckpt_keys - model_keys
    missing_in_ckpt = model_keys - ckpt_keys

    if missing_in_model:
        _log.warning(f'{len(missing_in_model)} checkpoint keys not in model:')
        for k in sorted(missing_in_model)[:30]:
            _log.warning(f'  EXTRA: {k}')
        if len(missing_in_model) > 30:
            _log.warning(f'  ... and {len(missing_in_model) - 30} more')

    if missing_in_ckpt:
        _log.warning(f'{len(missing_in_ckpt)} model keys not in checkpoint:')
        for k in sorted(missing_in_ckpt)[:30]:
            _log.warning(f'  MISSING: {k}')
        if len(missing_in_ckpt) > 30:
            _log.warning(f'  ... and {len(missing_in_ckpt) - 30} more')

    matched = ckpt_keys & model_keys
    _log.info(
        f'Weight matching: {len(matched)} matched, '
        f'{len(missing_in_ckpt)} missing from ckpt, '
        f'{len(missing_in_model)} extra in ckpt'
    )

    # Load weights (only keys that exist in model)
    loadable = {k: mx.array(v) for k, v in remapped.items() if k in model_keys}
    _log.info(f'Loading {len(loadable)} parameters into model')
    _set_params_from_flat(model, loadable)
    mx.eval(model.parameters())

    _log.info('Protenix MLX model weights loaded successfully')
    return model, ckpt
