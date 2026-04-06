"""
SE3-Transformer basis computation in MLX.

Computes spherical harmonics from closed-form formulas (no e3nn dependency)
and loads precomputed Clebsch-Gordan coefficients from .npz file.
"""

import os
import math
from typing import Dict, List, Optional

import numpy as np
import mlx.core as mx

from .fiber import degree_to_dim


# ---------------------------------------------------------------------------
# Real spherical harmonics matching e3nn.o3.spherical_harmonics convention
# ---------------------------------------------------------------------------
# Matches e3nn with normalize=True, normalization='integral' (the default).
# e3nn uses component-normalized formulas internally (factor sqrt(2l+1))
# with y as the "polar" axis, then divides by sqrt(4pi) for integral norm.
# The formulas below are ported directly from e3nn's _spherical_harmonics().
#
# Input: unit vectors (x, y, z) of shape (N, 3)
# Output per degree l: (N, 2l+1)


def _compute_sh(x, y, z, max_l):
    """Compute all SH components for degrees 0..max_l.

    Directly matches e3nn's _spherical_harmonics() with integral normalization.
    """
    # Normalization: e3nn component formulas / sqrt(4*pi)
    _C = 1.0 / math.sqrt(4.0 * math.pi)

    y2 = y * y
    x2z2 = x * x + z * z  # e3nn uses y-up convention

    components = []

    # l=0 (1 component)
    sh_0_0 = mx.ones_like(x) * _C
    components.append([sh_0_0])
    if max_l == 0:
        return components

    # l=1 (3 components): e3nn convention is [x, y, z]
    c1 = math.sqrt(3) * _C
    sh_1_0 = c1 * x
    sh_1_1 = c1 * y
    sh_1_2 = c1 * z
    components.append([sh_1_0, sh_1_1, sh_1_2])
    if max_l == 1:
        return components

    # l=2 (5 components)
    c2a = math.sqrt(15) * _C
    c2b = math.sqrt(5) * _C
    sh_2_0 = c2a * x * z
    sh_2_1 = c2a * x * y
    sh_2_2 = c2b * (y2 - 0.5 * x2z2)
    sh_2_3 = c2a * y * z
    sh_2_4 = 0.5 * c2a * (z * z - x * x)
    components.append([sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4])
    if max_l == 2:
        return components

    # l=3 (7 components) - recursive from l=2
    c3a = (1.0/6.0) * math.sqrt(42) * _C
    c3b = math.sqrt(7) * _C
    c3c = (1.0/8.0) * math.sqrt(168) * _C
    c3d = 0.5 * math.sqrt(7) * _C
    # Undo _C from sh_2 for recursion (e3nn builds recursively from component-norm SH)
    s2_0 = sh_2_0 / _C
    s2_4 = sh_2_4 / _C
    sh_3_0 = c3a * (s2_0 * z + s2_4 * x)
    sh_3_1 = c3b * s2_0 * y
    sh_3_2 = c3c * (4.0 * y2 - x2z2) * x
    sh_3_3 = c3d * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = c3c * z * (4.0 * y2 - x2z2)
    sh_3_5 = c3b * s2_4 * y
    sh_3_6 = c3a * (s2_4 * z - s2_0 * x)
    components.append([sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6])
    if max_l == 3:
        return components

    # l=4 (9 components) - recursive from l=3
    # Use component-norm versions for recursion
    s3_0 = sh_3_0 / _C
    s3_1 = sh_3_1 / _C
    s3_2 = sh_3_2 / _C
    s3_3 = sh_3_3 / _C
    s3_4 = sh_3_4 / _C
    s3_5 = sh_3_5 / _C
    s3_6 = sh_3_6 / _C

    sh_4_0 = _C * (3.0/4.0) * math.sqrt(2) * (s3_0 * z + s3_6 * x)
    sh_4_1 = _C * ((3.0/4.0) * s3_0 * y + (3.0/8.0) * math.sqrt(6) * s3_1 * z
                    + (3.0/8.0) * math.sqrt(6) * s3_5 * x)
    sh_4_2 = _C * (
        -3.0/56.0 * math.sqrt(14) * s3_0 * z
        + (3.0/14.0) * math.sqrt(21) * s3_1 * y
        + (3.0/56.0) * math.sqrt(210) * s3_2 * z
        + (3.0/56.0) * math.sqrt(210) * s3_4 * x
        + (3.0/56.0) * math.sqrt(14) * s3_6 * x
    )
    sh_4_3 = _C * (
        -3.0/56.0 * math.sqrt(42) * s3_1 * z
        + (3.0/28.0) * math.sqrt(105) * s3_2 * y
        + (3.0/28.0) * math.sqrt(70) * s3_3 * x
        + (3.0/56.0) * math.sqrt(42) * s3_5 * x
    )
    sh_4_4 = _C * (
        -3.0/28.0 * math.sqrt(42) * s3_2 * x
        + (3.0/7.0) * math.sqrt(7) * s3_3 * y
        - 3.0/28.0 * math.sqrt(42) * s3_4 * z
    )
    sh_4_5 = _C * (
        -3.0/56.0 * math.sqrt(42) * s3_1 * x
        + (3.0/28.0) * math.sqrt(70) * s3_3 * z
        + (3.0/28.0) * math.sqrt(105) * s3_4 * y
        - 3.0/56.0 * math.sqrt(42) * s3_5 * z
    )
    sh_4_6 = _C * (
        -3.0/56.0 * math.sqrt(14) * s3_0 * x
        - 3.0/56.0 * math.sqrt(210) * s3_2 * x
        + (3.0/56.0) * math.sqrt(210) * s3_4 * z
        + (3.0/14.0) * math.sqrt(21) * s3_5 * y
        - 3.0/56.0 * math.sqrt(14) * s3_6 * z
    )
    sh_4_7 = _C * (
        -3.0/8.0 * math.sqrt(6) * s3_1 * x
        + (3.0/8.0) * math.sqrt(6) * s3_5 * z
        + (3.0/4.0) * s3_6 * y
    )
    sh_4_8 = _C * (3.0/4.0) * math.sqrt(2) * (-s3_0 * x + s3_6 * z)
    components.append([sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4,
                       sh_4_5, sh_4_6, sh_4_7, sh_4_8])

    return components


def spherical_harmonics(relative_pos: mx.array, max_degree: int) -> List[mx.array]:
    """Compute real spherical harmonics for degrees 0..2*max_degree.

    Matches e3nn.o3.spherical_harmonics(degrees, pos, normalize=True) exactly.

    Args:
        relative_pos: (E, 3) relative position vectors (will be normalized)
        max_degree: Maximum SE3 degree (typically 2)

    Returns:
        List of (E, 2l+1) arrays for l in 0..2*max_degree
    """
    # Normalize to unit vectors
    norms = mx.linalg.norm(relative_pos, axis=-1, keepdims=True)
    norms = mx.maximum(norms, 1e-8)
    unit = relative_pos / norms

    x = unit[:, 0]
    y = unit[:, 1]
    z = unit[:, 2]

    max_l = 2 * max_degree
    components = _compute_sh(x, y, z, max_l)

    return [mx.stack(comp, axis=-1) for comp in components]


# ---------------------------------------------------------------------------
# Clebsch-Gordan coefficient loading
# ---------------------------------------------------------------------------

_CG_CACHE: Optional[Dict[str, mx.array]] = None
_CG_PATH = os.path.join(os.path.dirname(__file__), '..', 'cg_coefficients.npz')


def _load_cg_coefficients() -> Dict[str, mx.array]:
    """Load precomputed CG coefficients from .npz file."""
    global _CG_CACHE
    if _CG_CACHE is not None:
        return _CG_CACHE

    data = np.load(_CG_PATH)
    _CG_CACHE = {}
    for key in data.files:
        if key.startswith('cg_'):
            _CG_CACHE[key] = mx.array(data[key].astype(np.float32))
    return _CG_CACHE


def get_clebsch_gordon(J: int, d_in: int, d_out: int) -> mx.array:
    """Get the Q^{d_out,d_in}_J matrix."""
    cg = _load_cg_coefficients()
    key = f'cg_{d_in}_{d_out}_{J}'
    return cg[key]


def get_all_clebsch_gordon(max_degree: int) -> List[List[mx.array]]:
    """Get all CG coefficients as nested list matching PyTorch convention."""
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(get_clebsch_gordon(J, d_in, d_out))
            all_cb.append(K_Js)
    return all_cb


# ---------------------------------------------------------------------------
# Basis computation (replaces JIT-compiled functions)
# ---------------------------------------------------------------------------

def get_basis_script(max_degree: int,
                     use_pad_trick: bool,
                     sh: List[mx.array],
                     clebsch_gordon: List[List[mx.array]],
                     amp: bool) -> Dict[str, mx.array]:
    """Compute pairwise basis matrices for degrees up to max_degree.

    Pure MLX replacement for the JIT-compiled PyTorch version.
    """
    basis = {}
    idx = 0
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            key = f'{d_in},{d_out}'
            K_Js = []
            for freq_idx, J in enumerate(range(abs(d_in - d_out), d_in + d_out + 1)):
                Q_J = clebsch_gordon[idx][freq_idx]  # (k, l, f)
                sh_J = sh[J]  # (n, f)

                # Einsum 'n f, k l f -> n l k'
                # = (n, 1, 1, f) * (1, k, l, f) -> sum over f -> (n, k, l) -> transpose to (n, l, k)
                result = mx.einsum('nf,klf->nlk', sh_J, Q_J)
                K_Js.append(result)

            basis[key] = mx.stack(K_Js, axis=2)  # (n, l, freq, k)

            if use_pad_trick:
                # Pad the k dimension
                pad_width = [(0, 0)] * (basis[key].ndim - 1) + [(0, 1)]
                basis[key] = mx.pad(basis[key], pad_width)

            idx += 1

    return basis


def update_basis_with_fused(basis: Dict[str, mx.array],
                            max_degree: int,
                            use_pad_trick: bool,
                            fully_fused: bool) -> Dict[str, mx.array]:
    """Update basis dict with partially and optionally fully fused bases.

    Pure MLX replacement for the JIT-compiled PyTorch version.
    """
    num_edges = basis['0,0'].shape[0]
    dtype = basis['0,0'].dtype
    sum_dim = sum(degree_to_dim(d) for d in range(max_degree + 1))

    # Fused per output degree
    for d_out in range(max_degree + 1):
        sum_freq = sum(degree_to_dim(min(d, d_out)) for d in range(max_degree + 1))
        out_dim = degree_to_dim(d_out) + int(use_pad_trick)
        basis_fused = mx.zeros((num_edges, sum_dim, sum_freq, out_dim), dtype=dtype)
        acc_d, acc_f = 0, 0
        for d_in in range(max_degree + 1):
            dim_in = degree_to_dim(d_in)
            freq = degree_to_dim(min(d_out, d_in))
            dim_out = degree_to_dim(d_out)
            src = basis[f'{d_in},{d_out}'][:, :, :, :dim_out]
            # Slice assignment via concatenation
            basis_fused = _assign_slice(
                basis_fused, src,
                (slice(None), slice(acc_d, acc_d + dim_in),
                 slice(acc_f, acc_f + freq), slice(0, dim_out))
            )
            acc_d += dim_in
            acc_f += freq

        basis[f'out{d_out}_fused'] = basis_fused

    # Fused per input degree
    for d_in in range(max_degree + 1):
        sum_freq = sum(degree_to_dim(min(d, d_in)) for d in range(max_degree + 1))
        dim_in = degree_to_dim(d_in)
        basis_fused = mx.zeros((num_edges, dim_in, sum_freq, sum_dim), dtype=dtype)
        acc_d, acc_f = 0, 0
        for d_out in range(max_degree + 1):
            freq = degree_to_dim(min(d_out, d_in))
            dim_out = degree_to_dim(d_out)
            src = basis[f'{d_in},{d_out}'][:, :, :, :dim_out]
            basis_fused = _assign_slice(
                basis_fused, src,
                (slice(None), slice(None),
                 slice(acc_f, acc_f + freq), slice(acc_d, acc_d + dim_out))
            )
            acc_d += dim_out
            acc_f += freq

        basis[f'in{d_in}_fused'] = basis_fused

    if fully_fused:
        sum_freq = sum(
            sum(degree_to_dim(min(d_in, d_out))
                for d_in in range(max_degree + 1))
            for d_out in range(max_degree + 1)
        )
        basis_fused = mx.zeros((num_edges, sum_dim, sum_freq, sum_dim), dtype=dtype)
        acc_d, acc_f = 0, 0
        for d_out in range(max_degree + 1):
            b = basis[f'out{d_out}_fused']
            dim_out = degree_to_dim(d_out)
            basis_fused = _assign_slice(
                basis_fused, b[:, :, :, :dim_out],
                (slice(None), slice(None),
                 slice(acc_f, acc_f + b.shape[2]), slice(acc_d, acc_d + dim_out))
            )
            acc_f += b.shape[2]
            acc_d += dim_out

        basis['fully_fused'] = basis_fused

    del basis['0,0']  # Constant basis for l=k=0, not needed
    return basis


def _assign_slice(arr: mx.array, src: mx.array, slices: tuple) -> mx.array:
    """Assign src into a slice of arr. MLX arrays are immutable, so we
    rebuild via indexing. For the basis sizes involved this is efficient."""
    # Use at[] for slice assignment
    return arr.at[slices].add(src)


def get_basis(relative_pos: mx.array,
              max_degree: int = 4,
              use_pad_trick: bool = False,
              amp: bool = False) -> Dict[str, mx.array]:
    """Compute SE3 basis matrices from relative positions.

    Args:
        relative_pos: (E, 3) relative position vectors between graph nodes
        max_degree: Maximum representation degree
        use_pad_trick: Pad odd dimensions for hardware efficiency
        amp: Return in float16

    Returns:
        Dict of basis tensors keyed by degree pairs or fused labels
    """
    sh = spherical_harmonics(relative_pos, max_degree)
    cg = get_all_clebsch_gordon(max_degree)
    basis = get_basis_script(max_degree, use_pad_trick, sh, cg, amp)
    return basis
