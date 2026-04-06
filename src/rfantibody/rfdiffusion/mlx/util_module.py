"""
MLX utility modules for RFdiffusion.

Ports rbf, graph construction, coordinate computation, etc.
Heavy computation (argsort, scatter) runs on GPU. Only the final
mask→edge-list extraction uses numpy (a single C call, no Python loops).
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .graph_ops import SimpleGraph, simple_graph


def rbf(D: mx.array) -> mx.array:
    """Distance radial basis function."""
    D_min, D_max, D_count = 0., 20., 36
    D_mu = mx.linspace(D_min, D_max, D_count)
    D_mu = D_mu.reshape(1, -1)  # (1, 36)
    D_sigma = (D_max - D_min) / D_count
    D_expand = mx.expand_dims(D, -1)
    return mx.exp(-((D_expand - D_mu) / D_sigma) ** 2)


def cdist(a: mx.array, b: mx.array) -> mx.array:
    """Pairwise Euclidean distance: a (B, M, D), b (B, N, D) -> (B, M, N)."""
    a_sq = mx.sum(a * a, axis=-1, keepdims=True)  # (B, M, 1)
    b_sq = mx.sum(b * b, axis=-1, keepdims=True)  # (B, N, 1)
    ab = a @ mx.transpose(b, (0, 2, 1))  # (B, M, N)
    dist_sq = a_sq - 2 * ab + mx.transpose(b_sq, (0, 2, 1))
    return mx.sqrt(mx.maximum(dist_sq, 1e-12))


def get_seqsep(idx: mx.array) -> mx.array:
    """Sequence separation feature with sign. (B, L) -> (B, L, L, 1)"""
    seqsep = mx.expand_dims(idx, 1) - mx.expand_dims(idx, 2)
    sign = mx.sign(seqsep)
    neigh = mx.abs(seqsep).astype(mx.float32)
    neigh = mx.where(neigh > 1, 0.0, neigh)
    neigh = sign * neigh
    return mx.expand_dims(neigh, -1)


def cross(a: mx.array, b: mx.array) -> mx.array:
    """Cross product of two (..., 3) arrays."""
    return mx.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], axis=-1)


def _mask_to_edges(mask: mx.array, B: int, L: int):
    """Convert (B, L, L) boolean mask to (b, i, j) edge index arrays.

    Uses np.nonzero (a single C call) for the mask→index extraction.
    All heavy computation (argsort, scatter) stays on GPU.
    """
    mx.eval(mask)
    b, i, j = np.nonzero(np.array(mask))
    return (mx.array(b.astype(np.int32)),
            mx.array(i.astype(np.int32)),
            mx.array(j.astype(np.int32)))


def make_full_graph(xyz: mx.array, pair: mx.array, idx: mx.array,
                    top_k: int = 64, kmin: int = 9):
    """Build full graph (all non-self edges).

    xyz: (B, L, 3), pair: (B, L, L, E), idx: (B, L)
    """
    B, L = xyz.shape[:2]

    sep = mx.expand_dims(idx, 1) - mx.expand_dims(idx, 2)
    mask = mx.abs(sep) > 0

    b_mx, i_mx, j_mx = _mask_to_edges(mask, B, L)

    src = b_mx * L + i_mx
    tgt = b_mx * L + j_mx
    G = simple_graph((src, tgt), num_nodes=B * L)

    rel_pos = xyz[b_mx, j_mx, :] - xyz[b_mx, i_mx, :]
    G.edata = {'rel_pos': mx.stop_gradient(rel_pos)}

    edge_feats = mx.expand_dims(pair[b_mx, i_mx, j_mx], -1)
    return G, edge_feats


def make_topk_graph(xyz: mx.array, pair: mx.array, idx: mx.array,
                    top_k: int = 64, kmin: int = 32, eps: float = 1e-6):
    """Build top-k neighbor graph. Argsort + scatter on GPU.

    xyz: (B, L, 3), pair: (B, L, L, E), idx: (B, L)
    """
    B, L = xyz.shape[:2]

    # Distance matrix
    D = cdist(xyz, xyz) + mx.eye(L).reshape(1, L, L) * 999.9

    # Sequence separation
    sep = mx.expand_dims(idx, 1) - mx.expand_dims(idx, 2)
    sep_abs = mx.abs(sep).astype(mx.float32) + mx.eye(L).reshape(1, L, L) * 999.9
    D = D + sep_abs * eps

    # Top-k via argsort on GPU (eliminates numpy argsort + nested Python loops)
    k = min(top_k, L)
    E_idx = mx.argsort(D, axis=-1)[:, :, :k]  # (B, L, k)

    # Scatter top-k into (B, L, L) mask: one_hot(indices, L) summed over k
    topk_mask = mx.sum(
        mx.eye(L, dtype=mx.float32)[E_idx.astype(mx.int32)],
        axis=2
    ) > 0.0  # (B, L, L)

    # Include all sequence-local neighbors (|sep| < kmin)
    close_mask = sep_abs < kmin
    mask = mx.logical_or(topk_mask, close_mask)

    b_mx, i_mx, j_mx = _mask_to_edges(mask, B, L)

    src = b_mx * L + i_mx
    tgt = b_mx * L + j_mx
    G = simple_graph((src, tgt), num_nodes=B * L)
    G.edata = {'rel_pos': mx.stop_gradient(xyz[b_mx, j_mx, :] - xyz[b_mx, i_mx, :])}

    edge_feats = mx.expand_dims(pair[b_mx, i_mx, j_mx], -1)
    return G, edge_feats
