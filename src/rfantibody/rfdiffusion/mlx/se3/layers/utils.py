"""
Utility functions for SE3-Transformer layers in MLX.
"""

from typing import Dict, List

import mlx.core as mx

from ..fiber import degree_to_dim


def unfuse_features(features: mx.array, degrees: List[int]) -> Dict[str, mx.array]:
    """Split a fused feature tensor back into per-degree dict."""
    sizes = [degree_to_dim(d) for d in degrees]
    # mx.split takes split indices (not chunk sizes)
    indices = []
    acc = 0
    for s in sizes[:-1]:
        acc += s
        indices.append(acc)
    splits = mx.split(features, indices, axis=-1)
    return {str(d): s for d, s in zip(degrees, splits)}


def aggregate_residual(feats1: Dict[str, mx.array],
                       feats2: Dict[str, mx.array],
                       method: str) -> Dict[str, mx.array]:
    """Add or concatenate two fiber feature dicts."""
    if method in ('add', 'sum'):
        return {k: (v + feats1[k]) if k in feats1 else v
                for k, v in feats2.items()}
    elif method in ('cat', 'concat'):
        return {k: mx.concatenate([v, feats1[k]], axis=1) if k in feats1 else v
                for k, v in feats2.items()}
    else:
        raise ValueError('Method must be add/sum or cat/concat')
