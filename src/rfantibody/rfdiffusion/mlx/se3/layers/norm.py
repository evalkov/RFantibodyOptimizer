"""
SE(3)-equivariant norm-based nonlinearity in MLX.

              ┌──> feature_norm ──> LayerNorm/GroupNorm ──> ReLU ──┐
feature_in ──┤                                                      * ──> feature_out
              └──> feature_phase ──────────────────────────────────┘
"""

from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from ..fiber import Fiber


class NormSE3(nn.Module):
    """Norm-based SE(3)-equivariant nonlinearity."""

    NORM_CLAMP = 2 ** -24

    def __init__(self, fiber: Fiber):
        super().__init__()
        self.fiber = fiber

        if len(set(fiber.channels)) == 1:
            # Fuse all layer normalizations into a single group normalization
            self.group_norm = nn.GroupNorm(
                num_groups=len(fiber.degrees),
                dims=sum(fiber.channels),
                pytorch_compatible=True
            )
        else:
            # Per-degree layer normalizations
            self.layer_norms = {
                str(degree): nn.LayerNorm(channels)
                for degree, channels in fiber
            }

    def __call__(self, features: Dict[str, mx.array], *args, **kwargs) -> Dict[str, mx.array]:
        output = {}

        if hasattr(self, 'group_norm'):
            # Compute per-degree norms
            norms = []
            for d in self.fiber.degrees:
                feat = features[str(d)]
                # feat: (N, C, 2d+1) -> norm over last dim -> (N, C, 1)
                n = mx.linalg.norm(feat, axis=-1, keepdims=True)
                n = mx.maximum(n, self.NORM_CLAMP)
                norms.append(n)

            # Fuse norms: cat along channel dim -> (N, sum_C, 1)
            fused_norms = mx.concatenate(norms, axis=-2)

            # GroupNorm expects (N, ..., C) but we have (N, C, 1)
            # Squeeze last dim -> (N, sum_C), apply group_norm, unsqueeze back
            new_norms = self.group_norm(fused_norms.squeeze(-1))
            new_norms = nn.relu(new_norms)
            new_norms = mx.expand_dims(new_norms, axis=-1)

            # Split back per degree
            sizes = [features[str(d)].shape[-2] for d in self.fiber.degrees]
            split_indices = []
            acc = 0
            for s in sizes[:-1]:
                acc += s
                split_indices.append(acc)
            new_norms_split = mx.split(new_norms, split_indices, axis=-2)

            for norm, new_norm, d in zip(norms, new_norms_split, self.fiber.degrees):
                output[str(d)] = features[str(d)] / norm * new_norm
        else:
            for degree, feat in features.items():
                # feat: (N, C, 2d+1)
                norm = mx.linalg.norm(feat, axis=-1, keepdims=True)
                norm = mx.maximum(norm, self.NORM_CLAMP)
                # LayerNorm on (N, C) then unsqueeze
                new_norm = self.layer_norms[degree](norm.squeeze(-1))
                new_norm = nn.relu(mx.expand_dims(new_norm, axis=-1))
                output[degree] = new_norm * feat / norm

        return output
