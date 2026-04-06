"""
SE(3)-equivariant linear layer in MLX.

Per-degree matrix multiplication (1x1 convolution).
Maps a fiber to a fiber with the same degrees but possibly different channels.
"""

import math
from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from ..fiber import Fiber


class LinearSE3(nn.Module):
    """Graph Linear SE(3)-equivariant layer.

    type-k features (C_k channels) --> matmul --> type-k features (C'_k channels)
    No interaction between degrees.
    """

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber):
        super().__init__()
        # Dict of weight matrices keyed by degree string
        self.weights = {
            str(degree_out): mx.random.normal(
                shape=(channels_out, fiber_in[degree_out])
            ) / math.sqrt(fiber_in[degree_out])
            for degree_out, channels_out in fiber_out
        }

    def __call__(self, features: Dict[str, mx.array], *args, **kwargs) -> Dict[str, mx.array]:
        return {
            degree: self.weights[degree] @ features[degree]
            for degree in self.weights
        }
