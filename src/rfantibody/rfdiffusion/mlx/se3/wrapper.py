"""
SE3TransformerWrapper for RFdiffusion in MLX.

Bridge between RFdiffusion's track modules and the SE3Transformer.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .fiber import Fiber
from .transformer import SE3Transformer
from ..graph_ops import SimpleGraph


class SE3TransformerWrapper(nn.Module):
    """SE(3) equivariant GCN with attention."""

    def __init__(self, num_layers=2, num_channels=32, num_degrees=3,
                 n_heads=4, div=4,
                 l0_in_features=32, l0_out_features=32,
                 l1_in_features=3, l1_out_features=2,
                 num_edge_features=32,
                 final_layer='conv', populate_edge='lin'):
        super().__init__()
        self.l1_in = l1_in_features

        fiber_edge = Fiber({0: num_edge_features})
        if l1_out_features > 0:
            if l1_in_features > 0:
                fiber_in = Fiber({0: l0_in_features, 1: l1_in_features})
            else:
                fiber_in = Fiber({0: l0_in_features})
            fiber_hidden = Fiber.create(num_degrees, num_channels)
            fiber_out = Fiber({0: l0_out_features, 1: l1_out_features})
        else:
            if l1_in_features > 0:
                fiber_in = Fiber({0: l0_in_features, 1: l1_in_features})
            else:
                fiber_in = Fiber({0: l0_in_features})
            fiber_hidden = Fiber.create(num_degrees, num_channels)
            fiber_out = Fiber({0: l0_out_features})

        self.se3 = SE3Transformer(
            num_layers=num_layers,
            fiber_in=fiber_in,
            fiber_hidden=fiber_hidden,
            fiber_out=fiber_out,
            num_heads=n_heads,
            channels_div=div,
            fiber_edge=fiber_edge,
            final_layer=final_layer,
            populate_edge=populate_edge,
            use_layer_norm=True)

    def enable_fused_kernels(self):
        """Enable fused Metal kernels for SE3 convolutions."""
        self.se3.enable_fused_kernels()

    def __call__(self, G: SimpleGraph,
                 type_0_features: mx.array,
                 type_1_features: Optional[mx.array] = None,
                 edge_features: Optional[mx.array] = None,
                 basis: Optional[dict] = None):
        if self.l1_in > 0:
            node_features = {'0': type_0_features, '1': type_1_features}
        else:
            node_features = {'0': type_0_features}
        edge_features_dict = {'0': edge_features}
        return self.se3(G, node_features, edge_features_dict, basis=basis)
