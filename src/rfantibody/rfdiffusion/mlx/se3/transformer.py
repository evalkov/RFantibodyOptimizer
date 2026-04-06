"""
SE(3)-equivariant Transformer in MLX.

Main SE3Transformer module composing attention blocks, norm layers,
and a final convolution or linear layer.
"""

from typing import Dict, Optional, Literal

import mlx.core as mx
import mlx.nn as nn

from .fiber import Fiber
from .basis import get_basis, update_basis_with_fused
from .layers.attention import AttentionBlockSE3
from .layers.convolution import ConvSE3, ConvSE3FuseLevel
from .layers.linear import LinearSE3
from .layers.norm import NormSE3
from ..graph_ops import SimpleGraph


class GraphSequential(nn.Module):
    """Sequential that passes graph, edge_feats, and basis through all layers."""

    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)

    def __call__(self, x, edge_feats, graph, basis):
        for layer in self.layers:
            x = layer(x, edge_feats, graph=graph, basis=basis)
        return x


def get_populated_edge_features(relative_pos: mx.array,
                                edge_features: Optional[Dict[str, mx.array]] = None
                                ) -> Dict[str, mx.array]:
    """Add relative position norms to edge features."""
    edge_features = dict(edge_features) if edge_features else {}
    r = mx.linalg.norm(relative_pos, axis=-1, keepdims=True)
    if '0' in edge_features:
        edge_features['0'] = mx.concatenate(
            [edge_features['0'], mx.expand_dims(r, -1)], axis=1)
    else:
        edge_features['0'] = mx.expand_dims(r, -1)
    return edge_features


class SE3Transformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 num_heads: int,
                 channels_div: int,
                 fiber_edge: Fiber = Fiber({}),
                 return_type: Optional[int] = None,
                 final_layer: Optional[Literal['conv', 'lin', 'att']] = 'conv',
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = False,
                 populate_edge: Optional[Literal['lin', 'arcsin', 'log', 'zero']] = 'lin',
                 sum_over_edge: bool = True,
                 **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.return_type = return_type
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees,
                              *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory
        self.populate_edge = populate_edge

        fuse_level = (ConvSE3FuseLevel.FULL
                      if tensor_cores and not low_memory
                      else ConvSE3FuseLevel.PARTIAL)

        div = {str(degree): channels_div
               for degree in range(self.max_degree + 1)}
        div_fin = {str(degree): 1
                   for degree in range(self.max_degree + 1)}
        div_fin['0'] = channels_div

        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(AttentionBlockSE3(
                fiber_in=fiber_in, fiber_out=fiber_hidden,
                fiber_edge=fiber_edge, num_heads=num_heads,
                channels_div=div, use_layer_norm=use_layer_norm,
                max_degree=self.max_degree, fuse_level=fuse_level))
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        if final_layer == 'conv':
            graph_modules.append(ConvSE3(
                fiber_in=fiber_in, fiber_out=fiber_out,
                fiber_edge=fiber_edge, self_interaction=True,
                sum_over_edge=sum_over_edge,
                use_layer_norm=use_layer_norm,
                max_degree=self.max_degree))
        elif final_layer == 'lin':
            graph_modules.append(LinearSE3(
                fiber_in=fiber_in, fiber_out=fiber_out))
        else:
            graph_modules.append(AttentionBlockSE3(
                fiber_in=fiber_in, fiber_out=fiber_out,
                fiber_edge=fiber_edge, num_heads=1,
                channels_div=div_fin, use_layer_norm=use_layer_norm,
                max_degree=self.max_degree, fuse_level=fuse_level))

        self.graph_modules = GraphSequential(*graph_modules)

    def enable_fused_kernels(self):
        """Enable fused Metal kernels for final ConvSE3 layers.

        Only fuses the final convolution, not attention's to_key_value,
        because tiny numerical differences in attention keys compound
        through softmax and cause instability over multiple diffusion steps.
        """
        edge_dim = self.fiber_edge[0] + 1  # +1 for distance feature
        for layer in self.graph_modules.layers:
            if isinstance(layer, ConvSE3):
                layer.enable_fused_kernels(edge_dim)

    def __call__(self, graph: SimpleGraph,
                 node_feats: Dict[str, mx.array],
                 edge_feats: Optional[Dict[str, mx.array]] = None,
                 basis: Optional[Dict[str, mx.array]] = None):

        # Compute basis if not provided
        if basis is None:
            use_pad = self.tensor_cores and not self.low_memory
            basis = get_basis(graph.edata['rel_pos'],
                              max_degree=self.max_degree,
                              use_pad_trick=use_pad)

        # Add fused bases
        basis = update_basis_with_fused(
            basis, self.max_degree,
            use_pad_trick=self.tensor_cores and not self.low_memory,
            fully_fused=self.tensor_cores and not self.low_memory)

        # Populate edge features
        if self.populate_edge == 'lin':
            edge_feats = get_populated_edge_features(
                graph.edata['rel_pos'], edge_feats)
        elif self.populate_edge == 'arcsin':
            r = mx.linalg.norm(graph.edata['rel_pos'], axis=-1, keepdims=True)
            r = mx.maximum(r, mx.full(r.shape, 4.0, dtype=r.dtype)) - 4.0
            r = mx.arcsinh(r) / 3.0
            edge_feats['0'] = mx.concatenate(
                [edge_feats['0'], mx.expand_dims(r, -1)], axis=1)
        elif self.populate_edge == 'log':
            r = mx.log(1 + mx.linalg.norm(
                graph.edata['rel_pos'], axis=-1, keepdims=True))
            edge_feats['0'] = mx.concatenate(
                [edge_feats['0'], mx.expand_dims(r, -1)], axis=1)
        else:
            edge_feats['0'] = mx.concatenate(
                [edge_feats['0'],
                 mx.zeros_like(edge_feats['0'][:, :1, :])], axis=1)

        node_feats = self.graph_modules(
            node_feats, edge_feats, graph=graph, basis=basis)

        if self.return_type is not None:
            return node_feats[str(self.return_type)]

        return node_feats
