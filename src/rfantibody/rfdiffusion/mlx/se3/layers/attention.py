"""
SE(3)-equivariant multi-headed sparse graph self-attention in MLX.
"""

import math
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from ..fiber import Fiber, degree_to_dim
from .convolution import ConvSE3, ConvSE3FuseLevel
from .linear import LinearSE3
from .utils import unfuse_features, aggregate_residual
from ...graph_ops import SimpleGraph, copy_e_sum, e_dot_v, edge_softmax


class AttentionSE3(nn.Module):
    """Multi-headed sparse graph self-attention (SE(3)-equivariant)."""

    def __init__(self, num_heads: int, key_fiber: Fiber, value_fiber: Fiber):
        super().__init__()
        self.num_heads = num_heads
        self.key_fiber = key_fiber
        self.value_fiber = value_fiber

    def __call__(self,
                 value: Union[mx.array, Dict[str, mx.array]],
                 key: Union[mx.array, Dict[str, mx.array]],
                 query: Dict[str, mx.array],
                 graph: SimpleGraph):

        # Reshape keys and queries
        if isinstance(key, mx.array):
            # Fused case
            key = key.reshape(key.shape[0], self.num_heads, -1)
            out = mx.concatenate(
                [query[str(d)] for d in self.key_fiber.degrees], axis=-1)
            query = out.reshape(
                list(query.values())[0].shape[0], self.num_heads, -1)
        else:
            key = self.key_fiber.to_attention_heads(key, self.num_heads)
            query = self.key_fiber.to_attention_heads(query, self.num_heads)

        # Attention weights: dot product + softmax
        edge_weights = e_dot_v(graph, key, query).squeeze(-1)
        edge_weights = edge_weights / math.sqrt(self.key_fiber.num_features)
        edge_weights = edge_softmax(graph, edge_weights)
        edge_weights = mx.expand_dims(mx.expand_dims(edge_weights, -1), -1)

        # Weighted sum
        if isinstance(value, mx.array):
            # Fused case
            v = value.reshape(
                value.shape[0], self.num_heads, -1, value.shape[-1])
            weights = edge_weights * v
            feat_out = copy_e_sum(graph, weights)
            feat_out = feat_out.reshape(
                feat_out.shape[0], -1, feat_out.shape[-1])
            return unfuse_features(feat_out, self.value_fiber.degrees)
        else:
            out = {}
            for degree, channels in self.value_fiber:
                v = value[str(degree)].reshape(
                    -1, self.num_heads, channels // self.num_heads,
                    degree_to_dim(degree))
                weights = edge_weights * v
                res = copy_e_sum(graph, weights)
                out[str(degree)] = res.reshape(
                    -1, channels, degree_to_dim(degree))
            return out


class AttentionBlockSE3(nn.Module):
    """Multi-headed sparse graph self-attention block with skip connection
    and linear projection (SE(3)-equivariant)."""

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber,
                 fiber_edge: Optional[Fiber] = None,
                 num_heads: int = 4,
                 channels_div: Optional[Dict[str, int]] = None,
                 use_layer_norm: bool = False,
                 max_degree: int = 4,
                 fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.FULL,
                 **kwargs):
        super().__init__()
        if fiber_edge is None:
            fiber_edge = Fiber({})
        self.fiber_in = fiber_in

        if channels_div is not None:
            value_fiber = Fiber([
                (degree, channels // channels_div[str(degree)])
                for degree, channels in fiber_out
            ])
        else:
            value_fiber = Fiber([
                (degree, channels) for degree, channels in fiber_out
            ])

        key_query_fiber = Fiber([
            (fe.degree, fe.channels)
            for fe in value_fiber if fe.degree in fiber_in.degrees
        ])

        self.to_key_value = ConvSE3(
            fiber_in, value_fiber + key_query_fiber, pool=False,
            fiber_edge=fiber_edge, use_layer_norm=use_layer_norm,
            max_degree=max_degree, fuse_level=fuse_level,
            allow_fused_output=True)
        self.to_query = LinearSE3(fiber_in, key_query_fiber)
        self.attention = AttentionSE3(num_heads, key_query_fiber, value_fiber)
        self.project = LinearSE3(value_fiber + fiber_in, fiber_out)

    def __call__(self, node_features: Dict[str, mx.array],
                 edge_features: Dict[str, mx.array],
                 graph: SimpleGraph,
                 basis: Dict[str, mx.array]):

        fused_key_value = self.to_key_value(
            node_features, edge_features, graph, basis)
        key, value = self._get_key_value_from_fused(fused_key_value)

        query = self.to_query(node_features)

        z = self.attention(value, key, query, graph)
        z_concat = aggregate_residual(node_features, z, 'cat')
        return self.project(z_concat)

    def _get_key_value_from_fused(self, fused_key_value):
        if isinstance(fused_key_value, mx.array):
            # Fully fused: split along channel dim
            mid = fused_key_value.shape[-2] // 2
            value = fused_key_value[:, :mid, :]
            key = fused_key_value[:, mid:, :]
        else:
            key, value = {}, {}
            for degree, feat in fused_key_value.items():
                if int(degree) in self.fiber_in.degrees:
                    mid = feat.shape[-2] // 2
                    value[degree] = feat[:, :mid, :]
                    key[degree] = feat[:, mid:, :]
                else:
                    value[degree] = feat
        return key, value
