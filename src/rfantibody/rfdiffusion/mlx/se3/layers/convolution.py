"""
SE(3)-equivariant graph convolution (Tensor Field Network) in MLX.

Provides RadialProfile, VersatileConvSE3, and ConvSE3 with the same
fuse-level optimization hierarchy as the PyTorch version.
"""

import math
from enum import Enum
from itertools import product
from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from ..fiber import Fiber, degree_to_dim
from .utils import unfuse_features
from ...graph_ops import SimpleGraph, copy_e_sum, copy_e_mean


class ConvSE3FuseLevel(Enum):
    FULL = 2
    PARTIAL = 1
    NONE = 0


class RadialProfile(nn.Module):
    """Radial profile function: invariant edge features -> radial weights.

    MLP: edge_dim -> mid_dim -> mid_dim -> num_freq * channels_in * channels_out
    """

    def __init__(self, num_freq: int, channels_in: int, channels_out: int,
                 edge_dim: int = 1, mid_dim: int = 32,
                 use_layer_norm: bool = False):
        super().__init__()
        layers = [nn.Linear(edge_dim, mid_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(mid_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(mid_dim, mid_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(mid_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(mid_dim, num_freq * channels_in * channels_out,
                                bias=False))
        self.net = nn.Sequential(*layers)

    def __call__(self, features: mx.array) -> mx.array:
        return self.net(features)


class VersatileConvSE3(nn.Module):
    """Building block for TFN convolutions.

    Handles fully fused, partially fused, or pairwise convolutions.
    """

    def __init__(self, freq_sum: int, channels_in: int, channels_out: int,
                 edge_dim: int, use_layer_norm: bool,
                 fuse_level: ConvSE3FuseLevel):
        super().__init__()
        self.freq_sum = freq_sum
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.fuse_level = fuse_level
        self.radial_func = RadialProfile(
            num_freq=freq_sum, channels_in=channels_in,
            channels_out=channels_out, edge_dim=edge_dim,
            use_layer_norm=use_layer_norm)

    def __call__(self, features: mx.array, invariant_edge_feats: mx.array,
                 basis: mx.array) -> mx.array:
        # Dispatch to fused Metal kernel if available
        if hasattr(self, '_fused') and self._fused is not None and basis is not None:
            return self._fused(features, invariant_edge_feats, basis)
        num_edges = features.shape[0]
        in_dim = features.shape[2]

        radial_weights = self.radial_func(invariant_edge_feats)
        radial_weights = radial_weights.reshape(
            -1, self.channels_out, self.channels_in * self.freq_sum)

        if basis is not None:
            out_dim = basis.shape[-1]
            if self.fuse_level != ConvSE3FuseLevel.FULL:
                out_dim += out_dim % 2 - 1  # Account for padded basis

            basis_view = basis.reshape(num_edges, in_dim, -1)
            # features: (E, C_in, in_dim), basis_view: (E, in_dim, freq*out_dim)
            tmp = (features @ basis_view).reshape(num_edges, -1, basis.shape[-1])
            # radial_weights: (E, C_out, C_in*freq), tmp: (E, C_in*freq, out_dim)
            return (radial_weights @ tmp)[:, :, :out_dim]
        else:
            # k = l = 0 non-fused case
            return radial_weights @ features


class ConvSE3(nn.Module):
    """SE(3)-equivariant graph convolution (Tensor Field Network).

    Maps fiber_in to fiber_out using equivariant kernels built from
    spherical harmonics basis and learned radial profiles.
    """

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber, fiber_edge: Fiber,
                 pool: bool = True, use_layer_norm: bool = False,
                 self_interaction: bool = False, sum_over_edge: bool = True,
                 max_degree: int = 4,
                 fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.FULL,
                 allow_fused_output: bool = False):
        super().__init__()
        self.pool = pool
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.self_interaction = self_interaction
        self.sum_over_edge = sum_over_edge
        self.max_degree = max_degree
        self.allow_fused_output = allow_fused_output

        channels_in_set = set(
            f.channels + fiber_edge[f.degree] * (f.degree > 0)
            for f in self.fiber_in
        )
        channels_out_set = set(f.channels for f in self.fiber_out)
        unique_channels_in = (len(channels_in_set) == 1)
        unique_channels_out = (len(channels_out_set) == 1)
        degrees_up_to_max = list(range(max_degree + 1))
        common_args = dict(edge_dim=fiber_edge[0] + 1,
                           use_layer_norm=use_layer_norm)

        if (fuse_level.value >= ConvSE3FuseLevel.FULL.value
                and unique_channels_in
                and fiber_in.degrees == degrees_up_to_max
                and unique_channels_out
                and fiber_out.degrees == degrees_up_to_max):
            self.used_fuse_level = ConvSE3FuseLevel.FULL
            sum_freq = sum(
                degree_to_dim(min(d_in, d_out))
                for d_in, d_out in product(degrees_up_to_max, degrees_up_to_max)
            )
            self.conv = VersatileConvSE3(
                sum_freq, list(channels_in_set)[0],
                list(channels_out_set)[0],
                fuse_level=self.used_fuse_level, **common_args)

        elif (fuse_level.value >= ConvSE3FuseLevel.PARTIAL.value
              and unique_channels_in
              and fiber_in.degrees == degrees_up_to_max):
            self.used_fuse_level = ConvSE3FuseLevel.PARTIAL
            self.conv_out = {}
            for d_out, c_out in fiber_out:
                sum_freq = sum(
                    degree_to_dim(min(d_out, d))
                    for d in fiber_in.degrees
                )
                self.conv_out[str(d_out)] = VersatileConvSE3(
                    sum_freq, list(channels_in_set)[0], c_out,
                    fuse_level=self.used_fuse_level, **common_args)

        elif (fuse_level.value >= ConvSE3FuseLevel.PARTIAL.value
              and unique_channels_out
              and fiber_out.degrees == degrees_up_to_max):
            self.used_fuse_level = ConvSE3FuseLevel.PARTIAL
            self.conv_in = {}
            for d_in, c_in in fiber_in:
                sum_freq = sum(
                    degree_to_dim(min(d_in, d))
                    for d in fiber_out.degrees
                )
                self.conv_in[str(d_in)] = VersatileConvSE3(
                    sum_freq, c_in, list(channels_out_set)[0],
                    fuse_level=ConvSE3FuseLevel.FULL, **common_args)
        else:
            self.used_fuse_level = ConvSE3FuseLevel.NONE
            self.conv = {}
            for (degree_in, channels_in), (degree_out, channels_out) in (
                    self.fiber_in * self.fiber_out):
                dict_key = f'{degree_in},{degree_out}'
                channels_in_new = channels_in + fiber_edge[degree_in] * (degree_in > 0)
                sum_freq = degree_to_dim(min(degree_in, degree_out))
                self.conv[dict_key] = VersatileConvSE3(
                    sum_freq, channels_in_new, channels_out,
                    fuse_level=self.used_fuse_level, **common_args)

        if self_interaction:
            self.to_kernel_self = {}
            for degree_out, channels_out in fiber_out:
                if fiber_in[degree_out]:
                    self.to_kernel_self[str(degree_out)] = (
                        mx.random.normal(shape=(channels_out, fiber_in[degree_out]))
                        / math.sqrt(fiber_in[degree_out])
                    )

    def enable_fused_kernels(self, edge_dim: int):
        """Enable fused Metal kernels for VersatileConvSE3 instances.

        Only enables for configs where the fused kernel is faster than
        MLX's BLAS-backed implementation.
        """
        from ...kernels import FusedConvSE3

        def _try_fuse(conv, label=""):
            """Attach fused kernel to a VersatileConvSE3 if beneficial."""
            conv._fused = FusedConvSE3(conv, edge_dim=edge_dim)

        if self.used_fuse_level == ConvSE3FuseLevel.FULL:
            _try_fuse(self.conv, "full")
        elif hasattr(self, 'conv_out'):
            for key, conv in self.conv_out.items():
                _try_fuse(conv, f"conv_out[{key}]")
        elif hasattr(self, 'conv_in'):
            for key, conv in self.conv_in.items():
                _try_fuse(conv, f"conv_in[{key}]")
        else:
            for key, conv in self.conv.items():
                _try_fuse(conv, f"conv[{key}]")

    def __call__(self, node_feats: Dict[str, mx.array],
                 edge_feats: Dict[str, mx.array],
                 graph: SimpleGraph,
                 basis: Dict[str, mx.array]) -> Dict[str, mx.array]:
        invariant_edge_feats = edge_feats['0'].squeeze(-1)
        src, dst = graph.edges()
        out = {}
        in_features = []

        for degree_in in self.fiber_in.degrees:
            src_node_features = node_feats[str(degree_in)][src]
            if degree_in > 0 and str(degree_in) in edge_feats:
                src_node_features = mx.concatenate(
                    [src_node_features, edge_feats[str(degree_in)]], axis=1)
            in_features.append(src_node_features)

        if self.used_fuse_level == ConvSE3FuseLevel.FULL:
            in_features_fused = mx.concatenate(in_features, axis=-1)
            out = self.conv(in_features_fused, invariant_edge_feats,
                            basis['fully_fused'])
            if not self.allow_fused_output or self.self_interaction or self.pool:
                out = unfuse_features(out, self.fiber_out.degrees)

        elif (self.used_fuse_level == ConvSE3FuseLevel.PARTIAL
              and hasattr(self, 'conv_out')):
            in_features_fused = mx.concatenate(in_features, axis=-1)
            for degree_out in self.fiber_out.degrees:
                out[str(degree_out)] = self.conv_out[str(degree_out)](
                    in_features_fused, invariant_edge_feats,
                    basis[f'out{degree_out}_fused'])

        elif (self.used_fuse_level == ConvSE3FuseLevel.PARTIAL
              and hasattr(self, 'conv_in')):
            out = 0
            for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                out = out + self.conv_in[str(degree_in)](
                    feature, invariant_edge_feats,
                    basis[f'in{degree_in}_fused'])
            if not self.allow_fused_output or self.self_interaction or self.pool:
                out = unfuse_features(out, self.fiber_out.degrees)
        else:
            for degree_out in self.fiber_out.degrees:
                out_feature = 0
                for degree_in, feature in zip(self.fiber_in.degrees,
                                              in_features):
                    dict_key = f'{degree_in},{degree_out}'
                    out_feature = out_feature + self.conv[dict_key](
                        feature, invariant_edge_feats,
                        basis.get(dict_key, None))
                out[str(degree_out)] = out_feature

        for degree_out in self.fiber_out.degrees:
            if (self.self_interaction
                    and str(degree_out) in self.to_kernel_self):
                dst_features = node_feats[str(degree_out)][dst]
                kernel_self = self.to_kernel_self[str(degree_out)]
                out[str(degree_out)] = out[str(degree_out)] + kernel_self @ dst_features

            if self.pool:
                if self.sum_over_edge:
                    if isinstance(out, dict):
                        out[str(degree_out)] = copy_e_sum(
                            graph, out[str(degree_out)])
                    else:
                        out = copy_e_sum(graph, out)
                else:
                    if isinstance(out, dict):
                        out[str(degree_out)] = copy_e_mean(
                            graph, out[str(degree_out)])
                    else:
                        out = copy_e_mean(graph, out)

        return out
