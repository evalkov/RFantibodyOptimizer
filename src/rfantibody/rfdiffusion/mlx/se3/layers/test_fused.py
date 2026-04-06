#!/usr/bin/env python3
"""Test ConvSE3 with FULL fuse level and AttentionBlockSE3."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', '..', 'include', 'SE3Transformer'))

import torch
import mlx.core as mx


def _copy_radial_weights(torch_rp, mlx_rp):
    """Copy RadialProfile weights from PyTorch to MLX."""
    for i in range(len(torch_rp.net)):
        tl = torch_rp.net[i]
        if hasattr(tl, 'weight'):
            mlx_rp.net.layers[i].weight = mx.array(tl.weight.detach().numpy())
            if hasattr(tl, 'bias') and tl.bias is not None:
                mlx_rp.net.layers[i].bias = mx.array(tl.bias.detach().numpy())


def test_conv_full_fuse():
    from se3_transformer.model.fiber import Fiber as TorchFiber
    from se3_transformer.model.layers.convolution import (
        ConvSE3 as TorchConvSE3, ConvSE3FuseLevel as TorchFuse)
    from rfantibody.rfdiffusion.mlx.se3.fiber import Fiber as MLXFiber
    from rfantibody.rfdiffusion.mlx.se3.layers.convolution import (
        ConvSE3 as MLXConvSE3, ConvSE3FuseLevel as MLXFuse)
    from rfantibody.rfdiffusion.mlx.se3.basis import get_basis, update_basis_with_fused
    from rfantibody.rfdiffusion.mlx.graph_ops import SimpleGraph as MLXGraph
    from rfantibody.rfdiffusion.mps_graph import SimpleGraph as TorchGraph

    print("=" * 60)
    print("Test: ConvSE3 (FULL fuse level)")
    print("=" * 60)

    N, E = 20, 80
    max_degree = 2
    channels = 16

    # FULL fuse requires: same channels in/out, degrees 0..max_degree
    fiber_in = TorchFiber.create(max_degree + 1, channels)
    fiber_out = TorchFiber.create(max_degree + 1, channels)
    fiber_edge = TorchFiber({})

    torch_conv = TorchConvSE3(
        fiber_in, fiber_out, fiber_edge,
        pool=True, self_interaction=True,
        max_degree=max_degree, fuse_level=TorchFuse.FULL)
    torch_conv.eval()

    mlx_fiber_in = MLXFiber.create(max_degree + 1, channels)
    mlx_fiber_out = MLXFiber.create(max_degree + 1, channels)
    mlx_fiber_edge = MLXFiber({})

    mlx_conv = MLXConvSE3(
        mlx_fiber_in, mlx_fiber_out, mlx_fiber_edge,
        pool=True, self_interaction=True,
        max_degree=max_degree, fuse_level=MLXFuse.FULL)

    # Copy single conv radial weights
    _copy_radial_weights(torch_conv.conv.radial_func, mlx_conv.conv.radial_func)

    # Copy self-interaction weights
    for degree in torch_conv.to_kernel_self:
        mlx_conv.to_kernel_self[degree] = mx.array(
            torch_conv.to_kernel_self[degree].detach().numpy())

    # Create graph
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    torch_graph = TorchGraph(src, dst, N)
    mlx_graph = MLXGraph(mx.array(src.numpy()), mx.array(dst.numpy()), N)

    # Node and edge features
    torch_node_feats = {
        str(d): torch.randn(N, channels, 2 * d + 1)
        for d in range(max_degree + 1)
    }
    torch_edge_feats = {'0': torch.randn(E, 1, 1)}

    mlx_node_feats = {k: mx.array(v.numpy()) for k, v in torch_node_feats.items()}
    mlx_edge_feats = {k: mx.array(v.numpy()) for k, v in torch_edge_feats.items()}

    # Create basis with fused versions
    rel_pos = torch.randn(E, 3)
    mlx_basis = get_basis(mx.array(rel_pos.numpy()), max_degree=max_degree)
    mlx_basis = update_basis_with_fused(
        mlx_basis, max_degree, use_pad_trick=False, fully_fused=True)
    mx.eval(*mlx_basis.values())
    torch_basis = {k: torch.tensor(np.array(v)) for k, v in mlx_basis.items()}

    with torch.no_grad():
        torch_out = torch_conv(torch_node_feats, torch_edge_feats,
                               torch_graph, torch_basis)
    mlx_out = mlx_conv(mlx_node_feats, mlx_edge_feats, mlx_graph, mlx_basis)
    mx.eval(*mlx_out.values())

    for degree in torch_out:
        t_np = torch_out[degree].detach().numpy()
        m_np = np.array(mlx_out[degree])
        diff = np.max(np.abs(t_np - m_np))
        print(f"  degree {degree}: shape {t_np.shape}, max diff = {diff:.2e}")
        assert diff < 1e-3, f"ConvSE3 FULL degree {degree} diff too large: {diff}"

    print("  PASSED\n")


def test_attention_block():
    from se3_transformer.model.fiber import Fiber as TorchFiber
    from se3_transformer.model.layers.attention import (
        AttentionBlockSE3 as TorchAttBlock)
    from se3_transformer.model.layers.convolution import ConvSE3FuseLevel as TorchFuse
    from rfantibody.rfdiffusion.mlx.se3.fiber import Fiber as MLXFiber
    from rfantibody.rfdiffusion.mlx.se3.layers.attention import (
        AttentionBlockSE3 as MLXAttBlock)
    from rfantibody.rfdiffusion.mlx.se3.layers.convolution import (
        ConvSE3FuseLevel as MLXFuse)
    from rfantibody.rfdiffusion.mlx.se3.basis import get_basis, update_basis_with_fused
    from rfantibody.rfdiffusion.mlx.graph_ops import SimpleGraph as MLXGraph
    from rfantibody.rfdiffusion.mps_graph import SimpleGraph as TorchGraph

    print("=" * 60)
    print("Test: AttentionBlockSE3")
    print("=" * 60)

    N, E = 20, 80
    max_degree = 2
    channels = 16
    num_heads = 4

    fiber_in = TorchFiber.create(max_degree + 1, channels)
    fiber_out = TorchFiber.create(max_degree + 1, channels)
    fiber_edge = TorchFiber({})

    torch_att = TorchAttBlock(
        fiber_in, fiber_out, fiber_edge=fiber_edge,
        num_heads=num_heads, use_layer_norm=False,
        max_degree=max_degree, fuse_level=TorchFuse.FULL)
    torch_att.eval()

    mlx_fiber_in = MLXFiber.create(max_degree + 1, channels)
    mlx_fiber_out = MLXFiber.create(max_degree + 1, channels)
    mlx_fiber_edge = MLXFiber({})

    mlx_att = MLXAttBlock(
        mlx_fiber_in, mlx_fiber_out, fiber_edge=mlx_fiber_edge,
        num_heads=num_heads, use_layer_norm=False,
        max_degree=max_degree, fuse_level=MLXFuse.FULL)

    # Copy all weights from PyTorch to MLX
    # to_key_value conv
    _copy_radial_weights(
        torch_att.to_key_value.conv.radial_func,
        mlx_att.to_key_value.conv.radial_func)
    if hasattr(torch_att.to_key_value, 'to_kernel_self'):
        for degree in torch_att.to_key_value.to_kernel_self:
            mlx_att.to_key_value.to_kernel_self[degree] = mx.array(
                torch_att.to_key_value.to_kernel_self[degree].detach().numpy())

    # to_query linear
    for degree in torch_att.to_query.weights:
        mlx_att.to_query.weights[degree] = mx.array(
            torch_att.to_query.weights[degree].detach().numpy())

    # project linear
    for degree in torch_att.project.weights:
        mlx_att.project.weights[degree] = mx.array(
            torch_att.project.weights[degree].detach().numpy())

    # Graph
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    torch_graph = TorchGraph(src, dst, N)
    mlx_graph = MLXGraph(mx.array(src.numpy()), mx.array(dst.numpy()), N)

    # Features
    torch_node_feats = {
        str(d): torch.randn(N, channels, 2 * d + 1)
        for d in range(max_degree + 1)
    }
    torch_edge_feats = {'0': torch.randn(E, 1, 1)}
    mlx_node_feats = {k: mx.array(v.numpy()) for k, v in torch_node_feats.items()}
    mlx_edge_feats = {k: mx.array(v.numpy()) for k, v in torch_edge_feats.items()}

    # Basis
    rel_pos = torch.randn(E, 3)
    mlx_basis = get_basis(mx.array(rel_pos.numpy()), max_degree=max_degree)
    mlx_basis = update_basis_with_fused(
        mlx_basis, max_degree, use_pad_trick=False, fully_fused=True)
    mx.eval(*mlx_basis.values())
    torch_basis = {k: torch.tensor(np.array(v)) for k, v in mlx_basis.items()}

    with torch.no_grad():
        torch_out = torch_att(torch_node_feats, torch_edge_feats,
                              torch_graph, torch_basis)
    mlx_out = mlx_att(mlx_node_feats, mlx_edge_feats, mlx_graph, mlx_basis)
    mx.eval(*mlx_out.values())

    for degree in torch_out:
        t_np = torch_out[degree].detach().numpy()
        m_np = np.array(mlx_out[degree])
        diff = np.max(np.abs(t_np - m_np))
        print(f"  degree {degree}: shape {t_np.shape}, max diff = {diff:.2e}")
        assert diff < 1e-3, f"AttentionBlockSE3 degree {degree} diff too large: {diff}"

    print("  PASSED\n")


if __name__ == '__main__':
    test_conv_full_fuse()
    test_attention_block()
    print("=" * 60)
    print("ALL FUSED TESTS PASSED")
    print("=" * 60)
