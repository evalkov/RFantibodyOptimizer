#!/usr/bin/env python3
"""
Test SE3 layers: compare MLX implementations against PyTorch originals.

Run from project root with the pilot_mps venv:
  pilot_mps/.venv/bin/python -m rfantibody.rfdiffusion.mlx.se3.layers.test_layers
"""

import sys
import os
import math
import numpy as np

# Add SE3Transformer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', '..', 'include', 'SE3Transformer'))

import torch
import mlx.core as mx

# =====================================================================
# Test 1: LinearSE3
# =====================================================================
def test_linear():
    from se3_transformer.model.fiber import Fiber as TorchFiber
    from se3_transformer.model.layers.linear import LinearSE3 as TorchLinearSE3
    from rfantibody.rfdiffusion.mlx.se3.fiber import Fiber as MLXFiber
    from rfantibody.rfdiffusion.mlx.se3.layers.linear import LinearSE3 as MLXLinearSE3

    print("=" * 60)
    print("Test 1: LinearSE3")
    print("=" * 60)

    fiber_in = TorchFiber([(0, 16), (1, 8)])
    fiber_out = TorchFiber([(0, 32), (1, 16)])
    N = 50

    torch_linear = TorchLinearSE3(fiber_in, fiber_out)

    # Create MLX version with same weights
    mlx_fiber_in = MLXFiber([(0, 16), (1, 8)])
    mlx_fiber_out = MLXFiber([(0, 32), (1, 16)])
    mlx_linear = MLXLinearSE3(mlx_fiber_in, mlx_fiber_out)

    # Copy weights
    for degree in torch_linear.weights:
        w_np = torch_linear.weights[degree].detach().cpu().numpy()
        mlx_linear.weights[degree] = mx.array(w_np)

    # Create input features
    torch_feats = {
        '0': torch.randn(N, 16, 1),
        '1': torch.randn(N, 8, 3),
    }
    mlx_feats = {k: mx.array(v.numpy()) for k, v in torch_feats.items()}

    # Forward pass
    torch_out = torch_linear(torch_feats)
    mlx_out = mlx_linear(mlx_feats)
    mx.eval(*mlx_out.values())

    for degree in torch_out:
        t_np = torch_out[degree].detach().numpy()
        m_np = np.array(mlx_out[degree])
        diff = np.max(np.abs(t_np - m_np))
        print(f"  degree {degree}: shape {t_np.shape}, max diff = {diff:.2e}")
        assert diff < 1e-5, f"LinearSE3 degree {degree} diff too large: {diff}"

    print("  PASSED\n")


# =====================================================================
# Test 2: NormSE3
# =====================================================================
def test_norm():
    from se3_transformer.model.fiber import Fiber as TorchFiber
    from se3_transformer.model.layers.norm import NormSE3 as TorchNormSE3
    from rfantibody.rfdiffusion.mlx.se3.fiber import Fiber as MLXFiber
    from rfantibody.rfdiffusion.mlx.se3.layers.norm import NormSE3 as MLXNormSE3

    print("=" * 60)
    print("Test 2: NormSE3")
    print("=" * 60)

    # Test GroupNorm path (all channels equal)
    fiber = TorchFiber([(0, 16), (1, 16)])
    N = 50

    torch_norm = TorchNormSE3(fiber)
    torch_norm.eval()

    mlx_fiber = MLXFiber([(0, 16), (1, 16)])
    mlx_norm = MLXNormSE3(mlx_fiber)

    # Copy group_norm weights
    gn = torch_norm.group_norm
    mlx_norm.group_norm.weight = mx.array(gn.weight.detach().numpy())
    mlx_norm.group_norm.bias = mx.array(gn.bias.detach().numpy())

    torch_feats = {
        '0': torch.randn(N, 16, 1),
        '1': torch.randn(N, 16, 3),
    }
    mlx_feats = {k: mx.array(v.numpy()) for k, v in torch_feats.items()}

    with torch.no_grad():
        torch_out = torch_norm(torch_feats)
    mlx_out = mlx_norm(mlx_feats)
    mx.eval(*mlx_out.values())

    for degree in torch_out:
        t_np = torch_out[degree].detach().numpy()
        m_np = np.array(mlx_out[degree])
        diff = np.max(np.abs(t_np - m_np))
        print(f"  GroupNorm path degree {degree}: max diff = {diff:.2e}")
        assert diff < 1e-4, f"NormSE3 GroupNorm degree {degree} diff too large: {diff}"

    # Test LayerNorm path (different channels)
    fiber2 = TorchFiber([(0, 16), (1, 8)])
    torch_norm2 = TorchNormSE3(fiber2)
    torch_norm2.eval()

    mlx_fiber2 = MLXFiber([(0, 16), (1, 8)])
    mlx_norm2 = MLXNormSE3(mlx_fiber2)

    for degree in torch_norm2.layer_norms:
        ln = torch_norm2.layer_norms[degree]
        mlx_norm2.layer_norms[degree].weight = mx.array(ln.weight.detach().numpy())
        mlx_norm2.layer_norms[degree].bias = mx.array(ln.bias.detach().numpy())

    torch_feats2 = {
        '0': torch.randn(N, 16, 1),
        '1': torch.randn(N, 8, 3),
    }
    mlx_feats2 = {k: mx.array(v.numpy()) for k, v in torch_feats2.items()}

    with torch.no_grad():
        torch_out2 = torch_norm2(torch_feats2)
    mlx_out2 = mlx_norm2(mlx_feats2)
    mx.eval(*mlx_out2.values())

    for degree in torch_out2:
        t_np = torch_out2[degree].detach().numpy()
        m_np = np.array(mlx_out2[degree])
        diff = np.max(np.abs(t_np - m_np))
        print(f"  LayerNorm path degree {degree}: max diff = {diff:.2e}")
        assert diff < 1e-4, f"NormSE3 LayerNorm degree {degree} diff too large: {diff}"

    print("  PASSED\n")


# =====================================================================
# Test 3: RadialProfile
# =====================================================================
def test_radial_profile():
    from rfantibody.rfdiffusion.mlx.se3.layers.convolution import (
        RadialProfile as MLXRadialProfile)

    # Also import PyTorch version
    from se3_transformer.model.layers.convolution import (
        RadialProfile as TorchRadialProfile)

    print("=" * 60)
    print("Test 3: RadialProfile")
    print("=" * 60)

    num_freq, c_in, c_out, edge_dim = 5, 16, 32, 4
    E = 200

    torch_rp = TorchRadialProfile(num_freq, c_in, c_out, edge_dim=edge_dim)
    torch_rp.eval()

    mlx_rp = MLXRadialProfile(num_freq, c_in, c_out, edge_dim=edge_dim)

    # Copy weights: Sequential layers 0, 2, 4 are Linear
    for i in range(len(torch_rp.net)):
        tl = torch_rp.net[i]
        if hasattr(tl, 'weight'):
            mlx_rp.net.layers[i].weight = mx.array(tl.weight.detach().numpy())
            if hasattr(tl, 'bias') and tl.bias is not None:
                mlx_rp.net.layers[i].bias = mx.array(tl.bias.detach().numpy())

    edge_feats = torch.randn(E, edge_dim)
    mlx_edge_feats = mx.array(edge_feats.numpy())

    with torch.no_grad():
        torch_out = torch_rp(edge_feats)
    mlx_out = mlx_rp(mlx_edge_feats)
    mx.eval(mlx_out)

    t_np = torch_out.detach().numpy()
    m_np = np.array(mlx_out)
    diff = np.max(np.abs(t_np - m_np))
    print(f"  output shape: {t_np.shape}, max diff = {diff:.2e}")
    assert diff < 1e-5, f"RadialProfile diff too large: {diff}"
    print("  PASSED\n")


# =====================================================================
# Test 4: VersatileConvSE3 (pairwise, non-fused)
# =====================================================================
def test_versatile_conv():
    from se3_transformer.model.layers.convolution import (
        VersatileConvSE3 as TorchVConv, ConvSE3FuseLevel as TorchFuse)
    from rfantibody.rfdiffusion.mlx.se3.layers.convolution import (
        VersatileConvSE3 as MLXVConv, ConvSE3FuseLevel as MLXFuse)

    print("=" * 60)
    print("Test 4: VersatileConvSE3")
    print("=" * 60)

    freq_sum, c_in, c_out, edge_dim = 3, 16, 32, 4
    E = 100

    torch_vc = TorchVConv(freq_sum, c_in, c_out, edge_dim,
                           use_layer_norm=False,
                           fuse_level=TorchFuse.NONE)
    torch_vc.eval()

    mlx_vc = MLXVConv(freq_sum, c_in, c_out, edge_dim,
                       use_layer_norm=False,
                       fuse_level=MLXFuse.NONE)

    # Copy RadialProfile weights
    for i in range(len(torch_vc.radial_func.net)):
        tl = torch_vc.radial_func.net[i]
        if hasattr(tl, 'weight'):
            mlx_vc.radial_func.net.layers[i].weight = mx.array(
                tl.weight.detach().numpy())
            if hasattr(tl, 'bias') and tl.bias is not None:
                mlx_vc.radial_func.net.layers[i].bias = mx.array(
                    tl.bias.detach().numpy())

    # Create test inputs
    features = torch.randn(E, c_in, 3)  # 3 = degree_to_dim(1)
    inv_edge = torch.randn(E, edge_dim)
    basis = torch.randn(E, 3, freq_sum, 3)  # (E, in_dim, freq, out_dim)

    mlx_features = mx.array(features.numpy())
    mlx_inv_edge = mx.array(inv_edge.numpy())
    mlx_basis = mx.array(basis.numpy())

    with torch.no_grad():
        torch_out = torch_vc(features, inv_edge, basis)
    mlx_out = mlx_vc(mlx_features, mlx_inv_edge, mlx_basis)
    mx.eval(mlx_out)

    t_np = torch_out.detach().numpy()
    m_np = np.array(mlx_out)
    diff = np.max(np.abs(t_np - m_np))
    print(f"  output shape: {t_np.shape}, max diff = {diff:.2e}")
    assert diff < 1e-4, f"VersatileConvSE3 diff too large: {diff}"
    print("  PASSED\n")


# =====================================================================
# Test 5: Full ConvSE3 (with graph and basis)
# =====================================================================
def test_conv_se3():
    from se3_transformer.model.fiber import Fiber as TorchFiber
    from se3_transformer.model.layers.convolution import (
        ConvSE3 as TorchConvSE3, ConvSE3FuseLevel as TorchFuse)
    from rfantibody.rfdiffusion.mlx.se3.fiber import Fiber as MLXFiber
    from rfantibody.rfdiffusion.mlx.se3.layers.convolution import (
        ConvSE3 as MLXConvSE3, ConvSE3FuseLevel as MLXFuse)
    from rfantibody.rfdiffusion.mlx.se3.basis import get_basis
    from rfantibody.rfdiffusion.mlx.graph_ops import SimpleGraph as MLXGraph

    from rfantibody.rfdiffusion.mps_graph import SimpleGraph as TorchGraph

    print("=" * 60)
    print("Test 5: ConvSE3 (NONE fuse level)")
    print("=" * 60)

    N, E = 20, 80
    max_degree = 2

    fiber_in = TorchFiber([(0, 16), (1, 8)])
    fiber_out = TorchFiber([(0, 16), (1, 8)])
    fiber_edge = TorchFiber({})

    # Use NONE fuse level for simplest test
    torch_conv = TorchConvSE3(
        fiber_in, fiber_out, fiber_edge,
        pool=True, self_interaction=False,
        max_degree=max_degree, fuse_level=TorchFuse.NONE)
    torch_conv.eval()

    mlx_fiber_in = MLXFiber([(0, 16), (1, 8)])
    mlx_fiber_out = MLXFiber([(0, 16), (1, 8)])
    mlx_fiber_edge = MLXFiber({})

    mlx_conv = MLXConvSE3(
        mlx_fiber_in, mlx_fiber_out, mlx_fiber_edge,
        pool=True, self_interaction=False,
        max_degree=max_degree, fuse_level=MLXFuse.NONE)

    # Copy weights from PyTorch to MLX
    for dict_key in torch_conv.conv:
        torch_rp = torch_conv.conv[dict_key].radial_func
        mlx_rp = mlx_conv.conv[dict_key].radial_func
        for i in range(len(torch_rp.net)):
            tl = torch_rp.net[i]
            if hasattr(tl, 'weight'):
                mlx_rp.net.layers[i].weight = mx.array(
                    tl.weight.detach().numpy())
                if hasattr(tl, 'bias') and tl.bias is not None:
                    mlx_rp.net.layers[i].bias = mx.array(
                        tl.bias.detach().numpy())

    # Create graph
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    torch_graph = TorchGraph(src, dst, N)
    mlx_graph = MLXGraph(mx.array(src.numpy()), mx.array(dst.numpy()), N)

    # Create node and edge features
    torch_node_feats = {
        '0': torch.randn(N, 16, 1),
        '1': torch.randn(N, 8, 3),
    }
    torch_edge_feats = {'0': torch.randn(E, 1, 1)}

    mlx_node_feats = {k: mx.array(v.numpy()) for k, v in torch_node_feats.items()}
    mlx_edge_feats = {k: mx.array(v.numpy()) for k, v in torch_edge_feats.items()}

    # Create basis
    rel_pos = torch.randn(E, 3)
    mlx_rel_pos = mx.array(rel_pos.numpy())
    mlx_basis = get_basis(mlx_rel_pos, max_degree=max_degree)
    mx.eval(*mlx_basis.values())

    # For PyTorch basis, use the same numerical values
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
        assert diff < 1e-3, f"ConvSE3 degree {degree} diff too large: {diff}"

    print("  PASSED\n")


# =====================================================================
# Test 6: unfuse_features / aggregate_residual
# =====================================================================
def test_utils():
    from rfantibody.rfdiffusion.mlx.se3.layers.utils import (
        unfuse_features, aggregate_residual)

    print("=" * 60)
    print("Test 6: Utility functions")
    print("=" * 60)

    # unfuse_features
    fused = mx.ones((10, 32, 4))  # 32 channels, last dim has 1+3=4
    result = unfuse_features(fused, [0, 1])
    assert result['0'].shape == (10, 32, 1), f"Got {result['0'].shape}"
    assert result['1'].shape == (10, 32, 3), f"Got {result['1'].shape}"
    print("  unfuse_features: PASSED")

    # aggregate_residual - cat
    f1 = {'0': mx.ones((10, 4, 1)), '1': mx.ones((10, 4, 3))}
    f2 = {'0': mx.ones((10, 8, 1)), '1': mx.ones((10, 8, 3))}
    cat = aggregate_residual(f1, f2, 'cat')
    assert cat['0'].shape == (10, 12, 1)
    assert cat['1'].shape == (10, 12, 3)
    print("  aggregate_residual (cat): PASSED")

    # aggregate_residual - sum
    f1 = {'0': mx.ones((10, 4, 1)), '1': mx.ones((10, 4, 3))}
    f2 = {'0': mx.ones((10, 4, 1)), '1': mx.ones((10, 4, 3))}
    summed = aggregate_residual(f1, f2, 'sum')
    assert np.allclose(np.array(summed['0']), 2.0)
    print("  aggregate_residual (sum): PASSED")

    print()


if __name__ == '__main__':
    test_utils()
    test_linear()
    test_norm()
    test_radial_profile()
    test_versatile_conv()
    test_conv_se3()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
