#!/usr/bin/env python3
"""
Integration test: compare full SE3Transformer and SE3TransformerWrapper
between PyTorch and MLX with identical weights.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'include', 'SE3Transformer'))

import torch
import mlx.core as mx


def _copy_weights_recursive(torch_mod, mlx_mod, prefix=''):
    """Recursively copy weights from PyTorch module to MLX module."""
    copied = 0

    # Handle direct parameter dicts (LinearSE3.weights, to_kernel_self)
    for attr_name in ['weights', 'to_kernel_self']:
        torch_dict = getattr(torch_mod, attr_name, None)
        mlx_dict = getattr(mlx_mod, attr_name, None)
        if torch_dict is not None and mlx_dict is not None:
            if isinstance(torch_dict, torch.nn.ParameterDict):
                for key in torch_dict:
                    if key in mlx_dict:
                        mlx_dict[key] = mx.array(torch_dict[key].detach().numpy())
                        copied += 1

    # Handle GroupNorm / LayerNorm
    if hasattr(torch_mod, 'group_norm') and hasattr(mlx_mod, 'group_norm'):
        gn = torch_mod.group_norm
        mlx_mod.group_norm.weight = mx.array(gn.weight.detach().numpy())
        mlx_mod.group_norm.bias = mx.array(gn.bias.detach().numpy())
        copied += 2

    if hasattr(torch_mod, 'layer_norms') and hasattr(mlx_mod, 'layer_norms'):
        for key in torch_mod.layer_norms:
            ln = torch_mod.layer_norms[key]
            mlx_mod.layer_norms[key].weight = mx.array(ln.weight.detach().numpy())
            mlx_mod.layer_norms[key].bias = mx.array(ln.bias.detach().numpy())
            copied += 2

    # Handle Sequential (RadialProfile.net)
    if hasattr(torch_mod, 'net') and hasattr(mlx_mod, 'net'):
        if isinstance(torch_mod.net, torch.nn.Sequential):
            for i in range(len(torch_mod.net)):
                tl = torch_mod.net[i]
                if hasattr(tl, 'weight'):
                    mlx_mod.net.layers[i].weight = mx.array(tl.weight.detach().numpy())
                    copied += 1
                    if hasattr(tl, 'bias') and tl.bias is not None:
                        mlx_mod.net.layers[i].bias = mx.array(tl.bias.detach().numpy())
                        copied += 1

    # Handle conv (single VersatileConvSE3)
    if hasattr(torch_mod, 'conv') and hasattr(mlx_mod, 'conv'):
        torch_conv = torch_mod.conv
        mlx_conv = mlx_mod.conv
        if hasattr(torch_conv, 'radial_func') and hasattr(mlx_conv, 'radial_func'):
            copied += _copy_weights_recursive(torch_conv, mlx_conv,
                                               f'{prefix}conv.')

    # Handle conv dict (pairwise convolutions)
    if hasattr(torch_mod, 'conv') and isinstance(getattr(torch_mod, 'conv', None), torch.nn.ModuleDict):
        for key in torch_mod.conv:
            if key in mlx_mod.conv:
                copied += _copy_weights_recursive(torch_mod.conv[key],
                                                   mlx_mod.conv[key],
                                                   f'{prefix}conv.{key}.')

    # Handle conv_out / conv_in dicts
    for attr_name in ['conv_out', 'conv_in']:
        torch_dict = getattr(torch_mod, attr_name, None)
        mlx_dict = getattr(mlx_mod, attr_name, None)
        if torch_dict is not None and mlx_dict is not None:
            if isinstance(torch_dict, torch.nn.ModuleDict):
                for key in torch_dict:
                    if key in mlx_dict:
                        copied += _copy_weights_recursive(
                            torch_dict[key], mlx_dict[key],
                            f'{prefix}{attr_name}.{key}.')

    # Handle radial_func directly
    if hasattr(torch_mod, 'radial_func') and hasattr(mlx_mod, 'radial_func'):
        copied += _copy_weights_recursive(torch_mod.radial_func,
                                           mlx_mod.radial_func,
                                           f'{prefix}radial_func.')

    # Handle sub-modules: to_key_value, to_query, attention, project
    for attr_name in ['to_key_value', 'to_query', 'attention', 'project']:
        torch_sub = getattr(torch_mod, attr_name, None)
        mlx_sub = getattr(mlx_mod, attr_name, None)
        if torch_sub is not None and mlx_sub is not None:
            copied += _copy_weights_recursive(torch_sub, mlx_sub,
                                               f'{prefix}{attr_name}.')

    # Handle graph_modules (Sequential of attention blocks + norms)
    if hasattr(torch_mod, 'graph_modules') and hasattr(mlx_mod, 'graph_modules'):
        torch_layers = list(torch_mod.graph_modules)
        mlx_layers = mlx_mod.graph_modules.layers
        for i, (tl, ml) in enumerate(zip(torch_layers, mlx_layers)):
            copied += _copy_weights_recursive(tl, ml,
                                               f'{prefix}graph_modules.{i}.')

    # Handle se3 sub-module
    if hasattr(torch_mod, 'se3') and hasattr(mlx_mod, 'se3'):
        copied += _copy_weights_recursive(torch_mod.se3, mlx_mod.se3,
                                           f'{prefix}se3.')

    return copied


def test_se3_transformer():
    from se3_transformer.model.transformer import SE3Transformer as TorchSE3
    from se3_transformer.model.fiber import Fiber as TorchFiber
    from rfantibody.rfdiffusion.mlx.se3.transformer import SE3Transformer as MLXSE3
    from rfantibody.rfdiffusion.mlx.se3.fiber import Fiber as MLXFiber
    from rfantibody.rfdiffusion.mps_graph import SimpleGraph as TorchGraph
    from rfantibody.rfdiffusion.mlx.graph_ops import SimpleGraph as MLXGraph

    print("=" * 60)
    print("Test: SE3Transformer full forward pass")
    print("=" * 60)

    N, max_degree, channels = 15, 2, 16
    num_layers, num_heads, channels_div = 2, 4, 2
    num_edge_features = 8

    # Build torch model
    torch_fiber_in = TorchFiber({0: 32, 1: 4})
    torch_fiber_hidden = TorchFiber.create(max_degree + 1, channels)
    torch_fiber_out = TorchFiber({0: 32, 1: 2})
    torch_fiber_edge = TorchFiber({0: num_edge_features})

    torch_model = TorchSE3(
        num_layers=num_layers,
        fiber_in=torch_fiber_in,
        fiber_hidden=torch_fiber_hidden,
        fiber_out=torch_fiber_out,
        num_heads=num_heads,
        channels_div=channels_div,
        fiber_edge=torch_fiber_edge,
        use_layer_norm=True,
        norm=True,
        tensor_cores=False)
    torch_model.eval()

    # Build MLX model
    mlx_fiber_in = MLXFiber({0: 32, 1: 4})
    mlx_fiber_hidden = MLXFiber.create(max_degree + 1, channels)
    mlx_fiber_out = MLXFiber({0: 32, 1: 2})
    mlx_fiber_edge = MLXFiber({0: num_edge_features})

    mlx_model = MLXSE3(
        num_layers=num_layers,
        fiber_in=mlx_fiber_in,
        fiber_hidden=mlx_fiber_hidden,
        fiber_out=mlx_fiber_out,
        num_heads=num_heads,
        channels_div=channels_div,
        fiber_edge=mlx_fiber_edge,
        use_layer_norm=True,
        norm=True,
        tensor_cores=False)

    # Copy all weights
    n_copied = _copy_weights_recursive(torch_model, mlx_model, '')
    print(f"  Copied {n_copied} parameter tensors")

    # Create graph
    k = 8
    src_list, dst_list = [], []
    for i in range(N):
        neighbors = torch.randint(0, N, (k,))
        for j in neighbors:
            src_list.append(i)
            dst_list.append(j.item())
    src = torch.tensor(src_list)
    dst = torch.tensor(dst_list)
    E = len(src_list)

    torch_graph = TorchGraph(src, dst, N)
    mlx_graph = MLXGraph(mx.array(src.numpy()), mx.array(dst.numpy()), N)

    # Relative positions (stored in graph edata)
    rel_pos = torch.randn(E, 3)
    torch_graph.edata['rel_pos'] = rel_pos
    mlx_graph.edata = {'rel_pos': mx.array(rel_pos.numpy())}

    # Node features
    torch_node_feats = {
        '0': torch.randn(N, 32, 1),
        '1': torch.randn(N, 4, 3),
    }
    mlx_node_feats = {k: mx.array(v.numpy()) for k, v in torch_node_feats.items()}

    # Edge features
    torch_edge_feats = {'0': torch.randn(E, num_edge_features, 1)}
    mlx_edge_feats = {k: mx.array(v.numpy()) for k, v in torch_edge_feats.items()}

    # Forward pass
    with torch.no_grad():
        torch_out = torch_model(torch_graph, torch_node_feats, torch_edge_feats)

    mlx_out = mlx_model(mlx_graph, mlx_node_feats, mlx_edge_feats)
    mx.eval(*mlx_out.values())

    for degree in torch_out:
        t_np = torch_out[degree].detach().numpy()
        m_np = np.array(mlx_out[degree])
        diff = np.max(np.abs(t_np - m_np))
        print(f"  degree {degree}: shape {t_np.shape}, max diff = {diff:.2e}")
        assert diff < 1e-2, f"SE3Transformer degree {degree} diff too large: {diff}"

    print("  PASSED\n")


def test_se3_wrapper():
    from rfantibody.rfdiffusion.SE3_network import SE3TransformerWrapper as TorchWrapper
    from rfantibody.rfdiffusion.mlx.se3.wrapper import SE3TransformerWrapper as MLXWrapper
    from rfantibody.rfdiffusion.mps_graph import SimpleGraph as TorchGraph
    from rfantibody.rfdiffusion.mlx.graph_ops import SimpleGraph as MLXGraph

    print("=" * 60)
    print("Test: SE3TransformerWrapper")
    print("=" * 60)

    N = 15
    l0_in, l0_out = 32, 32
    l1_in, l1_out = 3, 2
    num_edge = 8

    torch_wrapper = TorchWrapper(
        num_layers=2, num_channels=16, num_degrees=3,
        n_heads=4, div=2,
        l0_in_features=l0_in, l0_out_features=l0_out,
        l1_in_features=l1_in, l1_out_features=l1_out,
        num_edge_features=num_edge)
    torch_wrapper.eval()

    mlx_wrapper = MLXWrapper(
        num_layers=2, num_channels=16, num_degrees=3,
        n_heads=4, div=2,
        l0_in_features=l0_in, l0_out_features=l0_out,
        l1_in_features=l1_in, l1_out_features=l1_out,
        num_edge_features=num_edge)

    n_copied = _copy_weights_recursive(torch_wrapper, mlx_wrapper, '')
    print(f"  Copied {n_copied} parameter tensors")

    # Graph
    k = 8
    src_list, dst_list = [], []
    for i in range(N):
        for j in torch.randint(0, N, (k,)):
            src_list.append(i)
            dst_list.append(j.item())
    src = torch.tensor(src_list)
    dst = torch.tensor(dst_list)
    E = len(src_list)

    torch_graph = TorchGraph(src, dst, N)
    mlx_graph = MLXGraph(mx.array(src.numpy()), mx.array(dst.numpy()), N)

    rel_pos = torch.randn(E, 3)
    torch_graph.edata['rel_pos'] = rel_pos
    mlx_graph.edata = {'rel_pos': mx.array(rel_pos.numpy())}

    # Features
    t0 = torch.randn(N, l0_in, 1)
    t1 = torch.randn(N, l1_in, 3)
    edge = torch.randn(E, num_edge, 1)

    with torch.no_grad():
        torch_out = torch_wrapper(torch_graph, t0, t1, edge)

    mlx_out = mlx_wrapper(
        mlx_graph,
        mx.array(t0.numpy()),
        mx.array(t1.numpy()),
        mx.array(edge.numpy()))
    mx.eval(*mlx_out.values())

    for degree in torch_out:
        t_np = torch_out[degree].detach().numpy()
        m_np = np.array(mlx_out[degree])
        diff = np.max(np.abs(t_np - m_np))
        print(f"  degree {degree}: shape {t_np.shape}, max diff = {diff:.2e}")
        assert diff < 1e-2, f"Wrapper degree {degree} diff too large: {diff}"

    print("  PASSED\n")


if __name__ == '__main__':
    test_se3_transformer()
    test_se3_wrapper()
    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
