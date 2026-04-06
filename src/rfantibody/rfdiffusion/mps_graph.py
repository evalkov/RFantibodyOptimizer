"""
Drop-in replacement for DGL graph primitives using pure PyTorch.

This module provides a lightweight SimpleGraph class and the 5 DGL operations
used by the SE(3)-Transformer, implemented with scatter_add / index_select
so they work on any PyTorch backend (CPU, CUDA, MPS).

Replaces:
  dgl.graph()             -> SimpleGraph(src, dst, num_nodes)
  dgl.ops.copy_e_sum()    -> copy_e_sum()
  dgl.ops.copy_e_mean()   -> copy_e_mean()
  dgl.ops.e_dot_v()       -> e_dot_v()
  dgl.ops.edge_softmax()  -> edge_softmax()
"""

import torch
from torch import Tensor


class SimpleGraph:
    """Minimal graph container matching the DGL API surface used by SE3-Transformer."""

    def __init__(self, src: Tensor, dst: Tensor, num_nodes: int):
        self._src = src.long()
        self._dst = dst.long()
        self._num_nodes = num_nodes
        self.edata: dict = {}

    def edges(self):
        return self._src, self._dst

    def num_nodes(self):
        return self._num_nodes

    def num_edges(self):
        return self._src.shape[0]

    def to(self, device):
        self._src = self._src.to(device)
        self._dst = self._dst.to(device)
        self.edata = {k: v.to(device) for k, v in self.edata.items()}
        return self


def simple_graph(edge_pair, num_nodes: int) -> SimpleGraph:
    """Drop-in for ``dgl.graph((src, dst), num_nodes=N)``."""
    src, dst = edge_pair
    return SimpleGraph(src, dst, num_nodes)


# ---------------------------------------------------------------------------
# Message-passing primitives
# ---------------------------------------------------------------------------

def copy_e_sum(graph: SimpleGraph, edge_feat: Tensor) -> Tensor:
    """Scatter-add edge features to destination nodes.

    Equivalent to ``dgl.ops.copy_e_sum(graph, edge_feat)``.
    edge_feat: (E, *feat_shape)
    Returns:   (N, *feat_shape)
    """
    dst = graph._dst
    N = graph.num_nodes()
    out_shape = (N,) + edge_feat.shape[1:]
    out = torch.zeros(out_shape, dtype=edge_feat.dtype, device=edge_feat.device)
    idx = dst.view(-1, *([1] * (edge_feat.dim() - 1))).expand_as(edge_feat)
    out.scatter_add_(0, idx, edge_feat)
    return out


def copy_e_mean(graph: SimpleGraph, edge_feat: Tensor) -> Tensor:
    """Scatter-mean edge features to destination nodes.

    Equivalent to ``dgl.ops.copy_e_mean(graph, edge_feat)``.
    """
    summed = copy_e_sum(graph, edge_feat)
    dst = graph._dst
    ones = torch.ones(dst.shape[0], dtype=edge_feat.dtype, device=edge_feat.device)
    count = torch.zeros(graph.num_nodes(), dtype=edge_feat.dtype, device=edge_feat.device)
    count.scatter_add_(0, dst, ones)
    count = count.clamp(min=1.0)
    count = count.view(-1, *([1] * (summed.dim() - 1)))
    return summed / count


def e_dot_v(graph: SimpleGraph, edge_feat: Tensor, node_feat: Tensor) -> Tensor:
    """Per-edge dot product of edge feature with destination node feature.

    Equivalent to ``dgl.ops.e_dot_v(graph, edge_feat, node_feat)``.
    edge_feat: (E, H, D)
    node_feat: (N, H, D)
    Returns:   (E, H, 1)
    """
    _, dst = graph.edges()
    dst_feats = node_feat[dst]  # (E, H, D)
    return (edge_feat * dst_feats).sum(dim=-1, keepdim=True)


def edge_softmax(graph: SimpleGraph, edge_weights: Tensor) -> Tensor:
    """Softmax over edges grouped by destination node.

    Equivalent to ``dgl.ops.edge_softmax(graph, edge_weights)``.
    edge_weights: (E, H)  or (E,) or (E, H, ...)
    Returns:      same shape as edge_weights
    """
    _, dst = graph.edges()
    N = graph.num_nodes()
    idx = dst.view(-1, *([1] * (edge_weights.dim() - 1))).expand_as(edge_weights)

    # scatter_reduce 'amax' is not on MPS, so compute per-node max on CPU.
    # This is a tiny tensor (N nodes), so the transfer cost is negligible.
    idx_cpu = idx.cpu()
    w_cpu = edge_weights.cpu()
    max_cpu = torch.full((N,) + edge_weights.shape[1:], float('-inf'), dtype=w_cpu.dtype)
    max_cpu.scatter_reduce_(0, idx_cpu, w_cpu, reduce='amax', include_self=True)
    max_vals = max_cpu.to(edge_weights.device)

    edge_max = max_vals[dst]
    exp_weights = torch.exp(edge_weights - edge_max)

    # Sum of exp per destination node (scatter_add_ works on MPS)
    sum_exp = torch.zeros((N,) + edge_weights.shape[1:],
                          dtype=edge_weights.dtype, device=edge_weights.device)
    sum_exp.scatter_add_(0, idx, exp_weights)
    edge_sum = sum_exp[dst]

    return exp_weights / edge_sum.clamp(min=1e-12)
