"""
MLX graph primitives for SE3-Transformer message passing.

Drop-in replacement for mps_graph.py, ported from PyTorch to MLX.
Provides the same 5 operations used by the SE3-Transformer:
  SimpleGraph, copy_e_sum, copy_e_mean, e_dot_v, edge_softmax
"""

import mlx.core as mx


class SimpleGraph:
    """Minimal graph container for SE3-Transformer."""

    def __init__(self, src: mx.array, dst: mx.array, num_nodes: int):
        self._src = src.astype(mx.int32)
        self._dst = dst.astype(mx.int32)
        self._num_nodes = num_nodes
        self.edata: dict = {}

    def edges(self):
        return self._src, self._dst

    def num_nodes(self):
        return self._num_nodes

    def num_edges(self):
        return self._src.shape[0]


def simple_graph(edge_pair, num_nodes: int) -> SimpleGraph:
    """Create a SimpleGraph from (src, dst) arrays."""
    src, dst = edge_pair
    return SimpleGraph(src, dst, num_nodes)


# ---------------------------------------------------------------------------
# Message-passing primitives
# ---------------------------------------------------------------------------

def copy_e_sum(graph: SimpleGraph, edge_feat: mx.array) -> mx.array:
    """Scatter-add edge features to destination nodes.

    edge_feat: (E, *feat_shape)
    Returns:   (N, *feat_shape)
    """
    dst = graph._dst  # (E,)
    N = graph.num_nodes()
    out_shape = (N,) + edge_feat.shape[1:]
    out = mx.zeros(out_shape, dtype=edge_feat.dtype)

    # MLX .at[idx] with 1D index scatters along dim 0
    out = out.at[dst].add(edge_feat)
    return out


def copy_e_mean(graph: SimpleGraph, edge_feat: mx.array) -> mx.array:
    """Scatter-mean edge features to destination nodes."""
    summed = copy_e_sum(graph, edge_feat)
    dst = graph._dst
    N = graph.num_nodes()

    # Count edges per destination node
    ones = mx.ones((dst.shape[0],), dtype=edge_feat.dtype)
    count = mx.zeros((N,), dtype=edge_feat.dtype)
    count = count.at[dst].add(ones)
    count = mx.maximum(count, 1.0)

    # Reshape count for broadcasting
    for _ in range(summed.ndim - 1):
        count = mx.expand_dims(count, axis=-1)

    return summed / count


def e_dot_v(graph: SimpleGraph, edge_feat: mx.array, node_feat: mx.array) -> mx.array:
    """Per-edge dot product of edge feature with destination node feature.

    edge_feat: (E, H, D)
    node_feat: (N, H, D)
    Returns:   (E, H, 1)
    """
    _, dst = graph.edges()
    dst_feats = node_feat[dst]  # (E, H, D)
    return mx.sum(edge_feat * dst_feats, axis=-1, keepdims=True)


def edge_softmax(graph: SimpleGraph, edge_weights: mx.array) -> mx.array:
    """Softmax over edges grouped by destination node.

    edge_weights: (E, H) or (E,) or (E, H, ...)
    Returns:      same shape as edge_weights
    """
    _, dst = graph.edges()
    N = graph.num_nodes()

    # Per-node max for numerical stability
    neg_inf = mx.full((N,) + edge_weights.shape[1:], -1e30, dtype=edge_weights.dtype)
    max_vals = neg_inf.at[dst].maximum(edge_weights)
    edge_max = max_vals[dst]

    # Exponentiate
    exp_weights = mx.exp(edge_weights - edge_max)

    # Sum of exp per destination node
    sum_exp = mx.zeros((N,) + edge_weights.shape[1:], dtype=edge_weights.dtype)
    sum_exp = sum_exp.at[dst].add(exp_weights)
    edge_sum = sum_exp[dst]

    return exp_weights / mx.maximum(edge_sum, 1e-12)
