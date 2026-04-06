"""
Custom Metal kernels for SE3 convolution fusion.

Fuses the VersatileConvSE3 3-step pipeline:
  1. RadialProfile MLP(edge_feats) → radial_weights
  2. features @ basis → tmp
  3. radial_weights @ tmp → output

into a single Metal kernel that eliminates intermediate tensor allocations
and reduces kernel launch overhead.
"""

import mlx.core as mx
from mlx.core import fast

# --- Fused MLP + dual matmul kernel ---
# Each thread handles one edge. The MLP weights are shared across all edges
# (broadcast via Metal L2 cache).
#
# MLP architecture: in_dim → 32 (ReLU) → 32 (ReLU) → out_dim (no bias on last)
# The MLP weights are packed into a single flat buffer:
#   w0: (in_dim, 32), b0: (32,), w1: (32, 32), b1: (32,), w2: (32, out_dim)

_FUSED_CONV_HEADER = """
#include <metal_stdlib>
using namespace metal;

// Apply LayerNorm in-place: x = (x - mean) / sqrt(var + eps) * weight + bias
inline void layer_norm_32(
    thread float* x,
    const device float* weight,
    const device float* bias
) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < 32; i++) mean += x[i];
    mean /= 32.0f;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < 32; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= 32.0f;

    // Normalize
    float inv_std = 1.0f / sqrt(var + 1e-5f);
    for (int i = 0; i < 32; i++) {
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}
"""


def _make_fused_conv_source(c_out, c_in, freq_sum, edge_dim, out_dim):
    """Generate Metal source for fused VersatileConvSE3.

    Tiles the computation by c_out to limit register usage: instead of
    storing all freq*c_in*c_out radial weights in registers, only
    c_in*freq weights are live at a time. The MLP hidden states (h0, h1)
    are computed once, then the final MLP layer is evaluated per-tile.

    Args:
        c_out: Output channels
        c_in: Input channels
        freq_sum: Sum of frequency dimensions
        edge_dim: Edge feature dimension (typically 33)
        out_dim: Spatial output dimension (from basis)
    """
    mlp_out_dim = freq_sum * c_in * c_out
    radial_tile_size = c_in * freq_sum  # registers per c_out tile

    # Determine param layout based on use_layer_norm
    # Without LN: w0(edge_dim*32) + b0(32) + w1(32*32) + b1(32) + w2(32*mlp_out)
    # With LN:    w0(edge_dim*32) + b0(32) + ln0_w(32) + ln0_b(32)
    #           + w1(32*32) + b1(32) + ln1_w(32) + ln1_b(32)
    #           + w2(32*mlp_out)

    return f"""
    // One thread per edge
    uint edge = thread_position_in_grid.x;
    uint num_edges = out_shape[0];
    if (edge >= num_edges) return;

    const int C_OUT = {c_out};
    const int C_IN = {c_in};
    const int FREQ = {freq_sum};
    const int EDGE_DIM = {edge_dim};
    const int OUT_DIM = {out_dim};
    const int MLP_OUT = {mlp_out_dim};
    const int TILE = {radial_tile_size};  // C_IN * FREQ
    const int USE_LN = use_ln_param[0];

    // ---- MLP layers 0 and 1: compute hidden states once ----
    const device float* edge_f = edge_feats + edge * EDGE_DIM;
    const device float* p = mlp_params;

    // Layer 0: Linear(edge_dim, 32)
    const device float* w0 = p;
    const device float* b0 = w0 + EDGE_DIM * 32;
    float h0[32];
    for (int j = 0; j < 32; j++) {{
        float acc = b0[j];
        for (int i = 0; i < EDGE_DIM; i++) {{
            acc += edge_f[i] * w0[i * 32 + j];
        }}
        h0[j] = acc;
    }}
    p = b0 + 32;

    // Optional LayerNorm 0
    if (USE_LN) {{
        layer_norm_32(h0, p, p + 32);
        p += 64;
    }}

    // ReLU
    for (int j = 0; j < 32; j++) h0[j] = max(h0[j], 0.0f);

    // Layer 1: Linear(32, 32)
    const device float* w1 = p;
    const device float* b1 = w1 + 32 * 32;
    float h1[32];
    for (int j = 0; j < 32; j++) {{
        float acc = b1[j];
        for (int k = 0; k < 32; k++) {{
            acc += h0[k] * w1[k * 32 + j];
        }}
        h1[j] = acc;
    }}
    p = b1 + 32;

    // Optional LayerNorm 1
    if (USE_LN) {{
        layer_norm_32(h1, p, p + 32);
        p += 64;
    }}

    // ReLU
    for (int j = 0; j < 32; j++) h1[j] = max(h1[j], 0.0f);

    // w2 pointer for last layer (used in tiled loop below)
    const device float* w2 = p;

    // ---- Pointers into features and basis ----
    int in_dim = in_dim_param[0];
    const device float* feat_ptr = features + edge * C_IN * in_dim;
    const device float* basis_ptr = basis + edge * in_dim * FREQ * OUT_DIM;

    // ---- Tile over c_out: compute radial slice + matmul per tile ----
    for (int co = 0; co < C_OUT; co++) {{
        // Compute MLP last layer for this c_out slice only
        // radial_tile[j] = radial[co * TILE + j] for j in [0, TILE)
        float radial_tile[{radial_tile_size}];
        for (int j = 0; j < TILE; j++) {{
            float acc = 0.0f;
            int w_idx = co * TILE + j;
            for (int k = 0; k < 32; k++) {{
                acc += h1[k] * w2[k * MLP_OUT + w_idx];
            }}
            radial_tile[j] = acc;
        }}

        // Compute output[co, od] = sum over ci,f of radial_tile[ci*FREQ+f] * (features @ basis)
        for (int od = 0; od < OUT_DIM; od++) {{
            float acc = 0.0f;
            for (int ci = 0; ci < C_IN; ci++) {{
                for (int f = 0; f < FREQ; f++) {{
                    float tmp_val = 0.0f;
                    for (int k = 0; k < in_dim; k++) {{
                        tmp_val += feat_ptr[ci * in_dim + k] * basis_ptr[k * FREQ * OUT_DIM + f * OUT_DIM + od];
                    }}
                    acc += radial_tile[ci * FREQ + f] * tmp_val;
                }}
            }}
            result[edge * C_OUT * OUT_DIM + co * OUT_DIM + od] = acc;
        }}
    }}
"""


def make_fused_conv_kernel(c_out, c_in, freq_sum, edge_dim, out_dim):
    """Create a fused VersatileConvSE3 Metal kernel.

    Returns a callable that takes (edge_feats, mlp_params, features, basis, in_dim)
    and returns the convolution output.
    """
    source = _make_fused_conv_source(c_out, c_in, freq_sum, edge_dim, out_dim)

    kernel = fast.metal_kernel(
        name=f"fused_conv_c{c_out}_c{c_in}_f{freq_sum}_o{out_dim}",
        input_names=["edge_feats", "mlp_params", "features", "basis",
                     "in_dim_param", "out_shape", "use_ln_param"],
        output_names=["result"],
        source=source,
        header=_FUSED_CONV_HEADER,
    )
    return kernel


def pack_mlp_params(radial_profile):
    """Pack RadialProfile MLP weights into a single contiguous buffer.

    The MLP layers are: Linear [+ LayerNorm] + ReLU (×2), then Linear(bias=False).
    LayerNorm params (weight, bias) are packed between the Linear bias and ReLU.

    Layout without LayerNorm:
        w0(in×32) + b0(32) + w1(32×32) + b1(32) + w2(32×out)
    Layout with LayerNorm:
        w0(in×32) + b0(32) + ln0_w(32) + ln0_b(32)
        + w1(32×32) + b1(32) + ln1_w(32) + ln1_b(32)
        + w2(32×out)

    Returns: (packed_params, use_layer_norm)
    """
    import mlx.nn as nn

    layers = radial_profile.net.layers
    params = []
    use_layer_norm = False

    for layer in layers:
        if isinstance(layer, nn.Linear):
            # Weight: (out_features, in_features) in MLX — transpose to (in, out) for our kernel
            w = layer.weight  # (out, in)
            params.append(w.T.reshape(-1))  # (in, out) flattened
            if 'bias' in layer:
                params.append(layer.bias.reshape(-1))
        elif isinstance(layer, nn.LayerNorm):
            use_layer_norm = True
            params.append(layer.weight.reshape(-1))
            params.append(layer.bias.reshape(-1))
        # Skip ReLU layers

    return mx.concatenate(params), use_layer_norm


class FusedConvSE3:
    """Drop-in replacement for VersatileConvSE3 using a fused Metal kernel."""

    def __init__(self, versatile_conv, edge_dim):
        """
        Args:
            versatile_conv: VersatileConvSE3 instance to fuse
            edge_dim: Edge feature dimension (typically 33)
        """
        self.c_out = versatile_conv.channels_out
        self.c_in = versatile_conv.channels_in
        self.freq_sum = versatile_conv.freq_sum
        self.edge_dim = edge_dim

        # Pack MLP params (includes LayerNorm params if present)
        self.packed_params, self.use_layer_norm = pack_mlp_params(versatile_conv.radial_func)
        mx.eval(self.packed_params)

        # Kernel will be created lazily on first call (need out_dim from basis)
        self._kernels = {}  # keyed by (out_dim, in_dim)

    def __call__(self, features, invariant_edge_feats, basis):
        num_edges = features.shape[0]
        in_dim = features.shape[2]

        if basis is None:
            # k=l=0 non-fused case — fall back to standard computation
            radial_weights = self._compute_radial(invariant_edge_feats)
            return radial_weights @ features

        out_dim = basis.shape[-1]
        key = (out_dim, in_dim)

        if key not in self._kernels:
            self._kernels[key] = make_fused_conv_kernel(
                self.c_out, self.c_in, self.freq_sum,
                self.edge_dim, out_dim)

        kernel = self._kernels[key]

        # Reshape basis for the kernel: (E, in_dim, freq*out_dim)
        basis_view = basis.reshape(num_edges, in_dim, -1)

        in_dim_param = mx.array([in_dim], dtype=mx.int32)
        out_shape = mx.array([num_edges], dtype=mx.int32)
        use_ln_param = mx.array([int(self.use_layer_norm)], dtype=mx.int32)

        # Threadgroup size
        tg_size = min(256, num_edges)

        outputs = kernel(
            inputs=[invariant_edge_feats, self.packed_params,
                    features, basis_view, in_dim_param, out_shape, use_ln_param],
            output_shapes=[(num_edges, self.c_out, out_dim)],
            output_dtypes=[mx.float32],
            grid=(num_edges, 1, 1),
            threadgroup=(tg_size, 1, 1),
        )

        return outputs[0]
