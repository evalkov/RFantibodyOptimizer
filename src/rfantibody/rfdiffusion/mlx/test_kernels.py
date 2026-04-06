#!/usr/bin/env python3
"""
Test custom Metal kernels against Python reference implementations.

Usage:
    PYTHONPATH=src:include/SE3Transformer pilot_mps/.venv/bin/python \
        src/rfantibody/rfdiffusion/mlx/test_kernels.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'include', 'SE3Transformer'))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from rfantibody.rfdiffusion.mlx.se3.layers.convolution import VersatileConvSE3, ConvSE3FuseLevel, RadialProfile
from rfantibody.rfdiffusion.mlx.kernels import FusedConvSE3, pack_mlp_params


def test_pack_mlp_params():
    """Test that MLP params are packed correctly (with LayerNorm)."""
    print("Test: pack_mlp_params...")
    rp = RadialProfile(num_freq=2, channels_in=8, channels_out=16,
                       edge_dim=33, mid_dim=32, use_layer_norm=True)
    mx.eval(rp.parameters())

    packed, use_ln = pack_mlp_params(rp)

    # With LayerNorm: 33*32 + 32 + 32 + 32 + 32*32 + 32 + 32 + 32 + 32*(2*8*16)
    #              = 1056 + 32 + 64 + 1024 + 32 + 64 + 8192 = 10464
    expected = 33 * 32 + 32 + 32 + 32 + 32 * 32 + 32 + 32 + 32 + 32 * (2 * 8 * 16)
    assert packed.shape[0] == expected, f"Expected {expected}, got {packed.shape[0]}"
    assert use_ln == True, f"Expected use_layer_norm=True"
    print(f"  OK: packed {packed.shape[0]} params (expected {expected}, use_ln={use_ln})")


def test_fused_conv_basic():
    """Test fused conv against VersatileConvSE3 reference."""
    print("\nTest: fused_conv basic correctness...")

    # Create a VersatileConvSE3 with known parameters
    freq_sum = 2
    c_in = 8
    c_out = 16
    edge_dim = 33

    conv = VersatileConvSE3(
        freq_sum=freq_sum, channels_in=c_in, channels_out=c_out,
        edge_dim=edge_dim, use_layer_norm=True,
        fuse_level=ConvSE3FuseLevel.FULL)
    mx.eval(conv.parameters())

    # Create fused version
    fused = FusedConvSE3(conv, edge_dim=edge_dim)

    # Test with random inputs
    E = 100  # edges
    in_dim = 1  # degree 0

    features = mx.random.normal((E, c_in, in_dim))
    edge_feats = mx.random.normal((E, edge_dim))
    # Basis shape: (E, in_dim * freq_sum, out_dim)
    # When reshaped to (E, in_dim, freq_sum*out_dim), the matmul works
    out_dim = 1  # degree 0 output
    basis = mx.random.normal((E, in_dim * freq_sum, out_dim))

    mx.eval(features, edge_feats, basis)

    # Reference output
    ref_out = conv(features, edge_feats.squeeze(-1) if edge_feats.ndim == 3 else edge_feats, basis)
    mx.eval(ref_out)

    # Fused output
    fused_out = fused(features, edge_feats, basis)
    mx.eval(fused_out)

    print(f"  Reference shape: {ref_out.shape}")
    print(f"  Fused shape:     {fused_out.shape}")

    if ref_out.shape != fused_out.shape:
        print(f"  SHAPE MISMATCH! ref={ref_out.shape} vs fused={fused_out.shape}")
        return False

    diff = mx.abs(ref_out - fused_out)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    print(f"  Max abs diff:  {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("  OK: outputs match within tolerance")
        return True
    else:
        print("  FAIL: outputs differ!")
        return False


def test_fused_conv_shapes():
    """Test with realistic SE3 shapes from the model."""
    print("\nTest: fused_conv with realistic shapes...")

    configs = [
        # AttentionBlockSE3 to_key_value conv_in['0']
        {"name": "attn_in0", "freq_sum": 2, "c_in": 8, "c_out": 16,
         "edge_dim": 33, "E": 10432, "in_dim": 1, "out_dim": 4},
        # AttentionBlockSE3 to_key_value conv_in['1']
        {"name": "attn_in1", "freq_sum": 4, "c_in": 3, "c_out": 16,
         "edge_dim": 33, "E": 10432, "in_dim": 3, "out_dim": 4},
        # Final ConvSE3 conv_out['0'] (in_dim=4 = fused degree 0+1)
        {"name": "final_out0", "freq_sum": 2, "c_in": 32, "c_out": 8,
         "edge_dim": 33, "E": 10432, "in_dim": 4, "out_dim": 1},
        # Final ConvSE3 conv_out['1'] (in_dim=4 = fused degree 0+1)
        {"name": "final_out1", "freq_sum": 4, "c_in": 32, "c_out": 2,
         "edge_dim": 33, "E": 10432, "in_dim": 4, "out_dim": 3},
        # Larger E from kmin graph expansion
        {"name": "out0_bigE", "freq_sum": 2, "c_in": 32, "c_out": 8,
         "edge_dim": 33, "E": 15598, "in_dim": 4, "out_dim": 1},
        {"name": "out1_bigE", "freq_sum": 4, "c_in": 32, "c_out": 2,
         "edge_dim": 33, "E": 15598, "in_dim": 4, "out_dim": 3},
    ]

    for cfg in configs:
        print(f"\n  Config: {cfg['name']} (E={cfg['E']}, C_in={cfg['c_in']}, "
              f"C_out={cfg['c_out']}, freq={cfg['freq_sum']})")

        conv = VersatileConvSE3(
            freq_sum=cfg['freq_sum'], channels_in=cfg['c_in'],
            channels_out=cfg['c_out'], edge_dim=cfg['edge_dim'],
            use_layer_norm=True, fuse_level=ConvSE3FuseLevel.FULL)
        mx.eval(conv.parameters())

        fused = FusedConvSE3(conv, edge_dim=cfg['edge_dim'])

        E = cfg['E']
        in_dim = cfg['in_dim']
        out_dim = cfg['out_dim']
        features = mx.random.normal((E, cfg['c_in'], in_dim))
        edge_feats = mx.random.normal((E, cfg['edge_dim']))
        # Basis: (E, in_dim * freq_sum, out_dim)
        basis = mx.random.normal((E, in_dim * cfg['freq_sum'], out_dim))
        mx.eval(features, edge_feats, basis)

        # Reference
        ref = conv(features, edge_feats, basis)
        mx.eval(ref)

        # Fused
        fused_out = fused(features, edge_feats, basis)
        mx.eval(fused_out)

        max_diff = mx.max(mx.abs(ref - fused_out)).item()
        print(f"    Max diff: {max_diff:.6f}", end="")

        if max_diff < 1e-2:
            print("  OK")
        else:
            print("  FAIL")


def test_fused_conv_perf():
    """Performance comparison for all 4 real model configs."""
    import time
    print("\nTest: performance comparison (all configs)...")

    configs = [
        # Actual model shapes — conv_out (final conv, fused in production)
        {"name": "final_out0", "freq_sum": 2, "c_in": 32, "c_out": 8,
         "edge_dim": 33, "E": 10432, "in_dim": 4, "out_dim": 1},
        {"name": "final_out1", "freq_sum": 4, "c_in": 32, "c_out": 2,
         "edge_dim": 33, "E": 10432, "in_dim": 4, "out_dim": 3},
        # conv_in (attention — NOT fused in production due to stability)
        {"name": "attn_in0", "freq_sum": 2, "c_in": 8, "c_out": 16,
         "edge_dim": 33, "E": 10432, "in_dim": 1, "out_dim": 4},
        {"name": "attn_in1", "freq_sum": 4, "c_in": 3, "c_out": 16,
         "edge_dim": 33, "E": 10432, "in_dim": 3, "out_dim": 4},
    ]

    n_iters = 20
    print(f"\n  {'Config':<12} {'Ref(ms)':>8} {'Fused(ms)':>10} {'Speedup':>8}  "
          f"{'MLP_OUT':>8} {'Tile':>6}")
    print("  " + "-" * 62)

    for cfg in configs:
        conv = VersatileConvSE3(
            freq_sum=cfg['freq_sum'], channels_in=cfg['c_in'],
            channels_out=cfg['c_out'], edge_dim=cfg['edge_dim'],
            use_layer_norm=True, fuse_level=ConvSE3FuseLevel.FULL)
        mx.eval(conv.parameters())
        fused = FusedConvSE3(conv, edge_dim=cfg['edge_dim'])

        E = cfg['E']
        features = mx.random.normal((E, cfg['c_in'], cfg['in_dim']))
        edge_feats = mx.random.normal((E, cfg['edge_dim']))
        basis = mx.random.normal((E, cfg['in_dim'] * cfg['freq_sum'], cfg['out_dim']))
        mx.eval(features, edge_feats, basis)

        # Warmup
        for _ in range(5):
            r = conv(features, edge_feats, basis); mx.eval(r)
        for _ in range(5):
            f = fused(features, edge_feats, basis); mx.eval(f)

        # Time reference
        t0 = time.time()
        for _ in range(n_iters):
            r = conv(features, edge_feats, basis); mx.eval(r)
        t_ref = (time.time() - t0) / n_iters * 1000

        # Time fused
        t0 = time.time()
        for _ in range(n_iters):
            f = fused(features, edge_feats, basis); mx.eval(f)
        t_fused = (time.time() - t0) / n_iters * 1000

        mlp_out = cfg['freq_sum'] * cfg['c_in'] * cfg['c_out']
        tile = cfg['c_in'] * cfg['freq_sum']
        speedup = (t_ref - t_fused) / t_ref * 100
        print(f"  {cfg['name']:<12} {t_ref:>8.2f} {t_fused:>10.2f} {speedup:>+7.1f}%  "
              f"{mlp_out:>8} {tile:>6}")


if __name__ == '__main__':
    test_pack_mlp_params()
    test_fused_conv_basic()
    test_fused_conv_shapes()
    test_fused_conv_perf()
    print("\nAll tests done.")
