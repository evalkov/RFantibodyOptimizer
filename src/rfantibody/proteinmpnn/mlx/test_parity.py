#!/usr/bin/env python3
"""
Test MLX ProteinMPNN against PyTorch reference for numerical parity.

Usage:
    PYTHONPATH=src:include/SE3Transformer pilot_mps/.venv/bin/python \
        src/rfantibody/proteinmpnn/mlx/test_parity.py
"""
import os
import sys
import time
import logging

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Load PyTorch model ---
from rfantibody.proteinmpnn.model.protein_mpnn_utils import ProteinMPNN as TorchMPNN

ckpt_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'ProteinMPNN_v48_noise_0.2.pt')
ckpt_path = os.path.abspath(ckpt_path)
print(f"Loading checkpoint: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
num_edges = ckpt.get('num_edges', 48)
noise_level = ckpt.get('noise_level', 0.2)

torch_model = TorchMPNN(
    num_letters=21, node_features=128, edge_features=128,
    hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
    k_neighbors=num_edges, augment_eps=0.0, dropout=0.1)
torch_model.load_state_dict(ckpt['model_state_dict'])
torch_model.eval()
print(f"PyTorch model loaded ({sum(p.numel() for p in torch_model.parameters()):,} params)")

# --- Load MLX model ---
from rfantibody.proteinmpnn.mlx.weight_converter import load_checkpoint_to_mlx
import mlx.core as mx

mlx_model, _ = load_checkpoint_to_mlx(ckpt_path)
mlx_model.features.augment_eps = 0.0

# --- Create test inputs ---
B, L = 1, 50  # small test case
np.random.seed(42)
torch.manual_seed(42)

# Random coordinates (N, CA, C, O)
X_np = np.random.randn(B, L, 4, 3).astype(np.float32) * 3.0
S_np = np.random.randint(0, 21, (B, L)).astype(np.int64)
mask_np = np.ones((B, L), dtype=np.float32)
chain_M_np = np.ones((B, L), dtype=np.float32)
# Make some positions fixed
chain_M_np[0, :10] = 0.0
residue_idx_np = np.tile(np.arange(L), (B, 1)).astype(np.int64)
chain_encoding_np = np.ones((B, L), dtype=np.int64)
randn_np = np.random.randn(B, L).astype(np.float32)

X_torch = torch.from_numpy(X_np)
S_torch = torch.from_numpy(S_np)
mask_torch = torch.from_numpy(mask_np)
chain_M_torch = torch.from_numpy(chain_M_np)
residue_idx_torch = torch.from_numpy(residue_idx_np)
chain_enc_torch = torch.from_numpy(chain_encoding_np)
randn_torch = torch.from_numpy(randn_np)

X_mx = mx.array(X_np)
S_mx = mx.array(S_np.astype(np.int32))
mask_mx = mx.array(mask_np)
chain_M_mx = mx.array(chain_M_np)
residue_idx_mx = mx.array(residue_idx_np)
chain_enc_mx = mx.array(chain_encoding_np)
randn_mx = mx.array(randn_np)


# --- Test 1: Forward pass parity ---
print("\n" + "=" * 60)
print("Test 1: Forward pass (log_probs) parity")
print("=" * 60)

with torch.no_grad():
    log_probs_torch = torch_model(
        X_torch, S_torch, mask_torch, chain_M_torch,
        residue_idx_torch, chain_enc_torch, randn_torch)
log_probs_torch_np = log_probs_torch.numpy()

log_probs_mlx = mlx_model(
    X_mx, S_mx, mask_mx, chain_M_mx,
    residue_idx_mx, chain_enc_mx, randn_mx)
mx.eval(log_probs_mlx)
log_probs_mlx_np = np.array(log_probs_mlx)

diff = np.abs(log_probs_torch_np - log_probs_mlx_np)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
print(f"  Shape: torch={log_probs_torch_np.shape}, mlx={log_probs_mlx_np.shape}")
print(f"  Max abs diff:  {max_diff:.6f}")
print(f"  Mean abs diff: {mean_diff:.6f}")

if max_diff < 1e-3:
    print("  PASS: forward pass matches within 1e-3")
elif max_diff < 1e-2:
    print("  PASS (loose): forward pass matches within 1e-2")
else:
    print(f"  FAIL: max diff {max_diff:.4f} exceeds tolerance")
    # Debug: check intermediate values
    print(f"  Torch log_probs range: [{log_probs_torch_np.min():.4f}, {log_probs_torch_np.max():.4f}]")
    print(f"  MLX log_probs range:   [{log_probs_mlx_np.min():.4f}, {log_probs_mlx_np.max():.4f}]")


# --- Test 2: Encoder output parity ---
print("\n" + "=" * 60)
print("Test 2: Encoder output parity")
print("=" * 60)

# Run encoder separately
with torch.no_grad():
    E_torch, E_idx_torch = torch_model.features(
        X_torch, mask_torch, residue_idx_torch, chain_enc_torch)

E_mlx, E_idx_mlx = mlx_model.features(
    X_mx, mask_mx, residue_idx_mx, chain_enc_mx)
mx.eval(E_mlx, E_idx_mlx)

E_torch_np = E_torch.numpy()
E_mlx_np = np.array(E_mlx)
E_idx_torch_np = E_idx_torch.numpy()
E_idx_mlx_np = np.array(E_idx_mlx)

e_diff = np.max(np.abs(E_torch_np - E_mlx_np))
idx_match = np.all(E_idx_torch_np == E_idx_mlx_np)
print(f"  Edge features max diff: {e_diff:.6f}")
print(f"  E_idx match: {idx_match}")

if e_diff < 1e-4:
    print("  PASS")
else:
    print(f"  Feature diff is {e_diff:.6f} — check RBF/positional encoding")


# --- Test 3: Wrapper bridge test ---
print("\n" + "=" * 60)
print("Test 3: Wrapper bridge (torch in → torch out)")
print("=" * 60)

from rfantibody.proteinmpnn.mlx.model_wrapper import MLXMPNNWrapper
wrapper = MLXMPNNWrapper.from_checkpoint(ckpt_path)

log_probs_wrapper = wrapper(
    X_torch, S_torch, mask_torch, chain_M_torch,
    residue_idx_torch, chain_enc_torch, randn_torch)

wrapper_diff = torch.max(torch.abs(log_probs_torch - log_probs_wrapper)).item()
print(f"  Wrapper vs PyTorch max diff: {wrapper_diff:.6f}")
print(f"  Output type: {type(log_probs_wrapper)}, device: {log_probs_wrapper.device}")
if wrapper_diff < 1e-2:
    print("  PASS")
else:
    print("  FAIL")


print("\n" + "=" * 60)
print("All tests done.")
print("=" * 60)
