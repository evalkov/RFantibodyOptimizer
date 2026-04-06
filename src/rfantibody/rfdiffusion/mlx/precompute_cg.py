#!/usr/bin/env python3
"""
Precompute Clebsch-Gordan coefficients for SE3Transformer and save as .npz.

These are mathematical constants that depend only on max_degree.
They are computed once via e3nn (which requires PyTorch) and then loaded
as MLX arrays at runtime, eliminating the e3nn dependency from the MLX path.

Usage:
    python -m rfantibody.rfdiffusion.mlx.precompute_cg [--max_degree 2]
"""
import argparse
import os

import numpy as np


def precompute_cg_coefficients(max_degree: int = 2) -> dict:
    """Compute all CG coefficients needed for SE3Transformer basis computation.

    For each (d_in, d_out) pair with d_in, d_out in [0, max_degree],
    and for each J in [|d_in - d_out|, d_in + d_out], compute:
        Q^{d_out, d_in}_J = wigner_3j(J, d_in, d_out).permute(2, 1, 0)

    Returns:
        dict with keys like 'cg_{d_in}_{d_out}_{J}' -> numpy float64 array
    """
    import torch
    import e3nn.o3 as o3

    cg_dict = {}
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                cg = o3.wigner_3j(J, d_in, d_out, dtype=torch.float64, device='cpu')
                cg = cg.permute(2, 1, 0)  # Match get_clebsch_gordon() convention
                key = f'cg_{d_in}_{d_out}_{J}'
                cg_dict[key] = cg.numpy()

    # Store metadata
    cg_dict['max_degree'] = np.array(max_degree)

    return cg_dict


def main():
    parser = argparse.ArgumentParser(description='Precompute CG coefficients for MLX')
    parser.add_argument('--max_degree', type=int, default=2,
                        help='Maximum degree for SE3Transformer (default: 2)')
    args = parser.parse_args()

    cg_dict = precompute_cg_coefficients(args.max_degree)

    out_path = os.path.join(os.path.dirname(__file__), 'cg_coefficients.npz')
    np.savez_compressed(out_path, **cg_dict)

    n_coeffs = sum(1 for k in cg_dict if k.startswith('cg_'))
    total_elements = sum(v.size for k, v in cg_dict.items() if k.startswith('cg_'))
    print(f'Saved {n_coeffs} CG coefficient tensors ({total_elements} elements) to {out_path}')
    print(f'Max degree: {args.max_degree}')

    # Print summary
    for k in sorted(cg_dict.keys()):
        if k.startswith('cg_'):
            print(f'  {k}: shape {cg_dict[k].shape}')


if __name__ == '__main__':
    main()
