"""
Testing utilities for comparing PyTorch and MLX module outputs.
"""
import numpy as np


def torch_to_mlx(tensor):
    """Convert a PyTorch tensor to an MLX array."""
    import mlx.core as mx
    return mx.array(tensor.detach().cpu().numpy())


def mlx_to_numpy(arr):
    """Convert an MLX array to numpy."""
    import mlx.core as mx
    mx.eval(arr)
    return np.array(arr)


def assert_allclose(torch_out, mlx_out, atol=1e-5, rtol=1e-5, name=""):
    """Compare a PyTorch tensor and MLX array for numerical equivalence.

    Args:
        torch_out: PyTorch tensor
        mlx_out: MLX array
        atol: Absolute tolerance
        rtol: Relative tolerance
        name: Label for error messages
    """
    import torch
    if isinstance(torch_out, torch.Tensor):
        a = torch_out.detach().cpu().numpy()
    else:
        a = np.asarray(torch_out)

    b = mlx_to_numpy(mlx_out)

    if a.shape != b.shape:
        raise AssertionError(
            f"{name}: shape mismatch: PyTorch {a.shape} vs MLX {b.shape}"
        )

    max_diff = np.max(np.abs(a - b))
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name}: max diff = {max_diff:.2e} (atol={atol}, rtol={rtol}), "
            f"shape={a.shape}"
        )
    return max_diff
