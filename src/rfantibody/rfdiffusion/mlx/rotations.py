"""
MLX rotation conversion utilities for RFdiffusion inference.

Ports the essential functions from rotation_conversions.py:
  quaternion↔matrix, axis_angle↔matrix/quaternion, SLERP, rigid_from_3_points.

MLX lacks boolean indexing; all branching uses mx.where instead.
"""

import math

import mlx.core as mx


# ---------------------------------------------------------------------------
# Quaternion ↔ Matrix
# ---------------------------------------------------------------------------

def quaternion_to_matrix(quaternions: mx.array) -> mx.array:
    """Convert quaternions (real-first) to rotation matrices.

    Args:
        quaternions: (..., 4) real-first quaternions.
    Returns:
        (..., 3, 3) rotation matrices.
    """
    r = quaternions[..., 0]
    i = quaternions[..., 1]
    j = quaternions[..., 2]
    k = quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = mx.stack([
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ], axis=-1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: mx.array) -> mx.array:
    """Convert rotation matrices to quaternions (real-first).

    Uses Shepperd's method with mx.where for branch-free selection
    (no boolean indexing required).

    Args:
        matrix: (..., 3, 3) rotation matrices.
    Returns:
        (..., 4) real-first quaternions.
    """
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(batch_shape + (9,))
    m00 = m[..., 0]; m01 = m[..., 1]; m02 = m[..., 2]
    m10 = m[..., 3]; m11 = m[..., 4]; m12 = m[..., 5]
    m20 = m[..., 6]; m21 = m[..., 7]; m22 = m[..., 8]

    # Four candidate square roots
    q_abs = mx.sqrt(mx.maximum(
        mx.stack([
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ], axis=-1), 0.0))

    # Each candidate quaternion (scaled)
    quat_by_0 = mx.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1)
    quat_by_1 = mx.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1)
    quat_by_2 = mx.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1)
    quat_by_3 = mx.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1)

    flr = 0.1
    denom = 2.0 * mx.maximum(q_abs, flr)

    cand0 = quat_by_0 / denom[..., 0:1]
    cand1 = quat_by_1 / denom[..., 1:2]
    cand2 = quat_by_2 / denom[..., 2:3]
    cand3 = quat_by_3 / denom[..., 3:4]

    # Pick best-conditioned candidate using argmax
    best = mx.argmax(q_abs, axis=-1)
    # Expand for broadcasting: (...) -> (..., 1, 1) is not needed;
    # we use nested mx.where
    result = mx.where(
        (best == 0)[..., None], cand0,
        mx.where(
            (best == 1)[..., None], cand1,
            mx.where(
                (best == 2)[..., None], cand2,
                cand3)))
    return result


# ---------------------------------------------------------------------------
# Axis-angle ↔ Quaternion ↔ Matrix
# ---------------------------------------------------------------------------

def axis_angle_to_quaternion(axis_angle: mx.array) -> mx.array:
    """Convert axis-angle to quaternion (real-first).

    Args:
        axis_angle: (..., 3) rotation vectors.
    Returns:
        (..., 4) real-first quaternions.
    """
    angles = mx.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-6
    # For small angles: sin(x/2)/x ≈ 0.5 - x²/48
    # For large angles: sin(x/2)/x
    sin_ha = mx.sin(half_angles)
    ratio_large = sin_ha / mx.maximum(angles, eps)
    ratio_small = 0.5 - (angles * angles) / 48.0
    ratio = mx.where(angles.abs() < eps, ratio_small, ratio_large)
    return mx.concatenate([mx.cos(half_angles), axis_angle * ratio], axis=-1)


def quaternion_to_axis_angle(quaternions: mx.array) -> mx.array:
    """Convert quaternion (real-first) to axis-angle.

    Args:
        quaternions: (..., 4) real-first quaternions.
    Returns:
        (..., 3) rotation vectors.
    """
    norms = mx.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    half_angles = mx.arctan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    sin_ha = mx.sin(half_angles)
    ratio_large = sin_ha / mx.maximum(angles, eps)
    ratio_small = 0.5 - (angles * angles) / 48.0
    ratio = mx.where(angles.abs() < eps, ratio_small, ratio_large)
    return quaternions[..., 1:] / mx.maximum(ratio, eps)


def axis_angle_to_matrix(axis_angle: mx.array) -> mx.array:
    """Convert axis-angle to rotation matrix.

    Args:
        axis_angle: (..., 3) rotation vectors.
    Returns:
        (..., 3, 3) rotation matrices.
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: mx.array) -> mx.array:
    """Convert rotation matrix to axis-angle.

    Args:
        matrix: (..., 3, 3) rotation matrices.
    Returns:
        (..., 3) rotation vectors.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


# ---------------------------------------------------------------------------
# Quaternion SLERP
# ---------------------------------------------------------------------------

def quaternion_slerp(q0: mx.array, q1: mx.array, alpha) -> mx.array:
    """Batched quaternion SLERP.

    Args:
        q0, q1: (..., 4) real-first quaternions.
        alpha: interpolation parameter, scalar or broadcastable.
    Returns:
        (..., 4) interpolated quaternions.
    """
    if not isinstance(alpha, mx.array):
        alpha = mx.array(alpha, dtype=q0.dtype)
    while alpha.ndim < q0.ndim:
        alpha = mx.expand_dims(alpha, -1)

    dot = (q0 * q1).sum(axis=-1, keepdims=True)
    # Shortest path
    q1 = mx.where(dot < 0, -q1, q1)
    dot = mx.abs(dot)
    dot = mx.clip(dot, a_min=None, a_max=1.0 - 1e-6)

    theta = mx.arccos(dot)
    sin_theta = mx.sin(theta)

    s0 = mx.sin((1.0 - alpha) * theta) / mx.maximum(sin_theta, mx.array(1e-6))
    s1 = mx.sin(alpha * theta) / mx.maximum(sin_theta, mx.array(1e-6))

    result = s0 * q0 + s1 * q1
    # Fallback to linear for small angles
    small = mx.abs(sin_theta) < 1e-6
    linear = (1.0 - alpha) * q0 + alpha * q1
    result = mx.where(mx.broadcast_to(small, result.shape), linear, result)

    return result / mx.linalg.norm(result, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Rigid body from 3 points (N, Ca, C)
# ---------------------------------------------------------------------------

def rigid_from_3_points(N: mx.array, Ca: mx.array, C: mx.array,
                        eps: float = 1e-8):
    """Compute rotation matrices from backbone N, Ca, C coordinates.

    Args:
        N, Ca, C: (B, L, 3) backbone atom coordinates.
    Returns:
        R: (B, L, 3, 3) rotation matrices, Ca: (B, L, 3).
    """
    v1 = C - Ca
    v2 = N - Ca
    e1 = v1 / (mx.linalg.norm(v1, axis=-1, keepdims=True) + eps)
    u2 = v2 - mx.sum(e1 * v2, axis=-1, keepdims=True) * e1
    e2 = u2 / (mx.linalg.norm(u2, axis=-1, keepdims=True) + eps)
    e3 = mx.stack([
        e1[..., 1] * e2[..., 2] - e1[..., 2] * e2[..., 1],
        e1[..., 2] * e2[..., 0] - e1[..., 0] * e2[..., 2],
        e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0],
    ], axis=-1)
    R = mx.concatenate([
        mx.expand_dims(e1, -1),
        mx.expand_dims(e2, -1),
        mx.expand_dims(e3, -1),
    ], axis=-1)
    return R, Ca


# ---------------------------------------------------------------------------
# Gram-Schmidt re-orthogonalization
# ---------------------------------------------------------------------------

def reorthogonalize(R: mx.array) -> mx.array:
    """Modified Gram-Schmidt ensuring proper rotations (det=+1).

    Cross product for third column guarantees det=+1 without a determinant check.

    Args:
        R: (..., 3, 3) near-orthogonal matrices.
    Returns:
        (..., 3, 3) proper rotation matrices.
    """
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    c0 = c0 / mx.linalg.norm(c0, axis=-1, keepdims=True)
    c1 = c1 - mx.sum(c0 * c1, axis=-1, keepdims=True) * c0
    c1 = c1 / mx.linalg.norm(c1, axis=-1, keepdims=True)
    c2 = mx.stack([
        c0[..., 1] * c1[..., 2] - c0[..., 2] * c1[..., 1],
        c0[..., 2] * c1[..., 0] - c0[..., 0] * c1[..., 2],
        c0[..., 0] * c1[..., 1] - c0[..., 1] * c1[..., 0],
    ], axis=-1)
    return mx.stack([c0, c1, c2], axis=-1)
