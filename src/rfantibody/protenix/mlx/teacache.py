"""
TeaCache: adaptive caching for flow-matching ODE sampler.

At each ODE step, computes a modulated input signal. If accumulated
change is below threshold, skips the diffusion transformer forward
pass and reuses the previous velocity prediction.

This integrates with FlowMatchingODESampler in diffusion.py.

Reference: TeaCache (Training-free Efficient Acceleration Cache)
for diffusion transformers -- adapted for protein structure
flow-matching with 5-step ODE.
"""

from __future__ import annotations

import mlx.core as mx


class TeaCache:
    """Adaptive caching for flow-matching ODE sampler.

    At each ODE step, compute modulated input signal. If accumulated
    change is below threshold, skip the diffusion transformer forward
    pass and reuse the previous velocity prediction.

    The modulated signal combines the noisy coordinates with the noise
    level to create a fingerprint of the current diffusion state.
    When this fingerprint changes slowly between steps, the velocity
    prediction is unlikely to change significantly.

    Args:
        threshold: maximum accumulated change before forcing recompute.
            Lower values = more accurate but fewer cache hits.
            Typical range: 0.05-0.30 for 5-step ODE.
        warmup_steps: number of initial steps to always compute
            (no caching). Must be >= 1 to establish baseline.
        sigma_data: data standard deviation for modulation scaling.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        warmup_steps: int = 1,
        sigma_data: float = 16.0,
    ):
        self.threshold = threshold
        self.warmup_steps = max(warmup_steps, 1)
        self.sigma_data = sigma_data

        # Internal state
        self.accumulator: float = 0.0
        self.prev_modulated: mx.array | None = None
        self.prev_velocity: mx.array | None = None

        # Statistics
        self.hits: int = 0
        self.misses: int = 0

    def reset(self):
        """Reset cache state for a new sampling run."""
        self.accumulator = 0.0
        self.prev_modulated = None
        self.prev_velocity = None
        self.hits = 0
        self.misses = 0

    def _compute_modulated_signal(
        self, coords: mx.array, sigma: mx.array
    ) -> mx.array:
        """Compute the modulated input signal for cache comparison.

        The modulation follows the EDM c_in scaling:
            modulated = coords / sqrt(sigma_data^2 + sigma^2)

        This normalizes the signal so that changes in the modulated
        representation correlate with changes in the model output.

        Args:
            coords: [..., N_atoms, 3] current noisy coordinates
            sigma: scalar noise level

        Returns:
            [..., N_atoms, 3] modulated signal
        """
        c_in = 1.0 / mx.sqrt(self.sigma_data ** 2 + sigma ** 2)
        return coords * c_in

    def _compute_change(
        self, current: mx.array, previous: mx.array
    ) -> float:
        """Compute normalized L2 change between signals.

        Uses relative change: ||current - prev|| / (||prev|| + eps)

        Args:
            current: current modulated signal
            previous: previous modulated signal

        Returns:
            scalar change metric
        """
        diff = current - previous
        diff_norm = float(mx.sqrt(mx.sum(diff ** 2)))
        prev_norm = float(mx.sqrt(mx.sum(previous ** 2))) + 1e-8
        return diff_norm / prev_norm

    def should_compute(
        self,
        coords: mx.array,
        sigma: mx.array,
        step_idx: int,
    ) -> bool:
        """Decide whether to run the full diffusion transformer.

        Returns True (must compute) if:
          - We are in the warmup phase (step_idx < warmup_steps)
          - No previous signal exists
          - Accumulated change exceeds threshold

        Returns False (safe to skip) if:
          - Accumulated change is below threshold

        When skipping, the caller should use get_cached() to retrieve
        the previous velocity prediction.

        Args:
            coords: [..., N_atoms, 3] current noisy coordinates
            sigma: scalar noise level
            step_idx: current ODE step index (0-based)

        Returns:
            True if the diffusion transformer must be evaluated
        """
        # Always compute during warmup
        if step_idx < self.warmup_steps:
            modulated = self._compute_modulated_signal(coords, sigma)
            mx.eval(modulated)
            self.prev_modulated = modulated
            self.accumulator = 0.0
            self.misses += 1
            return True

        # No previous state -> must compute
        if self.prev_modulated is None:
            modulated = self._compute_modulated_signal(coords, sigma)
            mx.eval(modulated)
            self.prev_modulated = modulated
            self.accumulator = 0.0
            self.misses += 1
            return True

        # Compute change from previous step
        modulated = self._compute_modulated_signal(coords, sigma)
        mx.eval(modulated)
        change = self._compute_change(modulated, self.prev_modulated)
        self.accumulator += change
        self.prev_modulated = modulated

        if self.accumulator >= self.threshold:
            # Change is significant -> must recompute
            self.accumulator = 0.0
            self.misses += 1
            return True
        else:
            # Change is small -> safe to reuse cached velocity
            self.hits += 1
            return False

    def cache_result(self, velocity: mx.array):
        """Store velocity for potential reuse in the next step.

        Args:
            velocity: [..., N_atoms, 3] predicted velocity from
                the diffusion transformer
        """
        self.prev_velocity = velocity

    def get_cached(self) -> mx.array:
        """Return cached velocity from the previous step.

        Returns:
            [..., N_atoms, 3] cached velocity prediction

        Raises:
            RuntimeError: if no cached velocity is available
        """
        if self.prev_velocity is None:
            raise RuntimeError(
                "TeaCache: no cached velocity available. "
                "This should not happen -- should_compute() must "
                "return True on the first call."
            )
        return self.prev_velocity

    @property
    def hit_rate(self) -> float:
        """Fraction of steps where cached velocity was reused."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def __repr__(self) -> str:
        total = self.hits + self.misses
        return (
            f"TeaCache(threshold={self.threshold}, "
            f"warmup={self.warmup_steps}, "
            f"hits={self.hits}/{total}, "
            f"rate={self.hit_rate:.1%})"
        )
