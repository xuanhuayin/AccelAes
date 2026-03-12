"""
Noise prediction extrapolators for skip-update (Phase 4).

Supports four extrapolation spaces:
  - copy: reuse last prediction directly
  - linear: linear extrapolation in noise_pred space
  - velocity: linear extrapolation in velocity-norm space (v * sigma)
  - x0: linear extrapolation in x0-prediction space (x_t - sigma * v)
"""

import torch

# Minimum sigma clamp to avoid division by zero near the end of denoising
_SIGMA_EPS = 1e-6


class Extrapolator:
    """Base class for noise prediction extrapolators."""

    def store(self, step: int, prediction: torch.Tensor,
              sigma: float, latent: torch.Tensor):
        """Store a prediction for later extrapolation."""
        raise NotImplementedError

    def extrapolate(self, step: int, sigma: float,
                    latent: torch.Tensor) -> torch.Tensor:
        """Extrapolate a prediction for the current step."""
        raise NotImplementedError

    def can_extrapolate(self) -> bool:
        """Whether enough history exists to extrapolate."""
        raise NotImplementedError

    def reset(self):
        """Clear stored history."""
        raise NotImplementedError


class CopyExtrapolator(Extrapolator):
    """Reuse the last stored prediction directly."""

    def __init__(self):
        self.last = None

    def store(self, step, prediction, sigma=None, latent=None):
        self.last = prediction.clone()

    def extrapolate(self, step, sigma=None, latent=None):
        return self.last

    def can_extrapolate(self):
        return self.last is not None

    def reset(self):
        self.last = None


class LinearExtrapolator(Extrapolator):
    """Linear extrapolation in noise_pred space: 2*v[-1] - v[-2]."""

    def __init__(self):
        self.history = []  # list of noise_pred tensors, max 2

    def store(self, step, prediction, sigma=None, latent=None):
        self.history.append(prediction.clone())
        if len(self.history) > 2:
            self.history.pop(0)

    def extrapolate(self, step, sigma=None, latent=None):
        if len(self.history) < 2:
            return self.history[-1]
        return 2 * self.history[-1] - self.history[-2]

    def can_extrapolate(self):
        return len(self.history) > 0

    def reset(self):
        self.history.clear()


class VelocityExtrapolator(Extrapolator):
    """
    Linear extrapolation in velocity-norm space.

    Stores v_norm = v * sigma, extrapolates linearly, then converts back:
        v_norm_extrap = 2 * v_norm[-1] - v_norm[-2]
        v_extrap = v_norm_extrap / max(sigma_cur, eps)
    """

    def __init__(self):
        self.history = []  # list of v_norm tensors, max 2

    def store(self, step, prediction, sigma, latent=None):
        v_norm = prediction * sigma
        self.history.append(v_norm.clone())
        if len(self.history) > 2:
            self.history.pop(0)

    def extrapolate(self, step, sigma, latent=None):
        if len(self.history) < 2:
            v_norm = self.history[-1]
        else:
            v_norm = 2 * self.history[-1] - self.history[-2]
        return v_norm / max(sigma, _SIGMA_EPS)

    def can_extrapolate(self):
        return len(self.history) > 0

    def reset(self):
        self.history.clear()


class X0Extrapolator(Extrapolator):
    """
    Linear extrapolation in x0-prediction space.

    Stores x0 = x_t - sigma * v, extrapolates linearly, then converts back:
        x0_extrap = 2 * x0[-1] - x0[-2]
        v_extrap = (x_t_cur - x0_extrap) / max(sigma_cur, eps)
    """

    def __init__(self):
        self.history = []  # list of x0 tensors, max 2

    def store(self, step, prediction, sigma, latent):
        x0 = latent - sigma * prediction
        self.history.append(x0.clone())
        if len(self.history) > 2:
            self.history.pop(0)

    def extrapolate(self, step, sigma, latent):
        if len(self.history) < 2:
            x0 = self.history[-1]
        else:
            x0 = 2 * self.history[-1] - self.history[-2]
        return (latent - x0) / max(sigma, _SIGMA_EPS)

    def can_extrapolate(self):
        return len(self.history) > 0

    def reset(self):
        self.history.clear()


def make_extrapolator(method: str) -> Extrapolator:
    """Factory function for creating extrapolators by name."""
    constructors = {
        "copy": CopyExtrapolator,
        "linear": LinearExtrapolator,
        "velocity": VelocityExtrapolator,
        "x0": X0Extrapolator,
    }
    if method not in constructors:
        raise ValueError(
            f"Unknown extrapolation method: {method}. "
            f"Available: {list(constructors.keys())}"
        )
    return constructors[method]()
