"""
Skip-update cache for noise_pred reuse (Phase 3).

Caches full noise_pred tensors and provides copy / linear / quadratic
extrapolation for skipped denoising steps.

Quadratic extrapolation formulas (equally-spaced points p1, p2, p3):
  1-step ahead : 3*p3 - 3*p2 + p1
  2-step ahead : 6*p3 - 8*p2 + 3*p1
"""

import torch


class SkipUpdateCache:
    """Caches noise_pred and provides copy/linear/quadratic extrapolation."""

    def __init__(self, method="copy"):
        """
        Args:
            method: "copy"      — reuse last prediction (0th-order)
                    "linear"    — extrapolate from last 2 points (1st-order)
                    "quadratic" — extrapolate from last 3 points (2nd-order),
                                  supports lookahead > 1.
        """
        if method not in ("copy", "linear", "quadratic"):
            raise ValueError(f"Unknown extrapolation method: {method}")
        self.method = method
        self.history = []  # list of (step_idx, noise_pred), max 3

    def store(self, step, noise_pred):
        """Store noise_pred (cloned). Keep last 3 entries."""
        self.history.append((step, noise_pred.clone()))
        if len(self.history) > 3:
            self.history.pop(0)

    def has_cache(self):
        return len(self.history) > 0

    def get_prediction(self, step=None, lookahead=1):
        """
        Return extrapolated noise_pred.

        Args:
            step     : (unused, kept for backward compat)
            lookahead: how many steps ahead to predict (default 1).
                       quadratic supports 1 and 2; linear supports any int.
        """
        if not self.history:
            raise RuntimeError("No cached predictions available")

        if self.method == "copy":
            return self.history[-1][1]

        elif self.method == "linear":
            if len(self.history) < 2:
                return self.history[-1][1]
            p2, p1 = self.history[-1][1], self.history[-2][1]
            return p2 + lookahead * (p2 - p1)

        else:  # quadratic
            if len(self.history) < 3:
                # Fall back to linear
                if len(self.history) < 2:
                    return self.history[-1][1]
                p2, p1 = self.history[-1][1], self.history[-2][1]
                return p2 + lookahead * (p2 - p1)
            p1, p2, p3 = [h[1] for h in self.history[-3:]]
            if lookahead == 1:
                return 3 * p3 - 3 * p2 + p1
            elif lookahead == 2:
                return 6 * p3 - 8 * p2 + 3 * p1
            else:
                # General Newton forward-difference for equal spacing
                # p(2+k) = a*(2+k)^2 + b*(2+k) + c
                a = (p1 - 2 * p2 + p3) / 2
                b = (4 * p2 - 3 * p1 - p3) / 2
                n = 2 + lookahead
                return a * n * n + b * n + p1

    def reset(self):
        """Clear cache."""
        self.history.clear()


class TaylorSkipCache:
    """
    TaylorSeer-style multi-step prediction cache (arXiv 2503.06923).

    Stores noise_pred tensors with their step indices and predicts future
    values via Taylor expansion with correct spacing normalization:

        F(t+k) = F(t) + (Δ¹/N)·k + (Δ²/(2N²))·k²

    where  N = actual step-gap between the last two stored history points,
           k = how many denoising steps ahead to predict.

    Use with run-based scheduling in the denoising loop:
      - every (run+1)-th step: compute transformer, call store()
      - the run intermediate steps: call predict(target_step=i), skip transformer

    Args:
        order: 1 = linear (TaylorSeer 1st-order),
               2 = quadratic (TaylorSeer 2nd-order, default).
    """

    def __init__(self, order: int = 2):
        self.order = max(1, min(order, 2))
        self.history: list = []   # [(step_idx, noise_pred), ...]  max 3 entries

    def store(self, step: int, noise_pred):
        self.history.append((step, noise_pred.clone()))
        if len(self.history) > 3:
            self.history.pop(0)

    def ready(self) -> bool:
        """True once at least 2 history points are available."""
        return len(self.history) >= 2

    def predict(self, target_step: int):
        """
        Predict noise_pred at target_step using stored history.

        k = target_step − last_stored_step.

        Falls back gracefully: order-2 → order-1 if < 3 history points.
        """
        if not self.history:
            raise RuntimeError("TaylorSkipCache: no history to predict from")

        s_t, p_t = self.history[-1]           # most recent compute step
        k = target_step - s_t                 # steps ahead

        if k <= 0 or len(self.history) < 2:
            return p_t

        s_tm1, p_tm1 = self.history[-2]       # one step before that
        N = s_t - s_tm1                       # spacing (in denoising steps)
        if N <= 0:
            return p_t

        # 1st-order term: (Δ¹/N)·k  where  Δ¹ = p_t - p_tm1
        delta1 = p_t - p_tm1                  # unnormalized 1st difference
        pred = p_t + (delta1 / N) * k

        # 2nd-order correction (requires 3 history points)
        if self.order >= 2 and len(self.history) >= 3:
            s_tm2, p_tm2 = self.history[-3]
            N2 = s_tm1 - s_tm2                # spacing of the earlier interval
            N_eff = (N + N2) / 2.0            # average spacing for Δ²

            # Δ² ≈ (p_t - 2·p_tm1 + p_tm2) / N_eff²
            delta2_raw = p_t - 2.0 * p_tm1 + p_tm2
            pred = pred + (delta2_raw / (2.0 * N_eff * N_eff)) * (k * k)

        return pred

    def has_cache(self) -> bool:
        """Alias for ready() — API compatibility with SkipUpdateCache."""
        return self.ready()

    def get_prediction(self, step=None, lookahead=1):
        """
        API-compatible wrapper around predict().

        When called as get_prediction(step=i), delegates to predict(target_step=i).
        When called without step (step=None), falls back to last stored prediction.
        """
        if step is not None:
            return self.predict(target_step=step)
        return self.history[-1][1] if self.history else None

    def reset(self):
        self.history.clear()
