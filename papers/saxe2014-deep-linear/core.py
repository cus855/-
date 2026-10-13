"""Small, dependency-light pieces of Saxe et al. (2014), Eqs. (12) and (15)."""

from __future__ import annotations

import numpy as np


def eq12(t, s: float, u0: float, tau: float = 1.0):
    """Exact strength of one balanced mode in a three-layer network."""
    if not (s > 0 and 0 < u0 < s and tau > 0):
        raise ValueError("Require s > u0 > 0 and tau > 0")
    t = np.asarray(t, dtype=np.float64)
    return s / (1.0 + (s / u0 - 1.0) * np.exp(-2.0 * s * t / tau))


def eq15_rhs(u: float, s: float, depth: int, tau: float = 1.0) -> float:
    """du/dt for a balanced mode in a network with `depth` layers."""
    if depth < 3 or s <= 0 or tau <= 0 or u < 0:
        raise ValueError("Require depth >= 3, s > 0, tau > 0 and u >= 0")
    weights = depth - 1
    return weights * u ** (2.0 - 2.0 / weights) * (s - u) / tau


def integrate_eq15(times, s: float, u0: float, depth: int, tau: float = 1.0):
    """Fixed-grid fourth-order Runge-Kutta integration of Eq. (15)."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be increasing, start at zero, and contain two points")
    if not 0 < u0 < s:
        raise ValueError("Require s > u0 > 0")
    values = np.empty_like(times)
    values[0] = u0
    for i, h in enumerate(np.diff(times), start=1):
        u = values[i - 1]
        k1 = eq15_rhs(u, s, depth, tau)
        k2 = eq15_rhs(u + h * k1 / 2, s, depth, tau)
        k3 = eq15_rhs(u + h * k2 / 2, s, depth, tau)
        k4 = eq15_rhs(u + h * k3, s, depth, tau)
        values[i] = u + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return values
