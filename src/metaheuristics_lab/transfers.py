"""Binary GWO transfer rules with deliberately different bit semantics."""

from __future__ import annotations

import numpy as np

from .core import Vector


def s_shaped(signal: Vector) -> Vector:
    """Stable logistic probability: S(z)=1/(1+exp(-z))."""
    return 0.5 * (1 + np.tanh(np.asarray(signal, dtype=float) / 2))


def v_shaped(displacement: Vector) -> Vector:
    """Flip probability V(delta)=|tanh(delta)|; zero displacement means no flip."""
    return np.abs(np.tanh(np.asarray(displacement, dtype=float)))


def binary_update(
    current: Vector, target: Vector, rng: np.random.Generator, transfer: str
) -> Vector:
    """S resamples bits from S(2*target-1); V flips using V(target-current).

    S centers a target of 0.5 at probability 0.5. V acts on displacement,
    preserving bits at zero movement. These are explicit reference variants,
    not a claim to reproduce every published Binary GWO formulation.
    """
    if transfer == "s":
        return (rng.random(current.shape) < s_shaped(2 * target - 1)).astype(float)
    if transfer == "v":
        flips = rng.random(current.shape) < v_shaped(target - current)
        return np.where(flips, 1 - current, current)
    raise ValueError("transfer must be 's' or 'v'")
