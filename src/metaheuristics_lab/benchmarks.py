"""Standard unshifted continuous benchmarks, all with global minimum zero."""

from __future__ import annotations

import numpy as np

from .core import Problem, Vector, positive_int


def _vector(x: Vector, minimum: int = 1) -> Vector:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < minimum or not np.all(np.isfinite(x)):
        raise ValueError(f"benchmark input must be a finite vector of length >= {minimum}")
    return x


def sphere(x: Vector) -> float:
    """Sum x_i^2; optimum x=0."""
    x = _vector(x)
    return float(np.sum(x**2))


def rastrigin(x: Vector) -> float:
    """10d + sum(x_i^2 - 10 cos(2 pi x_i)); optimum x=0."""
    x = _vector(x)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: Vector) -> float:
    """Ackley with a=20, b=0.2, c=2pi; optimum x=0."""
    x = _vector(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        - np.exp(np.mean(np.cos(2 * np.pi * x)))
        + 20
        + np.e
    )


def rosenbrock(x: Vector) -> float:
    """Sum[100(x_{i+1}-x_i^2)^2 + (x_i-1)^2]; optimum x=1, d>=2."""
    x = _vector(x, 2)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


BENCHMARKS = {
    "sphere": (sphere, -5.12, 5.12),
    "rastrigin": (rastrigin, -5.12, 5.12),
    "ackley": (ackley, -32.768, 32.768),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
}


def benchmark_problem(name: str, dimension: int = 10) -> Problem:
    """Create one standard box-constrained benchmark by lowercase name."""
    dimension = positive_int(dimension, "dimension", 2 if name == "rosenbrock" else 1)
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; choose from {tuple(BENCHMARKS)}")
    objective, lower, upper = BENCHMARKS[name]
    return Problem.continuous(objective, dimension, lower, upper)
