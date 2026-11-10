"""Validated minimization problems and objective-evaluation accounting."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Objective = Callable[[Vector], float]


def positive_int(value: int, name: str, minimum: int = 1) -> int:
    """Reject booleans, non-integers and integers below the stated minimum."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def real_parameter(value: float, name: str, low: float, high: float) -> float:
    """Validate a finite real hyperparameter in an inclusive interval."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return float(value)


@dataclass(frozen=True)
class Problem:
    """Scalar minimization objective with box bounds or a binary domain.

    Use ``Problem.continuous`` or ``Problem.binary``. Objective arguments are
    defensive copies; bit 1 means selected in feature-mask applications. Empty
    masks are valid domain points; the application must assign their fitness.
    """

    objective: Objective
    dimension: int
    lower: Vector
    upper: Vector
    domain: str = "continuous"

    def __post_init__(self) -> None:
        if not callable(self.objective):
            raise ValueError("objective must be callable")
        dimension = positive_int(self.dimension, "dimension")
        if self.domain not in ("continuous", "binary"):
            raise ValueError("domain must be 'continuous' or 'binary'")
        bounds = []
        for name, raw in (("lower", self.lower), ("upper", self.upper)):
            try:
                array = np.broadcast_to(np.asarray(raw, dtype=float), (dimension,)).copy()
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{name} must be scalar or dimension-length") from exc
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain finite values")
            array.setflags(write=False)
            bounds.append(array)
        lower, upper = bounds
        if np.any(lower >= upper) or not np.all(np.isfinite(upper - lower)):
            raise ValueError("bounds must have finite positive widths")
        if self.domain == "binary" and (np.any(lower != 0) or np.any(upper != 1)):
            raise ValueError("binary bounds must be 0 and 1")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def continuous(
        cls, objective: Objective, dimension: int, lower: float | Vector, upper: float | Vector
    ) -> Problem:
        """Construct a box-constrained problem (scalar bounds broadcast)."""
        return cls(objective, dimension, lower, upper)

    @classmethod
    def binary(cls, objective: Objective, dimension: int) -> Problem:
        """Construct a problem on {0,1}^dimension."""
        return cls(objective, dimension, 0.0, 1.0, "binary")

    def sample(self, rng: np.random.Generator, size: int) -> Vector:
        if self.domain == "binary":
            return rng.integers(0, 2, (size, self.dimension)).astype(float)
        return rng.uniform(self.lower, self.upper, (size, self.dimension))


@dataclass(frozen=True)
class OptimizationResult:
    """Best observed point and best-so-far value after each objective call.

    ``history[i]`` corresponds to evaluation ``i+1``. Runtime includes optimizer
    initialization and objective calls, but excludes argument validation.
    """

    best_solution: Vector
    best_fitness: float
    n_evaluations: int
    history: Vector
    runtime_seconds: float

    @property
    def evaluations(self) -> NDArray[np.int64]:
        return np.arange(1, self.n_evaluations + 1)


class Evaluator:
    """Internal single gateway for validation, counting and incumbent tracking."""

    def __init__(self, problem: Problem, budget: int):
        self.problem = problem
        self.budget = budget
        self.history: list[float] = []
        self.best_fitness = float("inf")
        self.best_solution: Vector | None = None
        self.started = perf_counter()

    @property
    def remaining(self) -> int:
        return self.budget - len(self.history)

    def __call__(self, candidate: Vector) -> float:
        if not self.remaining:
            raise RuntimeError("objective evaluation budget exhausted")
        x = np.asarray(candidate, dtype=float)
        p = self.problem
        if x.shape != (p.dimension,) or not np.all(np.isfinite(x)):
            raise ValueError("candidate must be a finite dimension-length vector")
        if np.any(x < p.lower) or np.any(x > p.upper):
            raise ValueError("candidate is outside bounds")
        if p.domain == "binary" and not np.all((x == 0) | (x == 1)):
            raise ValueError("binary candidate must contain only 0 and 1")
        value = p.objective(x.copy())
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError("objective must return a finite real scalar")
        value = float(value)
        if value < self.best_fitness:
            self.best_fitness = value
            self.best_solution = x.copy()
        self.history.append(self.best_fitness)
        return value

    def result(self) -> OptimizationResult:
        assert self.best_solution is not None
        return OptimizationResult(
            self.best_solution.copy(),
            self.best_fitness,
            len(self.history),
            np.array(self.history),
            perf_counter() - self.started,
        )
