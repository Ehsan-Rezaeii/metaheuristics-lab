"""Reproducible NumPy optimization under explicit objective-evaluation budgets."""

from .benchmarks import ackley, benchmark_problem, rastrigin, rosenbrock, sphere
from .core import OptimizationResult, Problem
from .optimizers import GA, GWO, PSO, WOA, BinaryGA, BinaryGWO, RandomSearch

__version__ = "0.1.0"
__all__ = [
    "Problem",
    "OptimizationResult",
    "GA",
    "PSO",
    "WOA",
    "GWO",
    "BinaryGA",
    "BinaryGWO",
    "RandomSearch",
    "sphere",
    "rastrigin",
    "ackley",
    "rosenbrock",
    "benchmark_problem",
]

from .transfers import s_shaped, v_shaped

__all__ += ["s_shaped", "v_shaped"]
