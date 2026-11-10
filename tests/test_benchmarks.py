import numpy as np
import pytest

from metaheuristics_lab import ackley, benchmark_problem, rastrigin, rosenbrock, sphere


@pytest.mark.parametrize("dimension", [2, 5, 30])
def test_known_optima(dimension):
    zeros = np.zeros(dimension)
    assert sphere(zeros) == 0
    assert rastrigin(zeros) == 0
    assert ackley(zeros) == pytest.approx(0, abs=1e-14)
    assert rosenbrock(np.ones(dimension)) == 0


def test_nontrivial_equations():
    assert sphere(np.array([1, -2])) == 5
    assert rastrigin(np.array([0.5])) == pytest.approx(20.25)
    assert rosenbrock(np.array([0.0, 0.0])) == 1
    assert ackley(np.ones(3)) == pytest.approx(20 * (1 - np.exp(-0.2)))


@pytest.mark.parametrize("function", [sphere, rastrigin, ackley, rosenbrock])
@pytest.mark.parametrize("x", [[], [[1, 2]], [np.nan, 0], [np.inf, 1]])
def test_invalid_inputs(function, x):
    with pytest.raises(ValueError):
        function(x)


def test_problem_factory():
    for name in ["sphere", "rastrigin", "ackley", "rosenbrock"]:
        p = benchmark_problem(name, 3)
        assert p.dimension == 3
        assert np.all(p.lower < p.upper)
    for name, dimension in [("other", 2), ("rosenbrock", 1), ("sphere", 0)]:
        with pytest.raises(ValueError):
            benchmark_problem(name, dimension)
