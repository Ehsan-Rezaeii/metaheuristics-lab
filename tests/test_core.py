import numpy as np
import pytest

from metaheuristics_lab import GA, GWO, PSO, WOA, BinaryGA, BinaryGWO, Problem, RandomSearch
from metaheuristics_lab.core import Evaluator

CONTINUOUS = [GA, GWO, PSO, WOA, RandomSearch]
BINARY = [BinaryGA, BinaryGWO]


@pytest.mark.parametrize("optimizer", CONTINUOUS + BINARY)
@pytest.mark.parametrize("budget", [1, 2, 7, 23, 61])
def test_budget_bounds_history_reproducibility(optimizer, budget):
    seen = []
    binary = optimizer in BINARY

    def objective(x):
        seen.append(x.copy())
        assert x.shape == (4,)
        if binary:
            assert np.isin(x, [0, 1]).all()
        else:
            assert ((x >= [-3, 1, -1, 4]) & (x <= [2, 2, 1, 8])).all()
        return float(np.sum(x**2))

    problem = (
        Problem.binary(objective, 4)
        if binary
        else Problem.continuous(objective, 4, [-3, 1, -1, 4], [2, 2, 1, 8])
    )
    algorithm = optimizer(population_size=7)
    first = algorithm.optimize(problem, max_evaluations=budget, seed=17)
    assert len(seen) == budget == first.n_evaluations
    values = [np.sum(x**2) for x in seen]
    np.testing.assert_allclose(first.history, np.minimum.accumulate(values))
    assert first.best_fitness == min(values) == np.sum(first.best_solution**2)
    np.testing.assert_array_equal(first.evaluations, np.arange(1, budget + 1))
    assert first.runtime_seconds >= 0
    second = algorithm.optimize(problem, max_evaluations=budget, seed=17)
    np.testing.assert_array_equal(first.best_solution, second.best_solution)
    np.testing.assert_array_equal(first.history, second.history)


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True])
def test_bad_dimension(dimension):
    with pytest.raises(ValueError):
        Problem.binary(np.sum, dimension)


@pytest.mark.parametrize(
    "lower,upper", [(0, 0), (2, 1), (np.nan, 1), (-1, np.inf), ([0, 1], [1, 2])]
)
def test_bad_bounds(lower, upper):
    with pytest.raises(ValueError):
        Problem.continuous(np.sum, 3, lower, upper)


def test_problem_validation_and_bounds_copy():
    with pytest.raises(ValueError):
        Problem.binary(None, 4)
    with pytest.raises(ValueError):
        Problem(np.sum, 2, 0, 1, "unknown")
    with pytest.raises(ValueError):
        Problem(np.sum, 2, -1, 1, "binary")
    lower = np.array([-2.0, -1.0])
    problem = Problem.continuous(np.sum, 2, lower, 3)
    lower[:] = 7
    assert problem.lower[0] == -2
    with pytest.raises(ValueError):
        problem.lower[0] = 0


@pytest.mark.parametrize(
    "value", [np.nan, np.inf, -np.inf, [1], np.array([1]), np.array(1), True, 1 + 2j, "one", None]
)
def test_bad_objective_outputs(value):
    with pytest.raises(ValueError, match="finite real scalar"):
        GA().optimize(Problem.continuous(lambda x: value, 2, -1, 1), max_evaluations=1, seed=0)


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_bad_budget_and_population(value):
    with pytest.raises(ValueError):
        GA(population_size=value)
    with pytest.raises(ValueError):
        GA().optimize(Problem.continuous(np.sum, 2, 0, 1), max_evaluations=value)


@pytest.mark.parametrize("seed", [-1, True, 1.2])
def test_bad_seed(seed):
    with pytest.raises(ValueError):
        GA().optimize(Problem.continuous(np.sum, 2, 0, 1), max_evaluations=3, seed=seed)


def test_domain_errors_and_hyperparameters():
    for algorithm, problem in [
        (GA(), Problem.binary(np.sum, 3)),
        (BinaryGA(), Problem.continuous(np.sum, 3, -1, 1)),
    ]:
        with pytest.raises(ValueError):
            algorithm.optimize(problem, max_evaluations=3)
    for kwargs in [
        {"mutation_rate": -1},
        {"crossover_rate": 2},
        {"mutation_scale": np.nan},
        {"tournament_size": False},
    ]:
        with pytest.raises(ValueError):
            GA(**kwargs)
    with pytest.raises(ValueError):
        GWO(population_size=2)
    with pytest.raises(ValueError):
        RandomSearch(domain="other")


def test_defensive_objective_argument_and_exceptions():
    def mutate(x):
        x[:] = 999
        return 0.0

    result = GA().optimize(Problem.continuous(mutate, 2, -1, 1), max_evaluations=3, seed=0)
    assert (abs(result.best_solution) <= 1).all()

    def broken(x):
        raise RuntimeError("application failure")

    with pytest.raises(RuntimeError, match="application failure"):
        GA().optimize(Problem.continuous(broken, 2, -1, 1), max_evaluations=1)


def test_evaluator_guard_and_candidate_validation():
    evaluator = Evaluator(Problem.binary(np.sum, 2), 1)
    for candidate in [np.array([0.5, 1]), np.array([0, 2]), np.array([0]), np.array([np.nan, 0])]:
        with pytest.raises(ValueError):
            evaluator(candidate)
    assert evaluator(np.array([0, 1])) == 1
    with pytest.raises(RuntimeError, match="exhausted"):
        evaluator(np.array([0, 0]))


def test_global_random_state_untouched():
    np.random.seed(12)
    expected = np.random.random(4)
    np.random.seed(12)
    GA().optimize(
        Problem.continuous(np.sum, 3, -1, 1),
        max_evaluations=80,
        seed=0,
    )
    np.testing.assert_array_equal(np.random.random(4), expected)


def test_binary_random_search_and_empty_mask():
    result = RandomSearch(domain="binary").optimize(
        Problem.binary(lambda x: 2.0 if x.sum() == 0 else 1.0, 1), max_evaluations=100, seed=0
    )
    assert result.best_fitness == 1
    assert result.best_solution[0] == 1
