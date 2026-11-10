import numpy as np
import pytest

from metaheuristics_lab import GA, PSO, WOA, BinaryGWO, Problem, s_shaped, v_shaped
from metaheuristics_lab.optimizers import _wolf_target
from metaheuristics_lab.transfers import binary_update


def test_transfer_equations_and_semantics():
    np.testing.assert_allclose(s_shaped(np.array([-1e6, 0, 1e6])), [0, 0.5, 1])
    np.testing.assert_allclose(v_shaped(np.array([-1, 0, 1])), [np.tanh(1), 0, np.tanh(1)])
    current = np.array([[0.0, 1.0, 0.0]])
    np.testing.assert_array_equal(
        binary_update(current, current, np.random.default_rng(0), "v"), current
    )
    target = np.array([[-1e6, 1e6, 1e6]])
    np.testing.assert_array_equal(
        binary_update(current, target, np.random.default_rng(0), "s"), [[0, 1, 1]]
    )
    with pytest.raises(ValueError):
        binary_update(current, target, np.random.default_rng(0), "other")
    with pytest.raises(ValueError):
        BinaryGWO(transfer="other")


@pytest.mark.parametrize("transfer", ["s", "v"])
def test_binary_gwo_transfer_run(transfer):
    candidates = []

    def objective(x):
        candidates.append(x.copy())
        return float(-x.sum())

    p = Problem.binary(objective, 12)
    optimizer = BinaryGWO(9, transfer=transfer)
    first = optimizer.optimize(p, max_evaluations=103, seed=13)
    second = optimizer.optimize(p, max_evaluations=103, seed=13)
    assert first.n_evaluations == 103
    assert np.isin(candidates, [0, 1]).all()
    np.testing.assert_array_equal(first.history, second.history)


def test_gwo_zero_a_averages_current_top_three():
    pop = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 4.0], [8.0, 8.0]])
    target = _wolf_target(pop, np.array([1.0, 0.0, 2.0, 3.0]), np.random.default_rng(7), 1)
    np.testing.assert_allclose(target, np.tile(pop[:3].mean(axis=0), (4, 1)))


def test_ga_elite_and_zero_mutation():
    ga = GA(population_size=3, crossover_rate=0, mutation_rate=0)
    pop = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    fitness = np.array([2.0, 4.0, 6.0])
    p = Problem.continuous(np.sum, 2, 0, 10)
    proposed = ga._propose(pop, fitness, {}, p, np.random.default_rng(0), 0)
    assert all(any(np.array_equal(child, parent) for parent in pop) for child in proposed)
    children = np.full((3, 2), 4.0)
    values = np.full(3, 8.0)
    ga._survive(children, values, pop, fitness)
    assert values.min() == 2
    np.testing.assert_array_equal(children[values.argmin()], pop[0])


def test_pso_personal_memory_and_zero_velocity():
    p = Problem.continuous(np.sum, 2, -10, 10)
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = np.array([3.0, 7.0])
    pso = PSO(2, inertia=0, cognitive=0, social=0)
    state = pso._initialize(pop, values, p, np.random.default_rng(0))
    np.testing.assert_array_equal(
        pso._propose(pop, values, state, p, np.random.default_rng(0), 0), pop
    )
    pso._observe(pop + 1, values + 1, state)
    np.testing.assert_array_equal(state["personal"], pop)
    pso._observe(pop - 1, values - 1, state)
    np.testing.assert_array_equal(state["personal"], pop - 1)


def test_woa_retains_best_observed_point():
    p = Problem.continuous(np.sum, 2, -10, 10)
    pop = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = np.array([3.0, 7.0])
    woa = WOA(2)
    state = woa._initialize(pop, values, p, np.random.default_rng(0))
    woa._observe(pop + 1, values + 1, state)
    np.testing.assert_array_equal(state["best"], pop[0])
    woa._observe(pop - 1, values - 1, state)
    np.testing.assert_array_equal(state["best"], pop[0] - 1)


@pytest.mark.parametrize(
    "constructor,kwargs",
    [
        (PSO, {"inertia": np.nan}),
        (PSO, {"cognitive": -1}),
        (PSO, {"social": True}),
        (PSO, {"velocity_fraction": 2}),
        (WOA, {"spiral_shape": np.inf}),
    ],
)
def test_extension_parameters(constructor, kwargs):
    with pytest.raises(ValueError):
        constructor(**kwargs)


def test_pso_equation_and_absorbing_wall():
    class HalfRandom:
        def random(self, shape):
            return np.full(shape, 0.5)

    p = Problem.continuous(np.sum, 1, -10, 10)
    pop = np.array([[1.0], [3.0]])
    pso = PSO(2, inertia=0.5, cognitive=1, social=1)
    state = {
        "personal": np.array([[2.0], [4.0]]),
        "values": np.array([0.0, 1.0]),
        "velocity": np.ones((2, 1)),
        "limit": np.array([10.0]),
    }
    target = pso._propose(pop, None, state, p, HalfRandom(), 0)
    np.testing.assert_allclose(target, [[2.5], [3.5]])
    state["velocity"][:] = 100
    pso._propose(np.full((2, 1), 10.0), None, state, p, HalfRandom(), 0)
    np.testing.assert_array_equal(state["velocity"], 0)


def test_woa_encircling_peer_and_spiral_equations():
    class FixedRandom:
        def __init__(self):
            self.draws = iter(
                [
                    np.array([[0.5], [0.9], [0.5]]),
                    np.full((3, 1), 0.5),
                    np.array([[0.0], [0.0], [1.0]]),
                ]
            )

        def random(self, shape):
            return next(self.draws)

        def integers(self, n, size):
            return np.array([1, 0, 1])

        def uniform(self, low, high, size):
            return np.zeros(size)

    pop = np.array([[2.0], [4.0], [6.0]])
    p = Problem.continuous(np.sum, 1, -10, 10)
    target = WOA(3)._propose(pop, None, {"best": pop[0]}, p, FixedRandom(), 0)
    # A=0 encircles best; A=1.6 explores peer 2; l=0 spirals to 2+|2-6|.
    np.testing.assert_allclose(target, [[2.0], [-1.2], [6.0]])
