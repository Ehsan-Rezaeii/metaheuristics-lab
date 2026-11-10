"""NumPy reference optimizers; all objective calls pass through one budget gate."""

from __future__ import annotations

import numpy as np

from .core import Evaluator, OptimizationResult, Problem, Vector, positive_int, real_parameter
from .transfers import binary_update


class PopulationOptimizer:
    """Common synchronous population loop with a possibly partial final generation."""

    domain = "continuous"
    minimum_population = 2

    def __init__(self, population_size: int = 30):
        self.population_size = positive_int(
            population_size, "population_size", self.minimum_population
        )

    def optimize(
        self, problem: Problem, *, max_evaluations: int, seed: int | None = None
    ) -> OptimizationResult:
        """Minimize with exactly max_evaluations calls, including initialization.

        Seeds are nonnegative integers or None. A fresh local generator is made
        per call, so reusing an optimizer does not retain random/search state.
        Budgets smaller than the population evaluate only an initial prefix.
        """
        if not isinstance(problem, Problem) or problem.domain != self.domain:
            raise ValueError(f"{type(self).__name__} requires a {self.domain} Problem")
        budget = positive_int(max_evaluations, "max_evaluations")
        if seed is not None:
            seed = positive_int(seed, "seed", 0)
        rng = np.random.default_rng(seed)
        evaluator = Evaluator(problem, budget)
        pop = problem.sample(rng, min(self.population_size, budget))
        fitness = np.array([evaluator(x) for x in pop])
        state = self._initialize(pop, fitness, problem, rng)
        # Progress covers all planned update generations, including a partial last one.
        generations = max(1, int(np.ceil((budget - len(pop)) / self.population_size)))
        generation = 0
        while evaluator.remaining:
            progress = generation / max(1, generations - 1)
            proposals = self._propose(pop, fitness, state, problem, rng, progress)
            count = min(len(pop), evaluator.remaining)
            old_pop, old_fit = pop.copy(), fitness.copy()
            for i in range(count):
                pop[i] = proposals[i]
                fitness[i] = evaluator(pop[i])
            self._survive(pop, fitness, old_pop, old_fit)
            self._observe(pop, fitness, state)
            generation += 1
        return evaluator.result()

    def _initialize(self, pop, fitness, problem, rng) -> dict:
        return {}

    def _propose(self, pop, fitness, state, problem, rng, progress) -> Vector:
        raise NotImplementedError

    def _survive(self, pop, fitness, old_pop, old_fit) -> None:
        pass

    def _observe(self, pop, fitness, state) -> None:
        pass


class RandomSearch(PopulationOptimizer):
    """Independent uniform samples; population_size only controls batching."""

    minimum_population = 1

    def __init__(self, population_size: int = 30, *, domain: str = "continuous"):
        super().__init__(population_size)
        if domain not in ("continuous", "binary"):
            raise ValueError("domain must be 'continuous' or 'binary'")
        self.domain = domain

    def _propose(self, pop, fitness, state, problem, rng, progress):
        return problem.sample(rng, len(pop))


class GA(PopulationOptimizer):
    """Tournament GA with arithmetic crossover, Gaussian mutation and one elite."""

    def __init__(
        self,
        population_size: int = 30,
        *,
        crossover_rate: float = 0.9,
        mutation_rate: float | None = None,
        mutation_scale: float = 0.1,
        tournament_size: int = 3,
    ):
        super().__init__(population_size)
        self.crossover_rate = real_parameter(crossover_rate, "crossover_rate", 0, 1)
        self.mutation_rate = (
            None if mutation_rate is None else real_parameter(mutation_rate, "mutation_rate", 0, 1)
        )
        self.mutation_scale = real_parameter(mutation_scale, "mutation_scale", 0, 1)
        self.tournament_size = positive_int(tournament_size, "tournament_size")

    def _propose(self, pop, fitness, state, problem, rng, progress):
        n, d = pop.shape
        entrants = rng.integers(n, size=(2, n, self.tournament_size))
        winners = np.take_along_axis(
            entrants, np.argmin(fitness[entrants], axis=2)[..., None], axis=2
        )[..., 0]
        first, second = pop[winners[0]], pop[winners[1]]
        cross = rng.random((n, 1)) < self.crossover_rate
        rate = 1 / d if self.mutation_rate is None else self.mutation_rate
        mutate = rng.random((n, d)) < rate
        if self.domain == "binary":
            children = np.where(cross & (rng.random((n, d)) < 0.5), second, first)
            return np.where(mutate, 1 - children, children)
        alpha = rng.random((n, d))
        children = np.where(cross, alpha * first + (1 - alpha) * second, first)
        children = children + mutate * rng.normal(size=(n, d)) * self.mutation_scale * (
            problem.upper - problem.lower
        )
        return np.clip(children, problem.lower, problem.upper)

    def _survive(self, pop, fitness, old_pop, old_fit):
        elite, worst = np.argmin(old_fit), np.argmax(fitness)
        if old_fit[elite] < fitness[worst]:
            pop[worst], fitness[worst] = old_pop[elite], old_fit[elite]


class BinaryGA(GA):
    """Native bit-string GA: uniform crossover and independent bit-flip mutation.

    Transfer functions are unnecessary for native binary genetic operators.
    mutation_scale is intentionally absent from this public constructor.
    """

    domain = "binary"

    def __init__(
        self,
        population_size: int = 30,
        *,
        crossover_rate: float = 0.9,
        mutation_rate: float | None = None,
        tournament_size: int = 3,
    ):
        super().__init__(
            population_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
        )


def _wolf_target(pop, fitness, rng, progress):
    """Average three leader-directed positions with independent A/C per coordinate."""
    leaders = pop[np.argsort(fitness, kind="stable")[:3]]
    a = 2 * (1 - progress)
    shape = (3, *pop.shape)
    coefficient_a = 2 * a * rng.random(shape) - a
    coefficient_c = 2 * rng.random(shape)
    distance = np.abs(coefficient_c * leaders[:, None, :] - pop[None, :, :])
    return np.mean(leaders[:, None, :] - coefficient_a * distance, axis=0)


class GWO(PopulationOptimizer):
    """Grey Wolf Optimizer using the three best current-generation wolves."""

    minimum_population = 3

    def _propose(self, pop, fitness, state, problem, rng, progress):
        return np.clip(_wolf_target(pop, fitness, rng, progress), problem.lower, problem.upper)


class BinaryGWO(GWO):
    """Binary GWO with S resampling or V displacement-based bit flipping."""

    domain = "binary"

    def __init__(self, population_size: int = 30, *, transfer: str = "s"):
        super().__init__(population_size)
        if transfer not in ("s", "v"):
            raise ValueError("transfer must be 's' or 'v'")
        self.transfer = transfer

    def _propose(self, pop, fitness, state, problem, rng, progress):
        target = _wolf_target(pop, fitness, rng, progress)
        return binary_update(pop, target, rng, self.transfer)


class PSO(PopulationOptimizer):
    """Global-best PSO with personal memory, velocity clamp and absorbing walls."""

    def __init__(
        self,
        population_size: int = 30,
        *,
        inertia: float = 0.7298,
        cognitive: float = 1.49618,
        social: float = 1.49618,
        velocity_fraction: float = 0.2,
    ):
        super().__init__(population_size)
        self.inertia = real_parameter(inertia, "inertia", 0, 1)
        self.cognitive = real_parameter(cognitive, "cognitive", 0, 4)
        self.social = real_parameter(social, "social", 0, 4)
        self.velocity_fraction = real_parameter(velocity_fraction, "velocity_fraction", 0, 1)

    def _initialize(self, pop, fitness, problem, rng):
        limit = self.velocity_fraction * (problem.upper - problem.lower)
        return {
            "personal": pop.copy(),
            "values": fitness.copy(),
            "limit": limit,
            "velocity": rng.uniform(-limit, limit, size=pop.shape),
        }

    def _propose(self, pop, fitness, state, problem, rng, progress):
        personal = state["personal"]
        best = personal[np.argmin(state["values"])]
        velocity = (
            self.inertia * state["velocity"]
            + self.cognitive * rng.random(pop.shape) * (personal - pop)
            + self.social * rng.random(pop.shape) * (best - pop)
        )
        velocity = np.clip(velocity, -state["limit"], state["limit"])
        target = pop + velocity
        outside = (target < problem.lower) | (target > problem.upper)
        state["velocity"] = np.where(outside, 0, velocity)
        return np.clip(target, problem.lower, problem.upper)

    def _observe(self, pop, fitness, state):
        improved = fitness < state["values"]
        state["personal"][improved] = pop[improved]
        state["values"][improved] = fitness[improved]


class WOA(PopulationOptimizer):
    """Whale optimizer: encircling, random-peer exploration and logarithmic spiral."""

    def __init__(self, population_size: int = 30, *, spiral_shape: float = 1.0):
        super().__init__(population_size)
        self.spiral_shape = real_parameter(spiral_shape, "spiral_shape", 0, 5)

    def _initialize(self, pop, fitness, problem, rng):
        index = np.argmin(fitness)
        return {"best": pop[index].copy(), "value": fitness[index]}

    def _propose(self, pop, fitness, state, problem, rng, progress):
        n = len(pop)
        a = 2 * (1 - progress)
        coefficient_a = 2 * a * rng.random((n, 1)) - a
        coefficient_c = 2 * rng.random((n, 1))
        peer = pop[rng.integers(n, size=n)]
        reference = np.where(np.abs(coefficient_a) < 1, state["best"], peer)
        encircle = reference - coefficient_a * np.abs(coefficient_c * reference - pop)
        spiral_coordinate = rng.uniform(-1, 1, size=(n, 1))
        spiral = (
            np.abs(state["best"] - pop)
            * np.exp(self.spiral_shape * spiral_coordinate)
            * np.cos(2 * np.pi * spiral_coordinate)
            + state["best"]
        )
        target = np.where(rng.random((n, 1)) < 0.5, encircle, spiral)
        return np.clip(target, problem.lower, problem.upper)

    def _observe(self, pop, fitness, state):
        index = np.argmin(fitness)
        if fitness[index] < state["value"]:
            state["best"], state["value"] = pop[index].copy(), fitness[index]
