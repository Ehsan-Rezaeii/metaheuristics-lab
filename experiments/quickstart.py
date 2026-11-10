"""Run with python -m experiments.quickstart after installation."""

from metaheuristics_lab import GWO, benchmark_problem

if __name__ == "__main__":
    result = GWO(population_size=20).optimize(
        benchmark_problem("sphere", 5), max_evaluations=1000, seed=42
    )
    print(f"Best objective: {result.best_fitness:.6g}; calls: {result.n_evaluations}")
