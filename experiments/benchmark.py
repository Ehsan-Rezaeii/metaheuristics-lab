"""Equal-evaluation repeated benchmarks; run as python -m experiments.benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metaheuristics_lab import GA, GWO, PSO, WOA, RandomSearch, benchmark_problem
from metaheuristics_lab.benchmarks import BENCHMARKS
from metaheuristics_lab.core import positive_int

from .common import metadata, output_directory, sample_std, write_csv, write_json

ALGORITHMS = {"RandomSearch": RandomSearch, "GA": GA, "GWO": GWO, "PSO": PSO, "WOA": WOA}


def run(
    *,
    output: Path,
    dimension: int = 10,
    population: int = 30,
    budget: int = 3000,
    seeds: list[int] | None = None,
    label: str = "main",
) -> None:
    seeds = list(range(20)) if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct nonnegative integers")
    for seed in seeds:
        positive_int(seed, "seed", 0)
    positive_int(budget, "budget")
    for name in BENCHMARKS:
        benchmark_problem(name, dimension)
    methods = {name: factory(population) for name, factory in ALGORITHMS.items()}
    output = output_directory(output)
    config = {
        "label": label,
        "dimension": dimension,
        "population": population,
        "budget": budget,
        "seeds": seeds,
        "algorithms": {name: vars(method) for name, method in methods.items()},
        "functions": {name: {"lower": v[1], "upper": v[2]} for name, v in BENCHMARKS.items()},
    }
    write_json(output / "metadata.json", metadata(config))
    rows, summaries, histories = [], [], {}
    for function in BENCHMARKS:
        problem = benchmark_problem(function, dimension)
        for name, algorithm in methods.items():
            group = []
            for seed in seeds:
                result = algorithm.optimize(problem, max_evaluations=budget, seed=seed)
                row = {
                    "function": function,
                    "algorithm": name,
                    "seed": seed,
                    "best_fitness": result.best_fitness,
                    "evaluations": result.n_evaluations,
                    "runtime_seconds": result.runtime_seconds,
                    "best_solution": json.dumps(result.best_solution.tolist()),
                }
                rows.append(row)
                group.append(row)
                histories[f"{function}__{name}__{seed}"] = result.history
            values = [row["best_fitness"] for row in group]
            times = [row["runtime_seconds"] for row in group]
            summaries.append(
                {
                    "function": function,
                    "algorithm": name,
                    "runs": len(seeds),
                    "mean": float(np.mean(values)),
                    "std": sample_std(values),
                    "median": float(np.median(values)),
                    "min": min(values),
                    "max": max(values),
                    "mean_runtime_seconds": float(np.mean(times)),
                    "std_runtime_seconds": sample_std(times),
                    "evaluations_per_run": budget,
                }
            )
            print(
                f"{function:10s} {name:12s} mean={np.mean(values):.6g} runs={len(seeds)}",
                flush=True,
            )
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summaries)
    np.savez_compressed(output / "histories.npz", **histories)
    plot_results(output, histories, seeds, budget, label)


def plot_results(output, histories, seeds, budget, label):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    dist, dist_axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for function, ax, dax in zip(BENCHMARKS, axes.flat, dist_axes.flat, strict=True):
        for index, name in enumerate(ALGORITHMS):
            traces = np.array([histories[f"{function}__{name}__{seed}"] for seed in seeds])
            low, median, high = np.quantile(traces, [0.25, 0.5, 0.75], axis=0)
            x = np.arange(1, budget + 1)
            (line,) = ax.plot(x, np.maximum(median, 1e-16), label=name)
            ax.fill_between(
                x,
                np.maximum(low, 1e-16),
                np.maximum(high, 1e-16),
                color=line.get_color(),
                alpha=0.12,
            )
            # Deterministic horizontal spread; it does not change objective values.
            spread = np.linspace(-0.15, 0.15, len(seeds))
            dax.scatter(index + spread, np.maximum(traces[:, -1], 1e-16), s=15, alpha=0.7)
        ax.set(
            title=function,
            xlabel="Objective evaluations",
            ylabel="Best-so-far (median, IQR)",
            yscale="log",
        )
        dax.set(title=function, ylabel="Final best objective (each seed)", yscale="log")
        dax.set_xticks(range(len(ALGORITHMS)), ALGORITHMS, rotation=15)
        ax.grid(alpha=0.2)
        dax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{label}: convergence; log display floor 1e-16")
    dist.suptitle(f"{label}: final outcomes; log display floor 1e-16")
    fig.savefig(output / "convergence.png", dpi=140)
    dist.savefig(output / "final_objectives.png", dpi=140)
    plt.close(fig)
    plt.close(dist)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_main"))
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--label", default="main")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
