"""Audit saved batches without rerunning optimization or fitting classifiers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metaheuristics_lab import benchmark_problem

from .common import sample_std, write_json


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit(directory: Path) -> dict:
    """Check all saved traces, budget counts, solutions/masks and summary means/SDs."""
    config = json.loads((directory / "metadata.json").read_text())["config"]
    rows = read_csv(directory / "runs.csv")
    continuous = "function" in rows[0]
    budget, seeds = config["budget"], config["seeds"]
    methods = config["algorithms"]
    expected = len(seeds) * len(methods) * (4 if continuous else 1)
    assert len(rows) == expected + (0 if continuous else 2), "unexpected row count"
    frozen = (
        None
        if continuous
        else json.loads((directory / "frozen_selection.json").read_text())["selections"]
    )
    if frozen is not None:
        assert all("test_accuracy" not in row for row in frozen)
        assert [row["mask"] for row in rows] == [row["mask"] for row in frozen]
    with np.load(directory / "histories.npz") as histories:
        assert len(histories.files) == expected
        for row in rows:
            search = continuous or row["method"] in methods
            if continuous:
                problem = benchmark_problem(row["function"], config["dimension"])
                point = np.array(json.loads(row["best_solution"]))
                assert point.shape == (problem.dimension,)
                assert np.all((point >= problem.lower) & (point <= problem.upper))
                np.testing.assert_allclose(
                    problem.objective(point), float(row["best_fitness"]), rtol=1e-12, atol=1e-14
                )
                key = f"{row['function']}__{row['algorithm']}__{row['seed']}"
                final = float(row["best_fitness"])
                runtime = float(row["runtime_seconds"])
            else:
                mask = np.array(json.loads(row["mask"]))
                assert mask.shape == (16,) and np.isin(mask, [0, 1]).all()
                assert int(mask.sum()) == int(row["selected_features"])
                final = float(row["fitness"])
                computed = (
                    0.99 * (1 - float(row["cv_accuracy"])) + 0.01 * mask.mean() if mask.any() else 1
                )
                np.testing.assert_allclose(final, computed, rtol=1e-12)
                assert 0 <= float(row["test_accuracy"]) <= 1
                key = f"{row['method']}__{row['seed']}"
                runtime = float(row["search_seconds"])
            assert np.isfinite(runtime) and runtime >= 0
            if search:
                assert int(row["seed"]) in seeds
                assert int(row["evaluations"]) == budget
                trace = histories[key]
                assert trace.shape == (budget,) and np.isfinite(trace).all()
                assert np.all(np.diff(trace) <= 0)
                assert trace[-1] == final
        for summary in read_csv(directory / "summary.csv"):
            group = [
                row
                for row in rows
                if (
                    row["function"] == summary["function"]
                    and row["algorithm"] == summary["algorithm"]
                    if continuous
                    else row["method"] == summary["method"]
                )
            ]
            assert int(summary["runs"]) == len(group)
            mapping = (
                {
                    "best_fitness": ("mean", "std"),
                    "runtime_seconds": ("mean_runtime_seconds", "std_runtime_seconds"),
                }
                if continuous
                else {
                    metric: (f"mean_{metric}", f"std_{metric}")
                    for metric in [
                        "fitness",
                        "cv_accuracy",
                        "test_accuracy",
                        "selected_features",
                        "search_seconds",
                    ]
                }
            )
            for metric, (mean_key, std_key) in mapping.items():
                values = [float(row[metric]) for row in group]
                np.testing.assert_allclose(float(summary[mean_key]), np.mean(values), rtol=1e-12)
                std = sample_std(values)
                if std is None:
                    assert summary[std_key] == ""
                else:
                    np.testing.assert_allclose(float(summary[std_key]), std, rtol=1e-12)
    return {
        "directory": str(directory),
        "label": config["label"],
        "status": "passed",
        "search_runs": expected,
        "objective_calls": expected * budget,
        "baselines": 0 if continuous else 2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reports = [audit(directory) for directory in args.directories]
    if args.report:
        write_json(args.report, {"audits": reports})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
