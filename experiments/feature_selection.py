"""Leakage-safe synthetic feature-mask demonstration, deliberately small in scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metaheuristics_lab import BinaryGA, BinaryGWO, Problem, RandomSearch
from metaheuristics_lab.core import positive_int

from .common import metadata, output_directory, sample_std, write_csv, write_json


def classifier() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1, solver="liblinear", max_iter=1000, random_state=0)),
        ]
    )


class MaskFitness:
    """CV objective that owns training arrays only; no held-out data is accepted."""

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2027)
        self.calls = 0
        self.model_fits = 0
        self.records: dict[tuple, float] = {}

    def __call__(self, mask: np.ndarray) -> float:
        self.calls += 1
        selected = mask.astype(bool)
        if not selected.any():
            self.records[tuple(mask)] = 0.0
            return 1.0
        scores = cross_val_score(
            classifier(),
            self.x_train[:, selected],
            self.y_train,
            cv=self.cv,
            scoring="accuracy",
            error_score="raise",
            n_jobs=1,
        )
        self.model_fits += 3
        accuracy = float(scores.mean())
        # Record only for reporting, never skip a repeated objective evaluation.
        self.records[tuple(mask)] = accuracy
        return 0.99 * (1 - accuracy) + 0.01 * selected.mean()


def make_data():
    """Fixed synthetic generator and raw stratified split; no preprocessing here."""
    x, y = make_classification(
        n_samples=400,
        n_features=16,
        n_informative=6,
        n_redundant=4,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.01,
        random_state=2026,
    )
    train, test = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=2026)
    return x, y, train, test


def run(
    *,
    output: Path,
    population: int = 12,
    budget: int = 90,
    seeds: list[int] | None = None,
    label: str = "demo",
) -> None:
    seeds = [0, 1, 2] if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    for seed in seeds:
        positive_int(seed, "seed", 0)
    positive_int(budget, "budget")
    methods = {
        "BinaryGA": BinaryGA(population),
        "BinaryGWO-S": BinaryGWO(population, transfer="s"),
        "BinaryGWO-V": BinaryGWO(population, transfer="v"),
        "RandomSearch": RandomSearch(population, domain="binary"),
    }
    output = output_directory(output)
    x, y, train, test = make_data()
    x_train, y_train = x[train], y[train]
    config = {
        "label": label,
        "population": population,
        "budget": budget,
        "seeds": seeds,
        "algorithms": {name: vars(method) for name, method in methods.items()},
        "dataset": "sklearn.datasets.make_classification, fixed synthetic binary data",
        "generator": {
            "n_samples": 400,
            "n_features": 16,
            "n_informative": 6,
            "n_redundant": 4,
            "n_classes": 2,
            "class_sep": 1.0,
            "flip_y": 0.01,
            "random_state": 2026,
        },
        "dataset_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
        "split": {"test_size": 0.25, "stratified": True, "random_state": 2026},
        "cv": {"n_splits": 3, "shuffle": True, "random_state": 2027},
        "model": {
            "scaler": "StandardScaler",
            "classifier": "LogisticRegression",
            "C": 1,
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": 0,
        },
        "fitness": "0.99*(1-mean_CV_accuracy)+0.01*selected_fraction; empty=1",
        "simple_baseline": "SelectKBest(f_classif,k=8), fitted inside each fold",
    }
    write_json(output / "metadata.json", metadata(config))
    write_json(output / "split_indices.json", {"train": train.tolist(), "test": test.tolist()})
    rows, frozen, histories = [], [], {}
    # Stage 1: all selection and validation use training arrays exclusively.
    for name, algorithm in methods.items():
        for seed in seeds:
            objective = MaskFitness(x_train, y_train)
            result = algorithm.optimize(
                Problem.binary(objective, x.shape[1]), max_evaluations=budget, seed=seed
            )
            mask = result.best_solution.astype(bool)
            row = {
                "method": name,
                "seed": seed,
                "fitness": result.best_fitness,
                "cv_accuracy": objective.records[tuple(result.best_solution)],
                "selected_features": int(mask.sum()),
                "evaluations": result.n_evaluations,
                "cv_model_fits": objective.model_fits,
                "search_seconds": result.runtime_seconds,
                "mask": json.dumps(mask.astype(int).tolist()),
            }
            assert objective.calls == budget
            rows.append(row)
            frozen.append((row, mask, None))
            histories[f"{name}__{seed}"] = result.history
            print(f"{name} seed={seed} fitness={result.best_fitness:.5f}", flush=True)
    # Baseline selection is also inside CV, then fitted on all training examples.
    for name, selector in [("AllFeatures", None), ("SelectKBest", SelectKBest(f_classif, k=8))]:
        started = perf_counter()
        model = classifier()
        if selector is not None:
            model = Pipeline([("select", selector), *model.steps])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2027)
        accuracy = float(
            cross_val_score(
                model, x_train, y_train, cv=cv, scoring="accuracy", error_score="raise"
            ).mean()
        )
        model.fit(x_train, y_train)
        mask = (
            np.ones(x.shape[1], dtype=bool)
            if selector is None
            else model.named_steps["select"].get_support()
        )
        row = {
            "method": name,
            "seed": "",
            "fitness": 0.99 * (1 - accuracy) + 0.01 * mask.mean(),
            "cv_accuracy": accuracy,
            "selected_features": int(mask.sum()),
            "evaluations": 1,
            "cv_model_fits": 3,
            "search_seconds": perf_counter() - started,
            "mask": json.dumps(mask.astype(int).tolist()),
        }
        rows.append(row)
        frozen.append((row, mask, model))
    # Persist choices BEFORE touching held-out data; no seed chosen by test scores.
    write_json(output / "frozen_selection.json", {"selections": rows})
    # Stage 2: evaluate every frozen choice once on the held-out test set.
    for row, mask, fitted_model in frozen:
        if not mask.any():
            # Empty masks cannot train a classifier: training-majority fallback.
            majority = np.bincount(y_train).argmax()
            row["test_accuracy"] = float(np.mean(y[test] == majority))
        elif fitted_model is not None:
            row["test_accuracy"] = float(fitted_model.score(x[test], y[test]))
        else:
            model = classifier().fit(x_train[:, mask], y_train)
            row["test_accuracy"] = float(model.score(x[test][:, mask], y[test]))
    write_csv(output / "runs.csv", rows)
    summaries = []
    for name in [*methods, "AllFeatures", "SelectKBest"]:
        group = [row for row in rows if row["method"] == name]
        summary = {"method": name, "runs": len(group)}
        for metric in [
            "fitness",
            "cv_accuracy",
            "test_accuracy",
            "selected_features",
            "search_seconds",
        ]:
            values = [row[metric] for row in group]
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"std_{metric}"] = sample_std(values)
        summaries.append(summary)
    write_csv(output / "summary.csv", summaries)
    np.savez_compressed(output / "histories.npz", **histories)
    plot_results(output, rows, histories, methods, seeds, label)


def plot_results(output, rows, histories, methods, seeds, label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name in methods:
        traces = np.array([histories[f"{name}__{seed}"] for seed in seeds])
        axes[0].plot(np.arange(1, traces.shape[1] + 1), np.median(traces, axis=0), label=name)
    axes[0].set(xlabel="Objective evaluations", ylabel="Training-CV penalized fitness (median)")
    axes[0].legend(fontsize=8)
    names = list(dict.fromkeys(row["method"] for row in rows))
    for i, name in enumerate(names):
        group = [row for row in rows if row["method"] == name]
        axes[1].scatter([i] * len(group), [row["test_accuracy"] for row in group], s=35)
    axes[1].set_xticks(range(len(names)), names, rotation=35, ha="right")
    axes[1].set(ylabel="Held-out accuracy (every frozen run)", ylim=(0, 1))
    fig.suptitle(f"{label}: synthetic feature-selection demonstration")
    fig.savefig(output / "feature_selection.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/feature_demo"))
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--label", default="demo")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
