import csv
import json

import numpy as np
import pytest

from experiments import benchmark, feature_selection
from experiments.common import sample_std


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_benchmark_artifacts_and_summary(tmp_path):
    output = tmp_path / "benchmark"
    benchmark.run(output=output, dimension=2, population=3, budget=7, seeds=[0, 1], label="test")
    rows = read_csv(output / "runs.csv")
    assert len(rows) == 40
    histories = np.load(output / "histories.npz")
    assert len(histories.files) == 40
    for row in rows:
        key = f"{row['function']}__{row['algorithm']}__{row['seed']}"
        trace = histories[key]
        assert trace.shape == (7,)
        assert trace[-1] == float(row["best_fitness"])
        assert int(row["evaluations"]) == 7
    for summary in read_csv(output / "summary.csv"):
        values = [
            float(row["best_fitness"])
            for row in rows
            if row["function"] == summary["function"] and row["algorithm"] == summary["algorithm"]
        ]
        assert float(summary["mean"]) == pytest.approx(np.mean(values))
        assert float(summary["std"]) == pytest.approx(np.std(values, ddof=1))
    assert json.loads((output / "metadata.json").read_text())["config"]["budget"] == 7
    for image in ["convergence.png", "final_objectives.png"]:
        assert (output / image).stat().st_size > 1000
    with pytest.raises(ValueError, match="not empty"):
        benchmark.run(output=output, dimension=2, population=3, budget=7, seeds=[0])


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1], [True]])
def test_invalid_experiment_seeds(tmp_path, seeds):
    for runner in [benchmark.run, feature_selection.run]:
        with pytest.raises(ValueError):
            runner(output=tmp_path / "invalid", seeds=seeds)


def test_sample_std_one_run_is_undefined():
    assert sample_std([1]) is None
    assert sample_std([1, 3]) == pytest.approx(np.sqrt(2))


def test_empty_mask_costs_one_call_no_model_fit():
    x, y, train, _ = feature_selection.make_data()
    objective = feature_selection.MaskFitness(x[train], y[train])
    assert objective(np.zeros(16)) == 1
    assert objective.calls == 1
    assert objective.model_fits == 0
    mask = np.ones(16)
    value = objective(mask)
    assert value == pytest.approx(0.99 * (1 - objective.records[tuple(mask)]) + 0.01)
    assert objective.model_fits == 3
    assert objective(mask) == value
    assert objective.calls == 3
    assert objective.model_fits == 6  # Duplicate points still consume actual fits.


def test_feature_demo_and_heldout_label_independence(tmp_path, monkeypatch):
    original = feature_selection.make_data
    first, second = tmp_path / "first", tmp_path / "second"
    feature_selection.run(output=first, population=4, budget=5, seeds=[0], label="test")

    def changed_heldout_labels():
        x, y, train, test = original()
        assert not set(train) & set(test)
        y[test] = 1 - y[test]
        return x, y, train, test

    monkeypatch.setattr(feature_selection, "make_data", changed_heldout_labels)
    feature_selection.run(output=second, population=4, budget=5, seeds=[0], label="test")
    a, b = read_csv(first / "runs.csv"), read_csv(second / "runs.csv")
    assert len(a) == len(b) == 6
    for left, right in zip(a, b, strict=True):
        for field in ["mask", "cv_accuracy", "fitness", "selected_features"]:
            assert left[field] == right[field]
        assert float(left["test_accuracy"]) + float(right["test_accuracy"]) == pytest.approx(1)
    frozen = json.loads((first / "frozen_selection.json").read_text())["selections"]
    assert all("test_accuracy" not in row for row in frozen)
    assert (first / "feature_selection.png").stat().st_size > 1000
    traces = np.load(first / "histories.npz")
    assert len(traces.files) == 4
    assert all(traces[key].shape == (5,) for key in traces.files)
    split = json.loads((first / "split_indices.json").read_text())
    assert len(split["train"]) == 300
    assert len(split["test"]) == 100
    assert not set(split["train"]) & set(split["test"])
