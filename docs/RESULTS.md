# Measured results and verification

All numbers below come from the checked-in raw files, not planned or illustrative values.
The predeclared [protocol](EXPERIMENT_PROTOCOL.md) was executed without reducing the main batches.

## Continuous experiments

Executed 20 seeds (0–19) × four functions × five optimizers = **400 runs**,
dimension 10, population 30, **3,000 calls each**, totaling **1,200,000 objective calls**.
Every saved trace and solution was independently audited after the runs.
A separate 40-run pilot used dimension 3, population 6, 31 calls and seeds 0/1.

Final objective, arithmetic mean ± sample standard deviation (lower is better):

| Method | Sphere | Rastrigin | Ackley | Rosenbrock |
|---|---:|---:|---:|---:|
| RandomSearch | 16.86 ± 3.77 | 77.91 ± 6.86 | 17.72 ± 0.759 | 2.143e+04 ± 1.03e+04 |
| GA | 0.002076 ± 0.00132 | 6.138 ± 2.31 | 0.7499 ± 0.392 | 27.15 ± 26.4 |
| GWO | 4.053e-13 ± 4.94e-13 | 2.254 ± 1.99 | 6.871e-06 ± 5.39e-06 | 7.338 ± 0.69 |
| PSO | 4.78e-05 ± 3.96e-05 | 12.6 ± 5.72 | 0.07066 ± 0.0425 | 6.776 ± 2.5 |
| WOA | 1.238e-16 ± 3.53e-16 | 0.6029 ± 2.7 | 3.387e-08 ± 6.34e-08 | 7.479 ± 3.03 |

WOA had the smallest mean on the first three functions in this configuration;
PSO had the smallest Rosenbrock mean. This is a description of these runs, not
a general ranking. WOA Rastrigin had a median of 2.67e-12 but a maximum of
12.0574, so the near-zero median hides an unsuccessful run. The distribution
figure and sample SD make this visible. Rosenbrock remained difficult: GWO and
WOA often stagnated far from its zero optimum; GA had substantial seed variation.
The tiny pilot also includes negative comparisons: GWO Rastrigin mean 33.8762
versus Random Search 24.7540, and WOA Rosenbrock mean 22,303.9 versus Random
Search 1,619.39. Pilot and main use different settings and should not be pooled.

Per-method/function average runtime ranged from 0.0432 to 0.0744 seconds per
3,000-call run on this environment. These include validation/Python overhead;
fixed run order, shared host load and differing function costs limit timing
comparisons. Raw per-run timings and their grouped SDs are retained.

- [Raw runs](../results/benchmark_main/runs.csv), [summary](../results/benchmark_main/summary.csv), [metadata](../results/benchmark_main/metadata.json)
- [Median/IQR convergence](../results/benchmark_main/convergence.png), [individual outcomes](../results/benchmark_main/final_objectives.png)

## Synthetic feature-selection demonstration

Executed four binary methods × three seeds = **12 searches**, population 12,
90 calls each (**1,080 calls**, 3,240 training-CV model fits in this batch).
All methods used the same 300 training/100 held-out examples and fixed folds.
The two simple baselines each used three CV fits and one final training fit.
The pilot contains four five-call searches plus the same two baselines.

| Method | Mean features | CV accuracy | Held-out accuracy, mean ± sample SD | Search/assessment seconds |
|---|---:|---:|---:|---:|
| BinaryGA | 8.67 | 0.8000 | 0.7967 ± 0.0058 | 1.199 |
| BinaryGWO-S | 7.33 | 0.7978 | 0.8167 ± 0.0208 | 1.360 |
| BinaryGWO-V | 7.00 | 0.8033 | 0.8033 ± 0.0058 | 1.593 |
| RandomSearch | 7.67 | 0.7956 | 0.8067 ± 0.0208 | 1.212 |
| AllFeatures | 16.00 | 0.7800 | 0.8000 (one assessment) | 0.015 |
| SelectKBest | 8.00 | 0.7833 | 0.8100 (one assessment) | 0.020 |

Binary GA did not improve held-out accuracy over the all-features baseline on
this split. Binary GWO-V had the lowest mean training-CV penalized fitness
(0.199075), while Binary GWO-S had the highest mean held-out accuracy among
searches (0.8167). Those are different criteria; no method or seed was selected
using held-out scores. Random Search remained competitive (0.8067 accuracy).
Baseline times include CV plus final model fitting; search times include only
optimization/CV and exclude final held-out assessment, so times are not directly
equivalent. The baselines were also much cheaper than a 90-call search.

Seed variation here measures search variability on one fixed split. It is not
a confidence interval for population predictive accuracy. The small 100-example
holdout, adaptive CV overfitting, correlated/redundant synthetic features and
fixed penalty/classifier are important limitations. No clinical claim is made.

- [All raw masks and scores](../results/feature_demo/runs.csv), [summary](../results/feature_demo/summary.csv)
- [Frozen choices before test access](../results/feature_demo/frozen_selection.json), [split](../results/feature_demo/split_indices.json), [metadata](../results/feature_demo/metadata.json)
- [Convergence and held-out outcomes](../results/feature_demo/feature_selection.png)

## Executed commands and observations

From the checkout with the installed Python 3.12.14 environment:

```bash
python -m pip install -e '.[dev,experiments]'
ruff check .
ruff format --check .
python -m pytest --cov=metaheuristics_lab --cov-report=term-missing
python -m experiments.quickstart
python -m build
python -m experiments.audit_results results/benchmark_pilot results/benchmark_main results/feature_pilot results/feature_demo --report results/audit.json
```

Installation succeeded. MVP checks passed (78 tests), then extension checks
(100 tests). Final validation: **114 passed**, no skips, in 8.97 seconds; package statement
coverage **99%** (298/299 statements; the abstract proposal guard is not called).
Ruff lint and formatting checks passed, and git diff --check was clean.
Quickstart measured 4.39321e-12 on five-dimensional Sphere after exactly 1,000
calls with seed 42. Both wheel and source distribution built successfully.
A fresh virtual environment installed the wheel with only NumPy, then imported
every public symbol and ran all seven optimizers from /tmp, away from the
checkout. Neither scikit-learn nor Matplotlib was imported by the core package.
Artifact audit passed on all four batches (456 searches total, plus four
baseline assessments). Its [machine-readable report](../results/audit.json)
checks budget counts, trace monotonicity, final values, stored continuous
solutions, binary masks and summary mean/SD consistency without rerunning search.

Experiment commands actually executed are the four commands in the protocol,
with the checked-in output paths. Benchmark data/plots occupy a modest amount
of disk space; full histories are compressed, not downsampled.

Initial formatting checks found README code blocks needing formatting; fixed.
The initial build reported deprecated license metadata; replaced it with SPDX
MIT metadata and rebuilt. Pilot plotting emitted read-only home cache warnings
but produced valid figures; subsequent commands used writable /tmp caches.
No scientific protocol was skipped or replaced by estimated values.

## Provenance and remaining limits

Experiment metadata identifies source commit 82d2283 and exact parameters and
versions. The dirty flag includes untracked generated results (and later docs);
optimizer/search implementations were committed before execution and did not
change afterward. Later changes add audit/tests/docs and modernize license
metadata. The dependency pins are in requirements-lock.txt for Python 3.12.
GitHub Actions completed successfully on **Python 3.10, 3.12 and 3.13** for
commit c5e03fc. Every matrix job passed lint, formatting, the test suite,
quickstart, both experiment smokes, distribution build and isolated wheel import.
[Verified workflow run](https://github.com/Ehsan-Rezaeii/metaheuristics-lab/actions/runs/42794817758).
The connector initially returned no runs immediately after PR creation; the
subsequent queued run was monitored through successful completion. Direct API
access and the local gh token were unavailable, so the connected GitHub tools
were used to create the PR and verify CI. No workflow permissions or repository
settings were changed. Later documentation-only commits do not change the tested
optimizer or experiment implementations; current checks are visible on PR #1.
The saved-solution audit allows 1e-10 absolute roundoff on recomputed benchmark
values, including near-zero cancellation in Rastrigin across NumPy versions;
raw saved histories and their final values are still compared exactly.

No known unresolved defect was found in the implemented scope. This does not
establish correctness for every objective or numerical scale. Binary GWO uses
explicitly documented variants, bounds use simple clipping, benchmarks are
unshifted/unrotated, and no parameter tuning, statistical superiority tests,
nested CV or external datasets were claimed. Larger studies belong elsewhere.
