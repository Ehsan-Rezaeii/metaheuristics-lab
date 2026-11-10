# metaheuristics-lab

A reusable NumPy library implementing continuous **GA, GWO, PSO and WOA**, plus
**Binary GA and Binary GWO** for feature-mask optimization. All methods minimize
through one validated API with local random generators, exact objective-call
budgets and evaluation-indexed convergence histories. Random Search, four
standard benchmarks and a small leakage-safe synthetic feature-selection demo
support reproducible comparisons. Optimization logic is implemented from scratch.

**Status:** Project 1 MVP and approved extensions implemented. Measured artifacts
are checked in; see [results](docs/RESULTS.md) for findings and verification.
This is a readable reference library, not a state-of-the-art performance claim.
MIT licensed. Other portfolio projects are intentionally outside this repository.

## Installation

From a checkout, using Python 3.10 or newer (local validation used Python 3.12;
CI checks 3.10, 3.12 and 3.13):

```bash
git clone https://github.com/Ehsan-Rezaeii/metaheuristics-lab.git
cd metaheuristics-lab
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev,experiments]'
python -m experiments.quickstart
```

Only NumPy is required by the library. Matplotlib and scikit-learn are optional
experiment dependencies; pytest, coverage, Ruff and build are development tools.
For the exact Python 3.12 experiment environment, install the recorded pins first:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

The lock records one measured environment, not a cross-version dependency lock.
For another repository, install this checkout with `pip install /path/to/metaheuristics-lab`
or pin a reviewed commit using `pip install 'metaheuristics-lab @ git+https://github.com/Ehsan-Rezaeii/metaheuristics-lab.git@<commit>'`.
No PyPI release is claimed. Experiments run from the checkout root and import
the installed package; there are no source-path hacks.

## Quick start

```python
from metaheuristics_lab import GWO, benchmark_problem

problem = benchmark_problem("sphere", dimension=5)
result = GWO(population_size=20).optimize(problem, max_evaluations=1000, seed=42)
print(result.best_solution, result.best_fitness, result.n_evaluations)
# result.history[i] = best objective after evaluation i+1
```

Custom continuous and binary domains:

```python
import numpy as np
from metaheuristics_lab import GA, BinaryGA, BinaryGWO, Problem

box = Problem.continuous(lambda x: float(np.sum((x - 0.3) ** 2)), dimension=4, lower=-2, upper=2)
answer = GA(20).optimize(box, max_evaluations=1000, seed=0)

# A toy mask objective, not a predictive modeling experiment:
mask_problem = Problem.binary(lambda mask: float(-mask.sum()), dimension=8)
mask_answer = BinaryGA(12).optimize(mask_problem, max_evaluations=120, seed=0)
v_answer = BinaryGWO(12, transfer="v").optimize(mask_problem, max_evaluations=120, seed=0)
```

## API and conventions

- `Problem.continuous(objective, dimension, lower, upper)` accepts finite scalar
  or dimension-length bounds with strictly positive widths.
  `Problem.binary(objective, dimension)` uses bits 0/1, represented as floats.
  Bit 1 means selected. The application defines empty-mask fitness.
- `GA`, `GWO`, `PSO`, `WOA`, `BinaryGA`, `BinaryGWO` and `RandomSearch` share
  `optimize(problem, *, max_evaluations, seed=None)`. Random Search supports
  `domain="binary"`; other methods validate the required domain.
- `OptimizationResult` returns `best_solution`, `best_fitness`, `n_evaluations`,
  `history`, `evaluations` and `runtime_seconds`. The incumbent is always an
  actually evaluated solution; reporting never re-evaluates it.
- Every objective call counts, including initialization, duplicates and empty
  masks. Valid runs consume exactly the budget. A small budget can stop during
  initialization; otherwise a final partial generation evaluates a prefix.
  Algorithms do not silently round the budget up to a population multiple.
- Histories contain one best-so-far value per call, with x-axis 1…budget.
  Runtime uses `perf_counter`, including initialization and objective calls,
  excluding public argument validation. It is not reproducible bit-for-bit.
- An objective must return a finite real scalar. Inputs are defensive copies.
  Invalid inputs/outputs raise `ValueError`; application exceptions propagate,
  with no partial result returned. Local seeds do not control randomness inside
  a caller-supplied stochastic objective; reproducibility is the caller's duty.
- Population minimums: 3 for (Binary) GWO, 2 for GA/Binary GA/PSO/WOA, 1 for
  Random Search. Seeds are nonnegative integers or `None` (fresh entropy).
  No optimizer retains search state between calls.

Default population size is 30. GA uses tournament size 3, crossover probability
0.9, per-bit/coordinate mutation probability 1/d, Gaussian mutation SD 0.1 of
box width for the continuous variant, and one-elite replacement. PSO uses inertia
0.7298, cognitive/social coefficients 1.49618 and velocity limits 0.2 of box
width. WOA spiral shape is 1. GWO/WOA exploration coefficient decreases linearly
from 2 to 0 across planned update generations (a single update uses 2).

Binary GA has native uniform crossover/bit-flip mutation; it needs no transfer
function. Binary GWO is an explicitly defined reference variant:
`transfer="s"` resamples bits using logistic S(2*wolf_target−1), while `"v"`
flips current bits with probability |tanh(wolf_target−current)|. These have
*different semantics*. Binary PSO/WOA are not implemented.

| Benchmark | Box in each dimension | Minimum and location |
|---|---|---|
| Sphere | [-5.12, 5.12] | 0 at x=0 |
| Rastrigin | [-5.12, 5.12] | 0 at x=0 |
| Ackley | [-32.768, 32.768] | 0 at x=0 (floating-point tolerance) |
| Rosenbrock | [-5, 10] | 0 at x=1, dimension ≥2 |

## Experiments and measured artifacts

The [protocol](docs/EXPERIMENT_PROTOCOL.md) specifies questions, seeds, budgets,
parameters, timing, summaries and leakage precautions. It was written before
batches were run. Commands below create fresh directories; use another output
path for reruns because checked-in results are protected from overwriting.

```bash
python -m experiments.benchmark --dimension 3 --population 6 --budget 31 --seeds 0 1 --label pilot --output results/my_benchmark_pilot
python -m experiments.benchmark --output results/my_benchmark_main
python -m experiments.feature_selection --population 4 --budget 5 --seeds 0 --label pilot --output results/my_feature_pilot
python -m experiments.feature_selection --output results/my_feature_demo
```

The main batch runs 20 seeds × 4 functions × 5 algorithms, dimension 10,
population 30 and exactly 3,000 calls per run. Each output has raw `runs.csv`,
`summary.csv`, compressed full histories, metadata and figures. The feature demo
uses 400 synthetic examples, 16 features, one untouched 25% test split and
training-only three-fold CV. Four binary searches use 90 calls and three seeds;
all-features and fixed SelectKBest(k=8) baselines use the same model. All final
masks are saved before test evaluation; no winning seed is chosen by test score.

- [Main summaries](results/benchmark_main/summary.csv) and [individual runs](results/benchmark_main/runs.csv)
- [Convergence](results/benchmark_main/convergence.png) and [final outcomes](results/benchmark_main/final_objectives.png)
- [Feature demo](results/feature_demo/runs.csv) and [figure](results/feature_demo/feature_selection.png)
- [Pilot benchmarks](results/benchmark_pilot/summary.csv) and [pilot feature selection](results/feature_pilot/runs.csv)

On restricted/read-only home directories, set writable caches before plotting:
`export MPLCONFIGDIR=/tmp/metaheuristics-mpl XDG_CACHE_HOME=/tmp/metaheuristics-cache`.

## Verification

```bash
ruff check .
ruff format --check .
python -m pytest --cov=metaheuristics_lab --cov-report=term-missing
python -m experiments.quickstart
python -m build
```

Tests cover reproducibility, exact/partial budgets, bounds, binary validity,
equations, memory/elitism, errors, raw-summary consistency and held-out-label
independence of selected masks. GitHub Actions runs practical checks and tiny
experiment smokes, builds a wheel and imports it away from the checkout.
See [verification notes](docs/RESULTS.md) for observed commands and limitations.

## Limitations and further reading

Reference algorithms use simple clipping and fixed default parameters, with no
noise handling, constraints beyond boxes/bits, parallel search, or automatic
hyperparameter tuning. GWO chooses current-generation leaders, whereas WOA and
PSO use retained best positions; the reporting incumbent is retained for all.
Synchronous GWO/WOA schedules and Binary GWO transfer rules are documented
implementation choices, not exact reproductions of every paper's pseudocode.
Unshifted benchmarks favor origin-seeking methods and four functions cannot
justify general rankings. Twenty seeds quantify variability on these settings,
not statistical superiority. Three feature-search seeds on one synthetic split
are a demonstration, not an independent generalization study. Adaptive CV search
can overfit CV; the final held-out scores are reported without model selection.

Study the [implementation walkthrough](docs/IMPLEMENTATION_WALKTHROUGH.md) for
architecture, equations, edge cases and a file-by-file guide. Algorithm credit:

- Holland (1975), *Adaptation in Natural and Artificial Systems* (GA foundations).
- Mirjalili, Mirjalili & Lewis (2014), *Grey Wolf Optimizer*, Advances in Engineering Software 69, 46–61. [DOI](https://doi.org/10.1016/j.advengsoft.2013.12.007).
- Kennedy & Eberhart (1995), *Particle Swarm Optimization*, ICNN. [DOI](https://doi.org/10.1109/ICNN.1995.488968).
- Mirjalili & Lewis (2016), *The Whale Optimization Algorithm*, Advances in Engineering Software 95, 51–67. [DOI](https://doi.org/10.1016/j.advengsoft.2016.01.008).
- Mirjalili & Lewis (2013), *S-shaped versus V-shaped transfer functions for binary Particle Swarm Optimization*, Swarm and Evolutionary Computation 9, 1–14. [DOI](https://doi.org/10.1016/j.swevo.2012.09.002). Transfer-family motivation only; our Binary GWO is defined explicitly here.
- [scikit-learn make_classification documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) for synthetic-data provenance.

No external optimizer implementation was copied or used as an optimization backend.
