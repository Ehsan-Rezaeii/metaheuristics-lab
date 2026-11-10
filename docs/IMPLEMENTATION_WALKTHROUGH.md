# Implementation walkthrough

This is the technical study reference for completed Project 1. The separate
Persian teaching phase can follow this map after review; no other portfolio
project is implemented here. Start with the README examples, then core.py,
optimizers.py, their tests, and finally experiment drivers and saved results.

## Architecture and responsibilities

| File | Responsibility |
|---|---|
| `src/metaheuristics_lab/__init__.py` | Public imports and library version |
| `core.py` | Problem validation, bounds, local sampling, result and evaluation gate |
| `optimizers.py` | Common population loop and seven optimizer classes |
| `transfers.py` | S probabilities, V probabilities, distinct binary updates |
| `benchmarks.py` | Four scalar equations, input checks and standard boxes |
| `experiments/quickstart.py` | Minimal installed-package example |
| `experiments/common.py` | UTF-8 CSV/JSON, output protection, environment provenance |
| `experiments/benchmark.py` | Serial repeated continuous runs, summaries and plots |
| `experiments/audit_results.py` | Verify all saved runs, traces, solutions and summaries |
| `experiments/feature_selection.py` | Synthetic dataset, CV masks, frozen choices, final test |
| `tests/test_core.py` | Shared invariants, defensive copies, error cases |
| `tests/test_algorithms.py` | Transfer, leader, elitism and personal-memory checks |
| `tests/test_benchmarks.py` | Known optima and non-optimum equation checks |
| `tests/test_experiments.py` | Artifact consistency and test-label independence |
| `.github/workflows/ci.yml` | Lint, format, tests, smokes, wheel build/import |
| `requirements-lock.txt` | Exact Python 3.12 environment used for measured runs |

The package depends only on NumPy. The experiment layer depends on Matplotlib
and scikit-learn; importing the public library does not import either. An
installed src-layout package prevents accidentally succeeding via imports from
the repository root. Other projects should depend on an installed version or
reviewed commit rather than copy source files.

## Problem, result and evaluation lifecycle

`Problem` is a frozen dataclass with a callable, positive dimension, broadcast
lower/upper arrays and domain. Bounds are copied and made read-only; changes to
the caller's original bound array cannot change the problem. Finite strictly
positive box widths are required. Binary bounds are exactly 0 and 1. Objectives
receive one-dimensional float arrays, including masks of exact 0/1 values.

Each `optimize` call validates the domain, positive integer budget, population
and seed, creates `default_rng(seed)`, then initializes a uniformly sampled
population. No global random state or prior-run state is read. The evaluator
validates candidate shape/domain/bounds before passing a defensive copy to the
objective. Finite real scalars are accepted; arrays, complex values, booleans,
NaN and infinity are rejected. Application exceptions propagate. Failed runs
raise immediately and return no misleading partial result.

All successful objective calls pass through this evaluator. It retains the
best actually evaluated point, appends one best-so-far value, and refuses calls
after its budget. Initialization counts. Elite retention and best-memory updates
reuse known values and do not call the objective. Duplicate points do count.
A budget smaller than a population simply returns the best initial prefix.
Otherwise the last generation may evaluate only a prefix of proposals. Since
that ends the run, remaining proposals cannot affect the reported incumbent.

`OptimizationResult.history[i]` is the incumbent after evaluation i+1;
`evaluations` supplies the explicit 1…N x-axis. Arrays are independent copies
from the working population. Runtime starts at evaluator creation, includes
sampling, updates and objective work, and excludes public argument validation.
Only deterministic objective values and search traces, not wall time, are
expected to reproduce in the same numerical environment.

## Population loop and schedule

Population updates are synchronous: all proposals use the previous population.
New candidates are evaluated sequentially, optional elitism is applied, then
personal/best memories are updated. GWO has at least three wolves; GA/PSO/WOA
at least two individuals; Random Search at least one. Budgets can still be less
than these configured sizes because no update is attempted after truncated
initialization.

Let P be configured population, B budget, n=min(P,B), and
G=ceil((B−n)/P) planned update generations. For g=0,…,G−1 set
p=g/max(1,G−1), a=2(1−p). Thus multi-update GWO/WOA runs reach a=0 on the final
update, while a single update uses a=2. Including partial updates in this
schedule makes arbitrary budgets explicit. Changing a budget changes the
schedule and therefore need not preserve a shorter run as a trajectory prefix.
Random Search is independent of this schedule.

## Algorithms

All equations below minimize f. Coordinates are clipped to their problem box
before continuous evaluation. No objective gradient or external optimizer is used.

### Random Search

Draw every coordinate independently uniformly on [lower,upper), or each binary
bit uniformly from {0,1}. Keep the global incumbent. Population size batches
sampling; it does not add selection or memory. This supplies an honest baseline
under exactly the same objective-call budget.

### Continuous GA and Binary GA

For each child, select two parents by independent tournaments with replacement:
choose the minimum-fitness entrant from k=3 draws. With probability 0.9 apply
crossover. The real-valued variant uses independent alpha_j~U(0,1):

`child_j = alpha_j*parent1_j + (1-alpha_j)*parent2_j`.

Without crossover use parent1. Independently mutate each coordinate with default
probability 1/d by adding `Normal(0,1)*0.1*(upper_j-lower_j)`, then clip. Mutation
scale is relative to each coordinate's width, avoiding an accidental dependence
on raw units. There is no generation-dependent mutation annealing.

Binary GA chooses each crossed bit from either parent with equal probability,
then independently replaces selected bits b by 1−b with default probability 1/d.
No continuous-to-binary transfer is applied to native genetic operators.

After evaluation replace the worst new individual with the previous elite only
if the elite is strictly better. Its known fitness is reused. This is a simple
one-elite generational GA, not SBX, differential evolution or a tuned genetic
algorithm. Tournament ties and leader ranking are deterministic for a fixed run.

### GWO

Rank the current population by f and choose its three best distinct individuals
(alpha, beta, delta; their coordinates may coincide). For each leader L and wolf X,
independently per coordinate draw r1,r2~U(0,1):

`A = 2*a*r1 − a`, `C = 2*r2`, `D = abs(C*L − X)`, `Y_L = L − A*D`.

The target is `(Y_alpha+Y_beta+Y_delta)/3`, then clipped. Leaders are chosen from
the current generation, not an archive of three historical leaders. The global
reporting incumbent is retained independently even if the pack loses it. This
choice is explicit because published/code variants differ in leader retention.
The a=0 equation reduces to the mean of the three leaders and is tested.

### Binary GWO and transfer functions

Use the same leader-directed target Y computed from binary positions, without
clipping it to [0,1] before transfer. This repository defines two reference
variants, inspired by common transfer families rather than claiming an exact
published BGWO reproduction:

- **S:** z=2Y−1, S(z)=1/(1+exp(−z)); set b'=1 iff u<S(z).
  Target 0.5 is neutral. Stable computation uses (1+tanh(z/2))/2 to avoid exp
  overflow. Even no movement can resample a bit; this is not a flip rule.
- **V:** delta=Y−b, V(delta)=abs(tanh(delta)); flip b iff u<V(delta).
  Zero displacement preserves the bit. Using Y itself here would give different
  and unintended semantics; the displacement is essential.

For both, u is an independent uniform per bit. Bits remain exactly 0/1.
A generic empty mask is valid. The feature application assigns fitness 1 to it;
no hidden bit repair biases the search distribution. Centering S and scaling
these transfers are design choices that can affect selection pressure.

### PSO

Initialize positions uniformly, velocities uniformly in ±vmax, and personal
best positions to initial positions. With independent random vectors r1,r2:

`v' = w*v + c1*r1*(personal_best−x) + c2*r2*(global_personal_best−x)`

`v' = clip(v', −vmax, vmax)`, `x' = clip(x+v', lower, upper)`.

Use w=0.7298, c1=c2=1.49618, vmax=0.2*box_width. On crossing a wall zero the
corresponding velocity (absorbing boundary) instead of retaining an outward
velocity indefinitely. Personal memory updates only on strict improvement;
the swarm's global guide is the best personal memory. These are common
constricted-coefficient values expressed in inertia form, without applying a
second constriction factor. They are defaults, not optimized settings.

### WOA

Retain the best observed whale X*. For each whale draw scalar random coefficients
(broadcast across its coordinates), with A=2ar−a and C=2r' independently. With
probability 0.5 use:

- abs(A)<1: `Y = X* − A*abs(C*X*−X)` (encircling),
- otherwise choose a uniformly random current peer R (self permitted) and use
  `Y = R − A*abs(C*R−X)` (exploration).

Otherwise draw l~U(−1,1) and use the logarithmic spiral
`Y = abs(X*−X)*exp(b*l)*cos(2*pi*l) + X*`, b=1.
Clip, evaluate and update X* on strict improvement. The scalar coefficient and
fixed l range are documented variants; no Levy flights, hybrid local search
or binary whale optimizer is included.

## Benchmark equations

For a vector x of length d:

- Sphere: `sum(x_i^2)`.
- Rastrigin: `10d + sum(x_i^2−10*cos(2*pi*x_i))`.
- Ackley: `−20*exp(−0.2*sqrt(mean(x_i^2))) − exp(mean(cos(2*pi*x_i))) + 20 + e`.
- Rosenbrock: `sum_{i=1}^{d−1}[100*(x_{i+1}−x_i^2)^2+(x_i−1)^2]`.

First three attain zero at the origin; Rosenbrock at the all-ones vector for
d≥2. Rosenbrock d=1 is rejected rather than silently returning an empty sum.
Ackley's optimum can have a tiny floating-point residual. Tests check optima
with tolerance and nonzero points against independently simplified equations.
These are unshifted/unrotated functions and should not be used for broad claims.

## Experiment and artifact reading map

Read [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) before interpreting the
[measured results](RESULTS.md). `results/benchmark_pilot` contains the deliberately
cheap 40-run pilot, while `results/benchmark_main` contains 400 actual runs.
Each raw CSV row identifies function, algorithm and seed. Its best solution is
a JSON list. `histories.npz` keys use `function__algorithm__seed`; each array has
one entry per call. `summary.csv` groups the raw measurements without rerunning
anything. Means and sample standard deviations describe final values and runtime.
Median/IQR convergence uses evaluation count; the log display floor changes only
the plot, not saved data. Final-outcome figures expose individual seeds.

`results/feature_pilot` and `results/feature_demo` contain the small synthetic
exercise. `split_indices.json` identifies the fixed split;
`frozen_selection.json` records all chosen masks and validation metrics before
any held-out score is computed. `runs.csv` later adds test accuracy for every
frozen run. Feature histories use `method__seed`. Model-fit counts distinguish
objective calls from expensive CV fits; an empty mask costs a call but no fit.
Baselines have one reported CV assessment and are not budget-matched searches.

The objective's signature accepts only training arrays. CV creates fresh
preprocessing/model fits in each fold. SelectKBest is itself inside the baseline
pipeline during CV. It is fitted again only on full training data for final
assessment. Fixed classifier settings avoid a hidden tuning loop. No test scores
are read while selecting masks or choosing seeds. An integration test flips
held-out labels and verifies identical masks and validation scores/fitness while held-out accuracy changes accordingly. CV remains subject
to adaptive-search overfitting; a single held-out split is a modest safeguard,
not a nested-CV study.

## Reproduction, review and limitations

Follow README installation and protocol commands with new output directories.
Metadata records actual parameters, UTC time, package/Python versions, platform,
source commit and dirty flag; dirty may include untracked result artifacts.
`requirements-lock.txt` pins the full measured development environment for
Python 3.12. Timing may change with load, BLAS and hardware, and exact floats
can change with dependency versions. Deterministic histories are tested in the
same environment. No expensive success threshold is asserted for stochastic
optimization. Error paths, bounds and budgets are more robust acceptance checks.

Algorithms are reference implementations with O(B*d) history-independent search
work (plus sorting/selection and objective work), O(B) retained convergence
history, and O(P*d) population/memory. Histories can be large for huge budgets;
there is no streaming or distributed engine. Objective cost dominates in feature
selection; benchmark timing includes Python validation overhead. Extreme floating
point ranges can overflow user objectives and will fail finite-output validation.
Readability is favored over vectorized batch objectives and micro-optimization.

Credits and paper references are listed in the README. No external optimizer
source is embedded. Any future optimization or algorithm change should rerun
structural tests and regenerate results into a distinct, labeled batch before
making new empirical claims.
