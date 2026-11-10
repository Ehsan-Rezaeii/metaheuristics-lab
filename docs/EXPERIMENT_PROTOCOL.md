# Experiment protocol

This protocol is specified before the measured batches. It is a lightweight
reference-library validation, not a competition or a new algorithm study.

## Continuous question and predeclared configuration

How do five reference optimizers differ in objective quality, variation,
convergence and CPU wall time under equal objective-evaluation budgets?

- Functions: unshifted Sphere, Rastrigin, Ackley, Rosenbrock; dimension 10.
- Boxes: respectively [-5.12,5.12], [-5.12,5.12], [-32.768,32.768], [-5,10].
- Methods: independent uniform Random Search, GA, GWO, PSO and WOA.
- Default public hyperparameters, population 30, exactly 3,000 calls per run.
- Seeds 0 through 19 inclusive: 20 independent runs per method/function,
  400 runs total. Same seed labels pair methods, but differing random draws
  do not imply identical trajectories. Each run has a fresh local generator.
- Pilot first: dimension 3, population 6, 31 calls, seeds 0 and 1,
  all functions/methods. Pilot results are saved separately.
- No tuning on these results, early stopping, or objective caching.
- Fixed benchmark/method/seed execution order; timings are descriptive and
  include Python loop and validation overhead, not just mathematical updates.

Count initialization and every subsequent objective call, including repeated
points and zero-movement updates. A partial final generation stops exactly at
the budget. History entry i is the best objective after call i+1, with no extra
objective calls for reporting. All methods have the same boxes and budget;
comparison by iteration count would be inappropriate.

Save every final objective, runtime, evaluations and solution in runs.csv; full
per-evaluation histories in compressed histories.npz; configuration, algorithm
parameters, software versions, platform and source revision in metadata.json.
Summary statistics: arithmetic mean, sample standard deviation (ddof=1),
median, minimum, maximum and mean/sample-SD runtime. For a single run sample SD
is undefined and exported as an empty CSV cell. Convergence uses median and
25th–75th percentile best-so-far values across seeds, with a 1e-16 plotting
floor for the log axis; original raw values are untouched. Final distributions
show individual values. No significance tests or superiority claims are planned:
four standard, origin-favoring unshifted functions do not represent all problems.

## Feature-selection demonstration

Question: can small budgeted binary searches trade predictive validation
accuracy against mask size, compared with all features and fixed univariate
selection? This is a synthetic demonstration, not Project 2 or a clinical study.

- scikit-learn make_classification, random_state=2026: 400 examples, 16
  features, 6 informative, 4 redundant, 2 classes, class_sep=1, flip_y=0.01.
  Other generator parameters use the recorded scikit-learn defaults.
- One stratified 75/25 train/test split, random_state=2026. No preprocessing
  precedes splitting. Save indices and the generated-array hash for provenance.
- Fitness computed solely on training data using fixed 3-fold StratifiedKFold,
  shuffle=True, random_state=2027. Every fold fits its own StandardScaler and
  LogisticRegression(C=1, solver='liblinear', max_iter=1000, random_state=0).
- For a mask m, minimize 0.99*(1-mean_CV_accuracy) + 0.01*sum(m)/16.
  Bit 1 selects a feature. An empty mask has fitness 1 and needs no model fit.
  Every mask request still counts as an objective evaluation; no caching.
- Binary GA, Binary GWO-S, Binary GWO-V and binary Random Search, population
  12, exactly 90 calls each, seeds 0,1,2. Same folds throughout for comparison.
- Baselines: all features, and SelectKBest(f_classif,k=8), a fixed half-size
  subset. The latter is fitted inside each CV fold and only on full training
  data for its final model; it never uses test labels for feature selection.
- Complete every optimizer run and freeze all selected masks before any final
  test evaluation. Fit each final model on training data and score held-out
  data once per method/seed. Do not select a winning seed by test accuracy.
- Pilot: same synthetic data/split, population 4, five calls, seed 0.
- Report all seeds separately, CV fitness/accuracy, feature count, test
  accuracy, calls, CV fit counts and search runtime. Baselines are not budget-
  matched searches. Repeated seeds share one split, so they measure search
  variability, not independent estimates of generalization uncertainty.

## Commands

From an installed checkout (see README), first run the pilot commands, then
full commands. Output directories must be new/empty to prevent mixing batches.

```bash
python -m experiments.benchmark --dimension 3 --population 6 --budget 31 --seeds 0 1 --label pilot --output results/benchmark_pilot
python -m experiments.benchmark --output results/benchmark_main
python -m experiments.feature_selection --population 4 --budget 5 --seeds 0 --label pilot --output results/feature_pilot
python -m experiments.feature_selection --output results/feature_demo
```

Metadata records actual settings even if CLI overrides change this protocol.
Runs are serial, CPU-only; dataset generation and plotting are outside search
timings. Environment pins are saved with the artifacts. Numerical reproducibility
is expected in the same environment; wall-clock times naturally vary. Consult
RESULTS.md for measured findings and deviations, after execution.
