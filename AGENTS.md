# AGENTS.md — Codex Instructions for `metaheuristics-lab`

## 1. Mission and scope

This repository is **Project 1 of a five-project research-oriented GitHub portfolio** on metaheuristic optimization applied to machine learning and software engineering. Its purpose is to build a **reusable, installable, tested NumPy library** and to produce **reproducible, honestly reported experiments**. Other projects will consume this library as a dependency rather than copy its source code.

The intended audience includes researchers and software engineers reviewing the repository as evidence of engineering and experimental-methodology skills. Favor correctness, clarity, credible results, and maintainability over feature count or impressive-looking claims.

**Approved repository name:** `metaheuristics-lab`.

**Scope authority:** If `PROJECT_BLUEPRINT.md` or `DECISIONS.md` exists in this repository, read it before changing scope. This file restates the approved Project 1 requirements so it is useful even if those project-wide files have not yet been copied here. If instructions conflict materially, identify the conflict rather than silently changing the approved five-project roadmap. The user's current explicit request may revise the workflow or scope.

Do **not** implement the other portfolio projects here:
- `breast-cancer-feature-selection`
- `software-defect-prediction`
- `sqli-detection-lab`
- `search-based-test-generation`

## 2. Working mode: build first, teach afterward

The owner has explicitly requested **full Codex implementation first**, followed by a separate, comprehensive learning phase after the completed work has been reviewed and placed on GitHub.

- Do not stop after one small code change or wait for a teaching session after each milestone.
- Start by inspecting the repository. Write a short implementation plan with milestones and acceptance checks; then execute the approved milestones in order, within the authorized task and environment.
- Finish the **MVP before extensions**, but the target is the **complete approved Project 1 scope**, not MVP-only. Continue through tests, experiments, documentation, and review while feasible.
- When a decision is not material, choose the simplest defensible design and document the trade-off. Ask for a decision only for a genuine scope, permission, scientific-validity, or external-access blocker.
- Keep user-facing progress and the final handoff concise. Source code, docstrings, commits, READMEs, technical documentation, and experiment artifacts must be in professional English. If explaining directly to the owner conversationally, use Persian unless requested otherwise.
- Do not substitute generated explanations for working code or actual experimental evidence. Prepare a technical walkthrough for **later study**, but do not require the owner to understand each implementation detail before making progress.

## 3. Implementation requirements

### Phase A — Foundation and MVP (required first)

1. Create an installable `src/`-layout Python package named `metaheuristics_lab`, with sensible public imports and `pyproject.toml` configuration.
2. Define a consistent, documented optimization API with a problem/objective abstraction and a structured result. Support minimization, explicit bounds or binary domains, reproducible seeds, and an **objective-function evaluation budget**. Return at least the best solution, best objective value, evaluation count, and convergence history.
3. Implement **continuous Genetic Algorithm (GA)** and **Grey Wolf Optimizer (GWO)** from scratch with NumPy. Do not delegate their optimization logic to an existing metaheuristics package.
4. Implement **Random Search** as a meaningful reference baseline using the same problem definitions and evaluation budgets.
5. Implement **Sphere, Rastrigin, Ackley, and Rosenbrock** benchmark functions, with documented dimensions, bounds, and known optima where applicable.
6. Implement **Binary GA** and **Binary GWO** for feature-mask optimization, with a clear convention for bit meaning, fitness, invalid masks, and reproducibility.
7. Add unit tests, smoke/integration tests, and a minimal documented example using the installed package.

### Phase B — Approved extensions (complete after the MVP)

1. Implement **continuous Particle Swarm Optimization (PSO)** and **Whale Optimization Algorithm (WOA)** through the same API.
2. Add appropriate, documented **S-shaped and V-shaped transfer-function support** where meaningful for binary optimization. Explain the bit-update semantics; do not assume one transfer rule is valid for every algorithm.
3. Add benchmark runners for **independent repeated runs**, with configurable seeds, population sizes, evaluation budgets, and output directories.
4. Compare algorithms fairly using **equal objective-evaluation budgets**, and report objective quality, convergence, evaluation count, wall-clock runtime, mean, and standard deviation. Show distributions or individual runs when useful.
5. Produce reproducible result files and interpretable plots; use statistical comparisons only if their assumptions and the experimental question justify them.
6. Add a lightweight feature-selection **demonstration** of the binary algorithms, with an honest baseline and leakage-safe evaluation. This is not the complete research study reserved for Project 2.
7. Add GitHub Actions CI to run practical automated checks. Make local installation and reuse from the other repositories straightforward.

**Scope boundary:** Binary PSO/WOA, many benchmark suites, GPU support, distributed execution, dashboards, web apps, and additional algorithms are **not required** by the approved roadmap. Do not add them merely to make the project look larger. The required binary optimizers are GA and GWO; document exactly which variants are implemented.

## 4. Correctness and API invariants

- Use NumPy's local random generator (for example, `np.random.default_rng(seed)`); avoid hidden global random state. Equal seed + configuration + environment should produce repeatable runs.
- Keep continuous solutions inside their specified bounds. Binary candidates must contain only valid bits. Validate dimensions, finite bounds, population sizes, budgets, and objective outputs.
- Count **every objective evaluation**, including initialization. Avoid silently exceeding the configured evaluation budget. Document what is included in evaluation counting.
- Make comparison budgets truly equivalent; an equal number of iterations alone is not a fair comparison when algorithms perform different numbers of evaluations per iteration.
- Use a convergence history tied to **objective evaluations** (not just iteration numbers); specify whether entries are best-so-far and how their x-axis is defined.
- Verify benchmark equations and expected optimum locations; use numerical tolerances appropriate to floating-point computations.
- Test structural properties (bounds, repeatability, budgets, finite outputs, correct result shapes, corner cases) instead of relying only on brittle tests asserting that a stochastic algorithm always reaches a near-global optimum.
- Optimize first for correct, readable reference implementations, not for micro-optimization.

## 5. Experimental methodology and integrity

- State the question, baselines, configurations, evaluation budget, seeds, and number of independent runs **before** presenting conclusions. Use a cheap smoke/pilot run before scaling up experiments.
- Prefer a feasible number of genuinely independent runs that can be reproduced on a normal laptop. Do not pretend there were 30 runs if fewer were executed. If experiments are too costly, run a clearly labeled reduced protocol and document the unexecuted plan.
- Record actual configurations, package/Python versions, timing methodology, and raw per-run metrics in machine-readable files. Separate measured results from illustrative values.
- Do not assume an optimizer beats Random Search or another algorithm. Negative and mixed findings are acceptable. Avoid unsupported claims of novelty, state-of-the-art performance, or general superiority.
- Feature selection: define the fitness and feature-count penalty explicitly; handle empty masks; compare with at least an all-features baseline and an appropriate simple selection baseline where feasible. Fit preprocessing, feature selection, and model tuning only within the permitted training/validation data. Keep any final held-out test set untouched until final evaluation; avoid any train/test leakage. Do not make clinical claims.
- Use self-contained synthetic or built-in demonstration data where practical. Do not require external dataset downloads for the benchmark suite. Record dataset provenance if a built-in dataset is used.
- Verify and credit external algorithms, equations, dataset sources, and borrowed ideas or code. Never invent references, DOIs, results, test passes, or paper authorship.

## 6. Repository organization and engineering conventions

Prefer a straightforward layout, adapting to existing files rather than replacing them without reason:

```text
metaheuristics-lab/
  AGENTS.md
  README.md
  pyproject.toml
  LICENSE                  # retain the existing license, if present
  .gitignore
  src/metaheuristics_lab/ # public API, optimizers, benchmarks, utilities
  tests/                  # unit and small integration tests
  experiments/            # reproducible benchmark/feature-selection runners
  results/                # measured tables and modest figures
  docs/                   # methodology and implementation walkthrough
  .github/workflows/      # CI
```

- Use small functions/classes, accurate type hints and docstrings, idiomatic naming, clear separation of concerns, and consistent formatting. Avoid unnecessary frameworks or design patterns.
- Put algorithms and benchmark definitions in `src/`, not in notebooks. Experiments should import the **installed package**, not use copy-pasted implementations or `sys.path` hacks.
- Prefer `pyproject.toml` with a small, justified dependency set. Avoid conflicting duplicate dependency declarations. Pin or constrain dependencies appropriately and document a reproducible environment; check actual compatibility before claiming support for a Python version.
- Use portable repository-relative paths and standard `pathlib` operations. The project must run without a GPU. Do not assume access to external networks or to a particular operating system.
- Keep CI fast: tests and small smoke checks in CI; larger benchmark batches run separately and are documented.
- Never commit `.venv`, secrets, API keys, cached data, or unnecessary large artifacts. Preserve legitimate repository files and history; never force-push or rewrite history.
- Make focused commits when permitted. Prepare a pull request or GitHub changes for review when the task/environment authorizes it; do not silently merge into the default branch or claim the repository has been published if it has not.

## 7. Documentation and verification deliverables

Before declaring the work complete, provide:

1. **README.md:** accurate one-paragraph project pitch with the key algorithms and applications up front; project status; installation; quick-start code; documented API; benchmark setup; links to actual results; limitations; references; and instructions for reproducing experiments. Do not display CI badges or performance claims that are not verifiable.
2. **Tests and CI:** executable commands and observed pass/fail results. Do not call unexecuted code tested. State any skipped or environment-blocked checks.
3. **Experiment artifacts:** scripts/configuration, seeds, raw measurements, summary tables, and plots corresponding to actual runs. Label reduced/pilot experiments prominently.
4. **`docs/IMPLEMENTATION_WALKTHROUGH.md`:** architecture map, key files and responsibilities, algorithms and equations at a useful level of detail, API choices, edge cases, and a map from experiments to saved results. This is a reliable reference for the owner's **later comprehensive teaching phase**.
5. **`docs/EXPERIMENT_PROTOCOL.md`:** benchmark/feature-selection question, methods, budgets, baselines, leakage precautions, measurement definitions, run instructions, and limitations.
6. **Final handoff report:** what was implemented, what was *actually* executed, exact commands and outcomes, important design decisions, measured findings (including negative findings), unresolved defects, and any deviations from the approved roadmap.

### Definition of done

The complete Project 1 is done only when its required MVP and approved extensions are implemented or explicitly marked blocked; the package can be installed and imported; meaningful tests and applicable CI checks have been run; benchmark results are genuine and reproducible; documentation matches the code; and limitations/unverified items are clearly disclosed. **A polished README or a successful commit is not, by itself, proof of completion.**
