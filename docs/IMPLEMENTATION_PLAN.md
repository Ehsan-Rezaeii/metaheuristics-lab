# Project 1 implementation plan

1. **Foundation and MVP:** installable NumPy package, validated problem/result API,
   evaluation-budget engine, Random Search, continuous GA/GWO, Binary GA/GWO,
   four benchmarks, minimal example, structural and integration tests.
   Acceptance: installed imports and MVP tests pass, including partial budgets.
2. **Approved extensions:** continuous PSO/WOA and documented S/V binary GWO
   transfers. Acceptance: common invariants and equation-level tests pass.
3. **Research artifacts:** configurable repeated benchmark runner, pilot then
   20 seeds × 4 functions × 5 continuous algorithms at dimension 10 and 3,000
   evaluations; synthetic feature-selection demo with train-only CV, held-out
   evaluation and two baselines. Acceptance: real raw data, metadata, summaries,
   convergence/distribution figures; leakage and artifact tests pass.
4. **Delivery:** README, walkthrough, protocol, CI, wheel installation check,
   final scientific/code review, focused commits, push and PR against main.
   Acceptance: documented commands executed, limitations disclosed, PR unmerged.

Initial inspection: local checkout was the initial commit. Fetched origin/main
and read its newly added AGENTS.md in full before implementation. No blueprint
or decisions file exists. Existing license, ignore rules and history retained.
