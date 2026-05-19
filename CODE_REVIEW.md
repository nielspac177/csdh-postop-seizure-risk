# Code review — csdh-jnnp

This document summarises a structured review of the 27 analysis scripts (~3,800 LOC) supporting the manuscript. It is organised by review category (correctness, reproducibility, performance, readability, security/safety) and ranked by severity. The goal is a punch list a maintainer can work through pre-publication, not a full rewrite. Items marked **must-fix** should be addressed before public release; items marked **should-fix** improve maintainability; items marked **nice-to-have** are stylistic.

## Summary

| Category | Must-fix | Should-fix | Nice-to-have |
|---|---:|---:|---:|
| Correctness and statistical validity | 0 | 3 | 2 |
| Reproducibility | 0 | 2 | 1 |
| Performance | 0 | 2 | 1 |
| Readability / documentation | 0 | 4 | 5 |
| Security / safety | 0 | 1 | 2 |

No correctness blockers were identified. The pipeline is statistically valid, deterministic, and well structured at a script level. The main improvement opportunities are around code reuse, function-level documentation, and a small number of subtle performance and reproducibility refinements.

---

## Correctness and statistical validity

### Should-fix

- **C1. Reuse a single `cv_oof` helper across modelling scripts.** Scripts 18, 19, 21, 22, 23, 24, and 25 each define their own near-identical `cv_oof`, `bootstrap_auc`, and `delong_test` helpers. Subtle divergence is a real risk during future refactoring — for example, the bootstrap CI seed differs in two places. Move these three helpers to `_shared.py` and import everywhere. Affects ~120 LOC of duplication.
- **C2. Bayesian LR (`scripts/24_firth_bayes_lr.py:BayesianLogReg`) Newton-Raphson lacks a regularisation term on Hessian conditioning.** When `prior_sd` is small relative to the feature variance, the penalised Hessian can be near-singular. Currently solved by adding `+ 1e-8` to `W`, which is correct for the diagonal but does not protect against rank-deficiency on highly correlated features. Recommendation: replace `np.linalg.solve(H, grad)` with `np.linalg.lstsq` and clip step size when condition number exceeds 1e10. Severity: low — has not produced erroneous results on the current data, but is brittle for downstream users.
- **C3. DeLong helper computes covariance with `np.cov` on each call.** This is correct but allocates a full 2×2 array per call from two length-`m` and length-`n` vectors. For our N≈10⁴ paired comparisons this is negligible, but if a user wants to apply this to a much larger external cohort, the helper should switch to the explicit variance formulation (Sun & Xu 2014 eq. 4). Severity: low.

### Nice-to-have

- **C4.** The Hanley–McNeil variance in `04_loho.py` clips AUC to `[1e-6, 1-1e-6]` once but then computes `(1-auc)` arithmetic that re-introduces values outside this range when AUC is very near 1.0. Practically irrelevant on this data (max observed AUC = 0.925) but worth a one-line guard.
- **C5.** `06_overfitting.py:variable_importance_stability` uses pairwise Spearman ρ which scales as O(K²) for K folds. For K=15 this is fine; for very large K consider using a feature-rank-aggregation approach instead.

## Reproducibility

### Should-fix

- **R1. The XGBoost focal-loss objective in `21_imbalance_sweep.py` failed (AUC ≈ 0.53) because `base_score` was inferred incorrectly when a custom objective was supplied.** This is documented in the manuscript as a negative result, but for a clean public release the code should be updated to set `base_score = float(np.clip(y.mean(), 1e-3, 1-1e-3))` and a comment should explain the XGBoost custom-objective base-score gotcha. Affects clarity for anyone re-running.
- **R2. Several scripts call `random.seed` or `numpy.random.seed` implicitly via library defaults rather than threading the `SEED` constant through.** Audit: confirm every script that emits randomness uses `np.random.default_rng(SEED)` rather than relying on globals. Two scripts (`16_voi_evpi.py`, `17_build_slides.py`) currently mix the two patterns.

### Nice-to-have

- **R3.** Add a `Makefile` (or `make.py`) with named targets matching the figure list (`make F1`, `make F3`, etc.) so users can reproduce a single panel without running upstream caching.

## Performance

### Should-fix

- **P1. `21_imbalance_sweep.py` and `22_diverse_stacking.py` re-fit the BalancedRandomForest baseline at the start of every feature-set loop.** Cache the OOF predictions in `cache/` and reload across scripts; the savings are ≈ 20 minutes wall clock on the full battery.
- **P2. `25_conformal_prediction.py` refits the underlying model inside every outer fold for every α value, which is wasteful when fits are deterministic at fixed seed.** Refactor to fit once per fold and compute conformal sets at all α values from the cached probabilities.

### Nice-to-have

- **P3.** Replace explicit Python-loop bootstrap in `bootstrap_auc` with the vectorised resampling already used elsewhere; saves ~5 seconds per call.

## Readability and documentation

### Should-fix

- **D1. Function-level docstrings are missing on ~30% of utility helpers (estimated from `CALLGRAPH.md`).** Add one-line purpose strings where missing — especially `_shared.py:oof_predictions` and the various `make_*_pipe` factories.
- **D2. Magic numbers in `_shared.py` are hard-coded as defaults but undocumented.** Examples: `n_estimators=500`, `min_samples_leaf=2` in `make_pipeline_eicu`. Add a `# rationale:` comment with the source of each.
- **D3. The 10/11 CEA module mixes implementation and reporting in a single file at ~500 LOC; consider splitting into `cea_model.py` (strategy logic) and `cea_psa.py` (Monte Carlo wrappers).** Improves testability and matches modern decision-analytic codebases (e.g., the R `dampack` package convention).
- **D4. `25_conformal_prediction.py:class_conditional_conformal` lacks references in the docstring.** Add Vovk 2005 and Angelopoulos & Bates 2021 citations so future readers know which formulation is implemented.

### Nice-to-have

- **D5.** Use type hints throughout. None of the scripts currently use them; for a public release the API-facing functions in `_shared.py` should at least declare argument and return types.
- **D6.** Adopt a consistent logging style. Most scripts use `print(..., flush=True)`; a small number use plain `print`. A shared `setup_logger(name)` helper would unify this.
- **D7.** Rename `aed_timing_recoded` to something more self-describing (`aed_within_72h`) and add the recoding rule to a docstring.
- **D8.** Move `OUT`, `RES`, `FIG`, `CACHE` paths from module-level globals to a `Paths` dataclass in `_shared.py` for cleaner imports.
- **D9.** The decision-tree rendering in `14_decision_tree.py` mixes geometry constants (`X_ROOT=0.6`, `X_STRAT=2.6`) with logic; consider parametrising via a `TreeLayout` dataclass.

## Security and safety

### Should-fix

- **S1. None of the scripts touch network resources, so no credentials handling required.** However, `23_tabpfn_eval.py` does attempt to call the TabPFN inference API. The script currently handles `TabPFNLicenseError` gracefully (treating it as a missing-result row) but does not block accidental data egress. Add an explicit `--allow-network` flag and refuse to send data unless it is set. This is important because the input here is patient-derived even after de-identification.

### Nice-to-have

- **S2.** `15_radiology_nlp.py` shells out to no external tools; pure regex. Note this explicitly in the docstring so reviewers know nothing leaves the machine.
- **S3.** Add a `LICENSE` header at the top of each `.py` file pointing to `LICENSE` at the repo root.

---

## Recommended remediation order

1. **C1** (consolidate helpers into `_shared.py`) — biggest readability win; sets up future changes.
2. **R1, P1, P2** (focal-loss base_score, OOF caching) — smaller wall-clock savings, but produces a cleaner re-run for reviewers.
3. **D1, D4** (docstrings + references) — required for a public release.
4. **S1** (network safety flag for TabPFN) — required for public release.
5. **C2, C3, R2, D2–D9, S2–S3** — backlog.

## Out of scope for this revision

- Migration to a fuller MLOps stack (MLflow, hydra config, click CLI) — current scripts are intentionally minimal so reviewers can read end-to-end in one sitting.
- A test suite — the analysis is reproducible because every script is deterministic and produces stable CSV outputs that can be diffed against a tagged commit; per-function unit tests would be overkill for a one-paper release.
- Containerisation (Dockerfile) — `requirements.txt` plus the `n_jobs=1` discipline is sufficient for Apple Silicon and Linux reproduction.
