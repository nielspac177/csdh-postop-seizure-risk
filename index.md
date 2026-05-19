---
layout: default
title: csdh-jnnp — companion code for JNNP submission
---

# csdh-jnnp

Companion code and documentation for the manuscript:

> **A calibrated and conformally-deployable risk score for postoperative seizure after chronic subdural haematoma evacuation: a proof-of-concept multi-database study with value-of-information analysis.**

submitted to the *Journal of Neurology, Neurosurgery and Psychiatry*.

[![Release](https://img.shields.io/github/v/release/nielspac177/csdh-jnnp.svg)](https://github.com/nielspac177/csdh-jnnp/releases/tag/v1.0-JNNP-submission)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Navigate

- **[Full README](README.html)** — quickstart, repository layout, dependencies, and reproducibility instructions.
- **[Module dependency graph and function inventory](CALLGRAPH.html)** — 187 functions across 28 modules with their arguments and one-line purpose.
- **[Code review notes](CODE_REVIEW.html)** — prioritized review covering correctness, reproducibility, performance, readability, and safety. No must-fix blockers identified.
- **[Reviewer access protocol](docs/reviewer_access.html)** — procedure for accessing filtered working data subsets under IRB / CRD / HCUP control.
- **[Manuscript plan](docs/imrad_plan.html)** — IMRAD outline, five main messages, and critical-thinking review of common reviewer concerns.
- **[Literature review on small-cohort clinical ML](docs/literature_review_imbalance_smallcohort.html)** — 2022–2026 evidence base for the modelling choices.

## What this repository contains

- Twenty-eight deterministic analysis scripts (BIDMC + eICU + NIS), all `n_jobs = 1`.
- The Firth penalized logistic regression deployment model and a class-conditional Mondrian conformal prediction layer.
- An eleven-method modelling battery (SMOTE family, Optuna-tuned gradient boosting, diverse-base stacking, Bayesian logistic regression).
- A probabilistic cost-effectiveness analysis with the first published value-of-information ranking for postoperative-seizure prevention.
- A regex-pattern radiology NLP pipeline that achieves 91 % macro-averaged extraction accuracy on a validation set.
- Six main paper figures (F1–F6) in JNNP aesthetic and seven supplementary figures.
- TRIPOD-AI reporting checklist and reproducibility appendix.

## Data policy

No patient-level data are included in this repository. The `.gitignore` excludes the source CSV exports from BIDMC, the eICU Collaborative Research Database, and the NIS HCUP file. Filtered working subsets used in the manuscript are documented at [`docs/reviewer_access.md`](docs/reviewer_access.html) and made available to approved peer reviewers via the protocol described there.

## Citation

If you build on this code, please cite:

> Pacheco-Barrios N, et al. A calibrated and conformally-deployable risk score for postoperative seizure after chronic subdural haematoma evacuation: a proof-of-concept multi-database study with value-of-information analysis. *Journal of Neurology, Neurosurgery and Psychiatry* (submitted, 2026).

A DOI from Zenodo will be minted on acceptance.
