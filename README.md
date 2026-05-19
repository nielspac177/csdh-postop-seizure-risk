# csdh-jnnp — code release for a calibrated risk score for postoperative seizure after chronic subdural haematoma evacuation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#requirements)
[![Reproducibility](https://img.shields.io/badge/n__jobs-1-green.svg)](#reproducibility)

Companion code for the *Journal of Neurology, Neurosurgery and Psychiatry* submission **"A calibrated and conformally-deployable risk score for postoperative seizure after chronic subdural haematoma evacuation: a proof-of-concept multi-database study with value-of-information analysis."**

This repository contains the analysis code, figure-generation scripts, decision-analytic implementation, and reporting artefacts. **No patient-level data are included.** Filtered, de-identified working subsets used in the manuscript are held on a private branch and may be released to peer reviewers on request under the BIDMC IRB and eICU CRD data-use agreements.

## Why this repository exists

The manuscript is a proof of concept: small-cohort clinical machine learning can be honestly deployable when calibration and decision-integration replace AUC as the optimisation target. This repository lets reviewers and the wider community reproduce the eleven-method modelling battery, the conformal risk stratification, the cost-effectiveness analysis with value-of-information, and the supplementary radiology-NLP feature pipeline.

## What you can do with this code

- **Reproduce every figure and table in the JNNP submission.** Each script is a single-purpose, deterministic module that consumes either the BIDMC working CSV, the eICU CRD cohort export, or the NIS HCUP working file. With access to the underlying data the analyses run end-to-end on a laptop in under one hour.
- **Apply the Firth penalized logistic regression deployment model to new cohorts.** The model factory at `scripts/24_firth_bayes_lr.py` is sklearn-compatible.
- **Run class-conditional conformal prediction on any binary clinical outcome.** The Mondrian split-conformal implementation at `scripts/25_conformal_prediction.py` accepts any scikit-learn classifier as the base model.
- **Apply the regex-pattern radiology NLP extractor to free-text reports.** The pipeline at `scripts/15_radiology_nlp.py` ships with a synthetic validation set demonstrating 91% macro-averaged extraction accuracy and a documented schema.
- **Re-fit the four-strategy cost-effectiveness decision tree** at `scripts/14_decision_tree.py` and the probabilistic CEA + EVPI/EVPPI at `scripts/16_voi_evpi.py`.

## Repository layout

```
csdh-jnnp/
├── README.md                       — this file
├── LICENSE                         — MIT
├── requirements.txt                — exact package versions for reproducibility
├── .gitignore                      — excludes /data/, /cache/, *.csv, *.pkl
├── CODE_REVIEW.md                  — prioritized code-quality recommendations
├── CALLGRAPH.md                    — module dependency graph
├── scripts/                        — analysis scripts (numbered in execution order)
│   ├── _shared.py                  — pipeline factories, loaders, calibration metrics
│   ├── 02_calibration.py           — Brier, CITL, slope/intercept, ECE, MCE, HL
│   ├── 03_dca.py                   — decision-curve net benefit (Vickers)
│   ├── 04_loho.py                  — leave-one-hospital-out + random-effects pooling
│   ├── 05_temporal_leakage.py      — 24h / 48h / 72h time-window cohort cuts
│   ├── 06_overfitting.py           — nested CV + feature-importance stability
│   ├── 07_missing_data.py          — Little's MCAR + multiple imputation
│   ├── 08_eicu_cohort.py           — 4-stratum eICU cohort sensitivity
│   ├── 09_competing_risks.py       — cause-specific Cox + IPCW Fine-Gray
│   ├── 10_11_cea_pairwise.py       — base-case CEA + PSA + CEAC
│   ├── 12_nis_seizure_reclassify.py — NIS outcome reclassification
│   ├── 13_nis_grouped_lasso.py     — group-LASSO with λ-path tuning
│   ├── 14_decision_tree.py         — TreeAge-style decision-tree figure
│   ├── 15_radiology_nlp.py         — regex-pattern radiology NLP pipeline
│   ├── 16_voi_evpi.py              — EVPI / EVPPI via Strong-Oakley regression
│   ├── 17_build_slides.py          — 15-minute oral-presentation deck
│   ├── 18_bidmc_optimize.py        — XGBoost/LightGBM hyperparameter tuning
│   ├── 19_transfer_learning.py     — eICU → BIDMC transfer-learning failure
│   ├── 20_build_manuscript.py      — initial Word manuscript builder
│   ├── 21_imbalance_sweep.py       — SMOTE-family + cost-sensitive comparison
│   ├── 22_diverse_stacking.py      — LR + RF + XGB + KNN + RBF-SVM stacking
│   ├── 23_tabpfn_eval.py           — TabPFN v2 evaluation (API-credentialed)
│   ├── 24_firth_bayes_lr.py        — Firth penalized + Bayesian LR (deployment)
│   ├── 25_conformal_prediction.py  — class-conditional conformal sets
│   ├── 26_main_figures.py          — consolidated 6-panel main paper figures
│   └── 27_build_jnnp_manuscript.py — JNNP-format main + supplementary docs
├── figures/                        — all generated PNG and PDF figures
├── results/                        — generated CSV result tables
└── docs/                           — manuscript draft, supplementary, lit review
    ├── JNNP_main_manuscript.docx
    ├── JNNP_supplementary.docx
    ├── imrad_plan.md
    └── literature_review_imbalance_smallcohort.md
```

## Quickstart

```bash
git clone https://github.com/<author>/csdh-jnnp.git
cd csdh-jnnp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce all main paper figures (requires the working data files in /data/)
python scripts/02_calibration.py
python scripts/24_firth_bayes_lr.py
python scripts/25_conformal_prediction.py
python scripts/26_main_figures.py

# Build the JNNP manuscript and supplementary
python scripts/27_build_jnnp_manuscript.py
```

## What each module computes

| Module | Inputs | Outputs | Key function |
|---|---|---|---|
| `_shared.py` | — | feature lists, pipeline factories, OOF prediction helper, calibration metrics | `oof_predictions()`, `calibration_metrics()` |
| `02_calibration.py` | BIDMC CSV, eICU CSV | calibration metrics CSV, calibration-curve figure | `run_one()` |
| `03_dca.py` | cached OOF predictions | net-benefit table and decision-curve figure | — |
| `04_loho.py` | eICU CSV | per-hospital and pooled summary CSVs, forest plot | `loho_for()`, `random_effects_pool()` |
| `05_temporal_leakage.py` | eICU CSV | leakage-audit CSV | `is_leakage_suspect_feature()` |
| `09_competing_risks.py` | BIDMC CSV | Cox + Fine-Gray c-index, Schoenfeld diagnostics | `fine_gray()` |
| `10_11_cea_pairwise.py` | parameter defaults | PSA samples, CEAC | `run_psa()`, `run_strategy()` |
| `14_decision_tree.py` | parameter defaults | TreeAge-style figure, rollback CSV | `rollback_strategy()`, `render_tree()` |
| `15_radiology_nlp.py` | optional radiology CSV | extracted features CSV, validation report | `extract_features()`, `validate()` |
| `16_voi_evpi.py` | PSA samples | EVPI table, EVPPI tornado | `run_psa_tracked()`, `compute_evpi()`, `evppi_strong_oakley()` |
| `19_transfer_learning.py` | BIDMC + eICU | augmented OOF, DeLong test | `delong_test()`, `build_bidmc_transfer_X()` |
| `21_imbalance_sweep.py` | BIDMC CSV | 11-method comparison CSV | `pipe_with_sampler()`, `pipe_xgb_focal()` |
| `22_diverse_stacking.py` | BIDMC CSV | diverse-stack CSV | `make_diverse_stack()` |
| `24_firth_bayes_lr.py` | BIDMC + eICU CSV | Firth + Bayesian CSV | `BayesianLogReg`, `derive_eicu_priors()` |
| `25_conformal_prediction.py` | BIDMC CSV | conformal coverage and rule-out table | `class_conditional_conformal()` |
| `26_main_figures.py` | results CSVs and figures | F1–F6 consolidated figures | `figure_1` … `figure_6` |
| `27_build_jnnp_manuscript.py` | results, figures | `JNNP_main_manuscript.docx`, supplementary | `build_main()`, `build_supplementary()` |

## Package dependencies and why they were chosen

| Package | Version | Why this and not an alternative |
|---|---|---|
| `scikit-learn` | 1.5.2 | Industry-standard tabular ML; supports the full pipeline / `predict_proba` API used by the conformal layer. |
| `imbalanced-learn` | 0.12.4 | The canonical SMOTE-family implementation; `imblearn.ensemble.BalancedRandomForestClassifier` reproduces the published baseline; we did not switch frameworks to avoid behavioural drift relative to the original analysis. |
| `xgboost` | 2.1.4 | Used for `scale_pos_weight` and custom focal-loss objectives. We chose XGBoost over LightGBM as the primary GBM because of `scale_pos_weight` documentation and stability with small data. |
| `lightgbm` | 4.6.0 | Tested as a secondary GBM for sensitivity; ordered boosting variant of CatBoost was considered but was not necessary once the AUC ceiling was established. |
| `lifelines` | 0.30.0 | Cox proportional-hazards and Fine-Gray competing-risks fits; we required an explicit Schoenfeld residuals test which `lifelines` exposes natively. |
| `firthlogist` | 0.5.0 | Direct implementation of Firth's penalized likelihood. We chose it over the R `logistf` port for in-process integration and `predict_proba` compatibility. |
| `mapie` | 1.4.0 | Class-conditional and Mondrian conformal prediction with sklearn-compatible API. We elected MAPIE over `crepes` for tighter sklearn integration. |
| `optuna` | 4.8.0 | TPE sampler for hyperparameter tuning; used over scikit-learn's GridSearchCV for adaptive Bayesian optimization. |
| `python-docx` | 1.2.0 | Programmatic Word-document generation for the JNNP main + supplementary; we elected docx over LaTeX because JNNP's editorial workflow accepts Word natively. |
| `python-pptx` | 1.0.2 | Programmatic PowerPoint for the oral-presentation deck. |
| `pandas` | 2.3.3 | Tabular manipulation throughout. |
| `numpy` | 1.26.4 | Numerical core; pinned at 1.26 for matplotlib 3.9.4 compatibility. |
| `matplotlib` | 3.9.4 | Publication-grade figures; we use a consistent style sheet across all scripts. |
| `Pillow` | 11.3 | Used in `26_main_figures.py` to embed sub-figures into composite plates. |
| `scipy` | 1.13 | Statistical testing (chi-square, Fisher exact, DeLong via norm CDF). |
| `lxml` + `xlsxwriter` | latest | Required by `python-docx` for Word file generation. |

## Reproducibility

- All scripts force `n_jobs = 1` and set `SEED = 42` (defined in `_shared.py`).
- The shell environment variables `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` are set to `1` at the top of every numerical script for Apple-Silicon stability.
- Cross-validation splits are obtained from `RepeatedStratifiedKFold` with the same `random_state`, ensuring identical splits between scripts that consume cached out-of-fold predictions.
- All probabilistic sensitivity analyses use a single shared `numpy.random.Generator` seeded with `SEED`.
- A `requirements.txt` pin-file is provided.

## Reviewer access to filtered working data

We have prepared de-identified working subsets — the 21-feature BIDMC table, the eICU non-traumatic cohort frame with the 103-variable Set C, and the NIS 2016–2019 chronic-SDH file — under access controls described in `docs/reviewer_access.md`. Reviewers may request the filtered files via the corresponding author; they are released on a private branch (`reviewer-access-vN`) of this repository for the duration of the review process.

## Citation

If you use this code in your own research please cite:

> Pacheco-Barrios N, [Co-authors], [Senior author]. A calibrated and conformally-deployable risk score for postoperative seizure after chronic subdural haematoma evacuation: a proof-of-concept multi-database study with value-of-information analysis. *Journal of Neurology, Neurosurgery and Psychiatry* (submitted, 2026).

A formal DOI (Zenodo) will be minted on acceptance.

## License

Code is released under the MIT license (see `LICENSE`). The cleaned NIS ICD-10 outcome codeset, the radiology NLP regex library, and the TreeAge-style decision-tree implementation are released under the same license. Data files remain under their respective data-use agreements (BIDMC IRB, eICU CRD, NIS HCUP) and are not part of this distribution.
