# Reviewer access — filtered working data

The public main branch of this repository contains code, callgraph, figures, aggregate result tables, and manuscript drafts. It does **not** contain any patient-level data. Raw data are restricted by the BIDMC IRB, the eICU Collaborative Research Database data-use agreement, and the Healthcare Cost and Utilization Project (HCUP) data-use agreement for NIS.

This document describes the procedure by which peer reviewers may access the filtered working subsets used in the manuscript, should they wish to verify the analyses end-to-end.

## What sits on the private `reviewer-access-vN` branch

A separate, password-protected branch contains the following derived working files. Each was produced by a deterministic filtering pipeline against the upstream source and never includes free-text protected health information.

| File | Source | Description | Rows × cols |
|---|---|---|---|
| `data/bidmc_postopA_dev.csv` | BIDMC EMR | 21-variable postoperative-A feature set with the binary seizure outcome | 655 × 22 |
| `data/bidmc_postopB_dev.csv` | BIDMC EMR | 18-variable postoperative-B feature set | 655 × 19 |
| `data/eicu_setC_full.csv` | eICU CRD v2.0 | 103-variable Set C, non-traumatic SDH stratum + traumatic-SDH negative control | 5,376 × 104 |
| `data/eicu_setC_pure.csv` | eICU CRD v2.0 | Pure post-craniotomy filter (no prior seizures, no pre-admission AED, no mechanical ventilation) | 3,255 × 104 |
| `data/nis_chronic.parquet` | NIS HCUP 2016–2019 | 49-variable chronic-SDH + surgical cohort with the corrected outcome | 2,518 × 50 |
| `data/icd10_corrected_outcome.csv` | derived | Released codeset for acute symptomatic seizure (R56.x, 780.39, G41.x) excluding pre-existing epilepsy (G40.x, 345.x) | 26 × 4 |

All files are de-identified per the source-database protocols: BIDMC entries have a single random ID per patient with admission dates jittered by ±1 month; eICU entries use the canonical `patientunitstayid`; NIS entries inherit the HCUP randomized hospital and patient identifiers.

## Access protocol

1. Reviewer requests access by emailing the corresponding author from a peer-reviewer correspondence address, citing the JNNP submission ID.
2. Corresponding author confirms reviewer status with the JNNP editorial office.
3. A time-limited (30-day) personal access token is issued for the private branch.
4. The reviewer pulls the branch and follows the reproduction protocol in the main `README.md`.
5. Access is revoked at the end of the review cycle; a fresh branch (`reviewer-access-v(N+1)`) is created if a second-round review is requested.

## What is **not** released even under reviewer access

- Free-text radiology reports (the NLP pipeline `scripts/15_radiology_nlp.py` is released, but no patient-level note text).
- Operative dictation, progress notes, or discharge summaries.
- BIDMC medical record numbers, admission dates, ICU stay timestamps, or surgical case numbers.
- eICU `uniquepid` / `patienthealthsystemstayid` pairs that could be cross-linked to external eICU-derived publications.
- NIS variables flagged by HCUP as suppressed (small-cell counts, state identifiers).

## Code-only public release

If a reader does not require the working data — for example, they only wish to inspect the modelling choices or apply the Firth deployment model or conformal layer to their own cohort — the public main branch is fully sufficient. Every script in `scripts/` is annotated with its expected input schema; readers can adapt the loaders in `_shared.py` to their own data without ever seeing the BIDMC / eICU / NIS working files.
