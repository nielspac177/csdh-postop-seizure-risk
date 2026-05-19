# csdh-jnnp · `reviewer-access-template`

This branch is a **template**, not a data release. It documents the structure and access protocol for filtered working subsets used in the JNNP submission. The actual de-identified data files are held under institutional control (BIDMC IRB, eICU CRD, NIS HCUP) and are not committed to git on either the public `main` branch or any GitHub-hosted branch of this repository.

If you are a peer reviewer authorised by the *Journal of Neurology, Neurosurgery and Psychiatry* editorial office to inspect the manuscript's working data, please follow the protocol below.

## What the data release would contain

The following files constitute the full reviewer-access bundle. They are listed here so reviewers know exactly what to expect on request. Each file is the deterministic output of the filtering pipeline in [`scripts/`](https://github.com/nielspac177/csdh-jnnp/tree/main/scripts) on the `main` branch.

| File path | Source | Description | Approx. rows × cols |
|---|---|---|---|
| `data/bidmc_postopA_dev.csv` | BIDMC EMR | 21-variable postoperative-A feature set with the binary seizure outcome | 655 × 22 |
| `data/bidmc_postopB_dev.csv` | BIDMC EMR | 18-variable postoperative-B feature set (no AED-timing or EEG variables) | 655 × 19 |
| `data/eicu_setC_full.csv` | eICU CRD v2.0 | 103-variable Set C, non-traumatic SDH stratum plus traumatic-SDH negative control | 5,376 × 104 |
| `data/eicu_setC_pure.csv` | eICU CRD v2.0 | Pure post-craniotomy filter (no prior seizures, no pre-admission AED, no mechanical ventilation) | 3,255 × 104 |
| `data/nis_chronic.parquet` | NIS HCUP 2016–2019 | 49-variable chronic-SDH + surgical cohort with the corrected acute-symptomatic outcome | 2,518 × 50 |
| `data/icd10_corrected_outcome.csv` | derived | Released codeset distinguishing acute symptomatic seizure (R56.x, 780.39, G41.x) from pre-existing epilepsy (G40.x, 345.x) | 26 × 4 |
| `data/SHA256SUMS.txt` | derived | Per-file SHA-256 checksums for verification |

All files are de-identified according to the source databases' own protocols. BIDMC identifiers are replaced with a random per-patient ID and admission timestamps are jittered by ±1 month. eICU rows carry the canonical `patientunitstayid`. NIS rows inherit the HCUP-anonymised hospital and patient identifiers; HCUP-suppressed cells are kept blank.

The bundle never contains free-text radiology, operative dictation, progress, or discharge text. The Firth-LR coefficient table is included as `data/firth_coefficients.csv` (point estimates, profile-likelihood 95 % CIs, p-values) so that a reviewer can spot-check the deployment model against the manuscript without re-fitting.

## Access protocol

1. **Identification.** The reviewer emails the corresponding author from a peer-review correspondence address used by the JNNP editorial system, citing the JNNP submission ID.
2. **Verification.** The corresponding author confirms the reviewer's status with the JNNP editorial office before any data are released.
3. **Issue.** A time-limited (30-day) personal access token to a private GitHub repository at `nielspac177/csdh-jnnp-data-vN` is granted. The data bundle described above is uploaded to that private repository, not to this public branch.
4. **Reproduction.** The reviewer pulls the data bundle, follows the reproduction protocol in the main branch [`README.md`](https://github.com/nielspac177/csdh-jnnp/blob/main/README.md), and reports any inconsistencies through the JNNP editorial portal.
5. **Revocation.** Access is revoked at the end of the review cycle. A fresh release (`csdh-jnnp-data-v(N+1)`) is generated if a second-round review is requested, with a refreshed seed for the BIDMC pseudonymisation step.

## Why the data are not committed even to a private GitHub branch on this public repository

GitHub branches in a public repository are themselves public; setting a branch to "protected" controls who can push, not who can fetch. The cleanest path is a **separate private GitHub repository** that contains only the data bundle and is granted to approved reviewers as a collaborator with read-only access. This pattern keeps the public code base intact while satisfying the BIDMC IRB and the eICU CRD data-use agreement.

## What is **not** released even under reviewer access

- Free-text radiology, operative dictation, progress notes, or discharge summaries.
- BIDMC medical record numbers, real admission dates, ICU stay timestamps, or surgical case numbers.
- eICU `uniquepid` / `patienthealthsystemstayid` pairs that could be cross-linked to external eICU-derived publications.
- NIS variables flagged by HCUP as suppressed (small-cell counts, state-level identifiers).

## Self-contained code reproduction without data

If you do not require the working data and only wish to inspect the modelling choices or apply the Firth deployment model or the conformal layer to your own cohort, the public [`main`](https://github.com/nielspac177/csdh-jnnp/tree/main) branch is fully sufficient. Every script in `scripts/` is annotated with its expected input schema; the loaders in `_shared.py` can be adapted to other datasets without ever seeing the BIDMC, eICU, or NIS working files.

## Contact

Niels Pacheco-Barrios, MD — Department of Neurosurgery, Beth Israel Deaconess Medical Center, Harvard Medical School — `nielspacheco1997@gmail.com`.
