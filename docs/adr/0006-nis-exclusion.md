# 0006 — The NIS database is excluded from the primary analysis

- Status: Accepted
- Date: 2026-06-17

## Context

The Nationwide Inpatient Sample (NIS) was initially explored as a third database. Its seizure
outcome is captured only through administrative ICD codes, which conflate acute symptomatic
seizure (R56.9) with chronic epilepsy (G40.*) and cannot anchor the postoperative-seizure
phenotype to the evacuation episode. When the outcome is reclassified to isolate acute
postoperative seizure, discrimination collapses to chance (AUC ≈ 0.50), and a grouped-LASSO on
the ICD groups confirms no recoverable signal — i.e. the limitation is the coding, not the model.

## Decision

Exclude NIS from the **primary** analysis. Retain the NIS reclassification and grouped-LASSO
results in a supplementary appendix as a documented negative result explaining *why*
administrative-coding data are inadequate for this outcome.

## Consequences

- The primary external-validation claim rests on **eICU** (structured seizure flag), not NIS.
- The supplementary appendix turns the NIS dead-end into a useful methodological caution for
  readers tempted to use administrative codes for this phenotype.
- The NIS scripts (`12_nis_seizure_reclassify.py`, `13_nis_grouped_lasso.py`) remain in the
  repo as an `optional` pipeline stage (their source data are not redistributable).
