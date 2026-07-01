# 0001 — postop-B (leakage-safe, 18-variable) is the deployable feature set

- Status: Accepted
- Date: 2026-06-17

## Context

The BIDMC development cohort supports two postoperative feature sets:

- **postop-A** (21 variables) — includes three variables that encode information generated
  *around or after* the antiepileptic-drug (AED) / EEG decision: `aed_timing_recoded`,
  `prop_aed` (proportion of stay on AED), and `ab_eeg` (abnormal EEG).
- **postop-B** (18 variables) — postop-A minus those three.

A model intended to *inform* the AED/EEG decision at the end of haematoma evacuation cannot use
features that are only realized once that decision has been made. Including them is target
leakage: it inflates apparent performance and cannot be reproduced at the bedside.

The leakage test applied per variable: *"Could a clinician fill this value at OR-exit, before
the AED/EEG decision?"* All three flagged variables fail it.

## Decision

The **deployable** model is trained and reported on **postop-B**. postop-A is retained only as
a paired sensitivity comparison to quantify the leakage-attributable performance gap.

## Consequences

- Discrimination drops from AUC ≈ 0.681 (postop-A) to ≈ 0.645 (postop-B) — within the
  Bernoulli noise floor (±0.06 at AUC≈0.70, 48 events), so not a meaningful loss of signal.
- Decision-curve net benefit in the clinically relevant threshold band is preserved/improved.
- The cost-effectiveness conclusion changes (see [0005](0005-cea-operating-point-and-message.md)):
  the honest, leakage-free operating point is too weak to justify ML-guided allocation over
  universal prophylaxis. This is the correct, defensible result.
- All deployment artifacts (web calculator, conformal layer, CEA inputs) use postop-B.
