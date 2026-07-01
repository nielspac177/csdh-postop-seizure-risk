# Architecture Decision Records

These ADRs record the consequential **methodological and modeling** decisions behind this
project — the choices a reviewer, a future maintainer, or the author six months from now would
otherwise have to reverse-engineer from code and CSVs. Each record is immutable once
`Accepted`; a later decision that overrides it gets a new number and links back.

Format: lightweight [MADR](https://adr.github.io/madr/). Status ∈ {Proposed, Accepted,
Superseded}.

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-postop-b-deployable-feature-set.md) | postop-B (leakage-safe, 18-var) is the deployable feature set | Accepted |
| [0002](0002-firth-over-balanced-random-forest.md) | Firth penalized logistic regression is the deployment model | Accepted |
| [0003](0003-calibration-not-auc-objective.md) | Optimize for calibration & decision-integration, not AUC | Accepted |
| [0004](0004-conformal-to-action-mapping.md) | Conformal prediction sets map to explicit clinical actions | Accepted |
| [0005](0005-cea-operating-point-and-message.md) | CEA uses the base-rate operating point; universal AED is the NMB-optimal strategy | Accepted |
| [0006](0006-nis-exclusion.md) | The NIS database is excluded from the primary analysis | Accepted |
