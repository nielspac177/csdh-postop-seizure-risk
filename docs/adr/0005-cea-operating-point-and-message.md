# 0005 — CEA uses the base-rate operating point; universal AED is the NMB-optimal strategy

- Status: Accepted
- Date: 2026-06-17

## Context

The cost-effectiveness analysis compares four strategies — observation, universal AED,
ML-guided AED, ML-guided cEEG+targeted AED — via decision-tree rollback plus a 10,000-iteration
probabilistic sensitivity analysis at willingness-to-pay $50k/$100k/$150k per QALY.

The earlier CEA was parameterized with sensitivity/specificity (Se 0.842 / Sp 0.504) that came
from the **postop-A** model (i.e. partly from leaked variables; see
[0001](0001-postop-b-deployable-feature-set.md)). With the leakage-free **postop-B** model, the
narrow Firth probability range means no threshold reproduces Se 0.842 at non-trivial
specificity; an automatic "matched-Se" rule degenerated to a threshold that flags everyone
(Se 1.0 / Sp 0.0), trivially identical to universal AED.

## Decision

Use the **cohort base rate (0.073 = 48/655)** as the deployable operating point — the natural
cutoff for a calibrated probability and the value already used to define the "ML-guided AED"
action. At that point postop-B gives **Se 0.50 / Sp 0.70**. Degenerate operating points
(Sp < 0.05 or Se > 0.99) are excluded from selection.

## Decision outcome (the message)

The NMB-optimal strategy is **conditional on two uncertain, cSDH-specific parameters** — AED
efficacy (`aed_rrr`) and AED disutility (`utility_aed_decrement`) — quantified in the
threshold analysis (`scripts/39_aed_harm_threshold.py` → `results/39_aed_harm_threshold.csv`):

- Under the **base case** (RRR 0.45 imported from TBI/tumour; AED disutility 0.02, i.e. a
  lifetime penalty of ~0.0036 QALY), **universal AED** has the highest expected NMB — but by a
  small margin (≈$1k on a ≈$641k base; ML-cEEG retains ≈20% PSA probability of optimality).
- **ML-guided allocation becomes NMB-optimal** once AED efficacy is at cSDH-plausible levels
  (**RRR ≤ 0.30**) *or* AED disutility reflects real elderly harm (**≥ 0.10** over the treatment
  window) — and across essentially the whole cSDH-plausible region of the two-way grid.
- Universal AED is preferred in **only one corner**: high efficacy (RRR ≥ 0.40) *and* near-zero
  harm — the optimistic, other-population assumption set that reviewers explicitly questioned
  (item S11).
- Every *active* strategy beats **observation** throughout — prophylaxis beats watchful waiting.

This **supersedes the previous flat claim that "ML-guided cEEG is optimal at $100k"** (an
artifact of leakage-inflated sensitivity) **and** rejects the equally flat "universal AED wins"
(an artifact of optimistic, imported AED assumptions). The honest result is a *conditional*
one.

## Consequences

- Resolves revision item M3 (CEA over-claim) and connects to S11 (cSDH-specific AED efficacy).
  The abstract/conclusion state that ML-guided allocation is cost-effective under
  cSDH-plausible AED assumptions, that the optimum is sensitive to AED efficacy/harm, and that
  all active strategies beat observation.
- The conditional result *is* the contribution: a calibrated model + conformal layer enables
  harm-sensitive allocation precisely in the regime (uncertain/low AED efficacy, non-trivial
  elderly harm) where it matters.
- Strengthens the value-of-information contribution: VOI identifies AED efficacy and AED harm
  as the top-priority parameters to resolve. (VOI recomputed under postop-B in Phase 3.)
- **Open item:** the cSDH-specific AED RRR must be grounded in published cSDH evidence (S11/S13)
  before the wording is locked; do not assert RRR ≤ 0.30 without citation.
- Memos: `results/38_PHASE0_CEA_NUMBERS_MEMO.md`, `results/39_aed_harm_threshold.csv`.
