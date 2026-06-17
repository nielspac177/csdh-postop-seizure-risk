# 0004 — Conformal prediction sets map to explicit clinical actions

- Status: Accepted
- Date: 2026-06-17

## Context

Class-conditional (Mondrian) split-conformal prediction yields, per patient, a *prediction set*
with finite-sample, class-conditional coverage under exchangeability. At α = 0.10 on postop-B
the sets partition the cohort into singleton {no-seizure} (rule-out, ≈27%), singleton
{seizure} (rule-in, ≈11%), and the doubleton {both} (abstain/defer, ≈64%).

Reviewers correctly noted that the manuscript previously never stated **what a clinician does**
with each set — the conformal layer and the cost-effectiveness model were not connected, an
architectural gap (revision item M4).

## Decision

Map each conformal set to an explicit action:

| Conformal set | Interpretation | Clinical action |
|---|---|---|
| singleton {no-seizure} | confident rule-out | observation (no AED) |
| singleton {seizure} | confident rule-in | cEEG + targeted AED |
| doubleton {both} | abstain / insufficient evidence | default policy (universal AED) |

The doubleton→action choice is a policy lever, so it is subjected to a sensitivity analysis:
three doubleton mappings (→ universal AED, → observation, → cEEG) are each run through the full
PSA.

## Consequences

- The conformal layer becomes decision-relevant, not decorative.
- The doubleton-mapping sensitivity is reported as a supplementary appendix; the
  cost-effectiveness conclusion is shown to be stable across all three mappings
  (see [0005](0005-cea-operating-point-and-message.md)).
- In prospective deployment, conformal recalibration on a site's first ~50 cases is recommended
  to restore class-conditional coverage under domain shift.
