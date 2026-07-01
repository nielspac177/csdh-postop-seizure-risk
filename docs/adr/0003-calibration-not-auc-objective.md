# 0003 — Optimize for calibration & decision-integration, not AUC

- Status: Accepted
- Date: 2026-06-17

## Context

Postoperative seizure after cSDH is rare (≈7–12%), and the development cohort has 48 events.
At that event count, the Bernoulli noise floor on AUC is ±0.06 around 0.70 (Hanley–McNeil
variance). Chasing AUC in this regime optimizes noise: differences between methods inside the
floor are not real, and an unconstrained search will "improve" AUC by fitting sampling
idiosyncrasies.

What a bedside tool actually needs is (a) **trustworthy probabilities** (good calibration) so a
clinician can reason about a patient's risk, and (b) **honest uncertainty** so the tool can
abstain when it should not commit.

## Decision

Treat **calibration (Brier, calibration slope/intercept, ECE) and decision-integration
(decision-curve net benefit, conformal coverage)** as the optimization and reporting targets.
AUC is reported for comparability and to demonstrate the ceiling, **not** maximized.

This decision also governs the model-redo experiments: the autoresearch loop optimizes OOF
Brier (not AUC), and every run logs an overfitting panel (train−OOF optimism gap,
permutation-label null, learning curve) against the noise floor so that any "win" inside the
floor is flagged as noise rather than signal.

## Consequences

- The headline framing is a **proof-of-concept** about *how* to build deployable small-cohort
  clinical ML, not a claim of strong discrimination.
- Reviewers' over-claim concerns (title/abstract/conclusion "deployable", "optimal") are
  resolved by aligning the prose with this objective.
- The value-of-information analysis becomes central: it quantifies what evidence would actually
  move the decision, given that discrimination cannot.
