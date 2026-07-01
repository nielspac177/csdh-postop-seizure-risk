# 0002 — Firth penalized logistic regression is the deployment model

- Status: Accepted
- Date: 2026-06-17

## Context

The 11-method battery (six SMOTE-family oversamplers, Optuna-tuned XGBoost and LightGBM, a
diverse-base stacking ensemble, TabPFN, Bayesian logistic regression, BalancedRandomForest,
and Firth penalized logistic regression) converged on an AUC ceiling near 0.68 on BIDMC — all
methods statistically indistinguishable in discrimination (DeLong p > 0.05), consistent with
2022–2025 cSDH-seizure meta-evidence and with a 48-event sample.

When discrimination is at a noise ceiling, the differentiator is **calibration** and
**small-sample bias**, not AUC. BalancedRandomForest, although a useful anchor for DeLong
comparisons, is badly miscalibrated (Brier ≈ 0.228). Maximum-likelihood logistic regression is
biased toward extreme coefficients in rare-event data.

## Decision

Deploy **Firth penalized logistic regression**. Firth's penalty corrects small-sample
separation/bias; nested Platt scaling delivers a calibration slope ≈ 1 and |CITL| ≈ 0. Brier
≈ 0.069 — roughly 3.3× better calibrated than BalancedRandomForest at equivalent
discrimination.

## Consequences

- BalancedRandomForest is reported as a **reference/DeLong anchor**, never a deployment
  candidate (its poor calibration is stated explicitly).
- The deployed probabilities are trustworthy enough to drive a conformal layer
  ([0004](0004-conformal-to-action-mapping.md)) and a cost-effectiveness model.
- Coefficients (with SE/CI) and a calibration plot for the deployed postop-B model are
  released as a supplementary table/figure.
- See [0003](0003-calibration-not-auc-objective.md) for why calibration is the optimization
  target rather than AUC.
