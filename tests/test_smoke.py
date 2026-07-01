"""Synthetic-data smoke test for the shared modeling utilities.

The real cohorts (csdh_clean.csv, eicu_csdh_cohort_final.csv) are *not*
redistributed (patient data; see .gitignore). This test fabricates a small
synthetic dataset with the postop-B schema so CI can verify that the core
pipeline + evaluation helpers run end-to-end, are deterministic under SEED,
and return values in valid ranges — without any protected health information.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _shared import (  # noqa: E402
    POSTOP_B_FEATURES, SEED, oof_predictions, make_pipeline_postopB,
    bootstrap_auc, calibration_metrics, delong_test,
)


def _synthetic_cohort(n=240, event_rate=0.12, seed=SEED):
    """Build a synthetic postop-B feature matrix with a mild true signal."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.normal(size=n) for f in POSTOP_B_FEATURES})
    # Inject a weak linear signal on three features so AUC is > 0.5 but modest,
    # mirroring the real cohort's near-noise-floor discrimination.
    logit = 0.6 * X["age"] + 0.5 * X["epilepsy_hx"] - 0.4 * X["preop_gcs"]
    logit -= np.quantile(logit, 1 - event_rate)  # shift to target prevalence
    p = 1 / (1 + np.exp(-logit))
    y = pd.Series((rng.uniform(size=n) < p).astype(int), name="seizure")
    return X, y


def test_oof_predictions_runs_and_is_in_range():
    X, y = _synthetic_cohort()
    p = oof_predictions(make_pipeline_postopB, X, y, n_splits=3, n_repeats=1)
    assert p.shape == (len(X),)
    assert np.all((p >= 0) & (p <= 1))


def test_oof_is_deterministic():
    X, y = _synthetic_cohort()
    p1 = oof_predictions(make_pipeline_postopB, X, y, n_splits=3, n_repeats=1)
    p2 = oof_predictions(make_pipeline_postopB, X, y, n_splits=3, n_repeats=1)
    assert np.allclose(p1, p2), "OOF predictions must be reproducible under SEED"


def test_bootstrap_auc_returns_valid_ci():
    X, y = _synthetic_cohort()
    p = oof_predictions(make_pipeline_postopB, X, y, n_splits=3, n_repeats=1)
    auc, lo, hi = bootstrap_auc(y.values, p, n_boot=200, seed=SEED)
    assert 0.0 <= lo <= auc <= hi <= 1.0


def test_calibration_metrics_keys():
    X, y = _synthetic_cohort()
    p = oof_predictions(make_pipeline_postopB, X, y, n_splits=3, n_repeats=1)
    m = calibration_metrics(y.values, p)
    for key in ("brier", "slope", "intercept"):
        assert key in m


def test_delong_identical_models_is_null():
    X, y = _synthetic_cohort()
    p = oof_predictions(make_pipeline_postopB, X, y, n_splits=3, n_repeats=1)
    z, pval = delong_test(y.values, p, p)
    # Comparing a model with itself: zero AUC difference. The variance of the
    # difference is zero, so the test returns NaN (undefined) — and must never
    # report a significant difference.
    assert np.isnan(pval) or pval > 0.05
