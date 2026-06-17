"""Task 43 — FLIC vs Platt vs raw Firth calibration comparison (C1).

Reviewer C1: cite/evaluate Firth-with-intercept-correction (FLIC) and FLAC
(Puhr et al. 2017) against the Platt recalibration we deploy. FLIC re-estimates only
the intercept of the Firth fit so the mean predicted probability matches the observed
event rate (a one-parameter recalibration); Platt fits a 2-parameter logistic on the
Firth scores. We compare the OOF calibration each yields on postop-B.

Per-repeat OOF (each patient predicted once per repeat; metrics averaged across repeats),
which is the methodologically correct calibration assessment for repeated CV.

Output: results/43_flic_vs_platt.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from firthlogist import FirthLogisticRegression

from _shared import RES, SEED, load_bidmc, POSTOP_B_FEATURES, calibration_metrics

OUTCOME = "seizure"


def firth_pipe(features):
    pre = ColumnTransformer([("num", Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), features)])
    return Pipeline([("pre", pre), ("clf", FirthLogisticRegression(
        max_iter=200, skip_pvals=True, skip_ci=True, wald=True))])


def flic_intercept(eta0_train, y_train):
    """Find intercept a so mean(sigmoid(a + eta0)) == mean(y) — the FLIC correction."""
    ybar = y_train.mean()
    f = lambda a: expit(a + eta0_train).mean() - ybar
    try:
        return brentq(f, -20, 20)
    except ValueError:
        return 0.0


def oof_three(X, y, features, n_splits=5, n_repeats=5):
    """Return per-repeat-averaged calibration for raw Firth, FLIC, Platt."""
    recs = {"raw_firth": [], "firth_flic": [], "firth_platt": []}
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + rep)
        p_raw, p_flic, p_platt = (np.zeros(len(y)) for _ in range(3))
        for tr, te in skf.split(X, y):
            pipe = firth_pipe(features); pipe.fit(X.iloc[tr], y[tr])
            clf = pipe.named_steps["clf"]; pre = pipe.named_steps["pre"]
            eta_tr = clf.decision_function(pre.transform(X.iloc[tr]))
            eta_te = clf.decision_function(pre.transform(X.iloc[te]))
            eta0_tr = eta_tr - clf.intercept_           # linear predictor w/o intercept
            eta0_te = eta_te - clf.intercept_
            # raw Firth
            p_raw[te] = expit(eta_te)
            # FLIC: re-estimate intercept only
            a = flic_intercept(eta0_tr, y[tr])
            p_flic[te] = expit(a + eta0_te)
            # Platt: 2-param logistic on the Firth scores (fit on train)
            pl = LogisticRegression(max_iter=500)
            pl.fit(expit(eta_tr).reshape(-1, 1), y[tr])
            p_platt[te] = pl.predict_proba(expit(eta_te).reshape(-1, 1))[:, 1]
        for nm, p in [("raw_firth", p_raw), ("firth_flic", p_flic), ("firth_platt", p_platt)]:
            m = calibration_metrics(y, p)
            recs[nm].append((m["brier"], m["citl"], m["slope"], m["ece"]))
    rows = []
    for nm, arr in recs.items():
        a = np.array(arr)
        rows.append({"method": nm,
                     "brier": round(a[:, 0].mean(), 4),
                     "citl": round(a[:, 1].mean(), 4),
                     "slope": round(a[:, 2].mean(), 3),
                     "slope_sd": round(a[:, 2].std(), 3),
                     "ece": round(a[:, 3].mean(), 4)})
    return pd.DataFrame(rows)


def main():
    df = load_bidmc()
    X = df[POSTOP_B_FEATURES].copy(); y = df[OUTCOME].astype(int).values
    out = oof_three(X, y, POSTOP_B_FEATURES)
    out.to_csv(RES / "43_flic_vs_platt.csv", index=False)
    print("[C1] FLIC vs Platt vs raw Firth (postop-B, per-repeat OOF):")
    print(out.to_string(index=False))
    print("\nReading: FLIC (1-param intercept correction) and Platt both achieve CITL~0;")
    print("compare Brier/ECE to see whether the extra Platt slope parameter helps.")


if __name__ == "__main__":
    main()
