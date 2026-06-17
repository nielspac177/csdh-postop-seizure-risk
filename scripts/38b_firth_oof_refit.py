"""Task 38b — Generate fresh Firth OOF predictions for postop-A and postop-B.

The existing OOF cache (oof_bidmc_postopA.npz / oof_bidmc_postopB.npz) is from
BalancedRandomForest (the conformal base model).  The deployed model is Firth
penalized logistic regression — refit here via 5×5 repeated stratified CV with
nested 3-fold Platt scaling.

Outputs:
  cache/oof_bidmc_postopA_firth.npz   (y, p_raw, p_platt)
  cache/oof_bidmc_postopB_firth.npz   (y, p_raw, p_platt)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from firthlogist import FirthLogisticRegression

from _shared import (CACHE, SEED, load_bidmc,
                      POSTOP_A_FEATURES, POSTOP_B_FEATURES)

OUTCOME = "seizure"


def make_firth_pipe(features):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), features)])
    return Pipeline([("pre", pre),
                      ("clf", FirthLogisticRegression(max_iter=200,
                                                      skip_pvals=True,
                                                      skip_ci=True,
                                                      wald=True))])


def cv_oof_with_platt(X, y, features, n_splits=5, n_repeats=5):
    n = len(X)
    p_raw_acc   = np.zeros(n); p_platt_acc = np.zeros(n); n_acc = np.zeros(n)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=SEED)
    total_folds = n_splits * n_repeats
    for k, (tr, te) in enumerate(rskf.split(X, y), 1):
        # Outer fit
        outer = make_firth_pipe(features)
        outer.fit(X.iloc[tr], y[tr])
        p_te_raw = outer.predict_proba(X.iloc[te])[:, 1]

        # Inner CV for Platt scaling
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + k)
        platt_x, platt_y = [], []
        for itr, ite in inner.split(X.iloc[tr], y[tr]):
            inner_pipe = make_firth_pipe(features)
            inner_pipe.fit(X.iloc[tr].iloc[itr], y[tr][itr])
            platt_x.extend(inner_pipe.predict_proba(X.iloc[tr].iloc[ite])[:, 1])
            platt_y.extend(y[tr][ite])
        platt = LogisticRegression(max_iter=500)
        platt.fit(np.array(platt_x).reshape(-1, 1), np.array(platt_y))
        p_te_platt = platt.predict_proba(p_te_raw.reshape(-1, 1))[:, 1]

        p_raw_acc[te]   += p_te_raw
        p_platt_acc[te] += p_te_platt
        n_acc[te]       += 1

        if k % 5 == 0:
            print(f"    fold {k}/{total_folds} done")
    return p_raw_acc / np.maximum(n_acc, 1), p_platt_acc / np.maximum(n_acc, 1)


def main():
    df = load_bidmc()
    y_all = df[OUTCOME].astype(int).values
    print(f"BIDMC cohort: n={len(df)}, events={int(y_all.sum())}")

    for fset, features in [("postopA", POSTOP_A_FEATURES),
                            ("postopB", POSTOP_B_FEATURES)]:
        print(f"\n[Firth] Fitting on {fset} ({len(features)} features) — "
              f"5×5 repeated stratified CV with nested 3-fold Platt scaling")
        X = df[features].copy()
        p_raw, p_platt = cv_oof_with_platt(X, y_all, features)

        print(f"  AUC (raw):    {roc_auc_score(y_all, p_raw):.4f}")
        print(f"  AUC (Platt):  {roc_auc_score(y_all, p_platt):.4f}")
        print(f"  Brier (raw):  {brier_score_loss(y_all, p_raw):.4f}")
        print(f"  Brier (Platt):{brier_score_loss(y_all, p_platt):.4f}")
        print(f"  P-distribution (Platt): mean={p_platt.mean():.3f}, "
              f"min={p_platt.min():.3f}, max={p_platt.max():.3f}")
        np.savez(CACHE / f"oof_bidmc_{fset}_firth.npz",
                  y=y_all, p=p_platt, p_raw=p_raw)
        print(f"  ✓ Saved cache/oof_bidmc_{fset}_firth.npz")


if __name__ == "__main__":
    main()
