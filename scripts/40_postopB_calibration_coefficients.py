"""Task 40 — Deployed postop-B Firth model: calibration metrics (S6) + coefficient
table (S12) + calibration reliability plot (Supplementary Figure S8).

Addresses revision items S6 (post-Platt calibration metrics for the *deployed* model,
not the BalancedRandomForest conformal base) and S12 (Firth coefficients with SE/CI/p
and a calibration plot for the deployable postop-B model).

Outputs:
  results/40_postopB_calibration_post_platt.csv   — Brier/CITL/slope/intercept/ECE/HL
  results/40_postopB_firth_coefficients.csv        — coef, SE, 95% CI, p, OR per SD
  figures/40_postopB_calibration.{png,pdf}
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from firthlogist import FirthLogisticRegression

from _shared import (CACHE, RES, FIG, SEED, load_bidmc,
                     POSTOP_B_FEATURES, calibration_metrics)

OUTCOME = "seizure"


def _firth_pipe(features):
    pre = ColumnTransformer([("num", Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), features)])
    return Pipeline([("pre", pre), ("clf", FirthLogisticRegression(
        max_iter=200, skip_pvals=True, skip_ci=True, wald=True))])


def per_repeat_calibration(X, y, features, n_splits=5, n_repeats=5):
    """Assess calibration the methodologically correct way for repeated CV:
    within EACH repeat every patient gets exactly one nested-Platt OOF prediction;
    compute calibration on that complete partition; average metrics across repeats.
    (Averaging probabilities across repeats first compresses them and corrupts the
    calibration slope — that is the artifact behind any spurious 'slope ~1' claim.)"""
    rows = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + rep)
        p = np.zeros(len(y))
        for k, (tr, te) in enumerate(skf.split(X, y), 1):
            o = _firth_pipe(features); o.fit(X.iloc[tr], y[tr])
            praw = o.predict_proba(X.iloc[te])[:, 1]
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 100 * rep + k)
            px, py = [], []
            for itr, ite in inner.split(X.iloc[tr], y[tr]):
                ip = _firth_pipe(features); ip.fit(X.iloc[tr].iloc[itr], y[tr][itr])
                px.extend(ip.predict_proba(X.iloc[tr].iloc[ite])[:, 1]); py.extend(y[tr][ite])
            pl = LogisticRegression(max_iter=500)
            pl.fit(np.array(px).reshape(-1, 1), np.array(py))
            p[te] = pl.predict_proba(praw.reshape(-1, 1))[:, 1]
        m = calibration_metrics(y, p)
        rows.append({"brier": m["brier"], "citl": m["citl"], "slope": m["slope"],
                     "intercept": m["intercept"], "ece": m["ece"],
                     "auc": roc_auc_score(y, p)})
    df = pd.DataFrame(rows)
    return df.mean(), df.std()


def coefficient_table(df):
    """Fit Firth on median-imputed, standardized postop-B and tabulate coefficients.
    Reported on the deployment (per-SD) scale; OR = exp(coef) is the odds ratio per
    1-SD increase. Firth penalty corrects small-sample bias / separation."""
    X = df[POSTOP_B_FEATURES].copy()
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xz = sc.fit_transform(imp.fit_transform(X))
    y = df[OUTCOME].astype(int).values
    f = FirthLogisticRegression(max_iter=500, wald=True)
    f.fit(Xz, y)
    ci = np.asarray(f.ci_)               # (n_features, 2) on the log-odds scale
    rows = []
    for i, name in enumerate(POSTOP_B_FEATURES):
        lo, hi = ci[i, 0], ci[i, 1]
        rows.append({
            "feature": name,
            "coef": round(float(f.coef_[i]), 4),
            "se": round(float(f.bse_[i]), 4),
            "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4),
            "p_value": round(float(f.pvals_[i]), 4),
            "OR_per_SD": round(float(np.exp(f.coef_[i])), 3),
            "OR_lo": round(float(np.exp(lo)), 3),
            "OR_hi": round(float(np.exp(hi)), 3),
        })
    out = pd.DataFrame(rows)
    out.attrs["intercept"] = float(f.intercept_)
    return out


def main():
    df = load_bidmc()

    # ── S12: coefficient table ──
    coefs = coefficient_table(df)
    coefs.to_csv(RES / "40_postopB_firth_coefficients.csv", index=False)
    print("[S12] Firth coefficients (per-SD), top by |coef|:")
    print(coefs.reindex(coefs.coef.abs().sort_values(ascending=False).index)
          .head(8).to_string(index=False))

    # ── S6: post-Platt calibration — per-repeat (correct) AND averaged-cache (artifact) ──
    X = df[POSTOP_B_FEATURES].copy()
    y = df[OUTCOME].astype(int).values
    mean_m, sd_m = per_repeat_calibration(X, y, POSTOP_B_FEATURES)

    z = np.load(CACHE / "oof_bidmc_postopB_firth.npz")
    yc, pc = z["y"].astype(int), z["p"].astype(float)
    mc = calibration_metrics(yc, pc)

    cal = pd.DataFrame([
        {"method": "per_repeat_mean (correct)", "n": len(y), "events": int(y.sum()),
         "brier": round(mean_m["brier"], 4), "citl": round(mean_m["citl"], 4),
         "intercept": round(mean_m["intercept"], 4), "slope": round(mean_m["slope"], 4),
         "ece": round(mean_m["ece"], 4), "auc": round(mean_m["auc"], 4)},
        {"method": "per_repeat_sd", "n": len(y), "events": int(y.sum()),
         "brier": round(sd_m["brier"], 4), "citl": round(sd_m["citl"], 4),
         "intercept": round(sd_m["intercept"], 4), "slope": round(sd_m["slope"], 4),
         "ece": round(sd_m["ece"], 4), "auc": round(sd_m["auc"], 4)},
        {"method": "averaged_cache (compressed artifact)", "n": len(yc), "events": int(yc.sum()),
         "brier": round(mc["brier"], 4), "citl": round(mc["citl"], 4),
         "intercept": round(mc["intercept"], 4), "slope": round(mc["slope"], 4),
         "ece": round(mc["ece"], 4), "auc": round(roc_auc_score(yc, pc), 4)},
    ])
    cal.to_csv(RES / "40_postopB_calibration_post_platt.csv", index=False)
    print("\n[S6] Post-Platt calibration (deployed postop-B Firth):")
    print(cal.to_string(index=False))
    p = pc  # use cache predictions for the reliability-curve shape

    # ── S12 figure: reliability curve ──
    y = yc
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, max(mean_pred.max(), frac_pos.max()) * 1.1] * 1, [0, max(mean_pred.max(), frac_pos.max()) * 1.1], ls=":", color="gray", label="ideal")
    ax.plot(mean_pred, frac_pos, "o-", color="#b5482a", label="postop-B Firth (Platt)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Deployed postop-B calibration (per-repeat)\n"
                 f"Brier {mean_m['brier']:.3f} · slope {mean_m['slope']:.2f} · "
                 f"CITL {mean_m['citl']:+.3f}")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "40_postopB_calibration.png", dpi=200)
    fig.savefig(FIG / "40_postopB_calibration.pdf")
    plt.close(fig)
    print(f"\n[✓] Wrote figures/40_postopB_calibration.{{png,pdf}}")


if __name__ == "__main__":
    main()
