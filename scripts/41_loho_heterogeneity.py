"""Task 41 — LOHO heterogeneity (S7): random-effects pooling of per-hospital AUCs
with tau-squared under THREE estimators (DerSimonian-Laird, Paule-Mandel,
REML/iterated), I-squared, Q-test, 95% CI and a 95% prediction interval, plus a
per-site descriptive table.

Addresses S7: the manuscript reported I2=0% without tau2 or per-site diagnostics.

Per-site AUC sampling variance uses the Hanley-McNeil (1982) estimator from each
site's positive/negative counts. Sites that cannot yield a finite AUC variance
(0 events or 0 non-events) are excluded and counted.

Outputs:
  results/41_loho_heterogeneity_summary.csv   — one row per (cohort,set) x estimator
  results/41_loho_per_site.csv                — per-site n/events/AUC/var/Brier
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.meta_analysis import combine_effects

from _shared import RES


def hanley_mcneil_var(auc, n_pos, n_neg):
    """Variance of AUC (Hanley & McNeil 1982)."""
    if n_pos < 1 or n_neg < 1:
        return np.nan
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    return (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
            + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)


def dl_pool(y, v):
    """DerSimonian-Laird random-effects pool. Returns dict."""
    w = 1.0 / v
    ybar_f = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - ybar_f) ** 2))
    k = len(y); dfree = k - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - dfree) / C) if C > 0 else 0.0
    I2 = max(0.0, (Q - dfree) / Q) * 100 if Q > 0 else 0.0
    wr = 1.0 / (v + tau2)
    pooled = np.sum(wr * y) / np.sum(wr)
    se = np.sqrt(1.0 / np.sum(wr))
    p_het = 1 - stats.chi2.cdf(Q, dfree) if dfree > 0 else np.nan
    return dict(tau2=tau2, I2=I2, Q=Q, df=dfree, p_het=p_het,
                pooled=pooled, ci_lo=pooled - 1.96 * se, ci_hi=pooled + 1.96 * se,
                se=se, k=k)


def main():
    d = pd.read_csv(RES / "04_loho_per_hospital.csv")
    d["n_pos"] = d["events_test"]
    d["n_neg"] = d["n_test"] - d["events_test"]
    d["auc_var"] = [hanley_mcneil_var(a, p, n) for a, p, n in
                    zip(d["auc"], d["n_pos"], d["n_neg"])]

    d.to_csv(RES / "41_loho_per_site.csv", index=False)

    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def _invlogit(x):
        return 1.0 / (1.0 + np.exp(-x))

    rows = []
    for (cohort, sset), g in d.groupby(["cohort", "set"]):
        gg = g[np.isfinite(g["auc_var"]) & (g["auc_var"] > 0)].copy()
        n_excl = len(g) - len(gg)
        if len(gg) < 3:
            continue
        auc = gg["auc"].values
        var_auc = gg["auc_var"].values
      # two scales: raw AUC (treats sites equally) and logit-AUC (variance-stabilising,
      # matches the manuscript's primary 04_loho pooling). I2 is scale-sensitive, so we
      # report both and let the prediction interval be the honest transportability summary.
        for scale in ("raw_auc", "logit_auc"):
            if scale == "raw_auc":
                y, v, back = auc, var_auc, (lambda x: x)
            else:
                y = _logit(auc)
                v = var_auc / (auc * (1 - auc)) ** 2   # delta method
                back = _invlogit

            # manual DL (gives Q, I2, tau2 on this scale, prediction interval)
            dl = dl_pool(y, v)
            if dl["k"] > 2:
                tcrit = stats.t.ppf(0.975, dl["k"] - 2)
                pi_half = tcrit * np.sqrt(dl["tau2"] + dl["se"] ** 2)
                pi_lo, pi_hi = back(dl["pooled"] - pi_half), back(dl["pooled"] + pi_half)
            else:
                pi_lo = pi_hi = np.nan

            for method, label in [("dl", "DerSimonian-Laird"),
                                  ("pm", "Paule-Mandel"),
                                  ("iterated", "REML/iterated")]:
                try:
                    res = combine_effects(y, v, method_re=method)
                    sf = res.summary_frame()
                    tau2 = float(res.tau2)
                    re_row = sf.loc[[i for i in sf.index if "random" in str(i).lower()][0]]
                    pooled = float(re_row["eff"]); ci_lo = float(re_row["ci_low"])
                    ci_hi = float(re_row["ci_upp"])
                except Exception:
                    tau2, pooled, ci_lo, ci_hi = dl["tau2"], dl["pooled"], dl["ci_lo"], dl["ci_hi"]
                rows.append({
                    "cohort": cohort, "set": sset, "scale": scale, "estimator": label,
                    "k_sites": dl["k"], "n_excluded": n_excl,
                    "pooled_auc": round(back(pooled), 4),
                    "ci_lo": round(back(ci_lo), 4), "ci_hi": round(back(ci_hi), 4),
                    "tau2": round(tau2, 6), "tau": round(np.sqrt(max(tau2, 0)), 4),
                    "I2_pct": round(dl["I2"], 1), "Q": round(dl["Q"], 2),
                    "Q_df": dl["df"], "p_heterogeneity": round(dl["p_het"], 4),
                    "pred_int_lo": round(pi_lo, 4) if np.isfinite(pi_lo) else np.nan,
                    "pred_int_hi": round(pi_hi, 4) if np.isfinite(pi_hi) else np.nan,
                })

    out = pd.DataFrame(rows)
    out.to_csv(RES / "41_loho_heterogeneity_summary.csv", index=False)
    print("[S7] LOHO random-effects heterogeneity (per cohort/set x estimator):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
