"""Task 2 — Calibration analysis (BIDMC + eICU).

Outputs:
  results/02_calibration_metrics.csv
  figures/02_calibration_curves.png
  cache/oof_<model>.npz   (reused by DCA / temporal-leakage / etc.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _shared import (
    load_bidmc, load_eicu, load_eicu_pure,
    POSTOP_A_FEATURES, POSTOP_B_FEATURES, PREOP_FEATURES,
    EICU_SET_A, EICU_SET_B, EICU_SET_C,
    make_pipeline_postopA, make_pipeline_postopB, make_pipeline_preop,
    make_pipeline_eicu, oof_predictions, calibration_metrics,
    RES, FIG, CACHE, SEED,
)

def bootstrap_calibration_ci(y, p, n_boot=1000, seed=42):
    """Fix F: paired-bootstrap 95% CI for Brier, CITL, slope/intercept, AUC."""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    n = len(y); collect = {k: [] for k in ("brier", "citl", "slope", "intercept", "auc")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2: continue
        # Brier
        collect["brier"].append(brier_score_loss(yb, pb))
        # CITL = observed - mean predicted
        collect["citl"].append(yb.mean() - pb.mean())
        # AUC
        collect["auc"].append(roc_auc_score(yb, pb))
        # slope / intercept via logit-regression of outcome on logit(pred)
        p_safe = np.clip(pb, 1e-6, 1 - 1e-6)
        x_logit = np.log(p_safe / (1 - p_safe)).reshape(-1, 1)
        try:
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(x_logit, yb)
            collect["intercept"].append(float(lr.intercept_[0]))
            collect["slope"].append(float(lr.coef_[0, 0]))
        except Exception:
            pass
    ci = {}
    for k, v in collect.items():
        if v:
            lo, hi = np.percentile(v, [2.5, 97.5])
            ci[f"{k}_lo"] = float(lo); ci[f"{k}_hi"] = float(hi)
        else:
            ci[f"{k}_lo"] = float("nan"); ci[f"{k}_hi"] = float("nan")
    return ci

def run_one(name, make_pipe, X, y, n_splits=5, n_repeats=5):
    print(f"  {name}: n={len(y)}, events={int(y.sum())}, splits={n_splits}x{n_repeats}", flush=True)
    p = oof_predictions(make_pipe, X, y, n_splits=n_splits, n_repeats=n_repeats)
    np.savez(CACHE / f"oof_{name}.npz", y=y.values, p=p)
    m = calibration_metrics(y.values, p)
    m["model"] = name
    m["n"] = int(len(y))
    m["events"] = int(y.sum())
    # Fix F: bootstrap 95% CIs on the core calibration metrics
    m.update(bootstrap_calibration_ci(y.values, p, n_boot=1000, seed=42))
    return m, p

def main():
    rows = []
    cohorts = {}

    # BIDMC ──────────────────────────────────────────────
    df = load_bidmc()
    y = df["seizure"].astype(int)
    print(f"\nBIDMC n={len(df)}, events={y.sum()}")
    m, p = run_one("bidmc_postopA",
                    make_pipeline_postopA,
                    df[POSTOP_A_FEATURES], y, 5, 5)
    rows.append(m); cohorts["bidmc_postopA"] = (y.values, p)

    m, p = run_one("bidmc_postopB",
                    make_pipeline_postopB,
                    df[POSTOP_B_FEATURES], y, 5, 5)
    rows.append(m); cohorts["bidmc_postopB"] = (y.values, p)

    m, p = run_one("bidmc_preop",
                    make_pipeline_preop,
                    df[PREOP_FEATURES], y, 5, 5)
    rows.append(m); cohorts["bidmc_preop"] = (y.values, p)

    # eICU ──────────────────────────────────────────────
    df_e = load_eicu()  # full (filtering pre-ICU seizures only)
    y_e = df_e["seizure"].astype(int)
    print(f"\neICU full n={len(df_e)}, events={y_e.sum()}")
    m, p = run_one("eicu_setA",
                    lambda: make_pipeline_eicu(EICU_SET_A, "rf_balanced"),
                    df_e[EICU_SET_A], y_e, 5, 5)
    rows.append(m); cohorts["eicu_setA"] = (y_e.values, p)

    # Set C is heavy — use 5x3 = 15 folds
    m, p = run_one("eicu_setC",
                    lambda: make_pipeline_eicu(EICU_SET_C, "rf_balanced"),
                    df_e[EICU_SET_C], y_e, 5, 3)
    rows.append(m); cohorts["eicu_setC"] = (y_e.values, p)

    # Pure cohort ──────────────────────────────────────
    df_p = load_eicu_pure()
    y_p = df_p["seizure"].astype(int)
    print(f"\neICU pure n={len(df_p)}, events={y_p.sum()}")
    m, p = run_one("eicu_pure_setC",
                    lambda: make_pipeline_eicu(EICU_SET_C, "rf_balanced"),
                    df_p[EICU_SET_C], y_p, 5, 3)
    rows.append(m); cohorts["eicu_pure_setC"] = (y_p.values, p)

    # ──────────────────────────────────────────────────
    metrics_df = pd.DataFrame(rows)[
        ["model", "n", "events",
         "brier", "brier_lo", "brier_hi",
         "citl",  "citl_lo",  "citl_hi",
         "intercept", "intercept_lo", "intercept_hi",
         "slope",     "slope_lo",     "slope_hi",
         "auc_lo", "auc_hi",
         "ece", "mce", "hl_chi2", "hl_p"]
    ]
    metrics_df.to_csv(RES / "02_calibration_metrics.csv", index=False)
    print("\n", metrics_df.round(4).to_string(index=False))

    # save bin observations as JSON for plotting
    bins = {r["model"]: {"obs": r["bin_obs"], "pred": r["bin_pred"]} for r in rows}
    (RES / "02_calibration_bins.json").write_text(json.dumps(bins, indent=2))

    # ── Calibration plot (3x2 grid) ─────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(11, 14), sharex=True, sharey=True)
    axes = axes.ravel()
    titles = {
        "bidmc_postopA": "BIDMC postop A (BRF, 21 features)",
        "bidmc_postopB": "BIDMC postop B (LR-EN+SMOTE, 18 features)",
        "bidmc_preop":   "BIDMC preop (RF, 14 features)",
        "eicu_setA":     "eICU Set A (8 features)",
        "eicu_setC":     "eICU Set C (103 features)",
        "eicu_pure_setC":"eICU pure Set C (no prior sz/AED/MV)",
    }
    for ax, (k, (y_, p_)) in zip(axes, cohorts.items()):
        # quantile bins
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_, p_, n_bins=10, strategy="quantile")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(prob_pred, prob_true, "o-", lw=2, ms=6)
        # secondary axis: histogram of predicted probs
        ax2 = ax.twinx()
        ax2.hist(p_, bins=30, alpha=0.2, color="gray")
        ax2.set_yticks([])
        m = next(r for r in rows if r["model"] == k)
        ax.set_title(f"{titles[k]}\nBrier={m['brier']:.3f}  slope={m['slope']:.2f}  intercept={m['intercept']:.2f}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.set_xlim(0, max(0.6, p_.max() * 1.05))
        ax.set_ylim(0, max(0.6, prob_true.max() * 1.05))
    plt.tight_layout()
    plt.savefig(FIG / "02_calibration_curves.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "02_calibration_curves.pdf", bbox_inches="tight")
    plt.close()
    print("\n[OK] Saved: results/02_calibration_metrics.csv  figures/02_calibration_curves.{png,pdf}")

if __name__ == "__main__":
    main()
