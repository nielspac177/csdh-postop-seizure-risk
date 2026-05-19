"""Task 8 — eICU cohort re-definition (chronic vs acute / non-op vs postop).

Compare AUC across cohort definitions:
  Cohort 1: paper "non_traumatic" (n≈3300)         — current target
  Cohort 2: non_traumatic + operative (cran/burr)  — strict postop cSDH
  Cohort 3: apacheadmissiondx contains "surgery for" or "Burr hole placement"
  Cohort 4: traumatic-SDH (negative control — should differ)

Outputs:
  results/08_cohort_comparison.csv
  figures/08_cohort_auc.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from _shared import (
    load_eicu, EICU_SET_A, EICU_SET_C,
    make_pipeline_eicu, RES, FIG, SEED,
)

def cv_auc(make_fn, X, y, n_splits=5, n_repeats=3):
    """Repeated CV AUC + bootstrap 95% CI on first-repeat OOF predictions.

    Returns (mean_auc, sd_across_folds, ci_lo, ci_hi).  CI is from 1000-iteration
    bootstrap on the OOF probabilities of one stratified pass so it reflects
    the true sample-size-driven uncertainty in small cohorts (Fix C).
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    aucs = []
    for tr, te in rskf.split(X, y):
        if y.iloc[tr].sum() < 5: continue
        p = make_fn()
        p.fit(X.iloc[tr], y.iloc[tr])
        prob = p.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], prob))
    if not aucs:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # OOF predictions for bootstrap CI (single non-repeated pass for stability)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        if y.iloc[tr].sum() < 5: continue
        p = make_fn()
        p.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = p.predict_proba(X.iloc[te])[:, 1]
    mask = ~np.isnan(oof)
    y_arr = y.values[mask]; p_arr = oof[mask]
    rng = np.random.default_rng(SEED)
    boot_aucs = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if len(np.unique(y_arr[idx])) < 2: continue
        boot_aucs.append(roc_auc_score(y_arr[idx], p_arr[idx]))
    lo, hi = (np.percentile(boot_aucs, [2.5, 97.5])
              if boot_aucs else (float("nan"), float("nan")))
    return float(np.mean(aucs)), float(np.std(aucs)), float(lo), float(hi)

def main():
    full = load_eicu()
    print(f"Full eICU: n={len(full)}, sz={full['seizure'].sum()}", flush=True)

    cohorts = {}
    cohorts["1: paper (non_traumatic)"] = full[full["etiology"] == "non_traumatic"].reset_index(drop=True)
    cohorts["2: non_traumatic + op"]    = full[
        (full["etiology"] == "non_traumatic") &
        ((full["craniotomy"] == 1) | (full["burr_hole"] == 1))
    ].reset_index(drop=True)
    apdx = full["apacheadmissiondx"].fillna("")
    cohorts["3: apache dx surgery/burr"] = full[
        apdx.str.contains("surgery for|Burr hole", case=False, regex=True)
    ].reset_index(drop=True)
    cohorts["4: traumatic SDH (control)"] = full[full["etiology"] == "traumatic"].reset_index(drop=True)

    rows = []
    for name, c in cohorts.items():
        n = len(c); ev = int(c["seizure"].sum())
        print(f"\n[{name}] n={n}, sz={ev} ({ev/n*100:.1f}%)", flush=True)
        if n == 0 or ev < 10:
            print("  too small — skipping CV", flush=True)
            rows.append({"cohort":name, "set":"-", "n":n, "events":ev,
                         "auc_mean":None, "auc_sd":None,
                         "ci_lo":None, "ci_hi":None})
            continue
        for set_name, feats in [("Set_A", EICU_SET_A), ("Set_C", EICU_SET_C)]:
            X = c[feats]; y = c["seizure"].astype(int)
            m, s, lo, hi = cv_auc(lambda: make_pipeline_eicu(feats, "rf_balanced"), X, y)
            rows.append({"cohort":name, "set":set_name, "n":n, "events":ev,
                         "auc_mean":m, "auc_sd":s, "ci_lo":lo, "ci_hi":hi})
            print(f"  {set_name}: AUC = {m:.3f} ± {s:.3f}  "
                  f"(boot 95% CI {lo:.3f}-{hi:.3f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "08_cohort_comparison.csv", index=False)
    print("\n", df.round(3).to_string(index=False))

    # plot — bootstrap 95% CI (Fix C) rather than fold-SD
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = df.dropna(subset=["auc_mean"]).reset_index(drop=True)
    pos = np.arange(len(plot_df))
    xerr_lo = plot_df["auc_mean"] - plot_df["ci_lo"]
    xerr_hi = plot_df["ci_hi"] - plot_df["auc_mean"]
    ax.errorbar(plot_df["auc_mean"], pos,
                xerr=[xerr_lo, xerr_hi], fmt="o", capsize=4,
                color="tab:blue", ecolor="gray")
    ax.set_yticks(pos)
    ax.set_yticklabels([f'{r.cohort} | {r.set} (n={r.n}, ev={r.events})'
                        for r in plot_df.itertuples()])
    ax.invert_yaxis()
    ax.axvline(0.5, color="gray", ls=":")
    ax.set_xlim(0.35, 0.9)
    ax.set_xlabel("Cross-validated AUC (bootstrap 95% CI)")
    ax.set_title("eICU cohort definition sensitivity")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "08_cohort_auc.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "08_cohort_auc.pdf", bbox_inches="tight")
    plt.close()
    print("\n[OK] Saved: results/08_cohort_comparison.csv  figures/08_cohort_auc.{png,pdf}")

if __name__ == "__main__":
    main()
