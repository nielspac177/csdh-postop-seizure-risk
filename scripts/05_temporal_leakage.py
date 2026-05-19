"""Task 5 — Temporal-leakage audit.

For BIDMC: confirm that postop_B (without aed_timing, prop_aed, ab_eeg) gives
unbiased estimates; quantify the AUC drop from postop_A.

For eICU:
  (a) Quantify how many "seizure" cases have features extracted AFTER seizure onset
      (those with seizure_offset_min < 24h × 60 = 1440 min are exposed via *_24h /
       *_48h / *_first / *_mean features).
  (b) Re-fit Set C with only "strictly pre-seizure" features (drop *_24h, *_48h,
       *_mean, *_min/_max/_std, prophylactic_aed) and compare to original.
  (c) Re-fit Set C on cohort restricted to seizure_offset_min >= 1440 min
      OR seizure==0 (i.e., "late-seizure-only" comparator).

Outputs:
  results/05_leakage_audit.csv
  figures/05_auc_comparison.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from _shared import (
    load_bidmc, load_eicu, load_eicu_pure,
    POSTOP_A_FEATURES, POSTOP_B_FEATURES,
    EICU_SET_A, EICU_SET_C,
    make_pipeline_postopA, make_pipeline_postopB, make_pipeline_eicu,
    oof_predictions, RES, FIG, CACHE,
)

def auc_with_ci(y, p, n_boot=500, seed=42):
    a = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        b.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(b, [2.5, 97.5]) if b else (np.nan, np.nan)
    return a, lo, hi

def is_leakage_suspect_feature(c):
    if c.endswith("_24h") or c.endswith("_48h") or c.endswith("_mean"):
        return True
    if c.endswith("_first") or c.endswith("_min") or c.endswith("_max") or c.endswith("_std"):
        return True
    if c in ("prophylactic_aed", "any_aed", "n_aed_orders", "first_aed_offset",
             "n_labs_24h", "n_vitals_24h", "icp_available", "blood_transfusion",
             "icp_monitor", "mechanical_ventilation"):
        return True
    return False

def main():
    rows = []

    # ── BIDMC ──────────────────────────────────────────
    df = load_bidmc()
    y = df["seizure"].astype(int)

    # Use cached OOF if available
    cA = CACHE / "oof_bidmc_postopA.npz"
    cB = CACHE / "oof_bidmc_postopB.npz"
    if cA.exists():
        z = np.load(cA); pA = z["p"]
    else:
        pA = oof_predictions(make_pipeline_postopA, df[POSTOP_A_FEATURES], y, 5, 5)
    if cB.exists():
        z = np.load(cB); pB = z["p"]
    else:
        pB = oof_predictions(make_pipeline_postopB, df[POSTOP_B_FEATURES], y, 5, 5)

    a, lo, hi = auc_with_ci(y.values, pA)
    rows.append({"cohort":"BIDMC", "spec":"postop_A (with aed_timing+prop_aed+ab_eeg)",
                 "n":len(y), "events":int(y.sum()), "auc":a, "lo":lo, "hi":hi,
                 "n_features":len(POSTOP_A_FEATURES)})
    a, lo, hi = auc_with_ci(y.values, pB)
    rows.append({"cohort":"BIDMC", "spec":"postop_B (no AED-timing leakage)",
                 "n":len(y), "events":int(y.sum()), "auc":a, "lo":lo, "hi":hi,
                 "n_features":len(POSTOP_B_FEATURES)})

    # Fix G note: postop_gcs is captured at OR exit, BEFORE any postoperative
    # seizure can occur — it is not temporal leakage. The earlier "strict" run
    # removing postop_gcs was over-aggressive and is dropped here.

    # ── eICU full + Set C ─────────────────────────────
    df_e = load_eicu()
    y_e = df_e["seizure"].astype(int)
    n_total = len(df_e)
    n_sz = int(y_e.sum())
    sz_off = df_e["seizure_offset_min"]
    n_early = int(((y_e == 1) & (sz_off.fillna(np.inf) < 1440)).sum())
    n_very_early = int(((y_e == 1) & (sz_off.fillna(np.inf) < 60)).sum())
    print(f"eICU n={n_total}, sz={n_sz}; sz with onset <24h: {n_early} ({n_early/n_sz:.1%}); "
          f"<1h: {n_very_early}", flush=True)

    cC = CACHE / "oof_eicu_setC.npz"
    if cC.exists():
        z = np.load(cC); pC = z["p"]
    else:
        pC = oof_predictions(lambda: make_pipeline_eicu(EICU_SET_C, "rf_balanced"),
                              df_e[EICU_SET_C], y_e, 5, 3)
    a, lo, hi = auc_with_ci(y_e.values, pC)
    rows.append({"cohort":"eICU full", "spec":"Set C (103 features, original)",
                 "n":n_total, "events":n_sz, "auc":a, "lo":lo, "hi":hi,
                 "n_features":len(EICU_SET_C)})

    # Strict: drop time-window-derived features
    SC_STRICT = [c for c in EICU_SET_C if not is_leakage_suspect_feature(c)]
    print(f"eICU strict pre-seizure features: {len(SC_STRICT)} kept of {len(EICU_SET_C)}", flush=True)
    print(f"  kept: {SC_STRICT}", flush=True)
    pCstrict = oof_predictions(
        lambda: make_pipeline_eicu(SC_STRICT, "rf_balanced"),
        df_e[SC_STRICT], y_e, 5, 3,
    )
    a, lo, hi = auc_with_ci(y_e.values, pCstrict)
    rows.append({"cohort":"eICU full", "spec":"Set C strict (pre-seizure features only)",
                 "n":n_total, "events":n_sz, "auc":a, "lo":lo, "hi":hi,
                 "n_features":len(SC_STRICT)})
    np.savez(CACHE / "oof_eicu_setC_strict.npz", y=y_e.values, p=pCstrict)

    # Multiple time-window cohorts (Fix B):
    # 0-72h is clinically the standard early-postop seizure window.
    # ≥24h cut keeps too few events; ≥72h keeps almost none.
    for cut_min, cut_label in [(60, "≥1h"), (1440, "≥24h"), (4320, "≥72h")]:
        keep = (y_e == 0) | (sz_off.fillna(np.inf) >= cut_min)
        df_x = df_e.loc[keep].reset_index(drop=True)
        y_x = df_x["seizure"].astype(int)
        if y_x.sum() < 20:
            print(f"  [{cut_label}] only {y_x.sum()} events — skipping CV", flush=True)
            rows.append({"cohort":f"eICU exclude sz<{cut_label}",
                         "spec":"Set C original", "n":len(df_x),
                         "events":int(y_x.sum()), "auc":float("nan"),
                         "lo":float("nan"), "hi":float("nan"),
                         "n_features":len(EICU_SET_C)})
            continue
        print(f"  [{cut_label}] n={len(df_x)}, sz={y_x.sum()}", flush=True)
        p_x = oof_predictions(
            lambda: make_pipeline_eicu(EICU_SET_C, "rf_balanced"),
            df_x[EICU_SET_C], y_x, 5, 3,
        )
        a, lo, hi = auc_with_ci(y_x.values, p_x)
        rows.append({"cohort":f"eICU exclude sz<{cut_label}",
                     "spec":"Set C original", "n":len(df_x),
                     "events":int(y_x.sum()), "auc":a, "lo":lo, "hi":hi,
                     "n_features":len(EICU_SET_C)})

    # Clinically meaningful 0-72h "early postop seizure" cohort (positive case = sz within 72h):
    early_window = ((y_e == 1) & (sz_off.fillna(np.inf) <= 4320)) | (y_e == 0)
    df_pre72 = df_e.loc[early_window].reset_index(drop=True)
    y_pre72 = (df_pre72["seizure"].astype(int) &
                 (df_pre72["seizure_offset_min"].fillna(np.inf) <= 4320)).astype(int)
    if y_pre72.sum() >= 20:
        print(f"  [0-72h target] n={len(df_pre72)}, sz_in_window={y_pre72.sum()}", flush=True)
        p_pre = oof_predictions(
            lambda: make_pipeline_eicu(SC_STRICT, "rf_balanced"),
            df_pre72[SC_STRICT], y_pre72, 5, 3,
        )
        a, lo, hi = auc_with_ci(y_pre72.values, p_pre)
        rows.append({"cohort":"eICU 0-72h target (strict feats)",
                     "spec":"Set C strict, 0-72h sz outcome",
                     "n":len(df_pre72), "events":int(y_pre72.sum()),
                     "auc":a, "lo":lo, "hi":hi, "n_features":len(SC_STRICT)})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RES / "05_leakage_audit.csv", index=False)
    print("\n", df_out.round(3).to_string(index=False))

    # plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pos = np.arange(len(df_out))
    ax.errorbar(df_out["auc"], pos,
                xerr=[df_out["auc"] - df_out["lo"], df_out["hi"] - df_out["auc"]],
                fmt="o", capsize=4)
    ax.set_yticks(pos)
    ax.set_yticklabels([f'{r["cohort"]}: {r["spec"]}' for _, r in df_out.iterrows()])
    ax.invert_yaxis()
    ax.axvline(0.5, color="gray", ls=":")
    ax.set_xlabel("AUC (95% bootstrap CI)")
    ax.set_xlim(0.4, 0.9)
    ax.set_title("Temporal-leakage audit — AUC across feature specifications")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "05_auc_comparison.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "05_auc_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("\n[OK] Saved: results/05_leakage_audit.csv  figures/05_auc_comparison.{png,pdf}")

if __name__ == "__main__":
    main()
