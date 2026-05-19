"""Task 23 — TabPFN v2 evaluation on BIDMC.

TabPFN (Hollmann et al., Nature 637:319, 2025; doi:10.1038/s41586-024-08328-6)
is a transformer pre-trained on 130M synthetic tabular datasets, designed
specifically for small tabular problems (n<10,000 in v2). It performs
in-context inference without per-task gradient training.

Skeptical-frame: Feb 2026 medRxiv benchmark (doi:10.64898/2026.02.02.26345274v1)
showed TabPFN beat classical baselines in only 16.7% of clinical tasks. Manage
expectations.

Evaluation:
  • TabPFN v2 solo on BIDMC postop_A / postop_B
  • 5×5 repeated stratified CV
  • Paired DeLong test vs BalancedRandomForest baseline
  • Brier, PR-AUC, calibration

Outputs:
  results/23_tabpfn.csv
  figures/23_tabpfn.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier

try:
    from tabpfn import TabPFNClassifier
    HAVE_TABPFN = True
except ImportError as e:
    HAVE_TABPFN = False
    print(f"[WARN] TabPFN not importable: {e}")

from _shared import (load_bidmc, POSTOP_A_FEATURES, POSTOP_B_FEATURES,
                      RES, FIG, CACHE, SEED)

N_SPLITS, N_REPEATS = 5, 5

def make_prep(features):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), features)])

def cv_oof(make_pipe_fn, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    p_acc = np.zeros(len(X)); n_acc = np.zeros(len(X))
    for tr, te in rskf.split(X, y):
        p = make_pipe_fn()
        p.fit(X.iloc[tr], y.iloc[tr])
        prob = p.predict_proba(X.iloc[te])[:, 1]
        p_acc[te] += prob; n_acc[te] += 1
    return p_acc / np.maximum(n_acc, 1)

def bootstrap_auc(y, p, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed); bs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        bs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return float(roc_auc_score(y, p)), float(lo), float(hi)

def delong_test(y, p1, p2):
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2)
    pos = (y == 1); neg = (y == 0)
    m, n = pos.sum(), neg.sum()
    if m < 2 or n < 2: return float("nan"), float("nan")
    def struct(s):
        sp = s[pos]; sn = s[neg]
        V10 = np.array([(np.sum(sn < v) + 0.5 * np.sum(sn == v)) / n for v in sp])
        V01 = np.array([(np.sum(sp > v) + 0.5 * np.sum(sp == v)) / m for v in sn])
        return V10.mean(), V10, V01
    a1, v10_1, v01_1 = struct(p1); a2, v10_2, v01_2 = struct(p2)
    s10 = (np.var(v10_1, ddof=1) + np.var(v10_2, ddof=1)
            - 2 * np.cov(v10_1, v10_2, ddof=1)[0,1])
    s01 = (np.var(v01_1, ddof=1) + np.var(v01_2, ddof=1)
            - 2 * np.cov(v01_1, v01_2, ddof=1)[0,1])
    var_diff = s10 / m + s01 / n
    if var_diff <= 0: return float("nan"), float("nan")
    z = (a1 - a2) / np.sqrt(var_diff)
    from scipy.stats import norm
    return float(z), float(2*(1 - norm.cdf(abs(z))))


def main():
    if not HAVE_TABPFN:
        print("Skipping — TabPFN unavailable.")
        return
    df = load_bidmc(); y = df["seizure"].astype(int)
    print(f"BIDMC: n={len(df)}, events={int(y.sum())}\n", flush=True)
    rows = []

    for fset, features in [("postop_A", POSTOP_A_FEATURES), ("postop_B", POSTOP_B_FEATURES)]:
        X = df[features]
        print(f"══ Feature set: {fset} ({len(features)} features) ══", flush=True)

        # Baseline
        def mk_brf():
            return Pipeline([("prep", make_prep(features)),
                             ("clf", BalancedRandomForestClassifier(
                                 n_estimators=300, min_samples_leaf=2, n_jobs=1,
                                 random_state=SEED))])
        p_brf = cv_oof(mk_brf, X, y)
        a, lo, hi = bootstrap_auc(y.values, p_brf)
        rows.append({"feature_set": fset, "model": "BalancedRF (baseline)",
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "brier": brier_score_loss(y, p_brf),
                     "prauc": average_precision_score(y, p_brf),
                     "delong_p": np.nan})
        print(f"  baseline:  AUC = {a:.3f} ({lo:.3f}-{hi:.3f})", flush=True)

        # TabPFN — runs in-context, no separate fit overhead per fold
        def mk_tabpfn():
            return Pipeline([("prep", make_prep(features)),
                             ("clf", TabPFNClassifier(
                                  random_state=SEED, n_jobs=1,
                                  ignore_pretraining_limits=True))])
        print(f"  TabPFN v2 — 5x5 CV (transformer in-context inference)...", flush=True)
        try:
            p_tab = cv_oof(mk_tabpfn, X, y, n_repeats=N_REPEATS)
            a, lo, hi = bootstrap_auc(y.values, p_tab)
            z, pv = delong_test(y.values, p_tab, p_brf)
            rows.append({"feature_set": fset, "model": "TabPFN v2",
                         "auc": a, "ci_lo": lo, "ci_hi": hi,
                         "brier": brier_score_loss(y, p_tab),
                         "prauc": average_precision_score(y, p_tab),
                         "delong_p": pv})
            print(f"    AUC = {a:.3f} ({lo:.3f}-{hi:.3f})  "
                  f"Brier = {brier_score_loss(y, p_tab):.3f}  DeLong p = {pv:.3f}",
                  flush=True)
            np.savez(CACHE / f"oof_bidmc_{fset}_tabpfn.npz",
                     y=y.values, p_brf=p_brf, p_tab=p_tab)
        except Exception as e:
            print(f"  TabPFN failed: {e}", flush=True)
            rows.append({"feature_set": fset, "model": f"TabPFN v2 (FAILED: {type(e).__name__})",
                         "auc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "brier": np.nan, "prauc": np.nan, "delong_p": np.nan})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RES / "23_tabpfn.csv", index=False)
    print("\n" + "=" * 78)
    print(df_out.round(3).to_string(index=False))

    if df_out["auc"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 4))
        valid = df_out.dropna(subset=["auc"]).reset_index(drop=True)
        pos_y = np.arange(len(valid))
        colors = ["#1f77b4" if "baseline" in m else "#7c3aed" for m in valid["model"]]
        ax.errorbar(valid["auc"], pos_y,
                     xerr=[valid["auc"] - valid["ci_lo"],
                            valid["ci_hi"] - valid["auc"]],
                     fmt="o", capsize=4, color="black", ecolor="gray")
        for i, c in enumerate(colors):
            ax.scatter(valid["auc"].iloc[i], pos_y[i], color=c, s=80, zorder=3)
        ax.set_yticks(pos_y)
        ax.set_yticklabels([f'{r.feature_set}  ·  {r.model}' for r in valid.itertuples()])
        ax.invert_yaxis(); ax.axvline(0.5, color="gray", ls=":")
        ax.set_xlim(0.45, 0.85)
        ax.set_xlabel("Cross-validated AUC (bootstrap 95% CI)")
        ax.set_title("TabPFN v2 vs BalancedRF baseline — BIDMC")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "23_tabpfn.png", dpi=200, bbox_inches="tight")
        plt.savefig(FIG / "23_tabpfn.pdf", bbox_inches="tight")
        plt.close()
    print(f"\n[OK] results/23_tabpfn.csv  figures/23_tabpfn.{{png,pdf}}")


if __name__ == "__main__":
    main()
