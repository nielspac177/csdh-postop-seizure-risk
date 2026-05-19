"""Task 19 — Transfer learning: eICU → BIDMC.

Hypothesis: BIDMC has only 48 seizure events. eICU has 461 events across
overlapping features. We can leverage eICU as auxiliary signal via feature
augmentation: train an eICU model on shared features, score each BIDMC
patient with it, and add that score as a new feature to the BIDMC model.

Shared feature mapping (BIDMC → eICU column name):
  age          → age
  sex          → sex
  preop_gcs    → gcs_admission
  epilepsy_hx  → prior_seizures
  prop_aed     → prophylactic_aed

We also engineer a "surgery type" indicator from BIDMC's surg_decompression /
drainage / mma_embo flags into a binary craniotomy-or-burrhole flag matching
eICU.

Evaluation:
  • Compare AUC (postop_A + eicu_transfer_pred) vs (postop_A alone) via
    cross-validated DeLong test
  • Bootstrap 95% CIs and Brier
  • Report calibration after recalibration

Outputs:
  results/19_transfer_learning.csv
  figures/19_transfer_learning.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb

from _shared import (
    load_bidmc, load_eicu,
    POSTOP_A_FEATURES, POSTOP_B_FEATURES,
    RES, FIG, CACHE, SEED,
)

# ── Shared features for the transfer model (eICU side) ──────────
TRANSFER_FEATURES_EICU = ["age", "sex", "gcs_admission", "prior_seizures",
                          "prophylactic_aed", "craniotomy", "burr_hole"]
# ── BIDMC mapping
def build_bidmc_transfer_X(df_b):
    """Build BIDMC view that matches eICU TRANSFER_FEATURES."""
    df = pd.DataFrame(index=df_b.index)
    df["age"] = df_b["age"].astype(float)
    df["sex"] = df_b["sex"].astype(float)
    df["gcs_admission"] = df_b["preop_gcs"].astype(float)
    df["prior_seizures"] = df_b["epilepsy_hx"].astype(float)
    df["prophylactic_aed"] = df_b["prop_aed"].fillna(0).astype(float)
    # BIDMC surgery indicators (1 if any of these procedures):
    df["craniotomy"] = ((df_b["surg_decompression"] == 1) |
                         (df_b["mma_embo"] == 1)).astype(int)
    df["burr_hole"] = df_b["drainage"].fillna(0).astype(int)
    return df

# ── DeLong test for paired AUC comparison ─────────────────────
def delong_test(y, p1, p2):
    """Returns z-stat and two-sided p-value for paired DeLong (Sun & Xu 2014)."""
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2)
    pos = (y == 1); neg = (y == 0)
    m, n = pos.sum(), neg.sum()
    if m < 2 or n < 2: return float("nan"), float("nan")
    def aucs_and_struct(scores):
        s_pos = scores[pos]; s_neg = scores[neg]
        # placement values
        V10 = np.array([(np.sum(s_neg < s) + 0.5 * np.sum(s_neg == s)) / n for s in s_pos])
        V01 = np.array([(np.sum(s_pos > s) + 0.5 * np.sum(s_pos == s)) / m for s in s_neg])
        auc = V10.mean()
        return auc, V10, V01
    a1, V10_1, V01_1 = aucs_and_struct(p1)
    a2, V10_2, V01_2 = aucs_and_struct(p2)
    s10_11 = np.var(V10_1, ddof=1)
    s10_22 = np.var(V10_2, ddof=1)
    s10_12 = np.cov(V10_1, V10_2, ddof=1)[0, 1]
    s01_11 = np.var(V01_1, ddof=1)
    s01_22 = np.var(V01_2, ddof=1)
    s01_12 = np.cov(V01_1, V01_2, ddof=1)[0, 1]
    var_diff = (s10_11 + s10_22 - 2 * s10_12) / m + (s01_11 + s01_22 - 2 * s01_12) / n
    if var_diff <= 0: return float("nan"), float("nan")
    z = (a1 - a2) / np.sqrt(var_diff)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)

# ── Pipelines ───────────────────────────────────────────────────
def make_brf_pipe(features):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), features)])
    clf = BalancedRandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)])

def make_xgb_pipe(features, scale_pos_weight=1.0):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), features)])
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight, tree_method="hist",
        n_jobs=1, random_state=SEED, verbosity=0, eval_metric="auc")
    return Pipeline([("prep", prep), ("clf", clf)])

def cv_oof(make_pipe_fn, X, y, n_splits=5, n_repeats=5):
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

def main():
    # ── Step 1: Train transfer model on eICU ─────────────────────
    eicu = load_eicu()
    print(f"eICU full: n={len(eicu)}, events={eicu['seizure'].sum()}", flush=True)
    Xe = eicu[TRANSFER_FEATURES_EICU].copy()
    ye = eicu["seizure"].astype(int)
    pos = float(ye.sum()); neg = float((1 - ye).sum())

    print(f"\nTraining XGBoost transfer model on eICU (5x3 CV, {len(TRANSFER_FEATURES_EICU)} shared features)...", flush=True)
    transfer_model_xgb = make_xgb_pipe(TRANSFER_FEATURES_EICU,
                                         scale_pos_weight=neg/pos)
    # OOF eval on eICU for diagnostic
    p_eicu = cv_oof(lambda: make_xgb_pipe(TRANSFER_FEATURES_EICU,
                                            scale_pos_weight=neg/pos),
                     Xe, ye, n_splits=5, n_repeats=3)
    auc_eicu, lo, hi = bootstrap_auc(ye.values, p_eicu)
    print(f"  eICU transfer model AUC = {auc_eicu:.3f} ({lo:.3f}-{hi:.3f})", flush=True)

    # Fit on ALL eICU data, save model for BIDMC scoring
    transfer_model_xgb.fit(Xe, ye)

    # ── Step 2: Apply transfer model to BIDMC ────────────────────
    bidmc = load_bidmc()
    yb = bidmc["seizure"].astype(int)
    Xb_transfer = build_bidmc_transfer_X(bidmc)
    print(f"\nBIDMC: n={len(bidmc)}, events={yb.sum()}", flush=True)
    bidmc["eicu_transfer_pred"] = transfer_model_xgb.predict_proba(
        Xb_transfer[TRANSFER_FEATURES_EICU])[:, 1]
    print(f"  Transfer-pred distribution: mean={bidmc['eicu_transfer_pred'].mean():.3f}, "
          f"correlation with outcome r={np.corrcoef(bidmc['eicu_transfer_pred'], yb)[0,1]:.3f}",
          flush=True)

    # ── Step 3: Compare BIDMC AUC with vs without transfer feature ─
    rows = []
    AUGMENTED_A = POSTOP_A_FEATURES + ["eicu_transfer_pred"]
    AUGMENTED_B = POSTOP_B_FEATURES + ["eicu_transfer_pred"]

    print("\n══ BIDMC postop_A: baseline vs +eICU-transfer ══", flush=True)
    X_a_base = bidmc[POSTOP_A_FEATURES]
    X_a_aug  = bidmc[AUGMENTED_A]
    p_base = cv_oof(lambda: make_brf_pipe(POSTOP_A_FEATURES), X_a_base, yb)
    p_aug  = cv_oof(lambda: make_brf_pipe(AUGMENTED_A),       X_a_aug,  yb)
    a_b, lo_b, hi_b = bootstrap_auc(yb.values, p_base)
    a_a, lo_a, hi_a = bootstrap_auc(yb.values, p_aug)
    z, pv = delong_test(yb.values, p_aug, p_base)
    rows.append({"feature_set":"postop_A baseline (21 features)",
                 "auc": a_b, "ci_lo": lo_b, "ci_hi": hi_b,
                 "brier": brier_score_loss(yb, p_base),
                 "delong_z": None, "delong_p": None})
    rows.append({"feature_set":"postop_A + eICU transfer (22 features)",
                 "auc": a_a, "ci_lo": lo_a, "ci_hi": hi_a,
                 "brier": brier_score_loss(yb, p_aug),
                 "delong_z": z, "delong_p": pv})
    print(f"  baseline:  AUC = {a_b:.3f} ({lo_b:.3f}-{hi_b:.3f})", flush=True)
    print(f"  augmented: AUC = {a_a:.3f} ({lo_a:.3f}-{hi_a:.3f})", flush=True)
    print(f"  DeLong:    z = {z:.3f}, p = {pv:.4f}  "
          f"(Δ AUC = +{a_a - a_b:.3f})", flush=True)

    print("\n══ BIDMC postop_B: baseline vs +eICU-transfer ══", flush=True)
    X_b_base = bidmc[POSTOP_B_FEATURES]
    X_b_aug  = bidmc[AUGMENTED_B]
    p_base_b = cv_oof(lambda: make_brf_pipe(POSTOP_B_FEATURES), X_b_base, yb)
    p_aug_b  = cv_oof(lambda: make_brf_pipe(AUGMENTED_B),       X_b_aug,  yb)
    a_bb, lo_bb, hi_bb = bootstrap_auc(yb.values, p_base_b)
    a_ab, lo_ab, hi_ab = bootstrap_auc(yb.values, p_aug_b)
    z_b, pv_b = delong_test(yb.values, p_aug_b, p_base_b)
    rows.append({"feature_set":"postop_B baseline (18 features)",
                 "auc": a_bb, "ci_lo": lo_bb, "ci_hi": hi_bb,
                 "brier": brier_score_loss(yb, p_base_b),
                 "delong_z": None, "delong_p": None})
    rows.append({"feature_set":"postop_B + eICU transfer (19 features)",
                 "auc": a_ab, "ci_lo": lo_ab, "ci_hi": hi_ab,
                 "brier": brier_score_loss(yb, p_aug_b),
                 "delong_z": z_b, "delong_p": pv_b})
    print(f"  baseline:  AUC = {a_bb:.3f} ({lo_bb:.3f}-{hi_bb:.3f})", flush=True)
    print(f"  augmented: AUC = {a_ab:.3f} ({lo_ab:.3f}-{hi_ab:.3f})", flush=True)
    print(f"  DeLong:    z = {z_b:.3f}, p = {pv_b:.4f}  "
          f"(Δ AUC = +{a_ab - a_bb:.3f})", flush=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RES / "19_transfer_learning.csv", index=False)

    # ── Save augmented OOF predictions for downstream use
    np.savez(CACHE / "oof_bidmc_postopA_transfer.npz",
              y=yb.values, p_base=p_base, p_aug=p_aug)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pos_y = np.arange(len(df_out))
    colors = ["#1f77b4" if "baseline" in s else "#d6594a" for s in df_out["feature_set"]]
    ax.errorbar(df_out["auc"], pos_y,
                 xerr=[df_out["auc"] - df_out["ci_lo"],
                       df_out["ci_hi"] - df_out["auc"]],
                 fmt="o", capsize=4, color="black", ecolor="gray")
    for i, c in enumerate(colors):
        ax.scatter(df_out["auc"].iloc[i], pos_y[i], color=c, s=80, zorder=3)
    ax.set_yticks(pos_y); ax.set_yticklabels(df_out["feature_set"])
    ax.invert_yaxis()
    ax.axvline(0.5, color="gray", ls=":")
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("Cross-validated AUC (95% bootstrap CI)")
    ax.set_title("BIDMC — augmentation with eICU transfer-learning feature")
    for i, r in df_out.iterrows():
        if r["delong_p"] is not None and not np.isnan(r["delong_p"]):
            ax.text(r["ci_hi"] + 0.005, i,
                    f'  Δ AUC = +{r["auc"] - df_out.iloc[i-1]["auc"]:.3f}  '
                    f'(DeLong p = {r["delong_p"]:.3f})',
                    va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "19_transfer_learning.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "19_transfer_learning.pdf", bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 72)
    print(df_out.round(4).to_string(index=False))
    print("\n[OK] results/19_transfer_learning.csv  figures/19_transfer_learning.{png,pdf}")

if __name__ == "__main__":
    main()
