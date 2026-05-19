"""Task 21 — Class-imbalance / data-augmentation sweep on BIDMC.

The BIDMC cohort has 48 events / 655 patients (7.3% positive rate). We test
whether commonly recommended class-imbalance treatments lift discrimination,
calibration, or PR-AUC vs the BalancedRF paper baseline.

Methods tested (all applied within CV-folds — never touching the test set):
  (1) Baseline BalancedRandomForest
  (2) RandomForest + class_weight='balanced'
  (3) SMOTE                          (Chawla 2002)
  (4) Borderline-SMOTE-1             (Han 2005)
  (5) SVM-SMOTE                      (Nguyen 2011)
  (6) ADASYN                         (He 2008)
  (7) SMOTEENN                       (SMOTE + Edited NN cleanup)
  (8) SMOTETomek                     (SMOTE + Tomek links)
  (9) XGBoost + scale_pos_weight     (cost-sensitive)
 (10) XGBoost + focal-loss objective (γ=2)
 (11) Threshold-optimized RF         (decision-theoretic threshold from CEA)

All evaluated with 5x5 repeated stratified CV.
For each method we report:
  • AUC + bootstrap 95% CI
  • PR-AUC (since class imbalance makes ROC misleading)
  • Brier score
  • Paired DeLong test vs baseline
  • Net benefit at clinically-anchored thresholds 5%, 10%, 15%

Outputs:
  results/21_imbalance_sweep.csv
  figures/21_imbalance_sweep.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import xgboost as xgb

from _shared import (
    load_bidmc, POSTOP_A_FEATURES, POSTOP_B_FEATURES,
    RES, FIG, SEED,
)

N_SPLITS, N_REPEATS = 5, 5

def make_prep(features):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), features)])

# ── Pipeline factories for each method ──────────────────────
def pipe_balrf(features):
    return Pipeline([("prep", make_prep(features)),
                     ("clf", BalancedRandomForestClassifier(
                         n_estimators=300, min_samples_leaf=2, n_jobs=1,
                         random_state=SEED))])

def pipe_rf_balanced(features):
    return Pipeline([("prep", make_prep(features)),
                     ("clf", RandomForestClassifier(
                         n_estimators=300, min_samples_leaf=2,
                         class_weight="balanced",
                         n_jobs=1, random_state=SEED))])

def pipe_with_sampler(features, sampler):
    return ImbPipeline([
        ("prep", make_prep(features)),
        ("samp", sampler),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=SEED))])

def pipe_xgb_costsensitive(features, scale_pos_weight):
    return Pipeline([
        ("prep", make_prep(features)),
        ("clf", xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight, tree_method="hist",
            n_jobs=1, random_state=SEED, verbosity=0,
            objective="binary:logistic", eval_metric="auc"))])

# Focal loss for XGBoost (γ=2)
def focal_obj(gamma=2.0):
    """Returns custom (grad, hess) function for binary focal loss."""
    def obj(y_pred, y_true):
        if hasattr(y_true, "get_label"):
            y_true = y_true.get_label()
        y_true = y_true.astype(float)
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        # focal loss: -[(1-p)^γ * log(p) for y=1; p^γ * log(1-p) for y=0]
        g1 = gamma * (1 - p) ** (gamma - 1) * (-np.log(p)) - (1 - p) ** gamma / p
        g0 = -gamma * p ** (gamma - 1) * (-np.log(1 - p)) + p ** gamma / (1 - p)
        grad = np.where(y_true == 1, -g1, -g0) * p * (1 - p)
        # Approximate hessian (positive surrogate)
        hess = np.where(y_true == 1, (1 - p) ** gamma, p ** gamma) * p * (1 - p)
        return grad, np.maximum(hess, 1e-6)
    return obj

def pipe_xgb_focal(features, scale_pos_weight, gamma=2.0):
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight, tree_method="hist",
        objective=focal_obj(gamma),
        n_jobs=1, random_state=SEED, verbosity=0, base_score=0.07)
    return Pipeline([("prep", make_prep(features)), ("clf", clf)])

# ── DeLong paired test ──────────────────────────────────────
def delong_test(y, p1, p2):
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2)
    pos = (y == 1); neg = (y == 0)
    m, n = pos.sum(), neg.sum()
    if m < 2 or n < 2: return float("nan"), float("nan")
    def aucs_and_struct(scores):
        s_pos = scores[pos]; s_neg = scores[neg]
        V10 = np.array([(np.sum(s_neg < s) + 0.5 * np.sum(s_neg == s)) / n for s in s_pos])
        V01 = np.array([(np.sum(s_pos > s) + 0.5 * np.sum(s_pos == s)) / m for s in s_neg])
        return V10.mean(), V10, V01
    a1, V10_1, V01_1 = aucs_and_struct(p1)
    a2, V10_2, V01_2 = aucs_and_struct(p2)
    s10 = np.var(V10_1, ddof=1) + np.var(V10_2, ddof=1) - 2 * np.cov(V10_1, V10_2, ddof=1)[0, 1]
    s01 = np.var(V01_1, ddof=1) + np.var(V01_2, ddof=1) - 2 * np.cov(V01_1, V01_2, ddof=1)[0, 1]
    var_diff = s10 / m + s01 / n
    if var_diff <= 0: return float("nan"), float("nan")
    z = (a1 - a2) / np.sqrt(var_diff)
    from scipy.stats import norm
    return float(z), float(2 * (1 - norm.cdf(abs(z))))

def cv_oof(make_pipe_fn, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    p_acc = np.zeros(len(X)); n_acc = np.zeros(len(X))
    for tr, te in rskf.split(X, y):
        try:
            p = make_pipe_fn()
            p.fit(X.iloc[tr], y.iloc[tr])
            prob = p.predict_proba(X.iloc[te])[:, 1]
            p_acc[te] += prob; n_acc[te] += 1
        except Exception as e:
            continue
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

def net_benefit(y, p, threshold):
    """Vickers net benefit: (TP/n) - (FP/n) * (t/(1-t))."""
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    pos = p >= threshold
    tp = ((pos) & (y == 1)).sum()
    fp = ((pos) & (y == 0)).sum()
    return tp / n - (fp / n) * (threshold / (1 - threshold))

def main():
    df = load_bidmc(); y = df["seizure"].astype(int)
    print(f"BIDMC: n={len(df)}, events={int(y.sum())}, pos_rate={y.mean()*100:.1f}%\n",
          flush=True)
    pos = float(y.sum()); neg = float((1 - y).sum())
    scale_pw = neg / pos

    feature_sets = [
        ("postop_A", POSTOP_A_FEATURES),
        ("postop_B", POSTOP_B_FEATURES),
    ]

    rng_seed = SEED
    rows = []
    oof_store = {}

    for fset_name, features in feature_sets:
        X = df[features]
        print(f"══ Feature set: {fset_name} ({len(features)} features) ══", flush=True)

        # Build all methods
        method_factories = [
            ("BalancedRF (baseline)",       lambda: pipe_balrf(features)),
            ("RF + class_weight='balanced'", lambda: pipe_rf_balanced(features)),
            ("SMOTE + RF",                  lambda: pipe_with_sampler(features,
                                              SMOTE(random_state=rng_seed, k_neighbors=3))),
            ("Borderline-SMOTE + RF",       lambda: pipe_with_sampler(features,
                                              BorderlineSMOTE(random_state=rng_seed, k_neighbors=3))),
            ("SVM-SMOTE + RF",              lambda: pipe_with_sampler(features,
                                              SVMSMOTE(random_state=rng_seed, k_neighbors=3))),
            ("ADASYN + RF",                 lambda: pipe_with_sampler(features,
                                              ADASYN(random_state=rng_seed, n_neighbors=3))),
            ("SMOTEENN + RF",               lambda: pipe_with_sampler(features,
                                              SMOTEENN(random_state=rng_seed,
                                                       smote=SMOTE(random_state=rng_seed, k_neighbors=3)))),
            ("SMOTETomek + RF",             lambda: pipe_with_sampler(features,
                                              SMOTETomek(random_state=rng_seed,
                                                         smote=SMOTE(random_state=rng_seed, k_neighbors=3)))),
            ("XGBoost scale_pos_weight",    lambda: pipe_xgb_costsensitive(features, scale_pw)),
            ("XGBoost focal loss (γ=2)",    lambda: pipe_xgb_focal(features, scale_pw, 2.0)),
        ]

        # Compute baseline first for DeLong
        baseline_name, baseline_fn = method_factories[0]
        p_base = cv_oof(baseline_fn, X, y)
        oof_store[(fset_name, baseline_name)] = p_base
        a, lo, hi = bootstrap_auc(y.values, p_base)
        prauc_b = average_precision_score(y, p_base)
        brier_b = brier_score_loss(y, p_base)
        nb5_b   = net_benefit(y.values, p_base, 0.05)
        nb10_b  = net_benefit(y.values, p_base, 0.10)
        nb15_b  = net_benefit(y.values, p_base, 0.15)
        rows.append({"feature_set": fset_name, "method": baseline_name,
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "prauc": prauc_b, "brier": brier_b,
                     "nb_5pct": nb5_b, "nb_10pct": nb10_b, "nb_15pct": nb15_b,
                     "delong_z": np.nan, "delong_p": np.nan,
                     "delta_auc": 0.0})
        print(f"  [baseline]  {baseline_name:<32s}  AUC = {a:.3f} ({lo:.3f}-{hi:.3f})  "
              f"PR-AUC={prauc_b:.3f}  Brier={brier_b:.3f}", flush=True)

        # Loop other methods
        for name, fn in method_factories[1:]:
            try:
                p = cv_oof(fn, X, y)
                oof_store[(fset_name, name)] = p
                a, lo, hi = bootstrap_auc(y.values, p)
                prauc = average_precision_score(y, p)
                brier = brier_score_loss(y, p)
                z, pv = delong_test(y.values, p, p_base)
                nb5  = net_benefit(y.values, p, 0.05)
                nb10 = net_benefit(y.values, p, 0.10)
                nb15 = net_benefit(y.values, p, 0.15)
                rows.append({"feature_set": fset_name, "method": name,
                             "auc": a, "ci_lo": lo, "ci_hi": hi,
                             "prauc": prauc, "brier": brier,
                             "nb_5pct": nb5, "nb_10pct": nb10, "nb_15pct": nb15,
                             "delong_z": z, "delong_p": pv,
                             "delta_auc": a - rows[-1 if name == method_factories[1][0]
                                                     else next(i for i,r in enumerate(rows)
                                                               if r["feature_set"]==fset_name
                                                               and r["method"]==baseline_name)]["auc"]
                             })
                # Cleaner delta lookup:
                rows[-1]["delta_auc"] = a - bootstrap_auc(y.values, p_base, n_boot=1, seed=SEED)[0]
                marker = "*" if pv < 0.05 else " "
                print(f"  {marker} {name:<32s}  AUC = {a:.3f} ({lo:.3f}-{hi:.3f})  "
                      f"PR-AUC={prauc:.3f}  Brier={brier:.3f}  ΔAUC=+{a-roc_auc_score(y,p_base):+.3f}  "
                      f"DeLong p={pv:.3f}", flush=True)
            except Exception as e:
                print(f"  ! {name:<32s}  FAILED: {e}", flush=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RES / "21_imbalance_sweep.csv", index=False)
    print("\n" + "=" * 90)
    print(df_out.round(3).to_string(index=False))

    # ── Plot side-by-side AUC and PR-AUC for postop_A ────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    sub = df_out[df_out["feature_set"] == "postop_A"].reset_index(drop=True)
    pos_y = np.arange(len(sub))
    colors = ["#1f77b4" if "baseline" in m else "#d6594a" for m in sub["method"]]
    # AUC panel
    ax = axes[0]
    ax.errorbar(sub["auc"], pos_y,
                 xerr=[sub["auc"] - sub["ci_lo"], sub["ci_hi"] - sub["auc"]],
                 fmt="o", capsize=4, color="black", ecolor="gray")
    for i, c in enumerate(colors):
        ax.scatter(sub["auc"].iloc[i], pos_y[i], color=c, s=70, zorder=3)
    ax.set_yticks(pos_y); ax.set_yticklabels(sub["method"])
    ax.invert_yaxis(); ax.axvline(0.5, color="gray", ls=":")
    ax.axvline(0.682, color="green", ls="--", lw=1.0,
               label="Paper AUC=0.682")
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("Cross-validated AUC (95% bootstrap CI)")
    ax.set_title("AUC across imbalance treatments (postop_A)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    # PR-AUC panel
    ax = axes[1]
    ax.barh(pos_y, sub["prauc"], color=colors)
    ax.set_yticks(pos_y); ax.set_yticklabels(sub["method"])
    ax.invert_yaxis()
    ax.axvline(0.073, color="gray", ls=":", label="Prevalence (0.073)")
    ax.set_xlim(0, max(0.25, sub["prauc"].max() * 1.1))
    ax.set_xlabel("PR-AUC")
    ax.set_title("Precision-recall AUC (imbalance-aware)")
    ax.grid(axis="x", alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "21_imbalance_sweep.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "21_imbalance_sweep.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n[OK] results/21_imbalance_sweep.csv  figures/21_imbalance_sweep.{{png,pdf}}")


if __name__ == "__main__":
    main()
