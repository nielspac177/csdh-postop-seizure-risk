"""Task 22 — Diverse-base stacking ensemble.

Lit-review-motivated: our existing stack (BRF+XGB+LGBM) is bias-redundant —
all tree ensembles. Adding genuinely diverse inductive biases (linear, kernel,
instance-based) is the most evidence-grounded easy win for small clinical
cohorts (Mohammed et al. PLOS ONE 2023, PMC10165871).

Base learners (deliberately heterogeneous):
  • Logistic regression elastic-net (linear)
  • BalancedRandomForest         (tree ensemble, bagging)
  • XGBoost (tuned)              (tree ensemble, boosting)
  • k-NN (k=5)                   (instance-based)
  • RBF-SVM (probability=True)   (kernel)

Meta-learner: logistic regression with isotonic calibration on OOF predictions.

Evaluation: 5×5 repeated stratified CV; bootstrap 95% AUC CI;
paired DeLong vs BalancedRF baseline.

Outputs:
  results/22_diverse_stacking.csv
  figures/22_diverse_stacking.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb

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
    a1, v10_1, v01_1 = struct(p1)
    a2, v10_2, v01_2 = struct(p2)
    s10 = (np.var(v10_1, ddof=1) + np.var(v10_2, ddof=1)
            - 2 * np.cov(v10_1, v10_2, ddof=1)[0,1])
    s01 = (np.var(v01_1, ddof=1) + np.var(v01_2, ddof=1)
            - 2 * np.cov(v01_1, v01_2, ddof=1)[0,1])
    var_diff = s10 / m + s01 / n
    if var_diff <= 0: return float("nan"), float("nan")
    z = (a1 - a2) / np.sqrt(var_diff)
    from scipy.stats import norm
    return float(z), float(2*(1 - norm.cdf(abs(z))))


def make_diverse_stack(features, isotonic=False):
    pos_weight = None  # set externally
    base_estimators = [
        ("lr",   LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0,
                                      solver="saga", max_iter=5000, n_jobs=1,
                                      class_weight="balanced", random_state=SEED)),
        ("brf",  BalancedRandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                  n_jobs=1, random_state=SEED)),
        ("xgb",  xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                     tree_method="hist", n_jobs=1, random_state=SEED,
                                     verbosity=0)),
        ("knn",  KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=1)),
        ("svm",  SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                      class_weight="balanced", random_state=SEED)),
    ]
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    stack = StackingClassifier(estimators=base_estimators, final_estimator=meta,
                                 cv=3, n_jobs=1, passthrough=False)
    pipe = Pipeline([("prep", make_prep(features)), ("clf", stack)])
    if isotonic:
        pipe = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    return pipe


def main():
    df = load_bidmc(); y = df["seizure"].astype(int)
    print(f"BIDMC: n={len(df)}, events={int(y.sum())}, prevalence={y.mean()*100:.1f}%\n",
          flush=True)

    rows = []
    for fset, features in [("postop_A", POSTOP_A_FEATURES), ("postop_B", POSTOP_B_FEATURES)]:
        X = df[features]
        print(f"══ Feature set: {fset} ({len(features)} features) ══", flush=True)

        # Baseline: BalancedRF
        def mk_brf():
            return Pipeline([("prep", make_prep(features)),
                             ("clf", BalancedRandomForestClassifier(
                                 n_estimators=300, min_samples_leaf=2, n_jobs=1,
                                 random_state=SEED))])
        p_brf = cv_oof(mk_brf, X, y, n_repeats=N_REPEATS)
        a, lo, hi = bootstrap_auc(y.values, p_brf)
        rows.append({"feature_set": fset, "model": "BalancedRF (baseline)",
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "brier": brier_score_loss(y, p_brf),
                     "prauc": average_precision_score(y, p_brf),
                     "delong_p": np.nan})
        print(f"  baseline: AUC = {a:.3f} ({lo:.3f}-{hi:.3f})  "
              f"Brier = {brier_score_loss(y, p_brf):.3f}", flush=True)

        # Diverse stack
        print(f"  diverse-stack (LR+BRF+XGB+KNN+SVM) — 5x5 CV ...", flush=True)
        p_stack = cv_oof(lambda: make_diverse_stack(features, isotonic=False), X, y,
                          n_repeats=N_REPEATS)
        a, lo, hi = bootstrap_auc(y.values, p_stack)
        z, pv = delong_test(y.values, p_stack, p_brf)
        rows.append({"feature_set": fset, "model": "Diverse stack (5 learners)",
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "brier": brier_score_loss(y, p_stack),
                     "prauc": average_precision_score(y, p_stack),
                     "delong_p": pv})
        print(f"    AUC = {a:.3f} ({lo:.3f}-{hi:.3f})  "
              f"Brier = {brier_score_loss(y, p_stack):.3f}  "
              f"DeLong p = {pv:.3f}", flush=True)

        # Isotonic-calibrated diverse stack
        print(f"  diverse-stack + isotonic recalibration — 5x3 CV (slower) ...", flush=True)
        p_stack_iso = cv_oof(lambda: make_diverse_stack(features, isotonic=True), X, y,
                              n_repeats=3)
        a, lo, hi = bootstrap_auc(y.values, p_stack_iso)
        z, pv = delong_test(y.values, p_stack_iso, p_brf)
        rows.append({"feature_set": fset, "model": "Diverse stack + isotonic",
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "brier": brier_score_loss(y, p_stack_iso),
                     "prauc": average_precision_score(y, p_stack_iso),
                     "delong_p": pv})
        print(f"    AUC = {a:.3f} ({lo:.3f}-{hi:.3f})  "
              f"Brier = {brier_score_loss(y, p_stack_iso):.3f}  "
              f"DeLong p = {pv:.3f}", flush=True)

        # Cache OOF
        np.savez(CACHE / f"oof_bidmc_{fset}_diverse_stack.npz",
                 y=y.values, p_brf=p_brf, p_stack=p_stack, p_stack_iso=p_stack_iso)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RES / "22_diverse_stacking.csv", index=False)
    print("\n" + "=" * 78)
    print(df_out.round(3).to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pos_y = np.arange(len(df_out))
    colors = ["#1f77b4" if "baseline" in m else "#d6594a" for m in df_out["model"]]
    ax.errorbar(df_out["auc"], pos_y,
                 xerr=[df_out["auc"] - df_out["ci_lo"], df_out["ci_hi"] - df_out["auc"]],
                 fmt="o", capsize=4, color="black", ecolor="gray")
    for i, c in enumerate(colors):
        ax.scatter(df_out["auc"].iloc[i], pos_y[i], color=c, s=80, zorder=3)
    ax.set_yticks(pos_y)
    ax.set_yticklabels([f'{r.feature_set}  ·  {r.model}' for r in df_out.itertuples()])
    ax.invert_yaxis()
    ax.axvline(0.5, color="gray", ls=":")
    ax.set_xlabel("Cross-validated AUC (bootstrap 95% CI)")
    ax.set_title("Diverse-base stacking ensemble — BIDMC")
    ax.set_xlim(0.45, 0.85)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "22_diverse_stacking.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "22_diverse_stacking.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n[OK] results/22_diverse_stacking.csv  figures/22_diverse_stacking.{{png,pdf}}")


if __name__ == "__main__":
    main()
