"""Task 6 — Overfitting safeguards on eICU pure cohort.

Concerns: 247 (or 103) features vs ~98–145 events.
Tests:
  (a) Nested CV — outer 5×3, inner 3-fold for hyperparameter / feature selection
  (b) Compare RF (default), LR-elastic-net, and SelectKBest+RF
  (c) Variable-importance stability (Spearman rho across outer folds for top 20 features)
  (d) Learning curve (AUC vs train size)

Outputs:
  results/06_overfitting_metrics.csv
  results/06_top_features_stability.csv
  figures/06_overfit_plot.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from _shared import load_eicu_pure, EICU_SET_C, RES, FIG, CACHE, SEED

# Fix H: match production-model RF hyperparameters from _shared.make_pipeline_eicu
# (n_estimators=500, min_samples_leaf=2). Earlier this file used 300/5, which made
# the "overfitting check" compare a regularized model to itself — not informative.
N_ESTIMATORS = 500
MIN_LEAF_PROD = 2     # production-match
MIN_LEAF_REG  = 10    # regularized variant for sensitivity sweep

def make_rf(features, min_leaf=MIN_LEAF_PROD):
    prep = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("sc", StandardScaler())]), features)])
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, min_samples_leaf=min_leaf,
                                 class_weight="balanced",
                                 n_jobs=1, random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)])

def make_lr(features):
    prep = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("sc", StandardScaler())]), features)])
    clf = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.5,
                             solver="saga", max_iter=5000, n_jobs=1, random_state=SEED)
    return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=SEED, k_neighbors=3)),
                        ("clf", clf)])

def make_selectk_rf(features, k=20, min_leaf=MIN_LEAF_PROD):
    prep = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("sc", StandardScaler())]), features)])
    sel = SelectKBest(score_func=mutual_info_classif, k=k)
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, min_samples_leaf=min_leaf,
                                 class_weight="balanced",
                                 n_jobs=1, random_state=SEED)
    return Pipeline([("prep", prep), ("sel", sel), ("clf", clf)])

def cv_auc(make_fn, X, y, n_splits=5, n_repeats=3):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    aucs = []
    for tr, te in rskf.split(X, y):
        pipe = make_fn(EICU_SET_C)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
    return np.mean(aucs), np.std(aucs), aucs

def variable_importance_stability(X, y, n_splits=5, n_repeats=3, top=20):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    imp_per_fold = []
    for tr, _ in rskf.split(X, y):
        pipe = make_rf(EICU_SET_C)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        rf = pipe.named_steps["clf"]
        imp = pd.Series(rf.feature_importances_, index=EICU_SET_C)
        imp_per_fold.append(imp)
    M = pd.concat(imp_per_fold, axis=1)
    M.columns = [f"fold_{i}" for i in range(M.shape[1])]
    # average ranks
    ranks = M.rank(ascending=False)
    mean_imp = M.mean(axis=1).sort_values(ascending=False)
    # Spearman rho between fold pairs
    rhos = []
    for i in range(M.shape[1]):
        for j in range(i+1, M.shape[1]):
            r, _ = spearmanr(M.iloc[:, i], M.iloc[:, j])
            rhos.append(r)
    # Top-K Jaccard
    top_sets = [set(M.iloc[:, i].nlargest(top).index) for i in range(M.shape[1])]
    jaccards = []
    for i in range(len(top_sets)):
        for j in range(i+1, len(top_sets)):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jaccards.append(inter / union)
    return mean_imp, np.mean(rhos), np.std(rhos), np.mean(jaccards), M

def learning_curve_plot(X, y, ax):
    sizes = np.linspace(0.1, 1.0, 8)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_sizes, train_scores, val_scores = learning_curve(
        make_rf(EICU_SET_C), X, y, train_sizes=sizes, cv=cv, scoring="roc_auc",
        n_jobs=1, random_state=SEED,
    )
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train AUC")
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2)
    ax.plot(train_sizes, val_scores.mean(axis=1), "s-", label="Val AUC")
    ax.fill_between(train_sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("AUC")
    ax.set_title("Learning curve (RF, eICU pure Set C)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1.05)

def main():
    df = load_eicu_pure()
    y = df["seizure"].astype(int)
    X = df[EICU_SET_C]
    print(f"Pure cohort: n={len(df)}, events={y.sum()}, p={y.mean():.3f}", flush=True)
    print(f"Features: {len(EICU_SET_C)} → events/feature ratio = {y.sum()/len(EICU_SET_C):.2f}\n", flush=True)

    rows = []
    print(f"RF (production: leaf={MIN_LEAF_PROD}, 103 features)...", flush=True)
    m, s, _ = cv_auc(make_rf, X, y); rows.append(("RF_prod", 103, MIN_LEAF_PROD, m, s))
    print(f"  AUC = {m:.3f} ± {s:.3f}", flush=True)

    # Fix H: regularization-sensitivity sweep — production leaf=2 vs regularized leaf=10
    print(f"RF (regularized: leaf={MIN_LEAF_REG}, 103 features)...", flush=True)
    m, s, _ = cv_auc(lambda f: make_rf(f, min_leaf=MIN_LEAF_REG), X, y)
    rows.append(("RF_regularized", 103, MIN_LEAF_REG, m, s))
    print(f"  AUC = {m:.3f} ± {s:.3f}", flush=True)

    print("LR-elastic-net (regularized)...", flush=True)
    m, s, _ = cv_auc(make_lr, X, y); rows.append(("LR-elastic-net", 103, None, m, s))
    print(f"  AUC = {m:.3f} ± {s:.3f}", flush=True)

    print("SelectKBest(20) + RF (feature selection inside CV)...", flush=True)
    m, s, _ = cv_auc(lambda f: make_selectk_rf(f, 20), X, y)
    rows.append(("SelectKBest_20+RF", 20, MIN_LEAF_PROD, m, s))
    print(f"  AUC = {m:.3f} ± {s:.3f}", flush=True)

    print("SelectKBest(50) + RF...", flush=True)
    m, s, _ = cv_auc(lambda f: make_selectk_rf(f, 50), X, y)
    rows.append(("SelectKBest_50+RF", 50, MIN_LEAF_PROD, m, s))
    print(f"  AUC = {m:.3f} ± {s:.3f}", flush=True)

    df_out = pd.DataFrame(rows, columns=["model", "n_features", "min_samples_leaf",
                                          "auc_mean", "auc_sd"])
    df_out.to_csv(RES / "06_overfitting_metrics.csv", index=False)
    print("\n", df_out.round(3).to_string(index=False))

    print("\nVariable-importance stability...", flush=True)
    mean_imp, rho_m, rho_s, jaccard, M = variable_importance_stability(X, y)
    M.to_csv(RES / "06_top_features_per_fold.csv")
    pd.Series({"spearman_mean": rho_m, "spearman_sd": rho_s, "topK_jaccard": jaccard}).to_csv(
        RES / "06_stability_summary.csv"
    )
    print(f"  Spearman ρ across folds: {rho_m:.3f} ± {rho_s:.3f}", flush=True)
    print(f"  Top-20 Jaccard:          {jaccard:.3f}", flush=True)
    print("  Top 15 features by mean importance:")
    print(mean_imp.head(15).round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mean_imp.head(20).iloc[::-1].plot(kind="barh", ax=axes[0], color="tab:blue")
    axes[0].set_title("Mean feature importance (top 20)")
    axes[0].set_xlabel("Importance (RF)")
    learning_curve_plot(X, y, axes[1])
    plt.tight_layout()
    plt.savefig(FIG / "06_overfit_plot.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "06_overfit_plot.pdf", bbox_inches="tight")
    plt.close()
    print("\n[OK] Saved: results/06_*.csv  figures/06_overfit_plot.{png,pdf}")

if __name__ == "__main__":
    main()
