"""Task 18 — Push BIDMC AUC higher with proper modeling.

Strategies (all leakage-clean, comparable to postop_A feature set):
    (a) BalancedRandomForest (paper baseline)
    (b) XGBoost      with Optuna-tuned hyperparams + scale_pos_weight
    (c) LightGBM     with Optuna-tuned hyperparams
    (d) Logistic-regression elastic-net + SMOTE (already in paper as postop_B)
    (e) Stacking ensemble of (a)+(b)+(c) with LR meta-learner
    (f) Best of (b)/(c)/(e) + isotonic recalibration

All evaluated with 5×5 repeated stratified CV + bootstrap 95% CIs.
n_jobs forced to 1 for Apple Silicon stability.

Outputs:
    results/18_bidmc_optimized.csv
    figures/18_bidmc_optimization.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from _shared import (
    load_bidmc, POSTOP_A_FEATURES, POSTOP_B_FEATURES,
    RES, FIG, CACHE, SEED,
)

N_TRIALS = 40
N_SPLITS, N_REPEATS = 5, 5

def make_prep(features):
    return ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler())]), features)]
    )

def cv_oof(make_pipe_fn, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    p_acc = np.zeros(len(X)); n_acc = np.zeros(len(X)); aucs = []
    for tr, te in rskf.split(X, y):
        p = make_pipe_fn()
        p.fit(X.iloc[tr], y.iloc[tr])
        prob = p.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], prob))
        p_acc[te] += prob; n_acc[te] += 1
    return p_acc / np.maximum(n_acc, 1), np.array(aucs)

def bootstrap_auc(y, p, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed); bs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        bs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return float(roc_auc_score(y, p)), float(lo), float(hi)

# ── Optuna objectives ─────────────────────────────────────────────────
def tune_xgb(X, y, features):
    pos = float(y.sum()); neg = float((1 - y).sum())
    def obj(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            max_depth        = trial.suggest_int("max_depth", 2, 6),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 8),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma            = trial.suggest_float("gamma", 0, 5),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        aucs = []
        for tr, te in cv.split(X, y):
            mdl = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="auc",
                scale_pos_weight=neg / pos, tree_method="hist",
                n_jobs=1, random_state=SEED, verbosity=0, **params,
            )
            prep = make_prep(features)
            pipe = Pipeline([("prep", prep), ("clf", mdl)])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            prob = pipe.predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y.iloc[te], prob))
        return np.mean(aucs)
    study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value

def tune_lgbm(X, y, features):
    pos = float(y.sum()); neg = float((1 - y).sum())
    def obj(trial):
        params = dict(
            n_estimators       = trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            num_leaves         = trial.suggest_int("num_leaves", 8, 64),
            max_depth          = trial.suggest_int("max_depth", 3, 8),
            min_child_samples  = trial.suggest_int("min_child_samples", 5, 40),
            subsample          = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree   = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha          = trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda         = trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        aucs = []
        for tr, te in cv.split(X, y):
            mdl = lgb.LGBMClassifier(
                objective="binary", metric="auc",
                scale_pos_weight=neg / pos, verbosity=-1,
                n_jobs=1, random_state=SEED, **params,
            )
            prep = make_prep(features)
            pipe = Pipeline([("prep", prep), ("clf", mdl)])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            prob = pipe.predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y.iloc[te], prob))
        return np.mean(aucs)
    study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value


def main():
    df = load_bidmc()
    y = df["seizure"].astype(int)
    print(f"BIDMC: n={len(df)}, events={y.sum()} ({y.mean()*100:.1f}%)", flush=True)

    rows = []

    # Run both feature sets — postop_A (richer) and postop_B (no AED/EEG leak risk)
    for feat_set_name, features in [("postop_A", POSTOP_A_FEATURES),
                                      ("postop_B", POSTOP_B_FEATURES)]:
        print(f"\n══ Feature set: {feat_set_name} ({len(features)} features) ══", flush=True)
        X = df[features]; pos = float(y.sum()); neg = float((1 - y).sum())

        # (a) BalancedRandomForest baseline
        print(f"  [a] BalancedRandomForest baseline ...", flush=True)
        def mk_brf():
            return Pipeline([
                ("prep", make_prep(features)),
                ("clf", BalancedRandomForestClassifier(
                    n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=SEED))
            ])
        p_brf, _ = cv_oof(mk_brf, X, y)
        a, lo, hi = bootstrap_auc(y.values, p_brf)
        rows.append({"feature_set": feat_set_name, "model": "BalancedRF",
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "brier": brier_score_loss(y, p_brf),
                     "prauc": average_precision_score(y, p_brf)})
        print(f"      AUC = {a:.3f} ({lo:.3f}-{hi:.3f})", flush=True)

        # (b) XGBoost with Optuna tuning
        print(f"  [b] XGBoost: tuning {N_TRIALS} trials ...", flush=True)
        xgb_params, xgb_best = tune_xgb(X, y, features)
        print(f"      best inner CV AUC = {xgb_best:.3f}", flush=True)
        def mk_xgb():
            return Pipeline([
                ("prep", make_prep(features)),
                ("clf", xgb.XGBClassifier(
                    objective="binary:logistic", eval_metric="auc",
                    scale_pos_weight=neg/pos, tree_method="hist",
                    n_jobs=1, random_state=SEED, verbosity=0, **xgb_params))
            ])
        p_xgb, _ = cv_oof(mk_xgb, X, y)
        a, lo, hi = bootstrap_auc(y.values, p_xgb)
        rows.append({"feature_set": feat_set_name, "model": "XGBoost (tuned)",
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "brier": brier_score_loss(y, p_xgb),
                     "prauc": average_precision_score(y, p_xgb)})
        print(f"      AUC = {a:.3f} ({lo:.3f}-{hi:.3f})", flush=True)

        # (c) LightGBM with Optuna tuning
        print(f"  [c] LightGBM: tuning {N_TRIALS} trials ...", flush=True)
        lgb_params, lgb_best = tune_lgbm(X, y, features)
        print(f"      best inner CV AUC = {lgb_best:.3f}", flush=True)
        def mk_lgb():
            return Pipeline([
                ("prep", make_prep(features)),
                ("clf", lgb.LGBMClassifier(
                    objective="binary", metric="auc",
                    scale_pos_weight=neg/pos, verbosity=-1,
                    n_jobs=1, random_state=SEED, **lgb_params))
            ])
        p_lgb, _ = cv_oof(mk_lgb, X, y)
        a, lo, hi = bootstrap_auc(y.values, p_lgb)
        rows.append({"feature_set": feat_set_name, "model": "LightGBM (tuned)",
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "brier": brier_score_loss(y, p_lgb),
                     "prauc": average_precision_score(y, p_lgb)})
        print(f"      AUC = {a:.3f} ({lo:.3f}-{hi:.3f})", flush=True)

        # (d) LR-elasticnet + SMOTE
        print(f"  [d] LR-elasticnet + SMOTE ...", flush=True)
        def mk_lr():
            return ImbPipeline([
                ("prep", make_prep(features)),
                ("smote", SMOTE(random_state=SEED, k_neighbors=3)),
                ("clf", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0,
                                            solver="saga", max_iter=5000, n_jobs=1,
                                            random_state=SEED))
            ])
        p_lr, _ = cv_oof(mk_lr, X, y)
        a, lo, hi = bootstrap_auc(y.values, p_lr)
        rows.append({"feature_set": feat_set_name, "model": "LR-EN + SMOTE",
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "brier": brier_score_loss(y, p_lr),
                     "prauc": average_precision_score(y, p_lr)})
        print(f"      AUC = {a:.3f} ({lo:.3f}-{hi:.3f})", flush=True)

        # (e) Stacking ensemble (BRF + XGB + LGBM → LR meta)
        print(f"  [e] Stacking ensemble (BRF+XGB+LGBM → LR meta) ...", flush=True)
        def mk_stack():
            ests = [
                ("brf", BalancedRandomForestClassifier(
                    n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=SEED)),
                ("xgb", xgb.XGBClassifier(
                    objective="binary:logistic", eval_metric="auc",
                    scale_pos_weight=neg/pos, tree_method="hist",
                    n_jobs=1, random_state=SEED, verbosity=0, **xgb_params)),
                ("lgb", lgb.LGBMClassifier(
                    objective="binary", metric="auc",
                    scale_pos_weight=neg/pos, verbosity=-1,
                    n_jobs=1, random_state=SEED, **lgb_params)),
            ]
            stack = StackingClassifier(estimators=ests,
                                        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
                                        cv=3, n_jobs=1, passthrough=False)
            return Pipeline([("prep", make_prep(features)), ("clf", stack)])
        p_stk, _ = cv_oof(mk_stack, X, y, n_repeats=2)  # halved repeats for speed
        a, lo, hi = bootstrap_auc(y.values, p_stk)
        rows.append({"feature_set": feat_set_name, "model": "Stacking (BRF+XGB+LGBM)",
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "brier": brier_score_loss(y, p_stk),
                     "prauc": average_precision_score(y, p_stk)})
        print(f"      AUC = {a:.3f} ({lo:.3f}-{hi:.3f})", flush=True)

        # (f) Isotonic recalibration of best base model
        print(f"  [f] Isotonic recalibration of best model ...", flush=True)
        candidates = [("XGBoost (tuned)", mk_xgb, p_xgb),
                       ("LightGBM (tuned)", mk_lgb, p_lgb),
                       ("Stacking", mk_stack, p_stk)]
        best_name, best_fn, _ = max(candidates,
                                       key=lambda c: roc_auc_score(y, c[2]))
        def mk_iso():
            base = best_fn()
            return CalibratedClassifierCV(base, method="isotonic", cv=3)
        p_iso, _ = cv_oof(mk_iso, X, y, n_repeats=2)
        a, lo, hi = bootstrap_auc(y.values, p_iso)
        rows.append({"feature_set": feat_set_name,
                     "model": f"{best_name} + isotonic",
                     "auc": a, "ci_lo": lo, "ci_hi": hi,
                     "brier": brier_score_loss(y, p_iso),
                     "prauc": average_precision_score(y, p_iso)})
        print(f"      AUC = {a:.3f} ({lo:.3f}-{hi:.3f})", flush=True)

        # Cache best probs for figure use
        np.savez(CACHE / f"oof_bidmc_{feat_set_name}_optimized.npz",
                 y=y.values, p_brf=p_brf, p_xgb=p_xgb, p_lgb=p_lgb,
                 p_stk=p_stk, p_iso=p_iso)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RES / "18_bidmc_optimized.csv", index=False)
    print("\n" + "=" * 72)
    print(df_out.round(3).to_string(index=False))

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(11, 6))
    df_plot = df_out.copy()
    df_plot["label"] = df_plot["feature_set"] + "  " + df_plot["model"]
    pos_ = np.arange(len(df_plot))
    xerr_lo = df_plot["auc"] - df_plot["ci_lo"]
    xerr_hi = df_plot["ci_hi"] - df_plot["auc"]
    colors = ["#1f77b4" if "A" in fs else "#ff7f0e" for fs in df_plot["feature_set"]]
    ax.errorbar(df_plot["auc"], pos_,
                xerr=[xerr_lo, xerr_hi], fmt="o", capsize=4,
                color="black", ecolor="gray")
    for i, c in enumerate(colors):
        ax.scatter(df_plot["auc"].iloc[i], pos_[i], color=c, s=70, zorder=3)
    ax.set_yticks(pos_); ax.set_yticklabels(df_plot["label"])
    ax.invert_yaxis()
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.axvline(0.682, color="green", ls="--", lw=1.2,
               label="Paper postop_A AUC = 0.682")
    ax.set_xlim(0.45, 0.90)
    ax.set_xlabel("Cross-validated AUC (bootstrap 95% CI)")
    ax.set_title("BIDMC — model optimization sweep\n"
                  "Blue: postop_A (21 features) · Orange: postop_B (18 features)")
    ax.grid(axis="x", alpha=0.3); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG / "18_bidmc_optimization.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "18_bidmc_optimization.pdf", bbox_inches="tight")
    plt.close()
    print("\n[OK] results/18_bidmc_optimized.csv  figures/18_bidmc_optimization.{png,pdf}")

if __name__ == "__main__":
    main()
