"""Task 13 — Grouped (hierarchical) regularization on NIS ICD codes.

Compares:
  (a) Plain L2-LR (paper baseline)
  (b) Plain L1-LR (sparse)
  (c) Group LASSO over ICD chapter prefixes
  (d) Sparse-group LASSO

Group structure: each comorbidity / division / race / payer cluster is a group.
Hierarchical sparsity selects whole groups before individual codes within.

Outputs:
  results/13_nis_grouped_lasso.csv
  figures/13_nis_grouped_lasso.png
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from _shared import RES, FIG, CACHE, SEED

# Group structure: maps each feature to a group
# Hierarchy: race-block, payer-block, income-block, division-block,
#            comorbidity-block, antithrombotic-block, procedure-block
GROUPS = {
    "AGE": "demographics", "female": "demographics",
    "race_white": "race", "race_black": "race", "race_hispanic": "race",
    "race_asian": "race", "race_other": "race",
    "pay_medicare": "payer", "pay_medicaid": "payer",
    "pay_private": "payer", "pay_other": "payer",
    "income_q1": "income", "income_q2": "income",
    "income_q3": "income", "income_q4": "income",
    "elective": "admission", "weekend": "admission", "transfer_in": "admission",
    "proc_craniotomy": "procedure", "proc_burr_hole": "procedure",
    "is_chronic": "sdh_subtype", "is_acute": "sdh_subtype",
    "n_dx": "severity", "n_pr": "severity",
    "div_1": "division", "div_2": "division", "div_3": "division",
    "div_4": "division", "div_5": "division", "div_6": "division",
    "div_7": "division", "div_8": "division", "div_9": "division",
    "hypertension": "comorbidity", "diabetes": "comorbidity", "afib": "comorbidity",
    "heart_failure": "comorbidity", "ckd": "comorbidity", "coagulopathy": "comorbidity",
    "alcohol": "comorbidity", "liver_disease": "comorbidity", "dementia": "comorbidity",
    "obesity": "comorbidity", "tobacco": "comorbidity", "prior_stroke": "comorbidity",
    "cad": "comorbidity",
    "antiplatelet": "antithrombotic", "anticoagulant": "antithrombotic",
    "comorbidity_count": "severity",
}

# Group-LASSO via proximal gradient (custom implementation, n_jobs=1)
def group_lasso_logistic(X, y, groups, lam=0.01, alpha=1.0, max_iter=2000, tol=1e-5):
    """
    alpha=1.0 → pure group lasso; alpha=0.5 → sparse-group lasso.
    Uses proximal gradient with backtracking line search.
    """
    n, p = X.shape
    y = y.astype(float).values if hasattr(y, "values") else y.astype(float)
    X = X.values if hasattr(X, "values") else X
    w = np.zeros(p); b = 0.0
    eta = 1.0  # step size

    grp_ids = np.unique(groups)
    grp_idx = {g: np.where(groups == g)[0] for g in grp_ids}
    grp_w   = {g: np.sqrt(len(grp_idx[g])) for g in grp_ids}  # √|g| weighting

    def sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def loss(w_, b_):
        z = X @ w_ + b_
        return -np.mean(y * z - np.log1p(np.exp(np.clip(z, -30, 30))))

    def grad(w_, b_):
        z = X @ w_ + b_
        p = sigmoid(z)
        gw = X.T @ (p - y) / n
        gb = (p - y).mean()
        return gw, gb

    def prox_group(v, lam_eff):
        # group-lasso prox: v_g · max(0, 1 - lam·√|g|/||v_g||)
        out = v.copy()
        for g, idx in grp_idx.items():
            nv = np.linalg.norm(out[idx])
            if nv > 0:
                out[idx] *= max(0, 1 - lam_eff * grp_w[g] / nv)
        return out

    def prox_l1(v, lam_eff):
        return np.sign(v) * np.maximum(np.abs(v) - lam_eff, 0)

    prev_loss = loss(w, b)
    for it in range(max_iter):
        gw, gb = grad(w, b)
        # gradient step
        v = w - eta * gw
        # apply L1 part first (for sparse-group), then group-lasso prox
        if alpha < 1.0:
            v = prox_l1(v, eta * lam * (1 - alpha))
        v = prox_group(v, eta * lam * alpha)
        b_new = b - eta * gb
        new_loss = loss(v, b_new)
        if new_loss > prev_loss + 1e-3:
            eta *= 0.5
            if eta < 1e-8:
                break
            continue
        if abs(prev_loss - new_loss) < tol:
            w, b = v, b_new
            break
        w, b = v, b_new
        prev_loss = new_loss
    return w, b

def predict_proba(w, b, X):
    z = X.values @ w + b if hasattr(X, "values") else X @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def cv_eval(fit_fn, X, y, n_splits=5, n_repeats=3):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    aucs, praucs = [], []
    nz_per_fold = []
    for tr, te in rskf.split(X, y):
        w_or_pipe = fit_fn(X.iloc[tr], y.iloc[tr])
        if callable(w_or_pipe):
            p = w_or_pipe(X.iloc[te])
        else:
            p = w_or_pipe.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        praucs.append(average_precision_score(y.iloc[te], p))
    return aucs, praucs

def main():
    chronic_path = CACHE / "nis_chronic.parquet"
    if not chronic_path.exists():
        print("[ERROR] Run task 12 first to produce cache/nis_chronic.parquet")
        return

    df = pd.read_parquet(chronic_path)
    print(f"NIS chronic cohort: n={len(df):,}", flush=True)

    F = [c for c in df.columns if c in GROUPS]
    print(f"Modelable features: {len(F)}")
    # use the corrected outcome from task 12: acute symptomatic
    y = df["seizure_acute"].astype(int)
    X = df[F].copy()
    # standardize
    imp = SimpleImputer(strategy="median")
    X[F] = imp.fit_transform(X[F])
    sc = StandardScaler()
    X[F] = sc.fit_transform(X[F])

    print(f"Outcome (seizure_acute): {y.sum()} / {len(y)} ({y.mean():.2%})", flush=True)

    rows = []

    # (a) plain L2
    def fit_l2(Xtr, ytr):
        m = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                                max_iter=1000, n_jobs=1, random_state=SEED)
        m.fit(Xtr, ytr); return m
    aucs, praucs = cv_eval(fit_l2, X, y)
    rows.append({"method":"L2 LR", "auc_mean":np.mean(aucs), "auc_sd":np.std(aucs),
                 "prauc":np.mean(praucs)})
    print(f"L2 LR: AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}", flush=True)

    # (b) plain L1
    def fit_l1(Xtr, ytr):
        m = LogisticRegression(penalty="l1", C=1.0, solver="liblinear",
                                class_weight="balanced", max_iter=2000,
                                random_state=SEED)
        m.fit(Xtr, ytr); return m
    aucs, praucs = cv_eval(fit_l1, X, y)
    rows.append({"method":"L1 LR", "auc_mean":np.mean(aucs), "auc_sd":np.std(aucs),
                 "prauc":np.mean(praucs)})
    print(f"L1 LR: AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}", flush=True)

    # (c) Group LASSO — with Fix J: λ-path tuning via inner CV
    grp_arr = np.array([GROUPS[f] for f in F])

    LAMBDA_GRID = np.logspace(-3, 0, 12)   # 0.001 → 1.0 (12 values, log-spaced)

    def _balance(Xtr, ytr):
        pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("missing class")
        w_pos = len(neg) / max(len(pos), 1)
        rep = max(1, int(round(w_pos)))
        idx = np.concatenate([neg, np.tile(pos, rep)])
        return Xtr.iloc[idx], ytr.iloc[idx]

    def select_lambda(Xtr, ytr, alpha, n_inner=3):
        """Fix J: inner-CV λ selection on the log-spaced grid."""
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED)
        scores = np.zeros(len(LAMBDA_GRID))
        for itr, ite in skf.split(Xtr, ytr):
            X_i = Xtr.iloc[itr]; y_i = ytr.iloc[itr]
            X_v = Xtr.iloc[ite]; y_v = ytr.iloc[ite]
            try:
                Xb, yb = _balance(X_i, y_i)
            except ValueError:
                continue
            for li, lam in enumerate(LAMBDA_GRID):
                w, b = group_lasso_logistic(Xb, yb, grp_arr, lam=lam, alpha=alpha,
                                             max_iter=800)
                pred = predict_proba(w, b, X_v)
                if len(np.unique(y_v)) < 2:
                    continue
                scores[li] += roc_auc_score(y_v, pred)
        scores /= max(n_inner, 1)
        best = int(np.argmax(scores))
        return float(LAMBDA_GRID[best]), scores

    selected_lams = {"group_lasso": [], "sgl": []}

    def fit_glasso_tuned(Xtr, ytr):
        lam, _ = select_lambda(Xtr, ytr, alpha=1.0, n_inner=3)
        selected_lams["group_lasso"].append(lam)
        Xb, yb = _balance(Xtr, ytr)
        w, b = group_lasso_logistic(Xb, yb, grp_arr, lam=lam, alpha=1.0)
        return lambda Xte: predict_proba(w, b, Xte)

    aucs, praucs = cv_eval(fit_glasso_tuned, X, y)
    lam_med = float(np.median(selected_lams["group_lasso"])) if selected_lams["group_lasso"] else np.nan
    rows.append({"method":f"Group LASSO (α=1, λ tuned, med={lam_med:.3f})",
                 "auc_mean":np.mean(aucs), "auc_sd":np.std(aucs),
                 "prauc":np.mean(praucs)})
    print(f"GroupLASSO tuned: AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f} "
          f"(median λ across outer folds = {lam_med:.3f})", flush=True)

    # (d) Sparse-group LASSO — also tuned
    def fit_sgl_tuned(Xtr, ytr):
        lam, _ = select_lambda(Xtr, ytr, alpha=0.5, n_inner=3)
        selected_lams["sgl"].append(lam)
        Xb, yb = _balance(Xtr, ytr)
        w, b = group_lasso_logistic(Xb, yb, grp_arr, lam=lam, alpha=0.5)
        return lambda Xte: predict_proba(w, b, Xte)

    aucs, praucs = cv_eval(fit_sgl_tuned, X, y)
    lam_med_s = float(np.median(selected_lams["sgl"])) if selected_lams["sgl"] else np.nan
    rows.append({"method":f"Sparse-group LASSO (α=0.5, λ tuned, med={lam_med_s:.3f})",
                 "auc_mean":np.mean(aucs), "auc_sd":np.std(aucs),
                 "prauc":np.mean(praucs)})
    print(f"SGL tuned: AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f} "
          f"(median λ = {lam_med_s:.3f})", flush=True)

    # Persist λ paths
    pd.DataFrame({
        "lambda": LAMBDA_GRID,
    }).to_csv(RES / "13_nis_lambda_grid.csv", index=False)
    pd.DataFrame({
        "outer_fold": np.arange(len(selected_lams["group_lasso"])),
        "lambda_group_lasso": selected_lams["group_lasso"],
        "lambda_sgl": selected_lams["sgl"],
    }).to_csv(RES / "13_nis_selected_lambdas.csv", index=False)

    # final fit: group LASSO on full data → group-level coefficients
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rep = max(1, int(round(len(neg) / max(len(pos), 1))))
    idx = np.concatenate([neg, np.tile(pos, rep)])
    Xb = X.iloc[idx]; yb = y.iloc[idx]
    w_full, b_full = group_lasso_logistic(Xb, yb, grp_arr, lam=0.01, alpha=1.0)

    # group-level summary: ||w_g|| per group
    grp_norms = {}
    for g in np.unique(grp_arr):
        idx_g = np.where(grp_arr == g)[0]
        grp_norms[g] = float(np.linalg.norm(w_full[idx_g]))
    grp_df = pd.DataFrame.from_dict(grp_norms, orient="index", columns=["L2_norm"])\
        .sort_values("L2_norm", ascending=False)
    print("\nGroup-level effective magnitudes:")
    print(grp_df.round(3).to_string())
    grp_df.to_csv(RES / "13_nis_group_norms.csv")

    out = pd.DataFrame(rows)
    out.to_csv(RES / "13_nis_grouped_lasso.csv", index=False)
    print("\n", out.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(out["method"], out["auc_mean"], xerr=out["auc_sd"], capsize=4,
                 color=["tab:blue","tab:orange","tab:green","tab:red"])
    axes[0].axvline(0.5, color="gray", ls=":")
    axes[0].set_xlim(0.5, 0.7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Cross-validated AUROC")
    axes[0].set_title("NIS chronic cohort: regularization comparison")
    grp_df.iloc[:12].iloc[::-1].plot(kind="barh", ax=axes[1], color="tab:purple", legend=False)
    axes[1].set_xlabel("‖β_group‖₂  (group LASSO, α=1)")
    axes[1].set_title("Group-level coefficient magnitudes")
    plt.tight_layout()
    plt.savefig(FIG / "13_nis_grouped_lasso.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "13_nis_grouped_lasso.pdf", bbox_inches="tight")
    plt.close()
    print("\n[OK] Saved: results/13_*.csv  figures/13_nis_grouped_lasso.{png,pdf}")

if __name__ == "__main__":
    main()
