"""Task 7 — Missing-data sensitivity analysis.

(a) Missingness pattern (% missing per feature) for BIDMC and eICU Set C.
(b) Compare:
      - median imputation (baseline / current)
      - IterativeImputer (MICE-like)
      - missing-indicator + median
    on AUC (5x3 CV) using the eICU pure Set C and BIDMC postop_B.

Outputs:
  results/07_missingness.csv
  results/07_imputation_comparison.csv
  figures/07_missingness.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Required for IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy import stats

# ── Fix E helpers ───────────────────────────────────────
def missingness_vs_outcome(df, features, outcome_col):
    """Chi-square (or Fisher) test: does missingness of each feature
    associate with the outcome? Suggests MAR rather than MCAR if many sig."""
    y = df[outcome_col].astype(int).values
    rows = []
    for c in features:
        miss = df[c].isna().astype(int).values
        if miss.sum() == 0 or miss.sum() == len(miss):
            rows.append({"feature": c, "n_miss": int(miss.sum()),
                          "p_miss_y1": float("nan"), "p_miss_y0": float("nan"),
                          "chi2": float("nan"), "p_value": float("nan"), "test": "—"})
            continue
        tab = pd.crosstab(miss, y)
        if (tab.values < 5).any():
            try:
                _, p_value = stats.fisher_exact(tab.values)
                chi2 = float("nan"); test = "Fisher"
            except Exception:
                chi2, p_value, test = float("nan"), float("nan"), "—"
        else:
            chi2, p_value, _, _ = stats.chi2_contingency(tab.values)
            test = "chi2"
        rows.append({
            "feature": c, "n_miss": int(miss.sum()),
            "p_miss_y1": float(miss[y == 1].mean()) if (y == 1).any() else float("nan"),
            "p_miss_y0": float(miss[y == 0].mean()) if (y == 0).any() else float("nan"),
            "chi2": float(chi2) if not np.isnan(chi2) else float("nan"),
            "p_value": float(p_value), "test": test,
        })
    return pd.DataFrame(rows)

def littles_mcar_test(df, features):
    """Simplified Little's MCAR test (Little 1988).
    For each unique missingness pattern, compute squared standardized distance
    of observed means from grand means; sum gives chi² with appropriate df.
    Significant p => reject MCAR."""
    Xn = df[features].apply(pd.to_numeric, errors="coerce")
    mu = Xn.mean()
    cov = Xn.cov()
    # Patterns
    miss = Xn.isna()
    patterns = miss.apply(lambda r: tuple(r.values.astype(int)), axis=1)
    chi2_total = 0.0; df_total = 0
    for pat, grp_idx in patterns.groupby(patterns).groups.items():
        obs_cols = [features[i] for i, m in enumerate(pat) if not m]
        if not obs_cols: continue
        Xobs = Xn.loc[grp_idx, obs_cols]
        n_pat = len(grp_idx)
        if n_pat < 2: continue
        try:
            mu_obs = mu[obs_cols].values
            S_obs = cov.loc[obs_cols, obs_cols].values
            S_inv = np.linalg.pinv(S_obs)
            mean_pat = Xobs.mean().values
            d = mean_pat - mu_obs
            chi2_total += float(n_pat * d @ S_inv @ d)
            df_total += len(obs_cols)
        except Exception:
            continue
    # Subtract overall df (number of variables) per Little
    df_eff = max(df_total - len(features), 1)
    p = float(1 - stats.chi2.cdf(chi2_total, df_eff))
    return {"chi2": chi2_total, "df": df_eff, "p": p}

def rubin_pool_auc(aucs, ses):
    """Rubin's rules pooling: combine point estimates and SEs across M imputations."""
    aucs = np.asarray(aucs); ses = np.asarray(ses)
    M = len(aucs)
    q_bar = float(aucs.mean())
    u_bar = float((ses ** 2).mean())              # within-imputation variance
    b = float(((aucs - q_bar) ** 2).sum() / (M - 1)) if M > 1 else 0.0  # between
    t_var = u_bar + (1 + 1 / M) * b
    df = (M - 1) * (1 + u_bar / ((1 + 1 / M) * b)) ** 2 if b > 0 else 1e9
    se_pool = float(np.sqrt(t_var))
    return {"auc_pooled": q_bar, "se_pooled": se_pool,
            "lo": q_bar - 1.96 * se_pool, "hi": q_bar + 1.96 * se_pool,
            "fmi": (1 + 1 / M) * b / t_var if t_var > 0 else 0.0,  # fraction missing info
            "df": df}

from _shared import (
    load_bidmc, load_eicu_pure,
    POSTOP_B_FEATURES, EICU_SET_C, RES, FIG, SEED,
)

def make_pipe(imputer, features, model="rf"):
    if model == "rf":
        prep = ColumnTransformer([("num", Pipeline([("imp", imputer),
                                                    ("sc", StandardScaler())]), features)])
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=1, random_state=SEED)
        return Pipeline([("prep", prep), ("clf", clf)])
    elif model == "lr":
        prep = ColumnTransformer([("num", Pipeline([("imp", imputer),
                                                    ("sc", StandardScaler())]), features)])
        clf = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0,
                                 solver="saga", max_iter=5000, n_jobs=1, random_state=SEED)
        return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=SEED, k_neighbors=3)),
                            ("clf", clf)])

def cv_auc(pipe_fn, X, y, n_splits=5, n_repeats=3):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    aucs = []
    for tr, te in rskf.split(X, y):
        p = pipe_fn()
        p.fit(X.iloc[tr], y.iloc[tr])
        prob = p.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], prob))
    return np.mean(aucs), np.std(aucs)

def main():
    miss_rows = []

    # ── Missingness pattern ────────────────────────────
    bidmc = load_bidmc()
    for c in POSTOP_B_FEATURES:
        miss_rows.append({"cohort":"BIDMC", "feature":c, "n":len(bidmc),
                          "n_missing":int(bidmc[c].isna().sum()),
                          "pct_missing":bidmc[c].isna().mean()*100})

    eicu = load_eicu_pure()
    for c in EICU_SET_C:
        miss_rows.append({"cohort":"eICU pure", "feature":c, "n":len(eicu),
                          "n_missing":int(eicu[c].isna().sum()),
                          "pct_missing":eicu[c].isna().mean()*100})

    miss_df = pd.DataFrame(miss_rows)
    miss_df.to_csv(RES / "07_missingness.csv", index=False)

    # ── Fix E: MCAR diagnostics ──────────────────────────
    print("\nMissingness ↔ outcome (chi-square per feature):", flush=True)
    bidmc_assoc = missingness_vs_outcome(bidmc, POSTOP_B_FEATURES, "seizure")
    eicu_assoc  = missingness_vs_outcome(eicu, EICU_SET_C, "seizure")
    bidmc_assoc.to_csv(RES / "07_missingness_vs_outcome_bidmc.csv", index=False)
    eicu_assoc.to_csv(RES / "07_missingness_vs_outcome_eicu.csv", index=False)
    sig_bidmc = (bidmc_assoc["p_value"] < 0.05).sum()
    sig_eicu = (eicu_assoc["p_value"] < 0.05).sum()
    print(f"  BIDMC: {sig_bidmc} / {len(bidmc_assoc)} features show missingness ↔ outcome (p<0.05)", flush=True)
    print(f"  eICU : {sig_eicu} / {len(eicu_assoc)} features show missingness ↔ outcome (p<0.05)", flush=True)

    print("\nLittle's MCAR test (multivariate):", flush=True)
    bidmc_mcar = littles_mcar_test(bidmc, POSTOP_B_FEATURES)
    eicu_mcar  = littles_mcar_test(eicu,  EICU_SET_C)
    print(f"  BIDMC: chi²={bidmc_mcar['chi2']:.1f}, df={bidmc_mcar['df']}, p={bidmc_mcar['p']:.4f}", flush=True)
    print(f"  eICU : chi²={eicu_mcar['chi2']:.1f}, df={eicu_mcar['df']}, p={eicu_mcar['p']:.4f}", flush=True)
    pd.DataFrame([{"cohort":"BIDMC", **bidmc_mcar}, {"cohort":"eICU", **eicu_mcar}]).to_csv(
        RES / "07_mcar_test.csv", index=False
    )

    # plot top missingness
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, cohort in zip(axes, ["BIDMC", "eICU pure"]):
        sub = miss_df[miss_df["cohort"] == cohort].sort_values("pct_missing", ascending=False).head(20)
        ax.barh(sub["feature"], sub["pct_missing"], color="tab:red")
        ax.invert_yaxis()
        ax.set_xlabel("% missing")
        ax.set_title(f"{cohort} (top 20 by missingness)")
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "07_missingness.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Imputation sensitivity ─────────────────────────
    print("\nImputation sensitivity — eICU pure Set C", flush=True)
    Xe = eicu[EICU_SET_C]; ye = eicu["seizure"].astype(int)
    rows = []
    for label, imp in [
        ("median",     SimpleImputer(strategy="median")),
        ("mean",       SimpleImputer(strategy="mean")),
        ("iterative",  IterativeImputer(estimator=BayesianRidge(),
                                        max_iter=10, random_state=SEED, n_nearest_features=15)),
        ("indicator+median", SimpleImputer(strategy="median", add_indicator=True)),
    ]:
        m, s = cv_auc(lambda imp_=imp: make_pipe(imp_, EICU_SET_C, "rf"), Xe, ye)
        rows.append({"cohort":"eICU pure Set C", "imputer":label, "auc":m, "auc_sd":s})
        print(f"  {label}: AUC = {m:.3f} ± {s:.3f}", flush=True)

    print("\nImputation sensitivity — BIDMC postop_B", flush=True)
    Xb = bidmc[POSTOP_B_FEATURES]; yb = bidmc["seizure"].astype(int)
    for label, imp in [
        ("median",     SimpleImputer(strategy="median")),
        ("iterative",  IterativeImputer(estimator=BayesianRidge(),
                                        max_iter=10, random_state=SEED)),
        ("indicator+median", SimpleImputer(strategy="median", add_indicator=True)),
    ]:
        m, s = cv_auc(lambda imp_=imp: make_pipe(imp_, POSTOP_B_FEATURES, "lr"), Xb, yb, 5, 5)
        rows.append({"cohort":"BIDMC postop_B", "imputer":label, "auc":m, "auc_sd":s})
        print(f"  {label}: AUC = {m:.3f} ± {s:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "07_imputation_comparison.csv", index=False)
    print("\n", out.round(3).to_string(index=False))

    # ── Fix E: Multiple-imputation pooling via Rubin's rules ───────
    # M=10 stochastic imputations, refit model on each, pool AUCs.
    print("\nMI (M=10) + Rubin's rules pooling on eICU pure Set C:", flush=True)
    M = 10
    aucs_mi = []; ses_mi = []
    from sklearn.model_selection import StratifiedKFold
    for m_i in range(M):
        imp = IterativeImputer(estimator=BayesianRidge(),
                                max_iter=10, sample_posterior=True,
                                random_state=SEED + m_i, n_nearest_features=15)
        # CV with this imputer
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + m_i)
        fold_aucs = []
        for tr, te in skf.split(Xe, ye):
            p = make_pipe(imp, EICU_SET_C, "rf")
            p.fit(Xe.iloc[tr], ye.iloc[tr])
            prob = p.predict_proba(Xe.iloc[te])[:, 1]
            fold_aucs.append(roc_auc_score(ye.iloc[te], prob))
        aucs_mi.append(float(np.mean(fold_aucs)))
        ses_mi.append(float(np.std(fold_aucs) / np.sqrt(len(fold_aucs))))
        print(f"  imp {m_i+1}/M={M}: AUC = {aucs_mi[-1]:.3f} ± {ses_mi[-1]:.3f}", flush=True)
    pooled = rubin_pool_auc(aucs_mi, ses_mi)
    print(f"  Rubin's-rules pool: AUC = {pooled['auc_pooled']:.3f} "
          f"(95% CI {pooled['lo']:.3f}-{pooled['hi']:.3f}); "
          f"FMI = {pooled['fmi']:.2f}", flush=True)
    pd.DataFrame([{
        "cohort": "eICU pure Set C", "M": M,
        "imputation_aucs": ";".join(f"{a:.3f}" for a in aucs_mi),
        **pooled,
    }]).to_csv(RES / "07_rubin_pooled.csv", index=False)
    print("\n[OK] Saved: results/07_*.csv  figures/07_missingness.png")

if __name__ == "__main__":
    main()
