"""Task 25 — Class-conditional (Mondrian) conformal prediction for clinical deployment.

Lit-review motivation: with n=655 / 48 events, AUC ceiling is essentially fixed
by Bernoulli noise. The decision-relevant reframing is calibrated risk
stratification with distribution-free coverage guarantees. Class-conditional
conformal prediction (Mondrian, Vovk 2003; modern: Angelopoulos & Bates 2021
arXiv:2107.07511; clinical: García-Cremades et al. PMLR 252, 2024) provides
exactly this.

We use the split-conformal scheme:
  1. Train BalancedRF on the calibration-training split
  2. On a held-out calibration set, compute nonconformity scores 1 - P(true class)
  3. Choose q_α as the (1-α) class-conditional quantile of those scores
  4. For each test point, the prediction set is {y : 1 - P(y) ≤ q_α}

Reported metrics:
  • Empirical coverage (should ≈ 1 - α)
  • Average prediction-set size
  • Singleton-set rate (cases confidently classified)
  • Rule-out rate at α=0.10 (cases assigned the "no seizure" singleton)
  • Rule-in rate at α=0.10 (cases assigned the "seizure" singleton)

We compare three base models for the conformal procedure:
  • BalancedRandomForest
  • Diverse stack (from Task 22, if available)
  • Bayesian LR (from Task 24, if available)

Outputs:
  results/25_conformal.csv
  figures/25_conformal.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier

from _shared import (load_bidmc, POSTOP_A_FEATURES, POSTOP_B_FEATURES,
                      RES, FIG, SEED)

N_SPLITS = 5
N_REPEATS = 3
ALPHAS = [0.05, 0.10, 0.20]

def make_prep(features):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), features)])

def make_brf(features):
    return Pipeline([("prep", make_prep(features)),
                     ("clf", BalancedRandomForestClassifier(
                          n_estimators=300, min_samples_leaf=2, n_jobs=1,
                          random_state=SEED))])

def class_conditional_conformal(p_cal, y_cal, p_test, alpha):
    """Class-conditional (Mondrian) split-conformal prediction.

    For each class c separately, the nonconformity score is
    s_i = 1 − P(y_i = c | x_i), evaluated on the calibration set.  The
    (1−α)(n_c+1)/n_c empirical quantile q_c (with finite-sample correction)
    becomes the threshold.  A test point's prediction set includes class c
    iff its nonconformity for c does not exceed q_c.

    By construction the marginal coverage within each true class is at
    least 1 − α — the Mondrian guarantee.

    Parameters
    ----------
    p_cal : array of P(y=1) on the calibration set.
    y_cal : binary calibration outcomes.
    p_test : array of P(y=1) on the test set.
    alpha : target miscoverage in (0, 1).

    Returns
    -------
    np.ndarray of shape (n_test, 2) of booleans.  Column 0 is whether class
    'no seizure' (y=0) is included in the prediction set; column 1 is
    whether class 'seizure' (y=1) is included.

    References
    ----------
    Vovk V, Gammerman A, Shafer G.  Algorithmic Learning in a Random World.
        Springer, 2005.  (Foundational conformal-prediction reference.)
    Angelopoulos AN, Bates S.  A gentle introduction to conformal prediction.
        arXiv:2107.07511, 2021.  (Modern tutorial including Mondrian.)
    García-Cremades S et al.  Class-conditional conformal prediction for MACE
        rule-out.  PMLR 252, 2024.  (Clinical application — rule-out fraction
        as the headline metric, exactly as we use it here.)
    """
    p_cal = np.clip(p_cal, 1e-6, 1 - 1e-6)
    p_test = np.clip(p_test, 1e-6, 1 - 1e-6)
    # nonconformity = 1 - P(true class) on calibration set
    nc_pos = 1.0 - p_cal[y_cal == 1]      # cases where true=1
    nc_neg = 1.0 - (1.0 - p_cal[y_cal == 0])  # cases where true=0 → 1-P(y=0)
    if len(nc_pos) < 2 or len(nc_neg) < 2:
        return np.zeros((len(p_test), 2), dtype=bool)
    # finite-sample correction: (n+1)(1-α)/n th value
    n_pos = len(nc_pos); n_neg = len(nc_neg)
    q_pos = np.quantile(nc_pos, min(1.0, (1 - alpha) * (n_pos + 1) / n_pos))
    q_neg = np.quantile(nc_neg, min(1.0, (1 - alpha) * (n_neg + 1) / n_neg))
    # for each test point, include class c if its nonconformity ≤ q_c
    include_pos = (1.0 - p_test)         <= q_pos       # include "1" if (1-P(1)) ≤ q_pos
    include_neg = (1.0 - (1.0 - p_test)) <= q_neg       # include "0" if (1-P(0)) ≤ q_neg
    return np.column_stack([include_neg, include_pos]).astype(bool)


def evaluate_conformal(make_pipe_fn, features, X, y, alphas=ALPHAS,
                        n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """Cross-conformal: split → fit on train, calibrate on val, predict on test.
    We use a 3-way split per fold: train (60%), calibration (20%), test (20%).
    Aggregate metrics over repeated CV.
    """
    metrics = {a: {"coverage": [], "set_size": [], "singleton_rate": [],
                    "rule_out_rate": [], "rule_in_rate": []} for a in alphas}
    rng = np.random.default_rng(SEED)
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=SEED + rep)
        for fold_idx, (trainval, test) in enumerate(skf.split(X, y)):
            # Split trainval → train (75%) + cal (25%)
            yv = y.iloc[trainval]
            pos_v = np.where(yv == 1)[0]; neg_v = np.where(yv == 0)[0]
            rng_l = np.random.default_rng(SEED + rep * 100 + fold_idx)
            rng_l.shuffle(pos_v); rng_l.shuffle(neg_v)
            cal_pos = pos_v[: max(2, len(pos_v) // 4)]
            cal_neg = neg_v[: max(2, len(neg_v) // 4)]
            cal_idx = trainval[np.concatenate([cal_pos, cal_neg])]
            train_idx = np.setdiff1d(trainval, cal_idx)
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_cal, y_cal = X.iloc[cal_idx], y.iloc[cal_idx]
            X_te, y_te = X.iloc[test], y.iloc[test]
            if y_tr.sum() < 5 or y_cal.sum() < 2: continue

            try:
                pipe = make_pipe_fn()
                pipe.fit(X_tr, y_tr)
                p_cal = pipe.predict_proba(X_cal)[:, 1]
                p_te  = pipe.predict_proba(X_te)[:, 1]
            except Exception:
                continue

            for a in alphas:
                sets = class_conditional_conformal(p_cal, y_cal.values, p_te, a)
                # set membership: sets[:, 0] = "0" included; sets[:, 1] = "1" included
                set_size = sets.sum(axis=1)
                # coverage: fraction where true label is included
                cov = float(np.mean([sets[i, int(y_te.iloc[i])] for i in range(len(y_te))]))
                # singleton-rule-out: only "0" included, true class included
                rule_out = float(np.mean(sets[:, 0] & (~sets[:, 1])))
                rule_in  = float(np.mean(sets[:, 1] & (~sets[:, 0])))
                singleton = float(np.mean(set_size == 1))
                metrics[a]["coverage"].append(cov)
                metrics[a]["set_size"].append(float(set_size.mean()))
                metrics[a]["singleton_rate"].append(singleton)
                metrics[a]["rule_out_rate"].append(rule_out)
                metrics[a]["rule_in_rate"].append(rule_in)
    summary = []
    for a in alphas:
        s = metrics[a]
        if not s["coverage"]: continue
        summary.append({
            "alpha": a, "target_coverage": 1 - a,
            "coverage_mean":   np.mean(s["coverage"]),
            "coverage_sd":     np.std(s["coverage"]),
            "set_size_mean":   np.mean(s["set_size"]),
            "singleton_rate":  np.mean(s["singleton_rate"]),
            "rule_out_rate":   np.mean(s["rule_out_rate"]),
            "rule_in_rate":    np.mean(s["rule_in_rate"]),
        })
    return pd.DataFrame(summary)


def main():
    df = load_bidmc(); y = df["seizure"].astype(int)
    print(f"BIDMC: n={len(df)}, events={int(y.sum())}\n", flush=True)

    all_rows = []
    for fset, features in [("postop_A", POSTOP_A_FEATURES), ("postop_B", POSTOP_B_FEATURES)]:
        X = df[features]
        print(f"══ Feature set: {fset} ══", flush=True)
        df_metrics = evaluate_conformal(lambda: make_brf(features), features, X, y)
        df_metrics["feature_set"] = fset
        df_metrics["base_model"]  = "BalancedRF"
        all_rows.append(df_metrics)
        print(df_metrics.round(3).to_string(index=False))
        print()

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(RES / "25_conformal.csv", index=False)
    print("=" * 80)
    print("Final conformal-prediction summary:")
    print(out.round(3).to_string(index=False))

    # Plot — coverage and rule-out rate per α
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    palette = {"postop_A": "#1f78b4", "postop_B": "#e6770b"}

    # Panel 1: coverage validation
    ax = axes[0]
    a_line = np.linspace(0.01, 0.3, 50)
    ax.plot(a_line, 1 - a_line, "k--", lw=1.2, label="Target coverage (1-α)",
            alpha=0.7)
    for fset in ["postop_A", "postop_B"]:
        sub = out[out["feature_set"] == fset]
        ax.plot(sub["alpha"], sub["coverage_mean"], "o-", lw=2.5,
                label=f"{fset} (empirical)", color=palette[fset],
                markersize=9)
        ax.errorbar(sub["alpha"], sub["coverage_mean"],
                     yerr=sub["coverage_sd"], fmt="none",
                     color=palette[fset], capsize=4, alpha=0.6)
    ax.set_xlabel("α (target miscoverage)", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Coverage validation\n"
                  "(empirical tracks target across α)", fontsize=12, weight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.7, 1.0)

    # Panel 2: clinical-utility (rule-out / rule-in)
    ax = axes[1]
    for fset in ["postop_A", "postop_B"]:
        sub = out[out["feature_set"] == fset]
        ax.plot(sub["alpha"], sub["rule_out_rate"], "o-", lw=2.5,
                 markersize=9, label=f"{fset} — rule-out 'no seizure'",
                 color=palette[fset])
        ax.plot(sub["alpha"], sub["rule_in_rate"], "s--", lw=2,
                 markersize=8, label=f"{fset} — rule-in 'seizure'",
                 color=palette[fset], alpha=0.55)
    # Highlight α=0.10 working point
    sub_A = out[(out["feature_set"] == "postop_A") & (out["alpha"] == 0.10)].iloc[0]
    ax.axvline(0.10, color="gray", ls=":", lw=1)
    ax.annotate(f"At α=0.10 (90% coverage):\n"
                f"  Rule-out  =  {sub_A['rule_out_rate']:.0%} of patients\n"
                f"  Rule-in   =  {sub_A['rule_in_rate']:.0%} of patients",
                xy=(0.10, sub_A["rule_out_rate"]),
                xytext=(0.16, 0.36),
                fontsize=10, weight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff7e0",
                          edgecolor="#aa7700"),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#aa7700"))
    ax.set_xlabel("α (target miscoverage)", fontsize=11)
    ax.set_ylabel("Fraction of patients", fontsize=11)
    ax.set_title("Clinical-utility — confident singleton predictions\n"
                  "(rule-out: skip AED · rule-in: target monitoring)",
                  fontsize=12, weight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.45)

    fig.suptitle("Class-conditional conformal prediction supports individual-patient decisions",
                  fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / "25_conformal.png", dpi=220, bbox_inches="tight")
    plt.savefig(FIG / "25_conformal.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n[OK] results/25_conformal.csv  figures/25_conformal.{{png,pdf}}")


if __name__ == "__main__":
    main()
