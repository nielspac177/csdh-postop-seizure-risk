"""Task 4 — Leave-one-hospital-out CV on eICU.

For each hospital with >= MIN_EVENTS seizures, train on remaining 138 hospitals
and evaluate on the held-out hospital. Report AUC distribution.

Outputs:
  results/04_loho_per_hospital.csv
  results/04_loho_summary.csv
  figures/04_loho_distribution.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from _shared import (
    load_eicu, load_eicu_pure, EICU_SET_A, EICU_SET_C,
    make_pipeline_eicu, RES, FIG, CACHE, SEED,
)

MIN_EVENTS = 3   # held-out hospital must have ≥3 seizures for AUC
MIN_PATIENTS = 10

def make_pipe_light(features):
    """Lighter pipeline (200 trees) for LOHO speed."""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    prep = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler())]), features)]
    )
    clf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced",
        n_jobs=1, random_state=SEED,
    )
    return Pipeline([("prep", prep), ("clf", clf)])

def loho_for(df, features, set_name, cohort_name, ckpt_path):
    """Returns a DataFrame; supports incremental checkpointing."""
    y = df["seizure"].astype(int).values
    H = df["hospitalid"].values
    X = df[features]
    hospitals = sorted(df["hospitalid"].unique())
    print(f"  {cohort_name} / {set_name}: {len(hospitals)} hospitals, n={len(df)}, sz={y.sum()}", flush=True)

    if ckpt_path.exists():
        rows = pd.read_csv(ckpt_path).to_dict("records")
        done = {r["hospitalid"] for r in rows
                if r["cohort"] == cohort_name and r["set"] == set_name}
        print(f"    resuming, {len(done)} hospitals done", flush=True)
    else:
        rows = []; done = set()

    for i, h in enumerate(hospitals):
        if int(h) in done:
            continue
        te = (H == h)
        tr = ~te
        if y[te].sum() < MIN_EVENTS or te.sum() < MIN_PATIENTS:
            continue
        if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
            continue
        pipe = make_pipe_light(features)
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], p)
        prauc = average_precision_score(y[te], p)
        brier = brier_score_loss(y[te], p)
        rows.append({
            "cohort": cohort_name, "set": set_name, "hospitalid": int(h),
            "n_test": int(te.sum()), "events_test": int(y[te].sum()),
            "auc": auc, "prauc": prauc, "brier": brier,
        })
        # incremental checkpoint
        pd.DataFrame(rows).to_csv(ckpt_path, index=False)
        if len(rows) % 5 == 0:
            print(f"    {len(rows)} hospitals scored, last AUC={auc:.3f}", flush=True)
    return pd.DataFrame(rows)

def main():
    df_full = load_eicu()
    df_pure = load_eicu_pure()
    print(f"Full: n={len(df_full)} events={df_full.seizure.sum()} hospitals={df_full.hospitalid.nunique()}")
    print(f"Pure: n={len(df_pure)} events={df_pure.seizure.sum()} hospitals={df_pure.hospitalid.nunique()}")

    ckpt = CACHE / "04_loho_ckpt.csv"
    parts = []
    parts.append(loho_for(df_full, EICU_SET_A, "Set_A", "full", ckpt))
    parts.append(loho_for(df_full, EICU_SET_C, "Set_C", "full", ckpt))
    parts.append(loho_for(df_pure, EICU_SET_C, "Set_C", "pure", ckpt))
    perh = pd.concat(parts, ignore_index=True).drop_duplicates(["cohort","set","hospitalid"])
    perh.to_csv(RES / "04_loho_per_hospital.csv", index=False)

    # ── Fix D: Random-effects meta-analytic pooling (DerSimonian–Laird)
    # Treat each held-out hospital AUC as one study; transform to logit-AUC,
    # compute within-study variance via Hanley–McNeil, then DL estimator for
    # between-hospital heterogeneity τ².
    def hanley_mcneil_var(auc, n1, n0):
        """Approximate variance of AUC (Hanley & McNeil 1982)."""
        auc = float(np.clip(auc, 1e-6, 1 - 1e-6))
        n1 = max(int(n1), 1); n0 = max(int(n0), 1)
        Q1 = auc / (2.0 - auc); Q2 = 2.0 * auc**2 / (1.0 + auc)
        return (auc * (1 - auc)
                + (n1 - 1) * (Q1 - auc**2)
                + (n0 - 1) * (Q2 - auc**2)) / (n1 * n0)

    def logit(p):  return np.log(p / (1 - p))
    def invlogit(x): return 1.0 / (1.0 + np.exp(-x))

    def random_effects_pool(g):
        """DerSimonian–Laird pooling on logit-AUC scale."""
        auc = g["auc"].values
        n_te = g["n_test"].values
        ev = g["events_test"].values
        n1 = ev; n0 = n_te - ev
        # Within-study variance on AUC scale; delta-method to logit scale
        var_auc = np.array([hanley_mcneil_var(a, n1_, n0_)
                            for a, n1_, n0_ in zip(auc, n1, n0)])
        # var(logit(p)) ≈ var(p) / (p(1-p))²
        p = np.clip(auc, 1e-6, 1 - 1e-6)
        var_lt = var_auc / (p * (1 - p)) ** 2
        y = logit(p)
        w_fe = 1.0 / var_lt
        mu_fe = np.sum(w_fe * y) / np.sum(w_fe)
        Q = float(np.sum(w_fe * (y - mu_fe) ** 2))
        k = len(y); df = k - 1
        C = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
        tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
        w_re = 1.0 / (var_lt + tau2)
        mu_re = float(np.sum(w_re * y) / np.sum(w_re))
        se_re = float(np.sqrt(1.0 / np.sum(w_re)))
        I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
        return pd.Series({
            "n_hospitals": k,
            "auc_median": float(np.median(auc)),
            "auc_q1":     float(np.quantile(auc, .25)),
            "auc_q3":     float(np.quantile(auc, .75)),
            "auc_min":    float(np.min(auc)),
            "auc_max":    float(np.max(auc)),
            "auc_weighted_mean": float(np.average(auc, weights=n_te)),
            "auc_pooled_RE":     float(invlogit(mu_re)),
            "auc_pooled_RE_lo":  float(invlogit(mu_re - 1.96 * se_re)),
            "auc_pooled_RE_hi":  float(invlogit(mu_re + 1.96 * se_re)),
            "tau2_logit": tau2,
            "I2_pct":     I2,
            "Q":          Q,
            "Q_df":       df,
            "prauc_median": g["prauc"].median(),
            "brier_median": g["brier"].median(),
        })

    summary = (
        perh.groupby(["cohort", "set"]).apply(random_effects_pool).reset_index()
    )
    summary.to_csv(RES / "04_loho_summary.csv", index=False)
    print("\n", summary.round(3).to_string(index=False))

    # ── Forest plot per (cohort, set) ────────────────────
    for (coh, sn), g in perh.groupby(["cohort", "set"]):
        g = g.sort_values("auc").reset_index(drop=True)
        # per-hospital CI via Hanley–McNeil on AUC scale (symmetric ≈)
        ci = np.array([
            np.sqrt(hanley_mcneil_var(a, e, n - e))
            for a, n, e in zip(g["auc"], g["n_test"], g["events_test"])
        ])
        pooled = summary[(summary["cohort"] == coh) & (summary["set"] == sn)].iloc[0]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(g) + 2.5)))
        pos = np.arange(len(g))
        ax.errorbar(g["auc"], pos, xerr=1.96 * ci, fmt="s", capsize=2, ms=4,
                    color="tab:blue", ecolor="gray", alpha=0.85)
        # pooled diamond
        diamond_y = -1.5
        d_lo = pooled["auc_pooled_RE_lo"]; d_hi = pooled["auc_pooled_RE_hi"]
        d_mid = pooled["auc_pooled_RE"]
        ax.fill([d_lo, d_mid, d_hi, d_mid],
                [diamond_y, diamond_y - 0.4, diamond_y, diamond_y + 0.4],
                color="black", alpha=0.85)
        ax.axvline(d_mid, color="black", ls=":", lw=1, alpha=0.5)
        ax.axvline(0.5, color="gray", ls=":", lw=1)
        ax.set_yticks(list(pos) + [diamond_y])
        ax.set_yticklabels(
            [f'H{int(h)} (n={int(n_)}, ev={int(e_)})'
             for h, n_, e_ in zip(g["hospitalid"], g["n_test"], g["events_test"])]
            + [f'Pooled RE (I²={pooled["I2_pct"]:.0f}%, τ²={pooled["tau2_logit"]:.2f})']
        )
        ax.invert_yaxis()
        ax.set_xlabel("Held-out hospital AUC (95% CI)")
        ax.set_xlim(0.2, 1.0)
        ax.set_title(f"LOHO forest — cohort={coh}, set={sn}\n"
                     f"Pooled RE AUC = {d_mid:.3f} ({d_lo:.3f}–{d_hi:.3f}), "
                     f"k={pooled['n_hospitals']:.0f}")
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        fname = f"04_loho_forest_{coh}_{sn}"
        plt.savefig(FIG / f"{fname}.png", dpi=200, bbox_inches="tight")
        plt.savefig(FIG / f"{fname}.pdf", bbox_inches="tight")
        plt.close()

    # plot distributions
    jitter_rng = np.random.default_rng(SEED)  # seeded so the swarm jitter is reproducible
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (k, g) in zip(axes, perh.groupby(["cohort", "set"])):
        ax.boxplot([g["auc"]], showfliers=True)
        ax.scatter(jitter_rng.normal(1, 0.04, len(g)), g["auc"], alpha=0.5, s=20)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_title(f"{k[0]} / {k[1]}\nmedian AUC={g['auc'].median():.3f}, n={len(g)} hosp")
        ax.set_ylabel("Held-out hospital AUC")
        ax.set_ylim(0.3, 1.0)
        ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(FIG / "04_loho_distribution.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "04_loho_distribution.pdf", bbox_inches="tight")
    plt.close()
    print("\n[OK] Saved: results/04_loho_*.csv  figures/04_loho_distribution.{png,pdf}")

if __name__ == "__main__":
    main()
