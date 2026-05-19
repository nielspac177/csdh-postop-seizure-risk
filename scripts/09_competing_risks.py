"""Task 9 — Competing-risks / discrete-time survival.

Fix A (severity HIGH): sensitivity analysis for the 7 BIDMC seizures with
  missing `time_seizure`. Compare 4 imputation scenarios:
    1. median (= 3 days)  — current
    2. drop                — exclude these 7 cases
    3. day_1               — assume early seizure
    4. day_7               — assume horizon-edge seizure

Fix I (severity LOW):
  - Fine-Gray subdistribution model (lifelines.fitters.CRCSplineFitter)
  - Schoenfeld test for proportional-hazards assumption

Outputs:
  results/09_competing_risks.csv
  results/09_missing_time_sensitivity.csv
  figures/09_cumulative_incidence.png
  figures/09_missing_time_sensitivity.png
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
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from _shared import load_bidmc, POSTOP_B_FEATURES, RES, FIG, SEED

def make_event_time_data(df, horizon_days=7, missing_strategy="median"):
    """status: 1 = seizure, 2 = death, 0 = censored.
       missing_strategy ∈ {'median','drop','day_1','day_7'} controls how the 7
       BIDMC seizures with missing time_seizure are handled.
    """
    out = df.copy()
    out["time_seizure"] = pd.to_numeric(out["time_seizure"], errors="coerce")
    out["los"] = pd.to_numeric(out["los"], errors="coerce")
    out["mortality"] = pd.to_numeric(out["mortality"], errors="coerce").fillna(0)

    miss_idx = (out.seizure == 1) & out["time_seizure"].isna()
    n_missing = int(miss_idx.sum())
    if missing_strategy == "median":
        median_t = out.loc[out.seizure == 1, "time_seizure"].median()
        out.loc[miss_idx, "time_seizure"] = median_t
    elif missing_strategy == "day_1":
        out.loc[miss_idx, "time_seizure"] = 1.0
    elif missing_strategy == "day_7":
        out.loc[miss_idx, "time_seizure"] = horizon_days
    elif missing_strategy == "drop":
        out = out.loc[~miss_idx].reset_index(drop=True)
    else:
        raise ValueError(missing_strategy)

    status = np.zeros(len(out), dtype=int)
    time_  = np.full(len(out), float(horizon_days))
    sz_in_window = (out.seizure == 1) & (out.time_seizure <= horizon_days)
    out_idx = sz_in_window.values
    time_[out_idx] = out.loc[sz_in_window, "time_seizure"].values
    status[out_idx] = 1
    death_in_window = (out.mortality == 1) & (out.los <= horizon_days) & (~sz_in_window)
    time_[death_in_window.values] = out.loc[death_in_window, "los"].values
    status[death_in_window.values] = 2
    disch_before = (out.los < horizon_days) & (status == 0)
    time_[disch_before.values] = out.loc[disch_before, "los"].values

    out["surv_time"] = time_
    out["surv_status"] = status
    out["event_7d"] = (status == 1).astype(int)
    out._missing_n = n_missing  # type: ignore
    return out

def discrete_time_auc(d, n_splits=5, n_repeats=5):
    y = d["event_7d"].astype(int)
    X = d[POSTOP_B_FEATURES]
    pipe = Pipeline([
        ("prep", ColumnTransformer([("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())]), POSTOP_B_FEATURES)])),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                    max_iter=5000, n_jobs=1, random_state=SEED)),
    ])
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    aucs = []
    oof = np.zeros(len(d)); cnt = np.zeros(len(d))
    for tr, te in rskf.split(X, y):
        m = clone(pipe)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        oof[te] += p; cnt[te] += 1
    oof /= np.maximum(cnt, 1)
    return float(np.mean(aucs)), float(np.std(aucs)), oof

def cox_concordance(d, ph_test=False):
    """Cause-specific Cox via lifelines, treating death as censoring for seizure."""
    from lifelines import CoxPHFitter
    cox_feats = [c for c in ["age","preop_gcs","postop_gcs","sdh_thickness",
                              "csdh_size_change","mid_shift","epilepsy_hx",
                              "drainage","mma_embo","sex"] if c in d.columns]
    coxd = d[cox_feats + ["surv_time","event_7d"]].copy()
    for c in cox_feats:
        coxd[c] = pd.to_numeric(coxd[c], errors="coerce")
    coxd = coxd.fillna(coxd.median(numeric_only=True))
    coxd["surv_time"] = coxd["surv_time"] + np.random.default_rng(SEED).uniform(0, 1e-3, len(coxd))
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(coxd, duration_col="surv_time", event_col="event_7d")
    c_idx = cph.concordance_index_
    ph_summary = None
    if ph_test:
        try:
            from lifelines.statistics import proportional_hazard_test
            ph = proportional_hazard_test(cph, coxd, time_transform="rank")
            ph_summary = ph.summary if hasattr(ph, "summary") else ph
        except Exception as e:
            ph_summary = f"PH test failed: {e}"
    return float(c_idx), ph_summary

def fine_gray(d):
    """Fine-Gray subdistribution-hazard model via IPCW weighting.

    For each observation:
      - If event of interest (seizure): weight = 1.
      - If competing event (death):    weight 1/G(t_i, t) for t > t_i  (kept in risk set).
      - If censored:                   weight = 1 (standard contribution).

    We implement the Geskus subdistribution dataset trick: subjects who
    experienced the competing event (death) remain in the risk set for the
    seizure outcome with their administrative censoring time = horizon and
    weight from the censoring distribution G(t). Approximate using
    Kaplan-Meier of the censoring time. This gives the Fine-Gray
    subdistribution-hazard estimator (Fine & Gray, JASA 1999).
    """
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        # Step 1: estimate censoring-distribution G(t) — KM on
        # event = (status == 0)  (i.e., truly censored), reverse Kaplan-Meier.
        kmf = KaplanMeierFitter()
        cens_indicator = (d["surv_status"] == 0).astype(int)
        kmf.fit(d["surv_time"].values, event_observed=cens_indicator.values)

        # Step 2: build subdistribution dataset
        cox_feats = [c for c in ["age","preop_gcs","postop_gcs","sdh_thickness",
                                  "csdh_size_change","mid_shift","epilepsy_hx",
                                  "drainage","mma_embo","sex"] if c in d.columns]
        sub = d[cox_feats + ["surv_time","surv_status"]].copy()
        for c in cox_feats:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.fillna(sub.median(numeric_only=True))

        # competing-event subjects: censored at horizon (kept in risk set), weight = G(t)/G(t_i)
        horizon = 7.0
        T = sub["surv_time"].values
        S = sub["surv_status"].values
        T_new = T.copy()
        E_new = (S == 1).astype(int)
        W = np.ones(len(sub))
        # for competing-event subjects (status==2), advance time to horizon
        comp_mask = (S == 2)
        T_new[comp_mask] = horizon
        # weight: 1 at t_i for primary, smoothly down at horizon
        Gt_horizon = float(kmf.survival_function_at_times(horizon).iloc[0])
        Gt_event   = kmf.survival_function_at_times(T).values
        Gt_event   = np.maximum(Gt_event, 1e-3)
        W[comp_mask] = Gt_horizon / Gt_event[comp_mask]

        sub["surv_time"] = T_new + np.random.default_rng(SEED).uniform(0, 1e-3, len(sub))
        sub["sub_event"] = E_new
        sub["wt"]        = W

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(sub.drop(columns=["surv_status"]), duration_col="surv_time",
                event_col="sub_event", weights_col="wt", robust=True)
        return float(cph.concordance_index_), None
    except Exception as e:
        return float("nan"), f"Fine-Gray failed: {e}"

def main():
    df = load_bidmc()

    # ── Fix A: Sensitivity over missing-time-seizure imputation ──────────
    print("Sensitivity over time_seizure imputation strategies", flush=True)
    sens_rows = []
    for strategy in ["median", "drop", "day_1", "day_7"]:
        d = make_event_time_data(df, horizon_days=7, missing_strategy=strategy)
        n_missing = getattr(d, "_missing_n", 7)
        n_events = int(d["event_7d"].sum())
        auc_m, auc_s, _ = discrete_time_auc(d)
        c_idx, _ = cox_concordance(d, ph_test=False)
        sens_rows.append({
            "missing_strategy": strategy,
            "n": len(d), "n_missing_handled": n_missing,
            "events_7d": n_events,
            "discrete_lr_auc": auc_m, "discrete_lr_sd": auc_s,
            "cox_c_index": c_idx,
        })
        print(f"  [{strategy}] n={len(d)}, events={n_events}, "
              f"discrete-LR AUC={auc_m:.3f}±{auc_s:.3f}, Cox c={c_idx:.3f}", flush=True)
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(RES / "09_missing_time_sensitivity.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos = np.arange(len(sens_df))
    width = 0.35
    ax.barh(pos - width/2, sens_df["discrete_lr_auc"],
            xerr=sens_df["discrete_lr_sd"], height=width,
            label="Discrete-time LR AUC", color="tab:blue", capsize=3)
    ax.barh(pos + width/2, sens_df["cox_c_index"], height=width,
            label="Cox c-index", color="tab:orange")
    ax.set_yticks(pos)
    ax.set_yticklabels([f'{r.missing_strategy} (n={r.n}, ev={r.events_7d})'
                         for r in sens_df.itertuples()])
    ax.invert_yaxis()
    ax.axvline(0.5, color="gray", ls=":")
    ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("Predictive performance")
    ax.set_title("BIDMC 7-day seizure prediction — missing-time-seizure sensitivity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "09_missing_time_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "09_missing_time_sensitivity.pdf", bbox_inches="tight")
    plt.close()

    # ── Main results: median strategy (default) + Fine-Gray + PH test ──
    d = make_event_time_data(df, horizon_days=7, missing_strategy="median")
    print(f"\nDefault scenario: median imputation. BIDMC n={len(d)}")
    print("Status counts:")
    print(d["surv_status"].value_counts().sort_index().rename({0:"censored",1:"seizure",2:"death"}))

    rows = []
    auc_m, auc_s, oof = discrete_time_auc(d)
    rows.append({"analysis":"7-day discrete-time LR (postop_B feats)",
                 "n":len(d), "events":int(d["event_7d"].sum()),
                 "auc_mean":auc_m, "auc_sd":auc_s})

    c_idx, ph = cox_concordance(d, ph_test=True)
    rows.append({"analysis":"Cause-specific Cox (10 covars, penalized)",
                 "n":len(d), "events":int(d["event_7d"].sum()),
                 "auc_mean":c_idx, "auc_sd":float("nan")})
    if isinstance(ph, pd.DataFrame):
        ph.to_csv(RES / "09_cox_ph_assumption_test.csv")
        print("\nProportional hazards Schoenfeld test:")
        print(ph.round(3).to_string())
    elif ph is not None:
        print(f"\nPH test note: {ph}")

    fg_c, fg_msg = fine_gray(d)
    rows.append({"analysis":"Fine-Gray approx (IPCW-weighted Cox)",
                 "n":len(d), "events":int(d["event_7d"].sum()),
                 "auc_mean":fg_c, "auc_sd":float("nan")})
    if fg_msg: print(f"\nFine-Gray note: {fg_msg}")
    print(f"Fine-Gray approx c-index: {fg_c:.3f}")

    # ── CIF plot (unchanged from earlier) ─────────────────────────────────
    try:
        from lifelines import AalenJohansenFitter
        risk_tertile = pd.Series(pd.qcut(oof, 3, labels=["Low", "Mid", "High"]),
                                 index=d.index)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for label, color in [("Low", "tab:green"), ("Mid", "tab:orange"), ("High", "tab:red")]:
            mask = (risk_tertile == label).values
            if mask.sum() < 5: continue
            ajf = AalenJohansenFitter()
            t = pd.Series(d.loc[mask, "surv_time"].values.astype(float))
            s = pd.Series(d.loc[mask, "surv_status"].values.astype(int))
            ajf.fit(t, s, event_of_interest=1)
            ajf.plot(ax=axes[0], color=color, label=f"{label} (n={mask.sum()})")
        axes[0].set_xlabel("Days post-op")
        axes[0].set_ylabel("Cumulative incidence — seizure")
        axes[0].set_title("Seizure CIF by risk tertile (competing risk: death)")
        axes[0].set_xlim(0, 7); axes[0].grid(alpha=0.3)

        bins = pd.qcut(oof, 5, duplicates="drop")
        agg = d.assign(bin=bins, oof=oof).groupby("bin").agg(
            obs=("event_7d","mean"), pred=("oof","mean"), n=("event_7d","count"))
        axes[1].plot(agg["pred"], agg["obs"], "o-", lw=2, ms=8)
        axes[1].plot([0,1],[0,1], "k--", lw=1)
        axes[1].set_xlim(0, max(0.4, agg["pred"].max()*1.1))
        axes[1].set_ylim(0, max(0.4, agg["obs"].max()*1.1))
        axes[1].set_xlabel("Predicted 7-day risk")
        axes[1].set_ylabel("Observed 7-day rate")
        axes[1].set_title("7-day discrete-time calibration (quintiles)")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "09_cumulative_incidence.png", dpi=200, bbox_inches="tight")
        plt.savefig(FIG / "09_cumulative_incidence.pdf", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [WARN] AJ plot failed: {e}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RES / "09_competing_risks.csv", index=False)
    print("\nMain results (median imputation):")
    print(out_df.round(3).to_string(index=False))
    print("\nSensitivity across missing-time strategies:")
    print(sens_df.round(3).to_string(index=False))
    print("\n[OK] Saved: results/09_*.csv  figures/09_*.{png,pdf}")

if __name__ == "__main__":
    main()
