"""Task 12 — NIS seizure reclassification (R56.9 vs G40.*).

Reviewer Q9: separate acute symptomatic seizures (R56.x / 78039 / G41.x)
from pre-existing epilepsy (G40.x / 345.x) so the prediction target reflects
the postoperative-cSDH question rather than chronic epilepsy status.

Outputs:
  results/12_nis_seizure_codes.csv    — code-level seizure-flag breakdown
  results/12_nis_outcome_comparison.csv — AUC by outcome definition
  figures/12_nis_outcome_auc.png
  cache/nis_chronic.parquet           — reused by task 13
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re, gc, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from _shared import RES, FIG, CACHE, SEED

NIS_PATH = "/Volumes/Niels 2/NIS_new_version/NIS_cSDH_seizure/csdh_cohort.parquet"
DX_COLS = [f"DX{i}" for i in range(1, 31)]

# ── Code patterns ─────────────────────────────────────────────────────
RE_R56     = re.compile(r"^R56")        # ICD-10 convulsions (acute symptomatic)
RE_G40     = re.compile(r"^G40")        # ICD-10 pre-existing epilepsy
RE_G41     = re.compile(r"^G41")        # ICD-10 status epilepticus
ICD9_345   = "345"                       # ICD-9 epilepsy
ICD9_3453  = "3453"                      # ICD-9 status epilepticus (345.3)
ICD9_78039 = "78039"                     # ICD-9 convulsions

def flag_from_codes(df, dx_cols):
    """Return three boolean Series: acute_symptomatic, pre_existing_epilepsy, status_epil."""
    acute = pd.Series(False, index=df.index)
    epi   = pd.Series(False, index=df.index)
    se    = pd.Series(False, index=df.index)
    for c in dx_cols:
        if c not in df.columns: continue
        v = df[c].astype(str).str.strip()
        # acute symptomatic: R56.x, 780.39
        acute = acute | v.str.match(RE_R56.pattern, na=False) | (v == ICD9_78039)
        # pre-existing epilepsy: G40.x, 345.x except 345.3
        epi_mask = v.str.match(RE_G40.pattern, na=False) | (v.str.startswith("345") & (v != ICD9_3453))
        epi   = epi | epi_mask
        # status epilepticus: G41.x, 345.3
        se    = se | v.str.match(RE_G41.pattern, na=False) | (v == ICD9_3453)
    return acute, epi, se

def cv_auc(make_fn, X, y, n_splits=5, n_repeats=3, n_jobs=1):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    aucs, praucs, briers = [], [], []
    for tr, te in rskf.split(X, y):
        if y.iloc[tr].sum() < 5: continue
        m = make_fn()
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        praucs.append(average_precision_score(y.iloc[te], p))
        briers.append(brier_score_loss(y.iloc[te], p))
    return aucs, praucs, briers

def make_lr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", class_weight="balanced",
                                    max_iter=1000, n_jobs=1, random_state=SEED)),
    ])

def make_rf():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        max_depth=10, min_samples_leaf=20,
                                        n_jobs=1, random_state=SEED)),
    ])

# Clean features (mirroring 03_csdh_seizure_prediction)
CLEAN_FEATS = [
    "AGE","female","race_white","race_black","race_hispanic","race_asian","race_other",
    "pay_medicare","pay_medicaid","pay_private","pay_other",
    "income_q1","income_q2","income_q3","income_q4",
    "elective","weekend","transfer_in",
    "proc_craniotomy","proc_burr_hole",
    "is_chronic","is_acute",
    "n_dx","n_pr",
    "div_1","div_2","div_3","div_4","div_5","div_6","div_7","div_8","div_9",
    "hypertension","diabetes","afib","heart_failure","ckd",
    "coagulopathy","alcohol","liver_disease","dementia",
    "obesity","tobacco","prior_stroke","cad",
    "antiplatelet","anticoagulant",
    "comorbidity_count",
]

def engineer(cohort):
    c = cohort.copy()
    c["female"] = (c["FEMALE"] == 1).astype(int)
    c["race_white"]    = (c["RACE"] == 1).astype(int)
    c["race_black"]    = (c["RACE"] == 2).astype(int)
    c["race_hispanic"] = (c["RACE"] == 3).astype(int)
    c["race_asian"]    = (c["RACE"] == 4).astype(int)
    c["race_other"]    = (c["RACE"].isin([5, 6])).astype(int)
    c["pay_medicare"]  = (c["PAY1"] == 1).astype(int)
    c["pay_medicaid"]  = (c["PAY1"] == 2).astype(int)
    c["pay_private"]   = (c["PAY1"] == 3).astype(int)
    c["pay_other"]     = (c["PAY1"].isin([4, 5, 6])).astype(int)
    for q in [1, 2, 3, 4]:
        c[f"income_q{q}"] = (c["ZIPINC_QRTL"] == q).astype(int)
    c["elective"]    = (c["ELECTIVE"] == 1).astype(int)
    c["weekend"]     = (c["AWEEKEND"] == 1).astype(int)
    c["transfer_in"] = (c["TRAN_IN"]  == 1).astype(int)
    c["proc_craniotomy"] = c["craniotomy"].astype(int)
    c["proc_burr_hole"]  = c["burr_hole"].astype(int)
    c["is_chronic"] = (c["sdh_type"] == "Nontraumatic_SDH_chronic").astype(int)
    c["is_acute"]   = (c["sdh_type"] == "Nontraumatic_SDH_acute").astype(int)
    c["n_dx"] = pd.to_numeric(c["NDX"], errors="coerce")
    c["n_dx"] = c["n_dx"].fillna(c["n_dx"].median())
    c["n_pr"] = pd.to_numeric(c["NPR"], errors="coerce")
    c["n_pr"] = c["n_pr"].fillna(c["n_pr"].median())
    c["HOSP_DIVISION"] = c["HOSP_DIVISION"].fillna(0)
    for d in range(1, 10):
        c[f"div_{d}"] = (c["HOSP_DIVISION"] == d).astype(int)
    for col in ["hypertension","diabetes","afib","heart_failure","ckd",
                "coagulopathy","alcohol","liver_disease","dementia",
                "obesity","tobacco","prior_stroke","cad",
                "antiplatelet","anticoagulant"]:
        c[col] = c[col].astype(int)
    c["comorbidity_count"] = c["comorbidity_count"].astype(int)
    return c

def main():
    print("Loading NIS cSDH parquet...", flush=True)
    df = pd.read_parquet(NIS_PATH)
    print(f"  full SDH: {len(df):,}", flush=True)

    # Use SECONDARY DX (DX2-DX30) for outcome flags
    sec_cols = DX_COLS[1:]
    print("Re-flagging seizure types from DX2-DX30...", flush=True)
    acute, epi, se = flag_from_codes(df, sec_cols)

    # Code-level breakdown
    rows = [
        ["any_R56_or_78039 (acute symptomatic)", int(acute.sum()), float(acute.mean())],
        ["any_G40_or_345_excl_345.3 (pre-existing epilepsy)", int(epi.sum()), float(epi.mean())],
        ["any_G41_or_345.3 (status epilepticus)", int(se.sum()), float(se.mean())],
        ["acute OR SE (corrected outcome)", int((acute | se).sum()), float((acute | se).mean())],
        ["original collapsed flag (paper)", int(df["seizure"].sum()), float(df["seizure"].mean())],
    ]
    code_df = pd.DataFrame(rows, columns=["category", "n", "rate"])
    code_df.to_csv(RES / "12_nis_seizure_codes.csv", index=False)
    print(code_df.to_string(index=False))

    # ── Restrict to paper cohort (chronic + surgical) ────────────────────
    chronic_surg = (df["is_chronic"] if "is_chronic" in df.columns
                     else (df["sdh_type"] == "Nontraumatic_SDH_chronic")) & df["surgical"]
    chronic_surg = chronic_surg.fillna(False)
    sub = df.loc[chronic_surg].copy()
    print(f"\nChronic + surgical cohort: n={len(sub):,}", flush=True)

    # Recompute outcome flags within this subset (fresh, to be safe)
    acute_s, epi_s, se_s = flag_from_codes(sub, sec_cols)
    sub["seizure_orig"]    = sub["seizure"].astype(int)
    sub["seizure_acute"]   = (acute_s | se_s).astype(int)   # corrected: acute symptomatic ± SE
    sub["pre_epilepsy"]    = epi_s.astype(int)
    sub["se_only"]         = se_s.astype(int)

    # 4 outcome scenarios
    feats = engineer(sub)
    F = [f for f in CLEAN_FEATS if f in feats.columns]

    # Scenario 1: original collapsed seizure (paper baseline)
    S1 = feats.copy()
    # Scenario 2: acute symptomatic outcome (R56/78039/G41/345.3)
    S2 = feats.copy()
    # Scenario 3: acute symptomatic outcome AND drop patients with pre-existing epilepsy
    S3 = feats.loc[~feats.index.isin(feats[feats["seizure_orig"].fillna(0).astype(int)*0 + 0 == 1].index)].copy()
    # Actually filter on pre_epilepsy
    keep = (sub["pre_epilepsy"] == 0)
    S3 = feats.loc[keep.values].copy()

    print(f"\nCohort sizes:")
    print(f"  S1 (paper collapsed): n={len(S1):,}, events={int(sub['seizure_orig'].sum())}")
    print(f"  S2 (acute corrected): n={len(S2):,}, events={int(sub['seizure_acute'].sum())}")
    print(f"  S3 (acute, exclude pre-epi): n={len(S3):,}, events={int((sub.loc[keep,'seizure_acute']).sum())}")

    rows = []

    # S1
    print("\nS1 — original outcome", flush=True)
    y1 = sub["seizure_orig"].astype(int)
    aucs, praucs, briers = cv_auc(make_lr, S1[F], y1)
    rows.append({"scenario":"S1: original collapsed", "model":"LR",
                 "n":len(S1), "events":int(y1.sum()),
                 "auc_mean":np.mean(aucs), "auc_sd":np.std(aucs),
                 "prauc_mean":np.mean(praucs), "brier_mean":np.mean(briers)})
    print(f"  LR AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    aucs, praucs, briers = cv_auc(make_rf, S1[F], y1)
    rows.append({"scenario":"S1: original collapsed", "model":"RF",
                 "n":len(S1), "events":int(y1.sum()),
                 "auc_mean":np.mean(aucs), "auc_sd":np.std(aucs),
                 "prauc_mean":np.mean(praucs), "brier_mean":np.mean(briers)})
    print(f"  RF AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # S2
    print("\nS2 — acute symptomatic only outcome", flush=True)
    y2 = sub["seizure_acute"].astype(int)
    aucs, praucs, briers = cv_auc(make_lr, S2[F], y2)
    rows.append({"scenario":"S2: acute symptomatic", "model":"LR",
                 "n":len(S2), "events":int(y2.sum()),
                 "auc_mean":np.mean(aucs), "auc_sd":np.std(aucs),
                 "prauc_mean":np.mean(praucs), "brier_mean":np.mean(briers)})
    print(f"  LR AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    aucs, praucs, briers = cv_auc(make_rf, S2[F], y2)
    rows.append({"scenario":"S2: acute symptomatic", "model":"RF",
                 "n":len(S2), "events":int(y2.sum()),
                 "auc_mean":np.mean(aucs), "auc_sd":np.std(aucs),
                 "prauc_mean":np.mean(praucs), "brier_mean":np.mean(briers)})
    print(f"  RF AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # S3
    print("\nS3 — acute symptomatic, exclude pre-existing epilepsy", flush=True)
    y3 = sub.loc[keep, "seizure_acute"].astype(int).reset_index(drop=True)
    X3 = S3[F].reset_index(drop=True)
    aucs, praucs, briers = cv_auc(make_lr, X3, y3)
    rows.append({"scenario":"S3: acute, no pre-epi", "model":"LR",
                 "n":len(X3), "events":int(y3.sum()),
                 "auc_mean":np.mean(aucs), "auc_sd":np.std(aucs),
                 "prauc_mean":np.mean(praucs), "brier_mean":np.mean(briers)})
    print(f"  LR AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    aucs, praucs, briers = cv_auc(make_rf, X3, y3)
    rows.append({"scenario":"S3: acute, no pre-epi", "model":"RF",
                 "n":len(X3), "events":int(y3.sum()),
                 "auc_mean":np.mean(aucs), "auc_sd":np.std(aucs),
                 "prauc_mean":np.mean(praucs), "brier_mean":np.mean(briers)})
    print(f"  RF AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(RES / "12_nis_outcome_comparison.csv", index=False)
    print("\n", out.round(3).to_string(index=False))

    # Save chronic cohort with reclassified outcome for task 13
    keep_cols = ["KEY_NIS","DISCWT","HOSP_NIS","NIS_STRATUM","YEAR"] + F + \
                ["seizure_orig","seizure_acute","pre_epilepsy","se_only"]
    keep_cols = [c for c in keep_cols if c in sub.columns or c in feats.columns]
    save_df = pd.concat([feats[F].reset_index(drop=True),
                         sub[["seizure_orig","seizure_acute","pre_epilepsy","se_only"]].reset_index(drop=True)],
                         axis=1)
    save_df.to_parquet(CACHE / "nis_chronic.parquet")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    pos = np.arange(len(out))
    colors = ["tab:blue" if r.scenario.startswith("S1") else
              "tab:orange" if r.scenario.startswith("S2") else "tab:green"
              for r in out.itertuples()]
    ax.barh(pos, out["auc_mean"], xerr=out["auc_sd"], color=colors, capsize=4)
    ax.set_yticks(pos)
    ax.set_yticklabels([f'{r.scenario} | {r.model} (n={r.n}, ev={r.events})'
                        for r in out.itertuples()])
    ax.invert_yaxis()
    ax.axvline(0.5, color="gray", ls=":")
    ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("Cross-validated AUROC")
    ax.set_title("NIS chronic cohort — outcome-definition sensitivity")
    plt.tight_layout()
    plt.savefig(FIG / "12_nis_outcome_auc.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "12_nis_outcome_auc.pdf", bbox_inches="tight")
    plt.close()
    print("\n[OK] Saved: results/12_*.csv  figures/12_nis_outcome_auc.{png,pdf}")
    print("[OK] Saved: cache/nis_chronic.parquet (for task 13)")

if __name__ == "__main__":
    main()
