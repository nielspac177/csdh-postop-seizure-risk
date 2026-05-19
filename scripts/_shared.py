"""Shared utilities for revision analyses.

n_jobs is forced to 1 throughout for Apple Silicon stability.
"""
import os, sys, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_analyses"
RES = OUT / "results"
FIG = OUT / "figures"
CACHE = OUT / "cache"
for p in (RES, FIG, CACHE):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
N_JOBS = 1  # Apple Silicon stability — never change

# ──────────────────────────────────────────────────────────────────────────
# BIDMC
# ──────────────────────────────────────────────────────────────────────────
BIDMC_CSV = ROOT / "csdh_clean.csv"

POSTOP_A_FEATURES = [
    "sex", "age", "blood_type", "sdh_type", "sdh_thickness", "csdh_size_change",
    "mid_shift", "hematoma_lat", "collection_density", "preop_gcs",
    "epilepsy_hx", "num_prev_sdh", "demographic", "procedures",
    "surg_decompression", "mma_embo", "drainage", "postop_gcs",
    "aed_timing_recoded", "prop_aed", "ab_eeg",
]
POSTOP_B_FEATURES = [
    "sex", "age", "blood_type", "sdh_type", "sdh_thickness", "csdh_size_change",
    "mid_shift", "hematoma_lat", "collection_density", "preop_gcs",
    "epilepsy_hx", "num_prev_sdh", "demographic", "procedures",
    "surg_decompression", "mma_embo", "drainage", "postop_gcs",
]
PREOP_FEATURES = [
    "sex", "age", "blood_type", "sdh_type", "sdh_thickness", "csdh_size_change",
    "mid_shift", "hematoma_lat", "collection_density", "preop_gcs",
    "epilepsy_hx", "num_prev_sdh", "demographic", "procedures",
]

def load_bidmc():
    df = pd.read_csv(BIDMC_CSV)
    df["aed_timing_recoded"] = df["aed_timing"].replace({2: 1})
    return df

# ──────────────────────────────────────────────────────────────────────────
# eICU
# ──────────────────────────────────────────────────────────────────────────
EICU_CSV = ROOT / "eicu_csdh_cohort_final.csv"

EICU_SET_A = ["age", "sex", "gcs_admission", "prior_seizures", "craniotomy",
              "burr_hole", "prophylactic_aed", "pre_admission_aed"]

EICU_SET_B_ADD = ["apache_score", "apache_gcs", "any_anticoagulant", "any_antiplatelet",
                  "any_steroid", "mechanical_ventilation", "icp_monitor", "blood_transfusion",
                  "comorbidity_count", "hx_hypertension", "hx_diabetes", "hx_stroke",
                  "hx_dementia", "hx_afib", "hx_alcohol", "pupil_abnormality",
                  "icp_available", "n_labs_24h", "n_vitals_24h", "Hgb_first",
                  "PT___INR_first", "WBC_x_1000_first", "albumin_first", "calcium_first",
                  "creatinine_first", "glucose_first", "lactate_first", "magnesium_first",
                  "platelets_x_1000_first", "potassium_first", "sodium_first",
                  "total_bilirubin_first", "temperature_mean", "sao2_mean",
                  "heartrate_mean", "respiration_mean", "systemicsystolic_mean",
                  "systemicdiastolic_mean", "systemicmean_mean"]

EICU_SET_C_ADD = ["gcs_24h", "gcs_min_24h", "gcs_max_24h", "gcs_delta", "n_gcs_assessments",
                  "sodium_slope_48h", "sodium_delta_48h", "sodium_cv_48h",
                  "potassium_slope_48h", "potassium_delta_48h", "potassium_cv_48h",
                  "glucose_slope_48h", "glucose_delta_48h", "glucose_cv_48h",
                  "creatinine_slope_48h", "creatinine_delta_48h", "creatinine_cv_48h",
                  "platelets_x_1000_slope_48h", "platelets_x_1000_delta_48h", "platelets_x_1000_cv_48h",
                  "lactate_slope_48h", "lactate_delta_48h", "lactate_cv_48h",
                  "WBC_x_1000_slope_48h", "WBC_x_1000_delta_48h", "WBC_x_1000_cv_48h",
                  "Hgb_slope_48h", "Hgb_delta_48h", "Hgb_cv_48h",
                  "heartrate_slope_48h", "heartrate_delta_48h", "heartrate_cv_48h",
                  "systemicsystolic_slope_48h", "systemicsystolic_delta_48h", "systemicsystolic_cv_48h",
                  "systemicmean_slope_48h", "systemicmean_delta_48h", "systemicmean_cv_48h",
                  "temperature_slope_48h", "temperature_delta_48h", "temperature_cv_48h",
                  "sao2_slope_48h", "sao2_delta_48h", "sao2_cv_48h",
                  "temperature_min", "temperature_max", "temperature_std",
                  "sao2_min", "sao2_max", "sao2_std",
                  "heartrate_min", "heartrate_max", "heartrate_std",
                  "systemicsystolic_min", "systemicsystolic_max", "systemicsystolic_std"]

EICU_SET_B = EICU_SET_A + EICU_SET_B_ADD
EICU_SET_C = EICU_SET_B + EICU_SET_C_ADD

def load_eicu_pure(filter_post_seizure=True):
    """Pure cSDH cohort: no prior seizures, no pre-admission AED, no mechanical ventilation."""
    df = load_eicu(filter_post_seizure=filter_post_seizure)
    mask = (df["prior_seizures"] == 0) & (df["pre_admission_aed"] == 0) & (df["mechanical_ventilation"] == 0)
    return df.loc[mask].reset_index(drop=True)

def load_eicu(filter_postop_csdh=False, filter_post_seizure=True):
    df = pd.read_csv(EICU_CSV)
    if filter_post_seizure:
        # Drop patients whose seizure_offset_min < 0 (seizure before/at admission)
        # — they are not "postoperative" prediction targets.
        keep = ~((df["seizure"] == 1) & (df["seizure_offset_min"].fillna(0) < 0))
        df = df.loc[keep].reset_index(drop=True)
    if filter_postop_csdh:
        # Heuristic for postoperative chronic SDH:
        # craniotomy or burr_hole = 1 (used in feature set)
        df = df.loc[(df["craniotomy"] == 1) | (df["burr_hole"] == 1)].reset_index(drop=True)
    return df

# ──────────────────────────────────────────────────────────────────────────
# Pipeline factories — fresh fit, n_jobs=1
# ──────────────────────────────────────────────────────────────────────────
def make_pipeline_postopA():
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from imblearn.ensemble import BalancedRandomForestClassifier
    prep = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler())]), POSTOP_A_FEATURES)]
    )
    clf = BalancedRandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        n_jobs=N_JOBS, random_state=SEED,
    )
    return Pipeline([("prep", prep), ("clf", clf)])

def make_pipeline_postopB():
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.linear_model import LogisticRegression
    prep = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler())]), POSTOP_B_FEATURES)]
    )
    clf = LogisticRegression(
        penalty="elasticnet", l1_ratio=0.5, C=1.0, solver="saga",
        max_iter=5000, n_jobs=N_JOBS, random_state=SEED,
    )
    return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=SEED, k_neighbors=3)), ("clf", clf)])

def make_pipeline_preop():
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    prep = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler())]), PREOP_FEATURES)]
    )
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced",
        n_jobs=N_JOBS, random_state=SEED,
    )
    return Pipeline([("prep", prep), ("clf", clf)])

def make_pipeline_eicu(features, model="rf_balanced"):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    prep = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler())]), features)]
    )
    if model == "rf_balanced":
        clf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            n_jobs=N_JOBS, random_state=SEED,
        )
        return Pipeline([("prep", prep), ("clf", clf)])
    if model == "logreg_enet":
        clf = LogisticRegression(
            penalty="elasticnet", l1_ratio=0.5, C=1.0, solver="saga",
            max_iter=5000, n_jobs=N_JOBS, random_state=SEED,
        )
        return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=SEED, k_neighbors=3)), ("clf", clf)])
    raise ValueError(model)

# ──────────────────────────────────────────────────────────────────────────
# OOF predictions via repeated stratified CV (n_jobs=1)
# ──────────────────────────────────────────────────────────────────────────
def oof_predictions(make_pipe_fn, X, y, n_splits=5, n_repeats=5, groups=None):
    """Return mean OOF probability for each row across repeats."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.base import clone
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    p_acc = np.zeros(len(X))
    n_acc = np.zeros(len(X))
    for split_idx, (tr, te) in enumerate(rskf.split(X, y)):
        pipe = make_pipe_fn()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        p_acc[te] += p
        n_acc[te] += 1
    return p_acc / np.maximum(n_acc, 1)

# ──────────────────────────────────────────────────────────────────────────
# Calibration metrics
# ──────────────────────────────────────────────────────────────────────────
def calibration_metrics(y, p, n_bins=10):
    """Return dict with brier, citl, slope, intercept, ece, mce, hl_p."""
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve
    from scipy import stats
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {}
    out["brier"]  = brier_score_loss(y, p)
    out["citl"]   = float(p.mean() - y.mean())  # calibration-in-the-large (mean diff)
    # logistic recalibration: log(p/(1-p)) → outcome
    logit = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(logit), logit])
    # use sklearn LogisticRegression on logit (no penalty)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, fit_intercept=False, max_iter=1000)
    lr.fit(X, y)
    out["intercept"] = float(lr.coef_[0, 0])
    out["slope"]     = float(lr.coef_[0, 1])
    # calibration curve
    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    out["bin_obs"]  = prob_true.tolist()
    out["bin_pred"] = prob_pred.tolist()
    # ECE / MCE (quantile bins, weighted by bin size)
    quantiles = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    quantiles[0] = 0.0; quantiles[-1] = 1.0 + 1e-9
    bin_idx = np.digitize(p, quantiles) - 1
    ece = mce = 0.0
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        diff = abs(p[m].mean() - y[m].mean())
        w = m.sum() / len(p)
        ece += w * diff
        mce = max(mce, diff)
    out["ece"] = float(ece)
    out["mce"] = float(mce)
    # Hosmer-Lemeshow
    quantiles = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    quantiles[0] = 0.0; quantiles[-1] = 1.0 + 1e-9
    bin_idx = np.digitize(p, quantiles) - 1
    hl = 0.0; df_hl = 0
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        obs = y[m].sum()
        exp = p[m].sum()
        if exp <= 0 or exp >= m.sum():
            continue
        hl += (obs - exp) ** 2 / (exp * (1 - exp / m.sum()))
        df_hl += 1
    out["hl_chi2"] = float(hl)
    out["hl_p"]    = float(1 - stats.chi2.cdf(hl, max(df_hl - 2, 1)))
    return out
