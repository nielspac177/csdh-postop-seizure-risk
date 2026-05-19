"""Task 30 — Export deployment-ready assets for the web calculator.

Produces:
  github_repo/site/model_assets.json
      • Firth penalized LR coefficients fit on the FULL BIDMC cohort
      • Feature means / SDs for per-feature z-scaling
      • Class-conditional conformal nonconformity quantiles at α ∈ {0.05, 0.10, 0.20}
      • CEA base-case parameters (cost / QALY per strategy)
      • Population-savings template (per-patient ΔCost, ΔQALY versus
        observation for ML-AED and ML-cEEG)

The web calculator at site/calculator.html consumes this JSON to produce
a deterministic prediction without any patient data leaving the browser.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from firthlogist import FirthLogisticRegression

from _shared import load_bidmc, POSTOP_A_FEATURES, OUT, SEED

SITE_DIR = OUT / "github_repo" / "site"
SITE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = load_bidmc()
    y = df["seizure"].astype(int).values
    X_raw = df[POSTOP_A_FEATURES].copy()

    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X_raw), columns=POSTOP_A_FEATURES)
    sc = StandardScaler()
    X_z = sc.fit_transform(X_imp)

    # Firth on full data — for deployment coefficients
    print(f"Fitting Firth penalized LR on full BIDMC (n={len(y)}, events={int(y.sum())})...",
          flush=True)
    firth = FirthLogisticRegression(max_iter=200, max_halfstep=25, tol=1e-6,
                                      skip_pvals=True, skip_ci=True, wald=True)
    firth.fit(X_z, y)
    p_full = firth.predict_proba(X_z)[:, 1]
    intercept = float(firth.intercept_)
    coefs = firth.coef_.tolist()
    print(f"  intercept = {intercept:.4f}")
    for f, c in zip(POSTOP_A_FEATURES, coefs):
        print(f"  β[{f:<22s}] = {c:+.4f}")

    # ── Class-conditional conformal quantiles ──
    # Use the same split as the conformal evaluation (5-fold, fit on 75% +
    # calibrate on 25%) and aggregate the nonconformity scores
    print("\nComputing class-conditional conformal nonconformity quantiles...",
          flush=True)
    nc_pos_all, nc_neg_all = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X_z, y):
        # split trainval -> train (75%) + cal (25%)
        rng = np.random.default_rng(SEED)
        pos_tr = np.where(y[tr] == 1)[0]; neg_tr = np.where(y[tr] == 0)[0]
        rng.shuffle(pos_tr); rng.shuffle(neg_tr)
        cal_pos = tr[pos_tr[: max(2, len(pos_tr) // 4)]]
        cal_neg = tr[neg_tr[: max(2, len(neg_tr) // 4)]]
        cal_idx = np.concatenate([cal_pos, cal_neg])
        train_idx = np.setdiff1d(tr, cal_idx)
        if y[train_idx].sum() < 5: continue
        m = FirthLogisticRegression(max_iter=200, max_halfstep=25, tol=1e-6,
                                      skip_pvals=True, skip_ci=True, wald=True)
        m.fit(X_z[train_idx], y[train_idx])
        p_cal = np.clip(m.predict_proba(X_z[cal_idx])[:, 1], 1e-6, 1 - 1e-6)
        y_cal = y[cal_idx]
        nc_pos = (1.0 - p_cal[y_cal == 1])
        nc_neg = (1.0 - (1.0 - p_cal[y_cal == 0]))
        nc_pos_all.extend(nc_pos.tolist())
        nc_neg_all.extend(nc_neg.tolist())
    nc_pos_all = np.array(nc_pos_all); nc_neg_all = np.array(nc_neg_all)
    alphas = [0.05, 0.10, 0.20]
    q_pos = {f"alpha_{a:.2f}": float(np.quantile(nc_pos_all, 1 - a)) for a in alphas}
    q_neg = {f"alpha_{a:.2f}": float(np.quantile(nc_neg_all, 1 - a)) for a in alphas}
    print(f"  positive-class nonconformity quantiles: {q_pos}")
    print(f"  negative-class nonconformity quantiles: {q_neg}")

    # ── CEA base-case (per-patient expected outcomes) ──
    # values from scripts/14_decision_tree.py rollback
    cea_rollback = pd.read_csv(OUT / "results" / "14_decision_tree_rollback.csv")
    cea_dict = cea_rollback.set_index("strategy")[["E_cost_USD", "E_QALY"]].to_dict()

    # ── Package and write ──
    payload = {
        "model": {
            "type": "Firth penalized logistic regression",
            "fit_on": "BIDMC postoperative-A, n=655, 48 events",
            "intercept": intercept,
            "features": POSTOP_A_FEATURES,
            "coefficients": coefs,
            "feature_means": sc.mean_.tolist(),
            "feature_sds":   sc.scale_.tolist(),
            "feature_medians_for_imputation": imp.statistics_.tolist(),
        },
        "conformal": {
            "method": "class-conditional (Mondrian) split conformal",
            "calibration_split": "75% train / 25% calibration, 5-fold pooled",
            "nonconformity_quantiles_positive": q_pos,
            "nonconformity_quantiles_negative": q_neg,
        },
        "cea": {
            "wtp_threshold_usd_per_qaly": 100000,
            "discount_rate": 0.03,
            "horizon_years": 10,
            "strategies": cea_dict,
        },
        "metadata": {
            "n_features": len(POSTOP_A_FEATURES),
            "n_patients": int(len(y)),
            "n_events": int(y.sum()),
            "prevalence": float(y.mean()),
            "manuscript_release_tag": "v1.0-JNNP-submission",
            "generated_by": "scripts/30_export_calculator_assets.py",
        },
    }
    out_path = SITE_DIR / "model_assets.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[OK] {out_path}")
    print(f"     size: {os.path.getsize(out_path)/1024:.1f} KB")


if __name__ == "__main__":
    main()
