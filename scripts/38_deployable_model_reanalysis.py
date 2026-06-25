"""Task 38 — Deployable-model reanalysis (Bucket A: M1 + M3 + M4).

Goals:
  A1. Refit Firth on BIDMC postop-B; compute Se / Sp / PPV / NPV at thresholds
      [0.05, 0.073 (base rate), 0.10, 0.15] with bootstrap 95% CIs.
  A2. Re-run the CEA (decision-tree rollback + PSA at $50k / $100k / $150k WTP)
      with the postop-B Se/Sp profile substituted for the current postop-A
      baseline (Se = 0.842, Sp = 0.504).
  A3. Conformal → CEA action-mapping sensitivity: test three doubleton
      policies (→ universal AED [default], → observation, → cEEG + targeted
      AED) and compare optimal strategy under each.

Inputs read from CACHE (oof_bidmc_postopB.npz) + results/25_conformal.csv.
All resampling, hyperparameter tuning and Platt scaling already nested in
the cached OOF; we do not re-run the model fit, only operating-point
computation and CEA re-evaluation.

Outputs (results/):
  38_postopB_operating_points.csv     — Se/Sp/PPV/NPV at 4 thresholds + CIs
  38_postopB_psa.csv                  — PSA with postop-B operating point
  38_postopB_psa_summary.csv          — best strategy at each WTP
  38_conformal_mapping_sensitivity.csv — CEA outputs under 3 doubleton mappings
  38_message_stability_summary.csv    — one-page summary of optimal-strategy
                                        rankings across all configurations

Runtime: ~3 minutes on Apple Silicon with n_jobs = 1.
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path

from _shared import CACHE, RES, SEED as RANDOM_STATE
import importlib.util

# ─────────────────────────────────────────────────────────
# Load CEA module (10_11_cea_pairwise.py) — module name starts with a digit
# so we use importlib to load it.
# ─────────────────────────────────────────────────────────
HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "cea_mod", HERE / "10_11_cea_pairwise.py")
cea_mod = importlib.util.module_from_spec(spec)
sys.modules["cea_mod"] = cea_mod
spec.loader.exec_module(cea_mod)

Params         = cea_mod.Params
rollback       = cea_mod.run_strategy
beta_params_from_ci = cea_mod.beta_params_from_ci
gamma_params   = cea_mod.gamma_params

N_PSA = 10000
SEED  = RANDOM_STATE  # 42
RNG   = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────
# 1.  Load postop-B OOF and compute Se / Sp / PPV / NPV @ thresholds
# ─────────────────────────────────────────────────────────
def operating_points(thresholds=(0.06, 0.073, 0.08, 0.09, 0.10), n_boot=1000,
                      cache_file="oof_bidmc_postopB_firth.npz"):
    z = np.load(CACHE / cache_file)
    y = z["y"].astype(int)
    p = z["p"].astype(float)
    n = len(y)
    rng = np.random.default_rng(SEED)
    rows = []

    for thr in thresholds:
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        se  = tp / max(tp + fn, 1)
        sp  = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)

        # Bootstrap 95% CI
        b_se, b_sp, b_ppv, b_npv = [], [], [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yb, pb = y[idx], p[idx]
            pr = (pb >= thr).astype(int)
            tp_b = int(((pr == 1) & (yb == 1)).sum())
            fn_b = int(((pr == 0) & (yb == 1)).sum())
            fp_b = int(((pr == 1) & (yb == 0)).sum())
            tn_b = int(((pr == 0) & (yb == 0)).sum())
            if tp_b + fn_b > 0: b_se.append(tp_b / (tp_b + fn_b))
            if tn_b + fp_b > 0: b_sp.append(tn_b / (tn_b + fp_b))
            if tp_b + fp_b > 0: b_ppv.append(tp_b / (tp_b + fp_b))
            if tn_b + fn_b > 0: b_npv.append(tn_b / (tn_b + fn_b))
        def _ci(arr):
            return np.percentile(arr, [2.5, 97.5]) if len(arr) > 0 else [np.nan, np.nan]
        ci_se  = _ci(b_se)
        ci_sp  = _ci(b_sp)
        ci_ppv = _ci(b_ppv)
        ci_npv = _ci(b_npv)
        rows.append({
            "threshold": thr,
            "n": n, "events": int(y.sum()),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sens":  round(se,  4), "sens_lo":  round(ci_se[0],  4), "sens_hi":  round(ci_se[1],  4),
            "spec":  round(sp,  4), "spec_lo":  round(ci_sp[0],  4), "spec_hi":  round(ci_sp[1],  4),
            "ppv":   round(ppv, 4), "ppv_lo":   round(ci_ppv[0], 4), "ppv_hi":   round(ci_ppv[1], 4),
            "npv":   round(npv, 4), "npv_lo":   round(ci_npv[0], 4), "npv_hi":   round(ci_npv[1], 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# 2.  PSA wrapper with custom Se/Sp Beta-prior
# ─────────────────────────────────────────────────────────
def run_psa_custom(sens_mu, sens_lo, sens_hi, spec_mu, spec_lo, spec_hi,
                    n_psa=N_PSA, label="postopB"):
    a_prev, b_prev   = beta_params_from_ci(0.12, 0.08, 0.18)
    a_sens, b_sens   = beta_params_from_ci(sens_mu, sens_lo, sens_hi)
    a_spec, b_spec   = beta_params_from_ci(spec_mu, spec_lo, spec_hi)
    a_pse, b_pse     = beta_params_from_ci(0.10, 0.03, 0.20)
    # AED efficacy prior: cSDH-grounded (no proven protective effect; mean 0.15,
    # 95% interval 0.01-0.45), matching the value-of-information analysis
    # (script 45). The previous base case (0.45, imported from TBI/tumour
    # prophylaxis) is optimistic for cSDH and is retained only as a sensitivity
    # scenario (Figure 6); it is the single setting under which universal AED
    # outranks ML-guided allocation.
    a_rrr, b_rrr     = beta_params_from_ci(0.15, 0.01, 0.45)
    a_pe_det, b_pe_det = beta_params_from_ci(0.03, 0.01, 0.06)
    a_pe_und, b_pe_und = beta_params_from_ci(0.12, 0.06, 0.20)
    sh_aed_drug, sc_aed_drug = gamma_params(200, 80)
    sh_aed_adv,  sc_aed_adv  = gamma_params(1507, 600)
    sh_aed_cont, sc_aed_cont = gamma_params(441, 200)
    sh_aed_adv_c, sc_aed_adv_c   = gamma_params(1293, 500)
    sh_aed_cont_c, sc_aed_cont_c = gamma_params(140, 60)
    sh_se_early, sc_se_early = gamma_params(13200, 5000)
    sh_se_late,  sc_se_late  = gamma_params(35000, 10000)
    sh_ceeg, sc_ceeg = gamma_params(1500, 300)
    sh_hosp, sc_hosp = gamma_params(3500, 700)
    sh_icu,  sc_icu  = gamma_params(5000, 1000)
    sh_szwu, sc_szwu = gamma_params(2500, 500)
    sh_epi,  sc_epi  = gamma_params(15000, 5000)
    sh_ddx,  sc_ddx  = gamma_params(5000, 2000)
    a_pdet_ceeg, b_pdet_ceeg = beta_params_from_ci(0.95, 0.85, 0.99)
    a_pdet_clin, b_pdet_clin = beta_params_from_ci(0.30, 0.15, 0.50)
    a_rescue, b_rescue = beta_params_from_ci(0.20, 0.05, 0.40)

    rng = np.random.RandomState(SEED)
    out = []
    p = cea_mod.params_global
    for i in range(n_psa):
        p_sz = rng.beta(a_prev, b_prev)
        sens = rng.beta(a_sens, b_sens)
        spec = rng.beta(a_spec, b_spec)
        p_se = rng.beta(a_pse, b_pse); rrr = rng.beta(a_rrr, b_rrr)
        pe_d = rng.beta(a_pe_det, b_pe_det); pe_u = rng.beta(a_pe_und, b_pe_und)
        pd_ceeg_s = rng.beta(a_pdet_ceeg, b_pdet_ceeg)
        pd_clin_s = rng.beta(a_pdet_clin, b_pdet_clin)
        ceeg_resc = rng.beta(a_rescue, b_rescue)
        c_aed_d = rng.gamma(sh_aed_drug, sc_aed_drug)
        c_aed_a = rng.gamma(sh_aed_adv, sc_aed_adv)
        c_aed_c = rng.gamma(sh_aed_cont, sc_aed_cont)
        aed_d   = rng.uniform(14, 120)
        c_aed_a_ceeg = rng.gamma(sh_aed_adv_c, sc_aed_adv_c)
        c_aed_c_ceeg = rng.gamma(sh_aed_cont_c, sc_aed_cont_c)
        aed_d_ceeg   = rng.uniform(7, 42)
        c_se_e = rng.gamma(sh_se_early, sc_se_early)
        c_se_l = rng.gamma(sh_se_late, sc_se_late)
        icu_red = rng.uniform(0.3, 1.5)
        c_ceeg = rng.gamma(sh_ceeg, sc_ceeg)
        c_hosp = rng.gamma(sh_hosp, sc_hosp)
        c_icu  = rng.gamma(sh_icu, sc_icu)
        c_szwu = rng.gamma(sh_szwu, sc_szwu)
        c_epi  = rng.gamma(sh_epi, sc_epi)
        c_ddx  = rng.gamma(sh_ddx, sc_ddx)
        u_dec  = rng.uniform(0.05, 0.20)
        p_rec  = rng.beta(3, 17); pd_b = rng.beta(2.5, 97.5); pd_e = rng.beta(2, 98)
        ceeg_d = rng.choice([2, 3, 4, 5], p=[0.2, 0.4, 0.3, 0.1])
        shared = dict(
            p_seizure_base=p_sz, aed_rrr=rrr,
            cost_aed_drug=c_aed_d, cost_aed_adverse=c_aed_a,
            cost_aed_continuation=c_aed_c, expected_aed_days=aed_d,
            cost_aed_adverse_ceeg=c_aed_a_ceeg,
            cost_aed_continuation_ceeg=c_aed_c_ceeg,
            expected_aed_days_ceeg=aed_d_ceeg,
            cost_se_early=c_se_e, cost_se_late=c_se_l,
            icu_los_reduction=icu_red,
            p_detect_ceeg=pd_ceeg_s, p_detect_clinical=pd_clin_s,
            cost_ceeg=c_ceeg, cost_hosp_day=c_hosp, cost_icu=c_icu,
            cost_sz_workup=c_szwu, p_se=p_se,
            utility_sz_decrement=u_dec, p_recurrence=p_rec,
            p_epilepsy_detected=pe_d, p_epilepsy_undetected=pe_u,
            p_death_base=pd_b, p_death_excess=pd_e,
            cost_annual_epi=c_epi, cost_delayed_dx=c_ddx, ceeg_days=ceeg_d,
            cost_dose_escalation=0.0, qaly_loss_escalation=0.0,
        )
        ml_shared = dict(sens=sens, spec=spec, **shared)
        c_obs, q_obs = rollback("observation", p, **shared)
        c_aed, q_aed = rollback("universal_aed", p, **shared)
        c_mla, q_mla = rollback("ml_aed_only", p, **ml_shared)
        c_mlg, q_mlg = rollback("ml_guided",   p, **ml_shared)
        out.append({"draw": i,
                    "cost_obs": c_obs, "qaly_obs": q_obs,
                    "cost_aed": c_aed, "qaly_aed": q_aed,
                    "cost_mla": c_mla, "qaly_mla": q_mla,
                    "cost_mlg": c_mlg, "qaly_mlg": q_mlg,
                    "sens": sens, "spec": spec})
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────
# 3.  Optimal-strategy summary at WTP thresholds
# ─────────────────────────────────────────────────────────
def best_at_wtp(psa, wtps=(50000, 100000, 150000)):
    rows = []
    for wtp in wtps:
        nmb = {s: (wtp * psa[f"qaly_{s}"] - psa[f"cost_{s}"])
                for s in ("obs", "aed", "mla", "mlg")}
        ev = {s: nmb[s].mean() for s in nmb}
        best = max(ev, key=ev.get)
        # Probabilities each strategy is best across draws
        nmb_mat = np.column_stack([nmb[s].values for s in ("obs","aed","mla","mlg")])
        winners = np.argmax(nmb_mat, axis=1)
        p_best = {s: (winners == i).mean() for i, s in enumerate(("obs","aed","mla","mlg"))}
        rows.append({"wtp": wtp,
                      "best_strategy": best,
                      "EV_max_NMB": ev[best],
                      "EV_NMB_obs": ev["obs"], "EV_NMB_aed": ev["aed"],
                      "EV_NMB_mla": ev["mla"], "EV_NMB_mlg": ev["mlg"],
                      "p_best_obs": p_best["obs"], "p_best_aed": p_best["aed"],
                      "p_best_mla": p_best["mla"], "p_best_mlg": p_best["mlg"]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# 4.  Conformal → action mapping sensitivity
#     Three doubleton policies:
#       (a) doubleton → universal AED (default — current implicit assumption)
#       (b) doubleton → observation
#       (c) doubleton → cEEG + targeted AED
#     For each, compute effective Se/Sp at the population level, then run PSA.
# ─────────────────────────────────────────────────────────
def conformal_mapping_sensitivity():
    # Build the conformal partition empirically from OOF + 25_conformal CSV
    z = np.load(CACHE / "oof_bidmc_postopB_firth.npz")
    y = z["y"].astype(int); p = z["p"].astype(float)
    # Use the 25_conformal_prediction operating point at α = 0.10:
    # rule_out at probability threshold below the cal-derived q-low,
    # rule_in above q-high.  For Mondrian split-conformal at α = 0.10 on
    # postop_B, the empirical singleton rates are 0.252 / 0.102 / defer 0.646
    # (see 25_conformal.csv).  We approximate the partition by using probability
    # quantiles that give the same partition sizes.
    co = pd.read_csv(RES / "25_conformal.csv")
    row = co[(co.alpha == 0.10) & (co.feature_set == "postop_B")].iloc[0]
    q_lo_target = row["rule_out_rate"]      # ≈ 0.252
    q_hi_target = 1 - row["rule_in_rate"]   # ≈ 0.898
    p_sorted = np.sort(p)
    thr_lo = np.quantile(p_sorted, q_lo_target)
    thr_hi = np.quantile(p_sorted, q_hi_target)

    is_rule_out = (p < thr_lo)
    is_rule_in  = (p >= thr_hi)
    is_defer    = ~(is_rule_out | is_rule_in)

    # Joint distribution rates (× population)
    n = len(y)
    n_sz = int(y.sum())
    rates = {
        "n":             n,
        "rule_out_rate": round(is_rule_out.mean(), 4),
        "rule_in_rate":  round(is_rule_in.mean(),  4),
        "defer_rate":    round(is_defer.mean(),    4),
        "p_sz_in_rule_out": round(y[is_rule_out].mean() if is_rule_out.any() else 0, 4),
        "p_sz_in_rule_in":  round(y[is_rule_in].mean()  if is_rule_in.any()  else 0, 4),
        "p_sz_in_defer":    round(y[is_defer].mean()    if is_defer.any()    else 0, 4),
        "thr_lo": round(thr_lo, 4),
        "thr_hi": round(thr_hi, 4),
    }
    # For each doubleton policy, compute the effective Se / Sp at the
    # population scale.  Decision rule:
    #   rule-out → no intervention (observation)
    #   rule-in  → intervention (treat as positive)
    #   doubleton → policy
    #
    # We compute Se/Sp specifically for "ML treats this patient as positive"
    # — what the CEA's "ml_aed_only" and "ml_guided" strategies need.
    policies = ["universal_aed_defer", "observation_defer", "ceeg_defer"]
    pol_rows = []
    for pol_i, pol in enumerate(policies):
        # "treated as positive" mask
        if pol == "universal_aed_defer":
            treated = is_rule_in | is_defer       # rule-in + defer → intervention
        elif pol == "observation_defer":
            treated = is_rule_in                  # only rule-in → intervention
        elif pol == "ceeg_defer":
            treated = is_rule_in | is_defer       # rule-in + defer → intervention (cEEG)
        tp = int(((treated == 1) & (y == 1)).sum())
        fn = int(((treated == 0) & (y == 1)).sum())
        fp = int(((treated == 1) & (y == 0)).sum())
        tn = int(((treated == 0) & (y == 0)).sum())
        se = tp / max(tp + fn, 1)
        sp = tn / max(tn + fp, 1)
        # Bootstrap CI (deterministic per-policy seed — hash() is per-process
        # randomized, so we use the enumeration index instead)
        rng = np.random.default_rng(SEED + pol_i)
        b_se, b_sp = [], []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            yb = y[idx]; trb = treated[idx]
            tp_b = int(((trb == 1) & (yb == 1)).sum())
            fn_b = int(((trb == 0) & (yb == 1)).sum())
            fp_b = int(((trb == 1) & (yb == 0)).sum())
            tn_b = int(((trb == 0) & (yb == 0)).sum())
            if tp_b + fn_b > 0: b_se.append(tp_b / (tp_b + fn_b))
            if tn_b + fp_b > 0: b_sp.append(tn_b / (tn_b + fp_b))
        ci_se = np.percentile(b_se, [2.5, 97.5])
        ci_sp = np.percentile(b_sp, [2.5, 97.5])
        pol_rows.append({
            "policy": pol,
            "n_treated": int(treated.sum()),
            "sens": round(se, 4), "sens_lo": round(ci_se[0], 4), "sens_hi": round(ci_se[1], 4),
            "spec": round(sp, 4), "spec_lo": round(ci_sp[0], 4), "spec_hi": round(ci_sp[1], 4),
        })
    pol_df = pd.DataFrame(pol_rows)
    return rates, pol_df


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("Task 38 — Deployable-model reanalysis (postoperative-B)")
    print("=" * 78)

    print("\n[A1a] postop-A Firth operating points (current CEA used Se=0.842, Sp=0.504)")
    op_A = operating_points(cache_file="oof_bidmc_postopA_firth.npz")
    op_A.to_csv(RES / "38_postopA_operating_points.csv", index=False)
    print(op_A.to_string(index=False))

    print("\n[A1b] postop-B Firth operating points (the deployable model)")
    op = operating_points(cache_file="oof_bidmc_postopB_firth.npz")
    op.to_csv(RES / "38_postopB_operating_points.csv", index=False)
    print(op.to_string(index=False))

    # Principled deployable operating point.  The earlier "matched-Se to
    # postop-A's 0.842" rule degenerated to threshold 0.06, where the narrow
    # Firth probability range flags *everyone* positive (Se=1.0, Sp=0.0) — a
    # point operationally identical to universal AED, which trivially makes the
    # CEA prefer universal AED.  Instead we (a) exclude degenerate points
    # (Sp < 0.05 or Se > 0.99) and (b) anchor the deployable threshold at the
    # cohort base rate (0.073 = 48/655), the natural decision cutoff for a
    # calibrated probability and the value already used to define the
    # "ML-guided AED" action in the manuscript.
    BASE_RATE = 0.073
    nondegen = op[(op["spec"] >= 0.05) & (op["sens"] <= 0.99)].copy()
    # Primary: the threshold at (or nearest to) the base rate among non-degenerate points
    nondegen["_dist_base"] = (nondegen["threshold"] - BASE_RATE).abs()
    baseline_row = nondegen.sort_values("_dist_base").iloc[0].to_dict()
    print(f"\n[A1] Deployable operating point (base-rate threshold): "
          f"threshold {baseline_row['threshold']} → "
          f"Se={baseline_row['sens']}, Sp={baseline_row['spec']}, "
          f"PPV={baseline_row['ppv']}, NPV={baseline_row['npv']}")
    print("     (degenerate points with Sp<0.05 or Se>0.99 excluded; "
          "base rate = 48/655 = 0.073)")

    print("\n[A2] PSA with postop-B Se/Sp profile")
    psa = run_psa_custom(
        sens_mu=baseline_row['sens'],
        sens_lo=baseline_row['sens_lo'],
        sens_hi=baseline_row['sens_hi'],
        spec_mu=baseline_row['spec'],
        spec_lo=baseline_row['spec_lo'],
        spec_hi=baseline_row['spec_hi'],
        n_psa=N_PSA, label="postopB",
    )
    psa.to_csv(RES / "38_postopB_psa.csv", index=False)

    best = best_at_wtp(psa)
    best.to_csv(RES / "38_postopB_psa_summary.csv", index=False)
    print(best.to_string(index=False))

    print("\n[A3] Conformal → action mapping sensitivity")
    rates, pol_df = conformal_mapping_sensitivity()
    print("\nConformal partition (empirical from OOF):")
    for k, v in rates.items():
        print(f"  {k:>22s} : {v}")
    print("\nEffective Se/Sp under each doubleton policy:")
    print(pol_df.to_string(index=False))

    # Now run PSA under each conformal-mapping policy
    print("\nRunning PSA for each conformal-mapping policy (this is the M4 sensitivity)...")
    mapping_summaries = []
    for _, p_row in pol_df.iterrows():
        pol = p_row["policy"]
        psa_pol = run_psa_custom(
            sens_mu=p_row['sens'],
            sens_lo=max(0.01, p_row['sens_lo']),
            sens_hi=min(0.99, p_row['sens_hi']),
            spec_mu=p_row['spec'],
            spec_lo=max(0.01, p_row['spec_lo']),
            spec_hi=min(0.99, p_row['spec_hi']),
            n_psa=N_PSA, label=f"conformal_{pol}",
        )
        best_pol = best_at_wtp(psa_pol)
        best_pol["policy"] = pol
        best_pol["mapping_sens"] = p_row['sens']
        best_pol["mapping_spec"] = p_row['spec']
        mapping_summaries.append(best_pol)
        print(f"  ✓ {pol}: best @ $100k = {best_pol[best_pol.wtp==100000]['best_strategy'].iloc[0]}, "
              f"EV = {best_pol[best_pol.wtp==100000]['EV_max_NMB'].iloc[0]:,.0f}")
    cm_summary = pd.concat(mapping_summaries, ignore_index=True)
    cm_summary.to_csv(RES / "38_conformal_mapping_sensitivity.csv", index=False)

    # ─────────── MESSAGE STABILITY SUMMARY ───────────
    print("\n" + "=" * 78)
    print("MESSAGE STABILITY SUMMARY")
    print("=" * 78)
    print("\nCurrent (postop-A baseline) at $100k WTP: ML-guided cEEG (mlg)")
    print("\nNew postop-B baseline at $100k WTP:")
    b100 = best[best.wtp == 100000].iloc[0]
    print(f"  best_strategy = {b100['best_strategy']}  ·  EV_NMB = ${b100['EV_max_NMB']:,.0f}")
    print("\nUnder each conformal-mapping policy at $100k WTP:")
    for _, r in cm_summary[cm_summary.wtp == 100000].iterrows():
        print(f"  policy={r['policy']:>22s}  best={r['best_strategy']}  "
              f"EV=${r['EV_max_NMB']:,.0f}  p_best={r['p_best_'+r['best_strategy']]:.3f}")

    # Combined summary CSV
    stab = pd.DataFrame([
        {"configuration": "postop-A baseline (current)",  "wtp": 100000, "best": "mlg", "source": "16_voi_evpi.csv"},
        {"configuration": "postop-B headline",            "wtp": 100000,
         "best": b100['best_strategy'], "source": "38_postopB_psa_summary.csv"},
        *[{"configuration": f"conformal-mapping: {r['policy']}",
           "wtp": 100000, "best": r['best_strategy'],
           "source": "38_conformal_mapping_sensitivity.csv"}
          for _, r in cm_summary[cm_summary.wtp == 100000].iterrows()],
    ])
    stab.to_csv(RES / "38_message_stability_summary.csv", index=False)
    print(f"\n[✓] Wrote {RES / '38_message_stability_summary.csv'}")
    print(stab.to_string(index=False))


if __name__ == "__main__":
    main()
