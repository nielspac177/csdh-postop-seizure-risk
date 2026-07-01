"""Task 38c — Scenario cost-effectiveness acceptability curves.

The optimal management strategy after cSDH evacuation is assumption-dependent.
This script runs the deployable-model PSA (postoperative-B operating point,
Se 0.50 / Sp 0.70) under three clearly-labelled assumption scenarios and writes
the cost-effectiveness acceptability curve (probability each strategy is optimal
versus willingness-to-pay) for each, so a single figure can show how the
preferred strategy changes with what one assumes about AED efficacy, AED harm
and the cost/yield of continuous EEG monitoring.

Scenarios (priors differ ONLY in the three knobs; every other prior is identical
to the headline PSA in script 38):
  A  cSDH-grounded (base case)        AED RRR ~0.15 (no proven effect),
                                       cEEG cost-effective, AED disutility 0.02
  B  AED effective, monitoring costly AED RRR ~0.25, cEEG cost x2.5,
                                       AED disutility 0.02
  C  Optimistic AED (least supported) AED RRR ~0.45 (TBI/tumour-imported),
                                       AED disutility 0.0, cEEG cost-effective

Output: results/38c_scenario_ceacs.csv
"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np
import pandas as pd
from _shared import RES, SEED as RANDOM_STATE

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cea_mod", HERE / "10_11_cea_pairwise.py")
cea_mod = importlib.util.module_from_spec(spec)
sys.modules["cea_mod"] = cea_mod
spec.loader.exec_module(cea_mod)
rollback = cea_mod.run_strategy
beta_params_from_ci = cea_mod.beta_params_from_ci
gamma_params = cea_mod.gamma_params

N_PSA = 10_000
SEED = RANDOM_STATE
# deployable postop-B operating point (base-rate threshold 0.073)
SENS_MU, SENS_LO, SENS_HI = 0.50, 0.3617, 0.6364
SPEC_MU, SPEC_LO, SPEC_HI = 0.7018, 0.6639, 0.7372
WTPS = np.arange(0, 200_001, 5_000)
STRATS = ["obs", "aed", "mla", "mlg"]


def scenario_psa(rrr_ci, ceeg_cost_mult, aed_disutil, n_psa=N_PSA):
    """PSA identical to script 38 except for the three scenario knobs."""
    a_prev, b_prev   = beta_params_from_ci(0.12, 0.08, 0.18)
    a_sens, b_sens   = beta_params_from_ci(SENS_MU, SENS_LO, SENS_HI)
    a_spec, b_spec   = beta_params_from_ci(SPEC_MU, SPEC_LO, SPEC_HI)
    a_pse, b_pse     = beta_params_from_ci(0.10, 0.03, 0.20)
    a_rrr, b_rrr     = beta_params_from_ci(*rrr_ci)          # scenario knob 1
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
    p = cea_mod.params_global
    rows = []
    for _ in range(n_psa):
        p_sz = rng.beta(a_prev, b_prev)
        sens = rng.beta(a_sens, b_sens); spec = rng.beta(a_spec, b_spec)
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
        c_ceeg = rng.gamma(sh_ceeg, sc_ceeg) * ceeg_cost_mult   # scenario knob 2
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
            utility_aed_decrement=aed_disutil,                  # scenario knob 3
            p_epilepsy_detected=pe_d, p_epilepsy_undetected=pe_u,
            p_death_base=pd_b, p_death_excess=pd_e,
            cost_annual_epi=c_epi, cost_delayed_dx=c_ddx, ceeg_days=ceeg_d,
            cost_dose_escalation=0.0, qaly_loss_escalation=0.0,
        )
        ml_shared = dict(sens=sens, spec=spec, **shared)
        c_obs, q_obs = rollback("observation",   p, **shared)
        c_aed, q_aed = rollback("universal_aed", p, **shared)
        c_mla, q_mla = rollback("ml_aed_only",   p, **ml_shared)
        c_mlg, q_mlg = rollback("ml_guided",     p, **ml_shared)
        rows.append((c_obs, q_obs, c_aed, q_aed, c_mla, q_mla, c_mlg, q_mlg))
    cols = ["cost_obs", "qaly_obs", "cost_aed", "qaly_aed",
            "cost_mla", "qaly_mla", "cost_mlg", "qaly_mlg"]
    return pd.DataFrame(rows, columns=cols)


def ceac(psa):
    """Probability each strategy is optimal across the WTP grid."""
    out = {s: [] for s in STRATS}
    for w in WTPS:
        nmb = np.column_stack([w * psa[f"qaly_{s}"].values - psa[f"cost_{s}"].values
                               for s in STRATS])
        win = np.argmax(nmb, axis=1)
        for i, s in enumerate(STRATS):
            out[s].append((win == i).mean())
    return out


SCENARIOS = [
    ("A_base",       dict(rrr_ci=(0.15, 0.01, 0.45), ceeg_cost_mult=1.0,  aed_disutil=0.02)),
    ("B_ml_aed",     dict(rrr_ci=(0.20, 0.12, 0.30), ceeg_cost_mult=2.5,  aed_disutil=0.05)),
    ("C_universal",  dict(rrr_ci=(0.45, 0.30, 0.60), ceeg_cost_mult=1.0,  aed_disutil=0.0)),
]


def main():
    rows = []
    for name, knobs in SCENARIOS:
        psa = scenario_psa(**knobs)
        cv = ceac(psa)
        at100 = {s: cv[s][list(WTPS).index(100_000)] for s in STRATS}
        winner = max(at100, key=at100.get)
        print(f"[{name}] winner @ $100k = {winner}  "
              + "  ".join(f"{s}:{at100[s]:.2f}" for s in STRATS))
        for j, w in enumerate(WTPS):
            rows.append({"scenario": name, "wtp": w,
                         **{f"p_{s}": cv[s][j] for s in STRATS}})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "38c_scenario_ceacs.csv", index=False)
    print(f"[OK] wrote {RES / '38c_scenario_ceacs.csv'}")


if __name__ == "__main__":
    main()
