"""Tasks 10 & 11 — Incremental ICER ML+EEG vs ML-only AND treatment-failure pathway.

Reuses the CEA model from csdh_cea_modelA_exec.ipynb (cells 1, 2, 4, 6, 12)
to compute:
  (10) Pairwise ML+EEG vs ML-only ICER + CEAC at WTP $50k/$100k/$150k.
  (11) Modified PSA where breakthrough seizures despite AED trigger
       'dose escalation / cEEG-guided optimization' costs uniformly across
       strategies that use AED.

Outputs:
  results/10_pairwise_psa.csv
  results/10_pairwise_summary.csv
  results/11_treatment_failure_psa.csv
  figures/10_pairwise_plane.png
  figures/10_ceac_pairwise.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np, pandas as pd, copy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings; warnings.filterwarnings("ignore")
from _shared import RES, FIG, SEED

ROOT = "/Users/nielspacheco/Desktop/Research/Ogilvy research/Data Chronic Subdural Haematoma"
DATA_PATH = os.path.join(ROOT, "csdh_clean.csv")
RANDOM_STATE = SEED
N_PSA = 10_000

# ──────────────────────────────────────────────────────────────────────
# Helpers (from notebook cell 4)
# ──────────────────────────────────────────────────────────────────────
def beta_params_from_ci(mean, ci_lo, ci_hi):
    if mean <= 0 or mean >= 1: return 1, 1
    se = (ci_hi - ci_lo) / (2 * 1.96)
    var = se**2
    if var <= 0 or var >= mean * (1 - mean):
        var = mean * (1 - mean) * 0.1
    common = mean * (1 - mean) / var - 1
    a = mean * common; b = (1 - mean) * common
    return max(a, 0.5), max(b, 0.5)

def gamma_params(mean, se):
    if se <= 0: se = mean * 0.2
    shape = (mean / se) ** 2
    scale = se ** 2 / mean
    return max(shape, 0.1), max(scale, 0.01)

@dataclass
class Params:
    p_seizure_base: float = 0.12
    aed_rrr: float = 0.45
    model_sensitivity: float = 0.842
    model_specificity: float = 0.504
    p_detect_ceeg: float = 0.95
    p_detect_clinical: float = 0.30
    p_se_given_seizure: float = 0.10
    p_se_undetected_mult: float = 1.5
    p_epilepsy_detected: float = 0.03
    p_epilepsy_undetected: float = 0.12
    cost_delayed_epilepsy_dx: float = 5000.0
    cost_delayed_dx_se: float = 2000.0
    p_recurrence: float = 0.15
    p_epilepsy: float = 0.05
    p_death_baseline: float = 0.025
    p_death_seizure_excess: float = 0.02
    cost_aed_drug_admin: float = 200.0
    cost_aed_adverse: float = 1507.0
    cost_aed_continuation: float = 441.0
    expected_aed_days: float = 66.0
    cost_aed_adverse_ceeg: float = 1293.0
    cost_aed_continuation_ceeg: float = 140.0
    expected_aed_days_ceeg: float = 21.0
    cost_se_early: float = 13200.0
    cost_se_late: float = 35000.0
    mortality_se_early: float = 0.18
    mortality_se_late: float = 0.33
    icu_los_reduction_ceeg: float = 0.8
    p_ceeg_changes_mgmt: float = 0.52
    ceeg_aed_rescue_rate: float = 0.20
    cost_aed_iv_reactive: float = 1000.0
    cost_aed_iv_se: float = 200.0
    cost_ceeg_per_day: float = 1500.0
    cost_hospital_day: float = 3500.0
    cost_icu_day: float = 5000.0
    cost_seizure_workup: float = 2500.0
    cost_annual_epilepsy: float = 15000.0
    cost_ml_tool: float = 50.0
    utility_baseline: float = 0.75
    utility_seizure_free: float = 0.82
    utility_seizure_decrement: float = 0.10
    utility_epilepsy: float = 0.72
    utility_aed_decrement: float = 0.02
    time_horizon_years: int = 10
    discount_rate: float = 0.03
    ceeg_monitoring_days: int = 3
    wtp_threshold: float = 100_000.0
    los_seizure: float = 20.3
    los_no_seizure: float = 9.8
    # NEW (Task 11): dose-escalation / cEEG-guided optimization triggered when
    # AED-treated patient still has breakthrough seizure
    cost_dose_escalation: float = 0.0   # 0 = task-10 baseline; flip on for task-11
    qaly_loss_escalation: float = 0.0

params_global = Params()

# ──────────────────────────────────────────────────────────────────────
# Markov + decision-tree (extracted from cell 6)
# ──────────────────────────────────────────────────────────────────────
def run_markov(initial_state, p, p_recurrence=None, p_epilepsy=None,
               p_death_base=None, p_death_excess=None, extra_initial_cost=0.0):
    pr = p_recurrence if p_recurrence is not None else p.p_recurrence
    pe = p_epilepsy if p_epilepsy is not None else p.p_epilepsy
    pd_base = p_death_base if p_death_base is not None else p.p_death_baseline
    pd_exc  = p_death_excess if p_death_excess is not None else p.p_death_seizure_excess
    T = p.time_horizon_years; dr = p.discount_rate
    state = np.zeros(4); state[initial_state] = 1.0
    total_qaly = 0.0; total_cost = extra_initial_cost
    utilities = [p.utility_seizure_free,
                 p.utility_baseline - p.utility_seizure_decrement,
                 p.utility_epilepsy, 0.0]
    costs = [0.0, p.cost_seizure_workup * 0.5, p.cost_annual_epilepsy, 0.0]
    for t in range(T):
        d_ = 1.0 / (1 + dr) ** t
        for s in range(4):
            total_qaly += state[s] * utilities[s] * d_
            total_cost += state[s] * costs[s] * d_
        new = np.zeros(4)
        new[3] += state[0] * pd_base
        new[0] += state[0] * (1 - pd_base)
        p1 = min(pd_base + pd_exc, 0.99)
        a1 = state[1] * (1 - p1)
        new[3] += state[1] * p1
        new[2] += a1 * pe
        new[1] += a1 * pr
        new[0] += a1 * max(1 - pe - pr, 0)
        p2 = min(pd_base + pd_exc, 0.99)
        new[3] += state[2] * p2
        new[2] += state[2] * (1 - p2)
        new[3] += state[3]
        state = new
    return total_qaly, total_cost

def run_strategy(strategy, p, **o):
    p_sz_base = o.get("p_seizure_base", p.p_seizure_base)
    aed_rrr   = o.get("aed_rrr", p.aed_rrr)
    sens      = o.get("sens", p.model_sensitivity)
    spec      = o.get("spec", p.model_specificity)
    pd_ceeg   = o.get("p_detect_ceeg", p.p_detect_ceeg)
    pd_clin   = o.get("p_detect_clinical", p.p_detect_clinical)
    p_se_val  = o.get("p_se", p.p_se_given_seizure)
    p_se_mult = o.get("p_se_undetected_mult", p.p_se_undetected_mult)
    c_ceeg    = o.get("cost_ceeg", p.cost_ceeg_per_day)
    c_hosp    = o.get("cost_hosp_day", p.cost_hospital_day)
    c_icu     = o.get("cost_icu", p.cost_icu_day)
    c_sz_wu   = o.get("cost_sz_workup", p.cost_seizure_workup)
    c_aed_iv  = o.get("cost_aed_iv", p.cost_aed_iv_reactive)
    c_epi     = o.get("cost_annual_epi", p.cost_annual_epilepsy)
    c_ml      = o.get("cost_ml", p.cost_ml_tool)
    c_delayed = o.get("cost_delayed_dx", p.cost_delayed_epilepsy_dx)
    u_base    = o.get("utility_baseline", p.utility_baseline)
    u_sz_dec  = o.get("utility_sz_decrement", p.utility_seizure_decrement)
    u_aed_dec = o.get("utility_aed_decrement", p.utility_aed_decrement)
    pr        = o.get("p_recurrence", p.p_recurrence)
    pe_det    = o.get("p_epilepsy_detected", p.p_epilepsy_detected)
    pe_undet  = o.get("p_epilepsy_undetected", p.p_epilepsy_undetected)
    pd_b      = o.get("p_death_base", p.p_death_baseline)
    pd_e      = o.get("p_death_excess", p.p_death_seizure_excess)
    ceeg_days = o.get("ceeg_days", p.ceeg_monitoring_days)
    los_sz    = o.get("los_seizure", p.los_seizure)
    los_no    = o.get("los_no_seizure", p.los_no_seizure)
    c_aed_drug = o.get("cost_aed_drug", p.cost_aed_drug_admin)
    c_aed_adv  = o.get("cost_aed_adverse", p.cost_aed_adverse)
    c_aed_cont = o.get("cost_aed_continuation", p.cost_aed_continuation)
    aed_days   = o.get("expected_aed_days", p.expected_aed_days)
    c_aed_total = c_aed_drug + c_aed_adv + c_aed_cont
    c_aed_adv_ceeg  = o.get("cost_aed_adverse_ceeg", p.cost_aed_adverse_ceeg)
    c_aed_cont_ceeg = o.get("cost_aed_continuation_ceeg", p.cost_aed_continuation_ceeg)
    aed_days_ceeg   = o.get("expected_aed_days_ceeg", p.expected_aed_days_ceeg)
    c_aed_total_ceeg = c_aed_drug + c_aed_adv_ceeg + c_aed_cont_ceeg
    c_se_early = o.get("cost_se_early", p.cost_se_early)
    c_se_late  = o.get("cost_se_late", p.cost_se_late)
    icu_los_red = o.get("icu_los_reduction", p.icu_los_reduction_ceeg)
    ceeg_rescue = o.get("ceeg_rescue", p.ceeg_aed_rescue_rate)
    # NEW (Task 11)
    cost_escalation = o.get("cost_dose_escalation", p.cost_dose_escalation)
    qaly_loss_escalation = o.get("qaly_loss_escalation", p.qaly_loss_escalation)

    aed_qaly_penalty = u_aed_dec * (aed_days / 365)
    aed_qaly_penalty_ceeg = u_aed_dec * (aed_days_ceeg / 365)
    p_sz_with_aed = p_sz_base * (1 - aed_rrr)
    los_extra = los_sz - los_no
    cost_sz_detected = c_sz_wu + los_extra * c_hosp + c_aed_iv
    cost_sz_undetected = cost_sz_detected * 1.3
    cost_se_event_ceeg = c_se_early
    cost_se_event_clin = c_se_late
    cost_icu_savings_ceeg = icu_los_red * c_icu
    qaly_sz_detected = (u_base - u_sz_dec * 0.5) * (los_extra / 365)
    qaly_sz_undetected = (u_base - u_sz_dec * 0.8) * (los_extra * 1.3 / 365)
    qaly_no_sz = u_base * (los_no / 365)
    p_m = copy.copy(p)
    p_m.cost_seizure_workup = c_sz_wu
    p_m.cost_annual_epilepsy = c_epi
    p_m.utility_seizure_decrement = u_sz_dec
    markov_q_sz_det, markov_c_sz_det = run_markov(1, p_m, pr, pe_det, pd_b, pd_e)
    markov_q_sz_undet, markov_c_sz_undet = run_markov(1, p_m, pr, pe_undet, pd_b, pd_e,
                                                      extra_initial_cost=c_delayed)
    markov_q_no, markov_c_no = run_markov(0, p_m, pr, pe_det, pd_b, pd_e)

    if strategy == "observation":
        p_sz = p_sz_base; p_det = pd_clin
        sz_det = p_sz * p_det; sz_undet = p_sz * (1 - p_det); no_sz = 1 - p_sz
        acute_cost = (sz_det * (cost_sz_detected + p_se_val * cost_se_event_clin) +
                      sz_undet * (cost_sz_undetected + p_se_val * p_se_mult * cost_se_event_clin))
        acute_qaly = (sz_det * qaly_sz_detected + sz_undet * qaly_sz_undetected + no_sz * qaly_no_sz)
        total_cost = (acute_cost + sz_det * markov_c_sz_det + sz_undet * markov_c_sz_undet +
                      no_sz * markov_c_no)
        total_qaly = (acute_qaly + sz_det * markov_q_sz_det + sz_undet * markov_q_sz_undet +
                      no_sz * markov_q_no)

    elif strategy == "universal_aed":
        p_sz = p_sz_with_aed; p_det = pd_clin
        sz_det = p_sz * p_det; sz_undet = p_sz * (1 - p_det); no_sz = 1 - p_sz
        # Task 11: breakthrough = patient on AED but seized → escalation
        breakthrough = sz_det + sz_undet
        upfront = c_aed_total + breakthrough * cost_escalation
        acute_cost = (upfront +
                      sz_det * (cost_sz_detected + p_se_val * cost_se_event_clin) +
                      sz_undet * (cost_sz_undetected + p_se_val * p_se_mult * cost_se_event_clin))
        acute_qaly = (-aed_qaly_penalty - breakthrough * qaly_loss_escalation +
                      sz_det * qaly_sz_detected + sz_undet * qaly_sz_undetected + no_sz * qaly_no_sz)
        total_cost = (acute_cost + sz_det * markov_c_sz_det + sz_undet * markov_c_sz_undet +
                      no_sz * markov_c_no)
        total_qaly = (acute_qaly + sz_det * markov_q_sz_det + sz_undet * markov_q_sz_undet +
                      no_sz * markov_q_no)

    elif strategy == "ml_aed_only":
        tp = sens * p_sz_base; fn = (1 - sens) * p_sz_base
        fp = (1 - spec) * (1 - p_sz_base); tn = spec * (1 - p_sz_base)
        tp_prevented = tp * aed_rrr
        tp_still = tp * (1 - aed_rrr)
        tp_sz_det = tp_still * pd_clin
        tp_sz_undet = tp_still * (1 - pd_clin)
        breakthrough = tp_still
        cost_tp_prevented = c_aed_total
        cost_tp_sz_det = c_aed_total + cost_sz_detected + p_se_val * cost_se_event_clin + cost_escalation
        cost_tp_sz_undet = c_aed_total + cost_sz_undetected + p_se_val * p_se_mult * cost_se_event_clin + cost_escalation
        qaly_tp_prevented = qaly_no_sz - aed_qaly_penalty
        qaly_tp_sz_det = qaly_sz_detected - aed_qaly_penalty - qaly_loss_escalation
        qaly_tp_sz_undet = qaly_sz_undetected - aed_qaly_penalty - qaly_loss_escalation
        cost_fp = c_aed_total
        qaly_fp = qaly_no_sz - aed_qaly_penalty
        fn_det = fn * pd_clin; fn_undet = fn * (1 - pd_clin)
        cost_fn_det = cost_sz_detected + p_se_val * cost_se_event_clin
        cost_fn_undet = cost_sz_undetected + p_se_val * p_se_mult * cost_se_event_clin
        cost_tn = 0.0; qaly_tn = qaly_no_sz
        acute_cost = (c_ml + tp_prevented * cost_tp_prevented +
                      tp_sz_det * cost_tp_sz_det + tp_sz_undet * cost_tp_sz_undet +
                      fp * cost_fp + fn_det * cost_fn_det + fn_undet * cost_fn_undet + tn * cost_tn)
        acute_qaly = (tp_prevented * qaly_tp_prevented + tp_sz_det * qaly_tp_sz_det +
                      tp_sz_undet * qaly_tp_sz_undet + fp * qaly_fp +
                      fn_det * qaly_sz_detected + fn_undet * qaly_sz_undetected + tn * qaly_tn)
        total_sz_det = tp_sz_det + fn_det
        total_sz_undet = tp_sz_undet + fn_undet
        total_no_sz = tp_prevented + fp + tn
        total_cost = (acute_cost + total_sz_det * markov_c_sz_det +
                      total_sz_undet * markov_c_sz_undet + total_no_sz * markov_c_no)
        total_qaly = (acute_qaly + total_sz_det * markov_q_sz_det +
                      total_sz_undet * markov_q_sz_undet + total_no_sz * markov_q_no)

    elif strategy == "ml_guided":
        tp = sens * p_sz_base; fn = (1 - sens) * p_sz_base
        fp = (1 - spec) * (1 - p_sz_base); tn = spec * (1 - p_sz_base)
        cost_hr_upfront = c_aed_total_ceeg + c_ceeg * ceeg_days
        tp_prev_aed = tp * aed_rrr
        tp_break = tp * (1 - aed_rrr)
        tp_rescued = tp_break * ceeg_rescue
        tp_prevented = tp_prev_aed + tp_rescued
        tp_still = tp_break - tp_rescued
        tp_sz_det = tp_still * pd_ceeg
        tp_sz_undet = tp_still * (1 - pd_ceeg)
        cost_tp_prevented = cost_hr_upfront
        # cEEG-guided optimization is built-in for ml_guided; escalation cost = 0 here
        cost_tp_sz_det = cost_hr_upfront + cost_sz_detected + p_se_val * cost_se_event_ceeg - cost_icu_savings_ceeg
        cost_tp_sz_undet = cost_hr_upfront + cost_sz_undetected + p_se_val * p_se_mult * cost_se_event_clin
        qaly_tp_prevented = qaly_no_sz - aed_qaly_penalty_ceeg
        qaly_tp_sz_det = qaly_sz_detected - aed_qaly_penalty_ceeg
        qaly_tp_sz_undet = qaly_sz_undetected - aed_qaly_penalty_ceeg
        cost_fp = cost_hr_upfront
        qaly_fp = qaly_no_sz - aed_qaly_penalty_ceeg
        fn_det = fn * pd_clin; fn_undet = fn * (1 - pd_clin)
        cost_fn_det = cost_sz_detected + p_se_val * cost_se_event_clin
        cost_fn_undet = cost_sz_undetected + p_se_val * p_se_mult * cost_se_event_clin
        cost_tn = 0.0; qaly_tn = qaly_no_sz
        acute_cost = (c_ml + tp_prevented * cost_tp_prevented +
                      tp_sz_det * cost_tp_sz_det + tp_sz_undet * cost_tp_sz_undet +
                      fp * cost_fp + fn_det * cost_fn_det + fn_undet * cost_fn_undet + tn * cost_tn)
        acute_qaly = (tp_prevented * qaly_tp_prevented + tp_sz_det * qaly_tp_sz_det +
                      tp_sz_undet * qaly_tp_sz_undet + fp * qaly_fp +
                      fn_det * qaly_sz_detected + fn_undet * qaly_sz_undetected + tn * qaly_tn)
        total_sz_det = tp_sz_det + fn_det
        total_sz_undet = tp_sz_undet + fn_undet
        total_no_sz = tp_prevented + fp + tn
        total_cost = (acute_cost + total_sz_det * markov_c_sz_det +
                      total_sz_undet * markov_c_sz_undet + total_no_sz * markov_c_no)
        total_qaly = (acute_qaly + total_sz_det * markov_q_sz_det +
                      total_sz_undet * markov_q_sz_undet + total_no_sz * markov_q_no)
    else:
        raise ValueError(strategy)
    return total_cost, total_qaly

# ──────────────────────────────────────────────────────────────────────
# PSA
# ──────────────────────────────────────────────────────────────────────
def run_psa(escalation_cost=0.0, escalation_qaly=0.0, n_psa=N_PSA, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    a_prev, b_prev = beta_params_from_ci(0.12, 0.08, 0.18)
    a_sens, b_sens = beta_params_from_ci(0.842, 0.50, 0.95)
    a_spec, b_spec = beta_params_from_ci(0.504, 0.30, 0.70)
    a_pse, b_pse   = beta_params_from_ci(0.10, 0.03, 0.20)
    a_rrr, b_rrr   = beta_params_from_ci(0.45, 0.25, 0.65)
    a_pe_det, b_pe_det = beta_params_from_ci(0.03, 0.01, 0.06)
    a_pe_und, b_pe_und = beta_params_from_ci(0.12, 0.06, 0.20)
    sh_aed_drug, sc_aed_drug = gamma_params(200, 80)
    sh_aed_adv, sc_aed_adv   = gamma_params(1507, 600)
    sh_aed_cont, sc_aed_cont = gamma_params(441, 200)
    sh_aed_adv_c, sc_aed_adv_c   = gamma_params(1293, 500)
    sh_aed_cont_c, sc_aed_cont_c = gamma_params(140, 60)
    sh_se_early, sc_se_early = gamma_params(13200, 5000)
    sh_se_late, sc_se_late   = gamma_params(35000, 10000)
    sh_ceeg, sc_ceeg = gamma_params(1500, 300)
    sh_hosp, sc_hosp = gamma_params(3500, 700)
    sh_icu, sc_icu   = gamma_params(5000, 1000)
    sh_szwu, sc_szwu = gamma_params(2500, 500)
    sh_epi, sc_epi   = gamma_params(15000, 5000)
    sh_ddx, sc_ddx   = gamma_params(5000, 2000)
    a_pdet_ceeg, b_pdet_ceeg = beta_params_from_ci(0.95, 0.85, 0.99)
    a_pdet_clin, b_pdet_clin = beta_params_from_ci(0.30, 0.15, 0.50)
    a_rescue, b_rescue = beta_params_from_ci(0.20, 0.05, 0.40)

    out = []
    p = params_global
    for i in range(n_psa):
        p_sz = rng.beta(a_prev, b_prev)
        sens = rng.beta(a_sens, b_sens)
        spec = rng.beta(a_spec, b_spec)
        p_se = rng.beta(a_pse, b_pse)
        rrr  = rng.beta(a_rrr, b_rrr)
        pe_d = rng.beta(a_pe_det, b_pe_det)
        pe_u = rng.beta(a_pe_und, b_pe_und)
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
        p_rec  = rng.beta(3, 17)
        pd_b   = rng.beta(2.5, 97.5)
        pd_e   = rng.beta(2, 98)
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
            cost_dose_escalation=escalation_cost,
            qaly_loss_escalation=escalation_qaly,
        )
        ml_shared = dict(sens=sens, spec=spec, **shared)
        c_o, q_o = run_strategy("observation", p, **shared)
        c_a, q_a = run_strategy("universal_aed", p, **shared)
        c_ma, q_ma = run_strategy("ml_aed_only", p, **ml_shared)
        c_mg, q_mg = run_strategy("ml_guided", p, **ml_shared)
        out.append({
            "cost_obs": c_o, "qaly_obs": q_o,
            "cost_aed": c_a, "qaly_aed": q_a,
            "cost_mla": c_ma, "qaly_mla": q_ma,
            "cost_mlg": c_mg, "qaly_mlg": q_mg,
        })
    df = pd.DataFrame(out)
    # all 6 pairwise comparisons
    pairs = [
        ("mla_obs","mla","obs"),
        ("mla_aed","mla","aed"),
        ("mlg_obs","mlg","obs"),
        ("mlg_aed","mlg","aed"),
        ("mlg_mla","mlg","mla"),     # ← TASK 10's MISSING COMPARISON
        ("aed_obs","aed","obs"),
    ]
    for k,a,b in pairs:
        df[f"dc_{k}"] = df[f"cost_{a}"] - df[f"cost_{b}"]
        df[f"dq_{k}"] = df[f"qaly_{a}"] - df[f"qaly_{b}"]
    return df

def summarize(psa_df, label, wtps=(50_000, 100_000, 150_000)):
    pairs = ["mla_obs","mla_aed","mlg_obs","mlg_aed","mlg_mla","aed_obs"]
    rows = []
    for k in pairs:
        dc = psa_df[f"dc_{k}"]; dq = psa_df[f"dq_{k}"]
        for w in wtps:
            nmb = w*dq - dc
            rows.append({
                "scenario": label, "comparison": k, "wtp": w,
                "dc_mean": dc.mean(), "dc_lo": dc.quantile(0.025), "dc_hi": dc.quantile(0.975),
                "dq_mean": dq.mean(), "dq_lo": dq.quantile(0.025), "dq_hi": dq.quantile(0.975),
                "nmb_mean": nmb.mean(), "p_ce": (nmb > 0).mean(),
            })
    return pd.DataFrame(rows)

def main():
    print("Task 10 — Baseline PSA (no escalation cost)", flush=True)
    psa10 = run_psa(escalation_cost=0.0, escalation_qaly=0.0)
    psa10.to_csv(RES / "10_pairwise_psa.csv", index=False)
    s10 = summarize(psa10, "baseline")
    s10.to_csv(RES / "10_pairwise_summary.csv", index=False)
    print("\nBaseline pairwise summary at $100k WTP:")
    print(s10[s10.wtp == 100_000].round(2).to_string(index=False))

    print("\nTask 11 — PSA with treatment-failure escalation ($1500 cost / 0.005 QALY loss)", flush=True)
    psa11 = run_psa(escalation_cost=1500.0, escalation_qaly=0.005)
    psa11.to_csv(RES / "11_treatment_failure_psa.csv", index=False)
    s11 = summarize(psa11, "with_escalation")
    s11.to_csv(RES / "11_treatment_failure_summary.csv", index=False)
    print("\nWith-escalation pairwise summary at $100k WTP:")
    print(s11[s11.wtp == 100_000].round(2).to_string(index=False))

    # ── CE plane (4 panels, focus on the missing comparison) ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, k, ttl in zip(axes.ravel(),
                          ["mla_obs", "mlg_obs", "mlg_aed", "mlg_mla"],
                          ["ML-AED vs Obs", "ML+cEEG vs Obs", "ML+cEEG vs Univ AED",
                           "ML+cEEG vs ML-AED  (TASK 10)"]):
        ax.scatter(psa10[f"dq_{k}"], psa10[f"dc_{k}"], s=2, alpha=0.2)
        for w, c in [(50_000,"tab:green"),(100_000,"tab:orange"),(150_000,"tab:red")]:
            xs = np.linspace(-0.05, 0.05, 100); ax.plot(xs, w*xs, "--", color=c, lw=0.8, label=f"WTP ${w/1000:.0f}k")
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("ΔQALYs"); ax.set_ylabel("ΔCost ($)")
        ax.set_title(ttl)
        if k == "mlg_mla": ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "10_pairwise_plane.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "10_pairwise_plane.pdf", bbox_inches="tight")
    plt.close()

    # CEAC for the missing pair
    wtp = np.arange(0, 300_001, 5_000)
    ceac = []
    for w in wtp:
        for k, label in [("mla_obs","ML-AED vs Obs"),
                          ("mlg_obs","ML+cEEG vs Obs"),
                          ("mlg_aed","ML+cEEG vs Univ AED"),
                          ("mlg_mla","ML+cEEG vs ML-AED")]:
            nmb = w*psa10[f"dq_{k}"] - psa10[f"dc_{k}"]
            ceac.append({"wtp": w, "comparison": label, "p_ce": float((nmb > 0).mean())})
    cdf = pd.DataFrame(ceac)
    cdf.to_csv(RES / "10_ceac_pairwise.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, sub in cdf.groupby("comparison"):
        ax.plot(sub["wtp"]/1000, sub["p_ce"], lw=2, label=label)
    ax.set_xlabel("WTP threshold ($1000)")
    ax.set_ylabel("P(cost-effective)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 300)
    for w in [50, 100, 150]:
        ax.axvline(w, color="gray", ls=":", lw=1)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title("Cost-effectiveness acceptability curves (BIDMC base case)")
    plt.tight_layout()
    plt.savefig(FIG / "10_ceac_pairwise.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "10_ceac_pairwise.pdf", bbox_inches="tight")
    plt.close()
    print("\n[OK] Saved: results/10_*.csv  results/11_*.csv  figures/10_*.{png,pdf}")

if __name__ == "__main__":
    main()
