"""Task 16 — Value-of-information (EVPI / EVPPI) analysis for the CEA.

EVPI quantifies the maximum expected health-and-cost-equivalent value of
resolving all parameter uncertainty:

    EVPI(λ) = E_θ [max_d NMB(d, θ; λ)] − max_d E_θ NMB(d, θ; λ)

EVPPI(φ; λ) is the analogue restricted to a single parameter subset φ, the
expected value of perfectly knowing φ. We estimate it via the Strong-Oakley
(MDM 2014) non-parametric regression approach: a smooth fit of NMB on the
focal parameter, with the fitted values approximating E[NMB | φ].

WTP base case is $100,000/QALY (Neumann 2014; Vanness 2021; Crespo 2023).
We also report EVPI at $50k and $150k.

Outputs:
  results/16_voi_evpi.csv
  results/16_voi_evppi.csv
  figures/16_voi_evpi.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Reuse PSA helpers from the existing CEA module
from importlib.util import spec_from_file_location, module_from_spec
_HERE = os.path.dirname(__file__)
_spec = spec_from_file_location("cea_mod", os.path.join(_HERE, "10_11_cea_pairwise.py"))
cea = module_from_spec(_spec); _spec.loader.exec_module(cea)

from _shared import RES, FIG, SEED

# ── Annual incident cSDH-surgery population (Move 3 lit input) ─────────
ANNUAL_POPULATION = 40_000      # midpoint of 30-50k cSDH operations / yr US
DECISION_HORIZON  = 10
DISCOUNT_RATE     = 0.03
WTPs              = (50_000, 100_000, 150_000)
N_PSA_VOI         = 5_000       # 5000 iterations — enough for stable EVPI

STRATEGIES = ["obs", "aed", "mla", "mlg"]   # column suffixes in PSA frame


def run_psa_tracked(n=N_PSA_VOI, seed=SEED):
    """Re-run the existing PSA but also persist the sampled parameters so
    we can fit per-parameter EVPPI regressions."""
    rng = np.random.default_rng(seed)
    a_prev, b_prev = cea.beta_params_from_ci(0.12, 0.05, 0.20)
    a_sens, b_sens = cea.beta_params_from_ci(0.842, 0.722, 0.918)
    a_spec, b_spec = cea.beta_params_from_ci(0.504, 0.439, 0.569)
    a_pse,  b_pse  = cea.beta_params_from_ci(0.10, 0.05, 0.18)
    a_rrr,  b_rrr  = cea.beta_params_from_ci(0.45, 0.25, 0.65)
    a_pe_det, b_pe_det = cea.beta_params_from_ci(0.03, 0.01, 0.06)
    a_pe_und, b_pe_und = cea.beta_params_from_ci(0.12, 0.05, 0.22)
    a_pdet_ceeg, b_pdet_ceeg = cea.beta_params_from_ci(0.95, 0.85, 0.99)
    a_pdet_clin, b_pdet_clin = cea.beta_params_from_ci(0.30, 0.15, 0.50)
    sh_aed_drug, sc_aed_drug = cea.gamma_params(200.0, 50)
    sh_aed_adv, sc_aed_adv   = cea.gamma_params(1507.0, 500)
    sh_aed_cont, sc_aed_cont = cea.gamma_params(441.0, 150)
    sh_aed_adv_c, sc_aed_adv_c   = cea.gamma_params(1293.0, 400)
    sh_aed_cont_c, sc_aed_cont_c = cea.gamma_params(140.0, 50)
    sh_se_early, sc_se_early = cea.gamma_params(13200.0, 4000)
    sh_se_late,  sc_se_late  = cea.gamma_params(35000.0, 10000)
    sh_ceeg, sc_ceeg = cea.gamma_params(1500.0, 500)
    sh_hosp, sc_hosp = cea.gamma_params(3500.0, 700)
    sh_icu,  sc_icu  = cea.gamma_params(5000.0, 1500)
    sh_szwu, sc_szwu = cea.gamma_params(2500.0, 800)
    sh_epi,  sc_epi  = cea.gamma_params(15000.0, 5000)

    rows = []
    p_global = cea.params_global
    for i in range(n):
        ps = dict(
            p_sz     = rng.beta(a_prev, b_prev),
            sens     = rng.beta(a_sens, b_sens),
            spec     = rng.beta(a_spec, b_spec),
            p_se     = rng.beta(a_pse,  b_pse),
            rrr      = rng.beta(a_rrr,  b_rrr),
            pe_d     = rng.beta(a_pe_det, b_pe_det),
            pe_u     = rng.beta(a_pe_und, b_pe_und),
            pd_ceeg  = rng.beta(a_pdet_ceeg, b_pdet_ceeg),
            pd_clin  = rng.beta(a_pdet_clin, b_pdet_clin),
            c_aed_d  = rng.gamma(sh_aed_drug, sc_aed_drug),
            c_aed_a  = rng.gamma(sh_aed_adv,  sc_aed_adv),
            c_aed_c  = rng.gamma(sh_aed_cont, sc_aed_cont),
            aed_d    = rng.uniform(14, 120),
            c_aed_ac = rng.gamma(sh_aed_adv_c,  sc_aed_adv_c),
            c_aed_cc = rng.gamma(sh_aed_cont_c, sc_aed_cont_c),
            aed_dc   = rng.uniform(7, 42),
            c_se_e   = rng.gamma(sh_se_early, sc_se_early),
            c_se_l   = rng.gamma(sh_se_late,  sc_se_late),
            icu_red  = rng.uniform(0.3, 1.5),
            c_ceeg   = rng.gamma(sh_ceeg, sc_ceeg),
            c_hosp   = rng.gamma(sh_hosp, sc_hosp),
            c_icu    = rng.gamma(sh_icu,  sc_icu),
            c_szwu   = rng.gamma(sh_szwu, sc_szwu),
            c_epi    = rng.gamma(sh_epi,  sc_epi),
            u_dec    = rng.uniform(0.05, 0.20),
            p_rec    = rng.beta(3, 17),
            pd_b     = rng.beta(2.5, 97.5),
            pd_e     = rng.beta(2, 98),
            ceeg_d   = float(rng.choice([2, 3, 4, 5], p=[0.2, 0.4, 0.3, 0.1])),
        )
        shared = dict(
            p_seizure_base=ps["p_sz"], aed_rrr=ps["rrr"],
            cost_aed_drug=ps["c_aed_d"], cost_aed_adverse=ps["c_aed_a"],
            cost_aed_continuation=ps["c_aed_c"], expected_aed_days=ps["aed_d"],
            cost_aed_adverse_ceeg=ps["c_aed_ac"],
            cost_aed_continuation_ceeg=ps["c_aed_cc"],
            expected_aed_days_ceeg=ps["aed_dc"],
            cost_se_early=ps["c_se_e"], cost_se_late=ps["c_se_l"],
            icu_los_reduction=ps["icu_red"],
            p_detect_ceeg=ps["pd_ceeg"], p_detect_clinical=ps["pd_clin"],
            cost_ceeg=ps["c_ceeg"], cost_hosp_day=ps["c_hosp"], cost_icu=ps["c_icu"],
            cost_sz_workup=ps["c_szwu"], p_se=ps["p_se"],
            utility_sz_decrement=ps["u_dec"], p_recurrence=ps["p_rec"],
            p_epilepsy_detected=ps["pe_d"], p_epilepsy_undetected=ps["pe_u"],
            p_death_base=ps["pd_b"], p_death_excess=ps["pd_e"],
            cost_annual_epi=ps["c_epi"], ceeg_days=ps["ceeg_d"],
        )
        ml_shared = dict(sens=ps["sens"], spec=ps["spec"], **shared)
        c_o, q_o = cea.run_strategy("observation", p_global, **shared)
        c_a, q_a = cea.run_strategy("universal_aed", p_global, **shared)
        c_ma, q_ma = cea.run_strategy("ml_aed_only", p_global, **ml_shared)
        c_mg, q_mg = cea.run_strategy("ml_guided", p_global, **ml_shared)
        row = dict(ps)
        row.update({
            "cost_obs": c_o, "qaly_obs": q_o,
            "cost_aed": c_a, "qaly_aed": q_a,
            "cost_mla": c_ma, "qaly_mla": q_ma,
            "cost_mlg": c_mg, "qaly_mlg": q_mg,
        })
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"  PSA {i+1:>5d}/{n}", flush=True)
    return pd.DataFrame(rows)


def compute_evpi(psa, wtp):
    """EVPI(λ) per patient and population-scaled."""
    nmb = np.column_stack([
        wtp * psa[f"qaly_{s}"].values - psa[f"cost_{s}"].values
        for s in STRATEGIES
    ])
    per_iter_max = nmb.max(axis=1).mean()
    best_strategy_evb = nmb.mean(axis=0).max()
    evpi = per_iter_max - best_strategy_evb
    # Population discounted EVPI
    discount = sum(1.0 / (1 + DISCOUNT_RATE) ** t for t in range(DECISION_HORIZON))
    pop_evpi = evpi * ANNUAL_POPULATION * discount
    best_idx = int(nmb.mean(axis=0).argmax())
    return {
        "wtp": wtp,
        "best_strategy": STRATEGIES[best_idx],
        "evpi_per_patient": float(evpi),
        "evpi_population": float(pop_evpi),
        "EV_max_NMB": float(per_iter_max),
        "best_NMB": float(best_strategy_evb),
    }


def evppi_strong_oakley(psa, wtp, focal_col, n_knots=5):
    """EVPPI for a single parameter via the Strong & Oakley (2014) GAM-like
    method, approximated here with a univariate smoothing spline fit of NMB on
    the focal parameter (per strategy). Cheaper than full GAM, same shape.
    """
    phi = psa[focal_col].values
    order = np.argsort(phi)
    phi_sorted = phi[order]
    nmb_cond_max = np.zeros(len(psa))
    nmb_strategies = []
    for s in STRATEGIES:
        nmb = wtp * psa[f"qaly_{s}"].values - psa[f"cost_{s}"].values
        nmb_sorted = nmb[order]
        # Smoothing spline (Strong-Oakley use GAM; we use a low-flex spline)
        try:
            sp = UnivariateSpline(phi_sorted, nmb_sorted, k=3,
                                   s=len(phi) * np.var(nmb_sorted))
            fitted = sp(phi_sorted)
        except Exception:
            # fallback: bin-mean if spline fails
            bins = np.linspace(phi_sorted.min(), phi_sorted.max(), n_knots + 1)
            idx = np.digitize(phi_sorted, bins[1:-1])
            fitted = np.array([nmb_sorted[idx == k].mean() if (idx == k).any() else np.nan
                                for k in range(n_knots)])[idx]
        nmb_strategies.append(fitted)
    nmb_strategies = np.column_stack(nmb_strategies)
    nmb_cond_max = nmb_strategies.max(axis=1).mean()
    best_NMB = (np.column_stack([
        wtp * psa[f"qaly_{s}"].values - psa[f"cost_{s}"].values
        for s in STRATEGIES
    ])).mean(axis=0).max()
    evppi = max(nmb_cond_max - best_NMB, 0.0)
    return evppi


def main():
    print("=" * 72)
    print("Task 16 — VOI / EVPI analysis (lit-review-refined WTP $100k base case)")
    print("=" * 72)

    psa = run_psa_tracked(N_PSA_VOI, seed=SEED)
    psa.to_csv(RES / "16_voi_psa_tracked.csv", index=False)

    # EVPI at multiple thresholds ──────────────────────────────────────
    evpi_rows = [compute_evpi(psa, w) for w in WTPs]
    evpi_df = pd.DataFrame(evpi_rows)
    evpi_df["annual_population"] = ANNUAL_POPULATION
    evpi_df["horizon_years"] = DECISION_HORIZON
    evpi_df.to_csv(RES / "16_voi_evpi.csv", index=False)
    print("\nEVPI table:")
    print(evpi_df.round(3).to_string(index=False))

    # EVPPI per parameter at WTP $100k ─────────────────────────────────
    focal_params = [
        ("p_sz",   "Seizure prevalence"),
        ("sens",   "ML model sensitivity"),
        ("spec",   "ML model specificity"),
        ("rrr",    "AED relative risk reduction"),
        ("u_dec",  "QALY decrement per seizure"),
        ("p_se",   "P(status epilepticus | seizure)"),
        ("pd_clin","P(clinical detection)"),
        ("pd_ceeg","P(cEEG detection)"),
        ("c_se_l", "Cost: late status epilepticus"),
        ("c_se_e", "Cost: early status epilepticus"),
        ("c_aed_a","Cost: AED adverse events"),
        ("c_ceeg", "Cost: cEEG monitoring/day"),
        ("c_epi",  "Cost: annual epilepsy care"),
    ]
    wtp_main = 100_000
    evppi_rows = []
    for col, label in focal_params:
        e = evppi_strong_oakley(psa, wtp_main, col)
        discount = sum(1.0 / (1 + DISCOUNT_RATE) ** t for t in range(DECISION_HORIZON))
        e_pop = e * ANNUAL_POPULATION * discount
        evppi_rows.append({"parameter": label, "column": col,
                            "evppi_per_patient": e,
                            "evppi_population": e_pop})
        print(f"  EVPPI[{label:<35s}] = ${e:>9,.0f}/pt  "
              f"(${e_pop/1e6:>6,.1f} M / 10-yr cohort)")
    evppi_df = pd.DataFrame(evppi_rows).sort_values("evppi_per_patient",
                                                      ascending=False)
    evppi_df.to_csv(RES / "16_voi_evppi.csv", index=False)

    # Plot: EVPPI tornado at WTP $100k ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    e = evppi_df.sort_values("evppi_per_patient")
    axes[0].barh(e["parameter"], e["evppi_per_patient"], color="#1f9e8b")
    axes[0].set_xlabel("Per-patient EVPPI ($) at WTP $100k/QALY")
    axes[0].set_title("Per-parameter expected value of\nperfect information (EVPPI)")
    axes[0].grid(axis="x", alpha=0.3)
    # EVPI vs WTP curve
    wtp_grid = np.arange(0, 200_001, 5_000)
    evpis = [compute_evpi(psa, w)["evpi_per_patient"] for w in wtp_grid]
    axes[1].plot(wtp_grid / 1000, evpis, lw=2, color="tab:purple")
    for w in WTPs:
        axes[1].axvline(w / 1000, color="gray", ls=":", lw=1)
    axes[1].set_xlabel("WTP threshold ($1000/QALY)")
    axes[1].set_ylabel("Per-patient EVPI ($)")
    axes[1].set_title("EVPI vs willingness-to-pay")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "16_voi_evpi.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "16_voi_evpi.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Saved: results/16_voi_*.csv  figures/16_voi_evpi.{{png,pdf}}")


if __name__ == "__main__":
    main()
