"""Task 45 — Value-of-information recomputed under the deployable postop-B model (F10).

The earlier VOI used the postop-A operating point (Se 0.842/Sp 0.504) and an AED
relative-risk reduction of 0.45. This recomputes EVPI and per-parameter EVPPI with the
deployable postop-B operating point (Se 0.50/Sp 0.70) and a cSDH-grounded AED-efficacy
prior (no proven effect: mean 0.15, 95% interval 0.01–0.45), so the value-of-information
ranking reflects the model and parameters actually deployed.

EVPPI for each parameter is estimated by the binned conditional-expectation method
(decile bins) — a transparent approximation to Strong–Oakley regression.

Output: results/45_voi_postopB.csv
"""
import os, sys, importlib.util
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from _shared import RES, SEED

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cea_mod", HERE / "10_11_cea_pairwise.py")
cea_mod = importlib.util.module_from_spec(spec); sys.modules["cea_mod"] = cea_mod
spec.loader.exec_module(cea_mod)
run_strategy = cea_mod.run_strategy
P = cea_mod.params_global
beta_ci = cea_mod.beta_params_from_ci
gamma_p = cea_mod.gamma_params

WTP = 100_000
N = 10_000
STRATS = ["observation", "universal_aed", "ml_aed_only", "ml_guided"]
POP = 40_000 * sum(1 / (1.03 ** t) for t in range(10))   # discounted 10-yr population


def main():
    rng = np.random.RandomState(SEED)
    a_prev, b_prev = beta_ci(0.12, 0.08, 0.18)
    a_rrr, b_rrr   = beta_ci(0.15, 0.01, 0.45)     # cSDH-grounded: no proven effect
    a_sens, b_sens = beta_ci(0.50, 0.36, 0.64)     # postop-B operating point
    a_spec, b_spec = beta_ci(0.70, 0.66, 0.74)
    a_pse, b_pse   = beta_ci(0.10, 0.03, 0.20)
    sh_ceeg, sc_ceeg = gamma_p(1500, 300)
    a_udec, b_udec = beta_ci(0.06, 0.02, 0.15)     # AED disutility

    params = {k: np.zeros(N) for k in
              ["prev", "rrr", "sens", "spec", "p_se", "ceeg_cost", "aed_dis"]}
    nmb = {s: np.zeros(N) for s in STRATS}

    for i in range(N):
        prev = rng.beta(a_prev, b_prev); rrr = rng.beta(a_rrr, b_rrr)
        sens = rng.beta(a_sens, b_sens); spec = rng.beta(a_spec, b_spec)
        p_se = rng.beta(a_pse, b_pse);   ceeg = rng.gamma(sh_ceeg, sc_ceeg)
        udec = rng.beta(a_udec, b_udec)
        shared = dict(p_seizure_base=prev, aed_rrr=rrr, p_se=p_se,
                      cost_ceeg=ceeg, utility_sz_decrement=0.10,
                      utility_aed_decrement=udec)
        mlsh = dict(sens=sens, spec=spec, **shared)
        for s, o in [("observation", shared), ("universal_aed", shared),
                     ("ml_aed_only", mlsh), ("ml_guided", mlsh)]:
            c, q = run_strategy(s, P, **o)
            nmb[s][i] = WTP * q - c
        for k, v in [("prev", prev), ("rrr", rrr), ("sens", sens), ("spec", spec),
                     ("p_se", p_se), ("ceeg_cost", ceeg), ("aed_dis", udec)]:
            params[k][i] = v

    M = np.column_stack([nmb[s] for s in STRATS])
    ev_means = M.mean(axis=0)
    evpi = M.max(axis=1).mean() - ev_means.max()

    # per-parameter EVPPI via decile-binned conditional expectation
    rows = []
    for k, x in params.items():
        order = np.argsort(x); bins = np.array_split(order, 10)
        cond_best = np.zeros(N)
        for b in bins:
            cond_means = M[b].mean(axis=0)
            cond_best[b] = cond_means.max()
        evppi = cond_best.mean() - ev_means.max()
        rows.append({"parameter": k, "EVPPI_per_patient": round(max(evppi, 0), 1),
                     "EVPPI_population_M": round(max(evppi, 0) * POP / 1e6, 1)})
    rows.sort(key=lambda r: -r["EVPPI_per_patient"])
    out = pd.DataFrame([{"metric": "EVPI", "EVPPI_per_patient": round(evpi, 1),
                         "EVPPI_population_M": round(evpi * POP / 1e6, 1)}] + rows)
    out.to_csv(RES / "45_voi_postopB.csv", index=False)
    print(f"[F10] VOI under deployable postop-B (WTP ${WTP:,}); "
          f"best-by-mean = {STRATS[int(np.argmax(ev_means))]}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
