"""Task 42 — One-way tornado + international cost transposition (S11).

Reviewer S11: AED efficacy was borrowed from TBI/tumour data; explore cSDH-specific
RRR uncertainty, and repeat the economics under GBP/EUR cost perspectives.

(1) TORNADO: at the deployable operating point (postop-B Se 0.50 / Sp 0.70) and
    WTP $100k/QALY, vary each key parameter across its plausible range and record the
    incremental net monetary benefit of the best ML strategy versus universal AED.
    Parameters whose range flips the sign of that increment are decision-relevant.
(2) INTERNATIONAL: re-evaluate the optimal strategy under US ($50k/$100k/$150k),
    UK NICE (costs x0.79 GBP, WTP 20k/30k) and Eurozone (costs x0.92 EUR, WTP 40k)
    perspectives.

Outputs:
  results/42_aed_tornado.csv
  results/42_international_perspectives.csv
"""
import os, sys, importlib.util
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from _shared import RES

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cea_mod", HERE / "10_11_cea_pairwise.py")
cea_mod = importlib.util.module_from_spec(spec); sys.modules["cea_mod"] = cea_mod
spec.loader.exec_module(cea_mod)
run_strategy = cea_mod.run_strategy
P = cea_mod.params_global

SENS, SPEC = 0.50, 0.70
STRATS = ["observation", "universal_aed", "ml_aed_only", "ml_guided"]
LABEL = {"observation": "obs", "universal_aed": "aed",
         "ml_aed_only": "ml-aed", "ml_guided": "ml-cEEG"}


def nmb(strategy, wtp, cost_scale=1.0, **over):
    o = dict(sens=SENS, spec=SPEC, **over)
    c, q = run_strategy(strategy, P, **o)
    return wtp * q - c * cost_scale


def best(wtp, cost_scale=1.0, **over):
    vals = {s: nmb(s, wtp, cost_scale, **over) for s in STRATS}
    return max(vals, key=vals.get), vals


# Parameter ranges for the tornado: (kwarg, low, high, base, label)
TORNADO = [
    ("aed_rrr",                0.0,  0.45, P.aed_rrr,                "AED efficacy (RRR)"),
    ("utility_aed_decrement",  0.02, 0.15, P.utility_aed_decrement,  "AED disutility"),
    ("p_seizure_base",         0.07, 0.18, P.p_seizure_base,         "Baseline seizure risk"),
    ("cost_ceeg",              900,  2200, P.cost_ceeg_per_day,       "cEEG cost/day"),
    ("cost_se_early",          8000, 20000, P.cost_se_early,          "Status epilepticus cost"),
    ("cost_aed_adverse",       700,  2500, P.cost_aed_adverse,        "AED adverse-event cost"),
    ("ceeg_rescue",            0.05, 0.40, P.ceeg_aed_rescue_rate,    "cEEG rescue rate"),
    ("p_detect_ceeg",          0.85, 0.99, P.p_detect_ceeg,           "cEEG detection prob."),
]
WTP_TORNADO = 100_000


def tornado():
    base_best, base_vals = best(WTP_TORNADO)
    base_incr = max(base_vals["ml_aed_only"], base_vals["ml_guided"]) - base_vals["universal_aed"]
    rows = []
    for kw, lo, hi, base, lbl in TORNADO:
        incrs = {}
        for tag, val in [("low", lo), ("high", hi)]:
            _, v = best(WTP_TORNADO, **{kw: val})
            incrs[tag] = max(v["ml_aed_only"], v["ml_guided"]) - v["universal_aed"]
        swing = abs(incrs["high"] - incrs["low"])
        rows.append({"parameter": lbl, "kwarg": kw, "base": base,
                     "low_val": lo, "high_val": hi,
                     "incr_NMB_low": round(incrs["low"], 1),
                     "incr_NMB_high": round(incrs["high"], 1),
                     "swing": round(swing, 1),
                     "ml_wins_low": incrs["low"] > 0, "ml_wins_high": incrs["high"] > 0})
    df = pd.DataFrame(rows).sort_values("swing", ascending=False)
    df.attrs["base_incr"] = base_incr
    return df


def international():
    perspectives = [
        ("US",  1.00, [50_000, 100_000, 150_000], "USD"),
        ("UK_NICE", 0.79, [20_000, 30_000], "GBP"),
        ("Eurozone", 0.92, [40_000], "EUR"),
    ]
    rows = []
    for name, scale, wtps, cur in perspectives:
        for wtp in wtps:
            b, vals = best(wtp, cost_scale=scale)
            incr = max(vals["ml_aed_only"], vals["ml_guided"]) - vals["universal_aed"]
            rows.append({"perspective": name, "currency": cur, "cost_scale": scale,
                         "wtp_local": wtp, "best_strategy": LABEL[b],
                         "incr_NMB_ml_vs_aed": round(incr, 1),
                         "ml_preferred": incr > 0})
    return pd.DataFrame(rows)


def main():
    t = tornado()
    t.to_csv(RES / "42_aed_tornado.csv", index=False)
    print(f"[S11] One-way tornado — incremental NMB (best ML − universal AED) @ $100k")
    print(f"      base-case increment = {t.attrs['base_incr']:+,.0f}  "
          f"(negative => universal AED preferred at base case)")
    print(t[["parameter", "incr_NMB_low", "incr_NMB_high", "swing",
             "ml_wins_low", "ml_wins_high"]].to_string(index=False))

    intl = international()
    intl.to_csv(RES / "42_international_perspectives.csv", index=False)
    print("\n[S11] International perspectives:")
    print(intl.to_string(index=False))


if __name__ == "__main__":
    main()
