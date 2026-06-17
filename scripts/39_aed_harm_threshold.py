"""Task 39 — AED-harm / AED-efficacy threshold analysis for the deployable model.

Phase-0 found that, at the deployable postop-B operating point (Se 0.50 / Sp 0.70)
and base-case parameters, *universal AED* has the highest expected NMB. That result
is driven by two base-case assumptions the reviewers flagged as uncertain (item S11):

  * AED disutility is tiny — utility_aed_decrement = 0.02 applied only over the ~66
    days on drug, i.e. a lifetime penalty of ~0.0036 QALY.
  * AED efficacy is borrowed from TBI/tumour data — aed_rrr = 0.45.

Because universal AED treats *everyone*, while ML-guided strategies treat only flagged
patients, raising AED disutility (or lowering AED efficacy) penalises universal AED most.
This script finds the threshold values at which harm-sensitive ML allocation becomes the
NMB-optimal strategy — answering "under what (plausible) assumptions is the model worth
using?" deterministically (point estimates; PSA is in script 38).

Output: results/39_aed_harm_threshold.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import RES

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cea_mod", HERE / "10_11_cea_pairwise.py")
cea_mod = importlib.util.module_from_spec(spec)
sys.modules["cea_mod"] = cea_mod
spec.loader.exec_module(cea_mod)
run_strategy = cea_mod.run_strategy
params_global = cea_mod.params_global

# Deployable postop-B operating point (base-rate threshold)
SENS, SPEC = 0.50, 0.70
WTP = 100_000
STRATS = ["observation", "universal_aed", "ml_aed_only", "ml_guided"]
LABEL = {"observation": "obs", "universal_aed": "aed",
         "ml_aed_only": "ml-aed", "ml_guided": "ml-cEEG"}


def nmb(strategy, **over):
    o = dict(sens=SENS, spec=SPEC, **over)
    c, q = run_strategy(strategy, params_global, **o)
    return WTP * q - c


def winner(**over):
    vals = {s: nmb(s, **over) for s in STRATS}
    best = max(vals, key=vals.get)
    return best, vals


def main():
    rows = []
    print("=" * 78)
    print(f"AED-harm / AED-efficacy threshold scan  (postop-B Se={SENS} Sp={SPEC}, "
          f"WTP=${WTP:,})")
    print("=" * 78)

    # ── One-way: AED disutility (utility_aed_decrement) ──
    print("\n[1] Vary AED disutility (base case 0.02); aed_rrr=0.45")
    for u in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        best, vals = winner(utility_aed_decrement=u)
        rows.append({"scan": "aed_disutility", "param": u, "aed_rrr": 0.45,
                     "winner": LABEL[best],
                     **{f"nmb_{LABEL[s]}": round(vals[s], 1) for s in STRATS}})
        mlbest = max(vals["ml_aed_only"], vals["ml_guided"])
        print(f"  u_aed_dec={u:>5}: winner={LABEL[best]:<7} "
              f"aed={vals['universal_aed']:>12,.0f}  "
              f"ml-best={mlbest:>12,.0f}  Δ(ml-aed)={mlbest-vals['universal_aed']:>+10,.0f}")

    # ── One-way: AED efficacy (aed_rrr) ──
    print("\n[2] Vary AED efficacy aed_rrr (base case 0.45); u_aed_dec=0.02")
    for rrr in [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]:
        best, vals = winner(aed_rrr=rrr)
        rows.append({"scan": "aed_rrr", "param": rrr, "aed_rrr": rrr,
                     "winner": LABEL[best],
                     **{f"nmb_{LABEL[s]}": round(vals[s], 1) for s in STRATS}})
        mlbest = max(vals["ml_aed_only"], vals["ml_guided"])
        print(f"  aed_rrr={rrr:>5}: winner={LABEL[best]:<7} "
              f"aed={vals['universal_aed']:>12,.0f}  "
              f"ml-best={mlbest:>12,.0f}  Δ(ml-best)={mlbest-vals['universal_aed']:>+10,.0f}")

    # ── Two-way grid: find the frontier ──
    print("\n[3] Two-way grid (winner at each cell)")
    us = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
    rrrs = [0.45, 0.35, 0.25, 0.15]
    hdr = "  u\\rrr   " + "".join(f"{r:>9}" for r in rrrs)
    print(hdr)
    for u in us:
        line = f"  {u:>6}  "
        for r in rrrs:
            best, vals = winner(utility_aed_decrement=u, aed_rrr=r)
            line += f"{LABEL[best]:>9}"
            rows.append({"scan": "grid", "param": u, "aed_rrr": r, "winner": LABEL[best],
                         **{f"nmb_{LABEL[s]}": round(vals[s], 1) for s in STRATS}})
        print(line)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "39_aed_harm_threshold.csv", index=False)
    print(f"\n[✓] Wrote {RES / '39_aed_harm_threshold.csv'}")

    # Headline: smallest u_aed_dec at base rrr where an ML strategy wins
    flip = df[(df.scan == "aed_disutility") & (df.winner.isin(["ml-aed", "ml-cEEG"]))]
    if len(flip):
        print(f"\nAt base rrr=0.45, ML allocation becomes NMB-optimal once AED disutility "
              f"≥ {flip.param.min()} QALY (base case 0.02).")
    else:
        print("\nNo ML win within the AED-disutility range scanned at rrr=0.45.")


if __name__ == "__main__":
    main()
