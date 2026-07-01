"""Task 44 — Address adversarial findings F7 and F9.

F7 (model-vs-random comparator): the red team argued that, under AED efficacy ≈ 0,
"ML-guided allocation wins" might only mean "treats fewer people with a useless drug."
To isolate the value of the model's DISCRIMINATION from the value of its treated
fraction, we add a 'random allocation at the matched treated fraction' comparator
(same number of patients treated, chosen without information: Se = f, Sp = 1 − f) and
report the discrimination premium = NMB(ML-guided) − NMB(random-matched) across AED
relative-risk-reduction scenarios.

F9 (conformal on the deployed model): recompute the class-conditional (Mondrian)
split-conformal partition on the DEPLOYED Firth/postoperative-B out-of-fold predictions
(the manuscript previously reported the BalancedRandomForest/postop-A partition).

Outputs:
  results/44_model_vs_random.csv
  results/44_conformal_postopB_firth.csv
"""
import os, sys, importlib.util
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from _shared import RES, CACHE

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cea_mod", HERE / "10_11_cea_pairwise.py")
cea_mod = importlib.util.module_from_spec(spec); sys.modules["cea_mod"] = cea_mod
spec.loader.exec_module(cea_mod)
run_strategy = cea_mod.run_strategy
P = cea_mod.params_global

WTP = 100_000
SENS, SPEC = 0.50, 0.70           # deployed operating point (threshold 7.3%)
PREV = P.p_seizure_base           # CEA prior prevalence (0.12)


def nmb(strategy, **over):
    c, q = run_strategy(strategy, P, **over)
    return WTP * q - c


def model_vs_random():
    # treated fraction of the ML rule at this operating point
    f = SENS * PREV + (1 - SPEC) * (1 - PREV)
    # random allocation treating the SAME fraction, without information:
    #   among true positives, fraction f treated -> Se = f
    #   among true negatives, fraction f treated -> 1 - Sp = f -> Sp = 1 - f
    se_r, sp_r = f, 1 - f
    rows = []
    for rrr in [0.0, 0.15, 0.30, 0.45]:
        vals = {
            "observation":   nmb("observation",   aed_rrr=rrr),
            "universal_aed": nmb("universal_aed", aed_rrr=rrr),
            "ml_guided":     nmb("ml_guided",     aed_rrr=rrr, sens=SENS, spec=SPEC),
            "random_matched": nmb("ml_guided",    aed_rrr=rrr, sens=se_r, spec=sp_r),
        }
        premium = vals["ml_guided"] - vals["random_matched"]
        best = max(vals, key=vals.get)
        rows.append({"aed_rrr": rrr, "treated_fraction": round(f, 3),
                     "NMB_obs": round(vals["observation"], 1),
                     "NMB_universal_aed": round(vals["universal_aed"], 1),
                     "NMB_ml_guided": round(vals["ml_guided"], 1),
                     "NMB_random_matched": round(vals["random_matched"], 1),
                     "discrimination_premium": round(premium, 1),
                     "best_strategy": best})
    return pd.DataFrame(rows)


def mondrian_conformal(alpha=0.10):
    """Class-conditional split-conformal partition on the deployed Firth postop-B OOF.
    Nonconformity: for a true positive, 1 − p; for a true negative, p. Label-conditional
    (1−α) quantiles with finite-sample correction give per-class thresholds; a test point's
    set includes each label whose nonconformity is within that label's threshold."""
    z = np.load(CACHE / "oof_bidmc_postopB_firth.npz")
    y = z["y"].astype(int); p = z["p"].astype(float)
    s1 = 1 - p[y == 1]          # nonconformity of true positives
    s0 = p[y == 0]              # nonconformity of true negatives
    n1, n0 = len(s1), len(s0)
    q1 = np.quantile(s1, min(1.0, np.ceil((n1 + 1) * (1 - alpha)) / n1))
    q0 = np.quantile(s0, min(1.0, np.ceil((n0 + 1) * (1 - alpha)) / n0))
    incl1 = (1 - p) <= q1       # include label "seizure"
    incl0 = p <= q0             # include label "no-seizure"
    rule_in  = incl1 & ~incl0   # singleton {seizure}
    rule_out = incl0 & ~incl1   # singleton {no-seizure}
    doubleton = incl1 & incl0
    empty = ~incl1 & ~incl0
    # empirical class-conditional coverage
    cov1 = incl1[y == 1].mean()
    cov0 = incl0[y == 0].mean()
    n = len(y)
    return pd.DataFrame([{
        "alpha": alpha, "model": "firth_postopB (deployed)", "n": n,
        "rule_out_rate": round(rule_out.mean(), 4),
        "rule_in_rate": round(rule_in.mean(), 4),
        "singleton_rate": round((rule_out | rule_in).mean(), 4),
        "doubleton_rate": round(doubleton.mean(), 4),
        "empty_rate": round(empty.mean(), 4),
        "coverage_class1": round(cov1, 4),
        "coverage_class0": round(cov0, 4),
    }])


def main():
    mr = model_vs_random()
    mr.to_csv(RES / "44_model_vs_random.csv", index=False)
    print("[F7] Model-vs-random comparator (matched treated fraction; WTP $100k):")
    print(mr.to_string(index=False))
    print(f"\n  -> discrimination premium positive => the MODEL adds value beyond "
          f"treating {mr['treated_fraction'].iloc[0]:.0%} at random.")

    cf = pd.concat([mondrian_conformal(a) for a in (0.05, 0.10, 0.20)], ignore_index=True)
    cf.to_csv(RES / "44_conformal_postopB_firth.csv", index=False)
    print("\n[F9] Conformal partition on the DEPLOYED Firth postop-B model:")
    print(cf.to_string(index=False))


if __name__ == "__main__":
    main()
