"""Task 3 — Decision-curve analysis / net benefit (BIDMC + eICU).

Reads OOF predictions cached by 02_calibration.py.
Outputs:
  results/03_dca_table.csv
  figures/03_dca_curves.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _shared import RES, FIG, CACHE

def net_benefit(y, p, t):
    """Net benefit at threshold t.
    NB = TP/n - FP/n * t/(1-t)
    """
    pred = (p >= t).astype(int)
    n = len(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    if t >= 1.0:
        return float("nan")
    return tp / n - fp / n * (t / (1 - t))

def treat_all_nb(y, t):
    prev = y.mean()
    if t >= 1.0:
        return float("nan")
    return prev - (1 - prev) * (t / (1 - t))

MODELS = ["bidmc_postopA", "bidmc_postopB", "bidmc_preop",
          "eicu_setA", "eicu_setC", "eicu_pure_setC"]
LABELS = {
    "bidmc_postopA": "BIDMC postop A (BRF)",
    "bidmc_postopB": "BIDMC postop B (LR-EN+SMOTE)",
    "bidmc_preop":   "BIDMC preop (RF)",
    "eicu_setA":     "eICU Set A (8 feat)",
    "eicu_setC":     "eICU Set C (103 feat)",
    "eicu_pure_setC":"eICU pure Set C",
}

def main():
    thresholds = np.arange(0.01, 0.91, 0.01)
    rows = []
    cohorts = {}
    for k in MODELS:
        f = CACHE / f"oof_{k}.npz"
        if not f.exists():
            print(f"[skip] {k} (no cache)")
            continue
        z = np.load(f)
        y, p = z["y"], z["p"]
        cohorts[k] = (y, p)
        for t in thresholds:
            rows.append({
                "model": k, "threshold": t,
                "nb_model":     net_benefit(y, p, t),
                "nb_treat_all": treat_all_nb(y, t),
                "nb_treat_none": 0.0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(RES / "03_dca_table.csv", index=False)

    # plots ───────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.ravel()
    for ax, k in zip(axes, MODELS):
        if k not in cohorts:
            ax.axis("off"); continue
        y, p = cohorts[k]
        sub = df[df.model == k]
        ax.plot(sub["threshold"], sub["nb_model"], lw=2, color="tab:blue", label="Model")
        ax.plot(sub["threshold"], sub["nb_treat_all"], lw=1, color="tab:orange", label="Treat all (universal AED)")
        ax.axhline(0, color="black", lw=1, ls=":", label="Treat none")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"{LABELS[k]}  (prev={y.mean():.2%})")
        ax.set_ylim(-0.05, max(0.15, y.mean() + 0.02))
        ax.set_xlim(0, 0.5)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "03_dca_curves.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIG / "03_dca_curves.pdf", bbox_inches="tight")
    plt.close()

    # ── summary table at clinically relevant thresholds ──
    pts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    summary = []
    for k, (y, p) in cohorts.items():
        for t in pts:
            summary.append({
                "model": k, "threshold": t,
                "model_nb":     net_benefit(y, p, t),
                "treat_all_nb": treat_all_nb(y, t),
                "incremental":  net_benefit(y, p, t) - max(0.0, treat_all_nb(y, t)),
            })
    sdf = pd.DataFrame(summary)
    sdf.to_csv(RES / "03_dca_summary_at_thresholds.csv", index=False)
    print("\nNet benefit at common thresholds:")
    print(sdf.round(4).to_string(index=False))
    print("\n[OK] Saved: results/03_dca_*.csv  figures/03_dca_curves.{png,pdf}")

if __name__ == "__main__":
    main()
